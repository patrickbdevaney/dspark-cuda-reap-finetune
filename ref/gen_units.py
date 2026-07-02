"""gen_units.py — micro-op golden generators that directly gate individual CUDA kernels.

Deterministic small inputs -> reference outputs, saved as safetensors the host CUDA gate reads.
Covers the two hardest net-new primitives first: FP8 128-block GEMM and HC/Sinkhorn.

  docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" -v $PWD/ref:/ref \
    vllm-dflash-thor:sglang python3 /ref/gen_units.py --out /ref/goldens
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel as K
from safetensors.torch import save_file


def _round_pow2(x):
    return torch.pow(2.0, torch.ceil(torch.log2(x.clamp_min(1e-30))))


def weight_quant_fp8(W, block=128):
    """Quantize [out,in] bf16 weight to fp8 e4m3 with per-128x128-block power-of-2 scale (e8m0-style)."""
    out, inn = W.shape
    Wb = W.float().view(out // block, block, inn // block, block)
    amax = Wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-6)
    s = _round_pow2(amax / K.FP8_MAX)                               # [out/128,1,in/128,1]
    q = torch.clamp(Wb / s, -K.FP8_MAX, K.FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(out, inn), s.view(out // block, inn // block)


def gen_fp8_gemm(out_dir, M=48, N=256, K_=512):
    torch.manual_seed(7)
    A = torch.randn(M, K_)                                          # bf16 activation
    A_fp8, a_s = K.act_quant(A, 128, None, torch.float32)          # [M,K] fp8, [M,K/128] f32
    W = torch.randn(N, K_) * 0.05
    b_fp8, b_s = weight_quant_fp8(W, 128)                          # [N,K] fp8, [N/128,K/128] f32
    C = K.fp8_gemm(A_fp8, a_s, b_fp8, b_s)                         # reference [M,N] (default bf16)
    save_file({
        "A_fp8": A_fp8.contiguous(), "a_s": a_s.contiguous().float(),
        "B_fp8": b_fp8.contiguous(), "b_s": b_s.contiguous().float(),
        "C_ref": C.contiguous().float(),
        "dims": torch.tensor([M, N, K_], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_fp8_gemm.safetensors"))
    print(f"[fp8_gemm] M={M} N={N} K={K_}  C {tuple(C.shape)} |C|max={C.abs().max():.4f}")


def gen_hc_sinkhorn(out_dir, n=64, hc_mult=4, iters=20, eps=1e-6):
    torch.manual_seed(11)
    mh = (2 + hc_mult) * hc_mult                                    # 24
    mixes = torch.randn(n, mh)
    hc_scale = torch.randn(3) * 0.5
    hc_base = torch.randn(mh) * 0.5
    pre, post, comb = K.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    save_file({
        "mixes": mixes.contiguous().float(), "hc_scale": hc_scale.contiguous().float(),
        "hc_base": hc_base.contiguous().float(),
        "pre": pre.contiguous().float(), "post": post.contiguous().float(),
        "comb": comb.contiguous().float(),
        "params": torch.tensor([n, hc_mult, iters], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_hc_sinkhorn.safetensors"))
    # sanity: comb rows/cols should be ~doubly-stochastic
    print(f"[hc_sinkhorn] n={n} comb row_sum~{comb.sum(-1).mean():.4f} col_sum~{comb.sum(-2).mean():.4f} "
          f"pre {tuple(pre.shape)} post {tuple(post.shape)} comb {tuple(comb.shape)}")


def gen_sparse_attn(out_dir, b=1, m=8, h=4, d=128, n=24, topk=16):
    """Gathered top-k attention with a learnable sink (kernel.py:276-368). Single latent KV shared
    across heads (MLA: 1 kv, h q-heads). Some topk_idxs = -1 (masked)."""
    torch.manual_seed(21)
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    attn_sink = torch.randn(h, dtype=torch.float32)
    # per (b,m): topk indices into [0,n), with a few -1 masks
    idx = torch.randint(0, n, (b, m, topk), dtype=torch.int32)
    mask = torch.rand(b, m, topk) < 0.25
    idx = torch.where(mask, torch.full_like(idx, -1), idx)
    scale = d ** -0.5
    o = K.sparse_attn(q, kv, attn_sink, idx, scale)          # ref -> [b,m,h,d] bf16
    save_file({
        "q": q.contiguous().float(), "kv": kv.contiguous().float(),
        "attn_sink": attn_sink.contiguous(), "topk_idxs": idx.contiguous(),
        "o_ref": o.contiguous().float(),
        "dims": torch.tensor([b, m, h, d, n, topk], dtype=torch.int32),
        "scale": torch.tensor([scale], dtype=torch.float32),
    }, os.path.join(out_dir, "unit_sparse_attn.safetensors"))
    print(f"[sparse_attn] b={b} m={m} h={h} d={d} n={n} topk={topk}  o {tuple(o.shape)} |o|max={o.abs().max():.4f}")


def _rope_ref(x, cos, sin, inverse=False):
    # x:[n,D] (D even), cos/sin:[n,D/2]; matches apply_rotary_emb (interleaved pairs, view_as_complex)
    xc = x.float().reshape(x.shape[0], -1, 2)
    xr, xi = xc[..., 0], xc[..., 1]
    s = -sin if inverse else sin
    yr = xr * cos - xi * s
    yi = xr * s + xi * cos
    return torch.stack([yr, yi], dim=-1).reshape(x.shape)


def gen_rope(out_dir, n=32, rope_dim=64):
    torch.manual_seed(31)
    x = torch.randn(n, rope_dim)
    ang = torch.randn(n, rope_dim // 2)          # arbitrary per-pos angles
    cos, sin = torch.cos(ang), torch.sin(ang)
    y_fwd = _rope_ref(x, cos, sin, False)
    y_inv = _rope_ref(x, cos, sin, True)
    save_file({
        "x": x.contiguous(), "cos": cos.contiguous(), "sin": sin.contiguous(),
        "y_fwd": y_fwd.contiguous(), "y_inv": y_inv.contiguous(),
        "dims": torch.tensor([n, rope_dim], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_rope.safetensors"))
    print(f"[rope] n={n} rope_dim={rope_dim}  fwd|max={y_fwd.abs().max():.4f}")


def gen_rmsnorm(out_dir, n=32, dim=512, eps=1e-6):
    torch.manual_seed(41)
    x = torch.randn(n, dim)
    w = torch.randn(dim) * 0.1 + 1.0
    yw = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * w)
    yn = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps))   # no-weight (per-head q norm)
    save_file({
        "x": x.contiguous(), "weight": w.contiguous(),
        "y_w": yw.contiguous(), "y_now": yn.contiguous(),
        "dims": torch.tensor([n, dim], dtype=torch.int32),
        "eps": torch.tensor([eps], dtype=torch.float32),
    }, os.path.join(out_dir, "unit_rmsnorm.safetensors"))
    print(f"[rmsnorm] n={n} dim={dim}")


def gen_act_quant(out_dir, n=32, dim=512, block=64):
    """Fused FP8 QAT-sim (quant->dequant back to input dtype), ue8m0 pow2 scale (kernel.py act_quant
    inplace=True; used on KV NoPE dims model.py:512)."""
    torch.manual_seed(51)
    x = torch.randn(n, dim, dtype=torch.bfloat16)
    y = K.act_quant(x.clone(), block, "ue8m0", torch.float8_e8m0fnu, inplace=True)   # in-place -> bf16
    save_file({
        "x": x.contiguous().float(), "y_ref": y.contiguous().float(),
        "dims": torch.tensor([n, dim, block], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_act_quant.safetensors"))
    print("[act_quant] n=%d dim=%d block=%d  maxdiff=%.4f" % (n, dim, block, (x.float()-y.float()).abs().max().item()))


_E2M1_MAG = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])   # nibble&7 -> magnitude


def weight_quant_fp4(W, block=32):
    """[N,K] bf16 -> packed fp4 [N,K/2] uint8 (2 nibbles/byte) + per-32 pow2 scale [N,K/32] f32.
    Nibble = sign(bit3) | nearest-E2M1-magnitude-index. Packing: element k even -> low nibble."""
    N, K = W.shape
    Wb = W.float().view(N, K // block, block)
    amax = Wb.abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -126)
    s = _round_pow2(amax / 6.0)
    scaled = torch.clamp((Wb / s).view(N, K), -6.0, 6.0)
    mags = _E2M1_MAG.to(scaled.device)
    idx = (scaled.abs().unsqueeze(-1) - mags).abs().argmin(-1).to(torch.uint8)     # [N,K]
    nib = idx | ((scaled < 0).to(torch.uint8) << 3)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()                      # [N,K/2]
    return packed, s.squeeze(-1).contiguous()


def gen_fp4_gemm(out_dir, M=32, N=128, K_=256):
    """FP8-act x FP4-weight GEMM (kernel.py fp4_gemm) — the MoE expert GEMM."""
    torch.manual_seed(71)
    A = torch.randn(M, K_)
    A_fp8, a_s = K.act_quant(A, 128, None, torch.float32)
    W = torch.randn(N, K_) * 0.1
    b_packed, b_s = weight_quant_fp4(W, 32)
    C = K.fp4_gemm(A_fp8, a_s, b_packed.view(torch.float4_e2m1fn_x2), b_s)
    save_file({
        "A_fp8": A_fp8.contiguous(), "a_s": a_s.contiguous().float(),
        "B_fp4": b_packed, "b_s": b_s.contiguous().float(), "C_ref": C.contiguous().float(),
        "dims": torch.tensor([M, N, K_], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_fp4_gemm.safetensors"))
    print("[fp4_gemm] M=%d N=%d K=%d  |C|max=%.4f" % (M, N, K_, C.abs().max().item()))


def gen_router(out_dir, n=16, dim=256, n_routed=8, topk=2, route_scale=1.5):
    """noaux_tc score router (sqrtsoftplus, bias for selection only, renorm, *route_scale)."""
    torch.manual_seed(81)
    x = torch.randn(n, dim)
    gate_w = torch.randn(n_routed, dim) * 0.1
    bias = torch.randn(n_routed) * 0.1
    scores = (x.float() @ gate_w.float().t())
    scores = F.softplus(scores).sqrt()
    orig = scores
    sel = scores + bias
    idx = sel.topk(topk, dim=-1)[1]
    wts = orig.gather(1, idx)
    wts = wts / wts.sum(-1, keepdim=True) * route_scale
    save_file({
        "x": x.contiguous(), "gate_w": gate_w.contiguous(), "bias": bias.contiguous(),
        "weights_ref": wts.contiguous().float(), "indices_ref": idx.contiguous().int(),
        "dims": torch.tensor([n, dim, n_routed, topk], dtype=torch.int32),
        "route_scale": torch.tensor([route_scale], dtype=torch.float32),
    }, os.path.join(out_dir, "unit_router.safetensors"))
    print("[router] n=%d dim=%d n_routed=%d topk=%d" % (n, dim, n_routed, topk))


def gen_ogroup_gemm(out_dir, bs=8, G=2, R=16, Kd=128):
    """Grouped o-LoRA einsum: out[bs,G,R] = sum_d o[bs,G,d]*wo_a[G,R,d]  (model.py:543-546, bf16)."""
    torch.manual_seed(61)
    o = torch.randn(bs, G, Kd, dtype=torch.bfloat16)
    wo_a = (torch.randn(G, R, Kd, dtype=torch.bfloat16) * 0.05)
    out = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float())
    save_file({
        "o": o.contiguous().float(), "wo_a": wo_a.contiguous().float(),
        "out_ref": out.contiguous().float(),
        "dims": torch.tensor([bs, G, R, Kd], dtype=torch.int32),
    }, os.path.join(out_dir, "unit_ogroup_gemm.safetensors"))
    print(f"[ogroup_gemm] bs={bs} G={G} R={R} Kd={Kd}  |out|max={out.abs().max():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="goldens")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gen_fp8_gemm(a.out)
    gen_hc_sinkhorn(a.out)
    gen_sparse_attn(a.out)
    gen_rope(a.out)
    gen_rmsnorm(a.out)
    gen_act_quant(a.out)
    gen_ogroup_gemm(a.out)
    gen_fp4_gemm(a.out)
    gen_router(a.out)
    print("units written to", a.out)
