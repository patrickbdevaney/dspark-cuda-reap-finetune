"""gen_units.py — micro-op golden generators that directly gate individual CUDA kernels.

Deterministic small inputs -> reference outputs, saved as safetensors the host CUDA gate reads.
Covers the two hardest net-new primitives first: FP8 128-block GEMM and HC/Sinkhorn.

  docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" -v $PWD/ref:/ref \
    vllm-dflash-thor:sglang python3 /ref/gen_units.py --out /ref/goldens
"""
import os, sys, argparse
import torch
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="goldens")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gen_fp8_gemm(a.out)
    gen_hc_sinkhorn(a.out)
    print("units written to", a.out)
