"""gen_golden.py — reference golden-activation generator for the CUDA port.

Modes:
  smoke  : tiny random model (dtype=bf16, no real weights) — validates that the pure-torch
           kernel replacements + the full deepseek_v4 architecture run end-to-end
           (HC/Sinkhorn, MLA, sparse_attn, compressor, indexer, MoE, DSpark). No GPU RAM for weights.
  block  : real config from config.json; instantiate embed + ONE Block at --layer, load that
           block's real weights, feed a FIXED-seed input hidden, dump boundary tensors that the
           CUDA kernels gate against (block-in, attn_norm, attn-out, ffn-out, block-out).

  usage:
    python gen_golden.py smoke
    python gen_golden.py block --ckpt /model --config /model/config.json --layer 2 --out goldens/
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local kernel.py / fast_hadamard shadow
import deepseek_v4_ref as M


def init_random_(model, n_routed):
    with torch.no_grad():
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if "tid2eid" in name:                       # hash-route table: valid expert ids
                t.random_(0, n_routed)
            elif t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                t.normal_(0, 0.02)


def smoke():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(dev)
    print(f"[smoke] device={dev}")
    args = M.ModelArgs(
        dtype="bf16", expert_dtype=None, scale_dtype="fp32", scale_fmt=None,
        # head_dim-rope_head_dim and index_head_dim-rope_head_dim must be multiples of 64 (act_quant block)
        dim=512, n_layers=4, n_heads=8, head_dim=128, rope_head_dim=64,
        q_lora_rank=128, o_lora_rank=128, o_groups=4,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
        moe_inter_dim=256, n_hash_layers=1, window_size=16,
        # length must cover layers 0..n_layers-1 PLUS the MTP layer index (n_layers); MTP => ratio 0
        compress_ratios=(0, 0, 4, 128, 0), index_n_heads=8, index_head_dim=128,
        index_topk=16, max_seq_len=256, max_batch_size=1,
        vocab_size=1024, score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
        dspark_block_size=4, dspark_target_layer_ids=(2, 3), dspark_noise_token_id=1000,
        compress_rope_theta=160000.0, original_seq_len=0, rope_theta=10000.0, rope_factor=16,
    )
    model = M.Transformer(args)
    init_random_(model, args.n_routed_experts)
    x = torch.randint(0, args.vocab_size, (1, 64), device=dev)

    out_ids, logits, main_hidden = model(x[:, :48])
    print(f"[smoke] prefill: logits {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()} "
          f"main_hidden {None if main_hidden is None else tuple(main_hidden.shape)}")
    model.forward_spec(out_ids, main_hidden)                      # prefill spec (builds draft KV)
    for i in range(48, 52):
        out_ids, logits, main_hidden = model(x[:, i:i + 1], i)
        spec = model.forward_spec(out_ids, main_hidden, i)
        if spec is not None:
            oids, slog, conf = spec
    print(f"[smoke] decode+spec: draft_ids {tuple(oids.shape)} conf {tuple(conf.shape)} "
          f"finite={torch.isfinite(slog).all().item()}")
    assert torch.isfinite(logits).all(), "non-finite logits"
    print("[smoke] PASS — architecture + swapped kernels run end-to-end.")


def block(ckpt, config, layer, seq, out_dir):
    from safetensors.torch import save_file
    import raw_loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(dev)
    cfg = json.load(open(config))
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})
    args.max_batch_size = 1; args.max_seq_len = max(256, seq * 2)
    print(f"[block] device={dev} layer={layer} compress_ratio={args.compress_ratios[layer]}")

    # build only embed + the single block (cheap); reuse Transformer's globals via a bare build
    M.world_size = 1; M.rank = 0
    M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    M.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    M.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    embed = M.ParallelEmbedding(args.vocab_size, args.dim)
    blk = M.Block(layer, args)

    prefix_e = ["embed."]
    prefix_b = [f"layers.{layer}."]
    # map our bare modules' names to checkpoint names
    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.embed = embed; s.layers = torch.nn.ModuleList()
    w = Wrap()
    # place block at index `layer` so named_parameters -> layers.<layer>.*
    for _ in range(layer): w.layers.append(torch.nn.Module())
    w.layers.append(blk)
    raw_loader.load_state_into(w, ckpt, only_prefixes=prefix_e + prefix_b)

    torch.manual_seed(1234)
    ids = torch.randint(0, args.vocab_size, (1, seq), device=dev)
    g = {}
    h = embed(ids)
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)             # HC expand
    g["block_in"] = h.reshape(1, seq, -1).clone()
    taps = {}
    def hook(name):
        def f(mod, inp, out): taps[name] = (out if isinstance(out, torch.Tensor) else out[0]).detach().clone()
        return f
    blk.attn_norm.register_forward_hook(hook("attn_norm_out"))
    blk.attn.register_forward_hook(hook("attn_out"))
    blk.ffn_norm.register_forward_hook(hook("ffn_norm_out"))
    blk.ffn.register_forward_hook(hook("ffn_out"))
    hout = blk(h, 0, ids)
    g["block_out"] = hout.reshape(1, seq, -1).clone()
    for k, v in taps.items():
        g[k] = v.reshape(1, seq, -1) if v.dim() >= 2 else v
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"golden_layer{layer}_seq{seq}.safetensors")
    save_file({k: v.contiguous().cpu().to(torch.float32) for k, v in g.items()}, path)
    print(f"[block] wrote {path}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in g.items()))


def mla(ckpt, config, layer, seq, out_dir):
    """Real-weights golden for a pure-sliding MLA layer (compress_ratio=0). Runs the reference Attention
    in fp32 (default dtype) so the gate isolates composition error, not bf16 noise. Dumps input + all
    weights (fp8 bytes+scale, bf16 wo_a, norms, sink) + freqs so tests/gate_mla.cu can replay it."""
    from safetensors.torch import save_file
    import raw_loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)          # fp32 activations to match the CUDA path
    torch.set_default_device(dev)
    cfg = json.load(open(config))
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})
    args.max_batch_size = 1; args.max_seq_len = max(256, seq * 2)
    assert args.compress_ratios[layer] == 0, f"layer {layer} is not pure-sliding (ratio={args.compress_ratios[layer]})"
    M.world_size = 1; M.rank = 0
    M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    M.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    M.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32

    attn = M.Attention(layer, args)
    class Wrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.layers = torch.nn.ModuleList()
    w = Wrap()
    for _ in range(layer): w.layers.append(torch.nn.Module())
    holder = torch.nn.Module(); holder.attn = attn; w.layers.append(holder)
    raw_loader.load_state_into(w, ckpt, only_prefixes=[f"layers.{layer}.attn."])
    # wo_a is declared bf16 but used in a raw einsum with fp32 activations -> cast to fp32 (matches CUDA path)
    attn.wo_a.weight.data = attn.wo_a.weight.data.float()

    torch.manual_seed(1234)
    x = torch.randn(1, seq, args.dim)
    o = attn(x, 0)                                   # [1,seq,dim]

    def u8(p): return p.detach().view(torch.uint8).contiguous().cpu()
    def f(p):  return p.detach().float().contiguous().cpu()
    fc = attn.freqs_cis[:seq]                         # complex [seq, rope_dim/2]
    g = {
        "x": f(x[0]), "o_ref": f(o[0]),
        "wq_a": u8(attn.wq_a.weight),   "wq_a_s": f(attn.wq_a.scale),
        "wq_b": u8(attn.wq_b.weight),   "wq_b_s": f(attn.wq_b.scale),
        "wkv":  u8(attn.wkv.weight),    "wkv_s":  f(attn.wkv.scale),
        "wo_b": u8(attn.wo_b.weight),   "wo_b_s": f(attn.wo_b.scale),
        "q_norm": f(attn.q_norm.weight), "kv_norm": f(attn.kv_norm.weight),
        "wo_a": f(attn.wo_a.weight.view(args.o_groups, args.o_lora_rank, -1)),
        "attn_sink": f(attn.attn_sink),
        "cos": torch.view_as_real(fc)[..., 0].contiguous().cpu(),
        "sin": torch.view_as_real(fc)[..., 1].contiguous().cpu(),
        "dims": torch.tensor([1, seq], dtype=torch.int32),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mla_layer{layer}_seq{seq}.safetensors")
    save_file(g, path)
    print(f"[mla] layer={layer} seq={seq} dev={dev} wrote {path}")
    print("   |o_ref|max=%.4f  x%s wq_b%s wo_a%s" % (o.abs().max().item(), tuple(x.shape), tuple(attn.wq_b.weight.shape), tuple(g['wo_a'].shape)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("smoke")
    b = sub.add_parser("block")
    b.add_argument("--ckpt", required=True); b.add_argument("--config", required=True)
    b.add_argument("--layer", type=int, default=2); b.add_argument("--seq", type=int, default=32)
    b.add_argument("--out", default="goldens")
    ml = sub.add_parser("mla")
    ml.add_argument("--ckpt", required=True); ml.add_argument("--config", required=True)
    ml.add_argument("--layer", type=int, default=1); ml.add_argument("--seq", type=int, default=16)
    ml.add_argument("--out", default="goldens")
    a = ap.parse_args()
    if a.mode == "smoke": smoke()
    elif a.mode == "block": block(a.ckpt, a.config, a.layer, a.seq, a.out)
    else: mla(a.ckpt, a.config, a.layer, a.seq, a.out)
