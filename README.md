# dspark-cuda-reap-finetune

**An end-to-end, pure-CUDA pipeline to fine-tune a DeepSeek "DSpark" speculative-decoding draft head onto
a REAP-pruned, NVFP4 180B MoE (`0xSero/DeepSeek-V4-Flash-180B`) and serve the pair at maximum decode on a
single NVIDIA Jetson Thor (Blackwell `sm_110a`) — every kernel hand-rolled and gated bit-exact against a
PyTorch oracle.**

Forked from `gemma-cuda-hybrid` (a from-scratch NVFP4 CUDA server that beat vLLM on Gemma-4; its README is
preserved at `reference/GEMMA_ENGINE_README.md`). This repo re-targets that engine to DeepSeek-V4's very
different architecture (MLA + DSA sparse attention + Hyper-Connections + hash/noaux_tc MoE + FP8/NVFP4 mixed
quant) and adds the **draft-head fine-tuning** pipeline the Gemma project never had.

---

## The North Star

Three products, one thesis: **push CUDA/SIMT down to the exact shape of a model on the exact shape of a
chip — for training *and* inference — and the wins compound.**

1. **An end-to-end pure-CUDA fine-tune of a spec-decode draft head.** Capture on-policy data from the real
   quantized target, train the DSpark head — all in hand-written CUDA. Not because a framework can't, but
   because the point is a *lean, wall-clock-optimal, auditable* reference for how this is actually done at
   the metal. The training-phase setup is chosen from first principles at build time (optimizer, precision,
   data pipeline — e.g. re-evaluating AdamW vs. a 2026-current choice like **Muon** for this specific
   ~3%-trainable head), not inherited by default. Before that phase runs, the capture step is re-reviewed for
   whether it is genuinely wall-clock-optimal and the correct CUDA solution.

2. **An optimal inference server** for the REAP model + fine-tuned head. It inherits `gemma-cuda-hybrid`'s
   abstractions (NVFP4 decode, Marlin-class `mma.sync` GEMM, KV mgmt, OpenAI API, prefix cache) and tunes
   them to the particulars of `sm_110a` interacting with the particulars of *this* model's attention and
   architecture — and of the DSpark head — to maximize speculative-decode decode speed for production
   serving.

3. **A transferable reference.** The end state is a worked example of efficient, wall-time-optimal capture +
   training of DSpark draft heads for **any** model in pure CUDA, and an intermediate step toward training
   such heads *from scratch* for NVFP4 MoE models.

### The transferability principle

What is learned here is not single-use:

- **Across models** — the capture→train→serve methodology for spec-decode draft heads transfers to adjacent
  scenarios on other LLMs.
- **Across NVIDIA devices** — the kernels are `sm_110a`-tuned but the shapes and roofline reasoning port to
  other Blackwell / Hopper / Orin parts via intrinsic substitution + retuning.
- **Across vendors, abstractly** — every major GPU stack is a near-clone of CUDA's SIMT model (threads/warps,
  a shared scratchpad, matrix cores, async copy). The patterns here reduce to SIMT primitives that map to
  HIP/ROCm, MUSA, and others (see `CUDA_ENGINEERING_CONSTITUTION.md`).

### The model-oriented gain

The deeper aim is **expanding the use of CUDA for model-hardware-optimized ML instructions** — writing
inference/training kernels tuned to the *specific* structure of varied model architectures, on my hardware
first and toward other NVIDIA / heterogeneous GPUs next. Each architecture stresses the metal differently and
rewards a hand-tuned instruction stream:

| Architecture | The structural particular to exploit |
|---|---|
| **DeepSeek-V4** (this repo) | MLA latent KV + DSA top-k sparse attention + Hyper-Connections + FP8/NVFP4 mix |
| **Gemma-4** (the fork base) | NVFP4 W4A4, hybrid sliding/global attention, PLE, DFlash block-diffusion draft |
| **Step-3.7-Flash** | sliding-window attention (SWA) |
| **MiniMax-M2.7** | linear / lightning attention |
| **Qwen-3.6 series** | gated delta-net (GDN) |
| **Nemotron-3 series** | Mamba-hybrid (SSM + attention) |

The through-line: **model–hardware co-specialization of the CUDA/SIMT instruction stream**, generalized across
architectures and, ultimately, across devices and vendors.

---

## Status (gate-by-gate, all bit-exact vs the PyTorch oracle)

Every kernel is validated against `ref/` (a pure-PyTorch reimplementation of the reference `model.py`), most
to 0.0 error; the deepest real-weights compositions to ≤1.6%.

**19 kernel primitives** — `fp8_block_gemm`, `fp4_gemm`, `gemm_fp32`, `hc_sinkhorn`, `moe_router_score`,
`sparse_attn`, `rope_interleaved`, `rmsnorm`, `act_quant_{fp8,fp8sim,fp4sim,fp4}`, `ogroup_gemm`, `hadamard`,
`index_score`, `compressor_pool(+overlap)`, `compressor_forward(+rotate)`.

**Composed & validated on REAL 180B-REAP weights** —
`mla_forward` (pure-sliding attn, 0.36%) · `compressed_attn_forward` (MLA + KV-compressor + DSA-indexer, 1.6%)
· `moe_forward` (hash + noaux_tc, bit-exact) · `hc_pre/post` · the **full transformer `block_forward` on
2.76 GB of real weights (0.23%)**.

**Loader** — `ShardedSafeTensors` mmaps the 46-shard checkpoint; footprint measured at 96.02 GiB (published
96.66), 1383 shape-checks pass (Gate-1 memory confirmed).

**Remaining** — YaRN freq precompute · `forward.cu` 43-layer loop → **Gate 1** (decode tok/s) → DSpark head →
**Gate 2** (unfine-tuned τ on REAP — the pivotal go/no-go) → capture → train → serve.

---

## Governing docs
- **`CONSTITUTION.md`** — the charter: two artifacts (portable weights + *preserved* Thor CUDA), the gate
  ladder, front-loaded correctness, no silent failures. The CUDA here is a first-class, version-controlled
  deliverable, never throwaway scratch.
- **`ADAPTATION_PLAN.md`** — Gemma-4 → DeepSeek-V4 arch delta, milestones, research-corrected training recipe.
- **`CAPTURE_TRAIN_PLAN.md`** — research-backed plan to minimize capture+train wall-clock (batching,
  lossless spec-decode bootstrap, cache-once-train-many, disk budget).
- **`reference/DEEPSEEK_V4_MODELING_NOTES.md`** — exact numeric spec (file:line-cited) driving every kernel.
- **`reference/GEMMA_ENGINE_README.md`** · **`CUDA_ENGINEERING_CONSTITUTION.md`** — the inherited engine and
  its kernel-craft discipline, roofline ledger, and SIMT-porting notes.

## Build & gate
```bash
bash scripts/build_gate.sh   && ./build/gate_units ref/goldens                                  # 19 primitives
bash scripts/build_block.sh  && ./build/gate_block ref/goldens/block_layer1_seq16.safetensors   # full Block
bash scripts/build_cmla.sh   && ./build/gate_cmla  ref/goldens/cmla_layer2_seq16.safetensors    # compressed attn
# goldens are generated in the torch container: python3 ref/gen_units.py ; ref/gen_golden.py {mla,block,cmla,...}
```

---
*Jetson Thor · Blackwell sm_110a · CUDA 13 · 128 GB unified LPDDR5x. One model, one chip, hand-tuned end to
end — as a template for many.*
