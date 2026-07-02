# Deep Research Brief — Absolute Decode-Speed Zenith for gemma-4-26B-A4B + DFlash on Jetson Thor

## Context (what we've already built — do NOT re-suggest these)
Pure-CUDA NVFP4 + DFlash spec-decode server for `google/gemma-4-26B-A4B-it` (26B MoE, 128 experts top-8, ~4B active) on **Jetson Thor sm_110a** (Blackwell, 20 SMs, 228KB smem/SM, TMEM+tcgen05+TMA, ~273 GB/s LPDDR5x, CUDA 13.0). Current champion: **DFlash 118 tok/s (beat vLLM's 107 by +10%), base ~45, tau 13.33, all bit-exact.** We already have: raw-mma.sync TC W4A16 GEMM (in-register FP4 dequant, offline repack, 16B int4 + __ldcs evict-first, WARPS=1 grid-fill) routing all M≤16 GEMMs INCLUDING lm_heads and draft bf16 linears; grouped weight-resident MoE (U=4 prefetch, no atomics); head-packed GQA attention; CUDA graphs; NVFP4 tied lm_head. Bottleneck now MoE (~40%, latency-bound 55% compute, resistant to ILP/16B/cp.async/TC-mma). Measured dead-ends: tcgen05/CUTLASS at M≤16 (padding, M=15 is 60× under compute roof), megakernel (grid-barrier overhead at M=1), cp.async on tc (max-grid-fill hides latency at block level), draft→FP4/FP8 (destroys tau).

## THE MISSION
Find the **complete optimization surface** to drive decode toward the theoretical zenith (our tau-moat ceiling is ~157 tok/s at vLLM's step time; true HBM roofline may allow more). Cover BOTH: (1) incremental kernel grinds we haven't tried, and (2) **non-obvious "black swan" high-value techniques** — algorithmic or systems integrations of the same magnitude as speculative decoding or Marlin that could unlock a step-change. Be exhaustive and specific; cite exact repos/files/papers/PTX.

## RESEARCH AXES (cover all; go deep, quote real code/results)

### 1. Inference-engine internals to mine (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLC-LLM, ExLlamaV2, ktransformers)
For EACH: the exact batch=1 decode path, kernel dispatch, MoE fused-GEMM at low arithmetic intensity, quant kernel (Marlin/Machete/tcgen05), spec-decode integration, KV layout, CUDA-graph/capture strategy, scheduler overhead elimination, any SoC/Jetson/edge-specific paths. What does each do at bs=1 MoE decode that a hand-CUDA server hasn't? llama.cpp/ggml specifically for edge/unified-memory tricks (mmap, tensor-split, the Metal/CUDA unified paths). ktransformers for MoE-offload/expert-caching ideas.

### 2. Speculative-decoding frontier (beyond linear DFlash)
EAGLE-1/2/3, Medusa, Hydra, Lookahead, SpecInfer/tree attention, ngram/prompt-lookup, self-speculative, multi-token-prediction (MTP, DeepSeek), Falcon/Griffin drafters, dynamic draft-length & draft-exit, cascade/staged speculation, block/diffusion drafting SOTA. Which raise ACCEPTANCE (tau) or cut DRAFT COST for a 26B MoE target on a bandwidth-bound edge device? Quantify tau vs draft-cost tradeoffs. Is there a draft architecture that beats our 5-layer qwen3 block-diffusion at tau 13.33? Any way to make the verify cheaper (partial verify, early-exit verify, hierarchical)?

### 3. CUDA / Blackwell / tcgen05 kernel frontier (the "black swan" axis)
Undertested low-level techniques for bs=1..16 on sm_110a: persistent-kernel / megakernel done RIGHT (single grid-resident kernel for the whole decode step — where does the Hazy/FlashDecoding-style persistent approach actually win vs our measured loss?), warp-specialization (producer/consumer warps), TMA + cluster/DSMEM (distributed shared mem across the cluster), tcgen05 for the VERIFY (M=15 padded — is there any regime it wins?), FP8/MXFP4 tricks, split-K/stream-K for the small-N GEMMs, kernel fusion opportunities (norm+proj+... ), async everything. What has NVIDIA published for Blackwell edge inference? Any Thor/Orin-specific SM-arbiter or LPDDR5x tricks.

### 4. MoE-at-bs=1 frontier (our current bottleneck, ~40%, latency-bound)
The hardest problem: 128 experts, top-8, ~1 token/expert at bs=1 verify (M=15 → ~120 assignments). SGLang's fused MoE, FlashInfer trtllm-gen MoE, the moe_align/sort, expert-parallel-at-low-batch, the router+glue elimination, expert weight prefetch/caching, grouped-GEMM tiling for M~1. Is there a fundamentally better MoE decode kernel (persistent per-expert, expert-major streaming, sorted-token megakernel)? KV-cache quantization (FP8/FP4 KV) impact. Activation quantization pitfalls (the 75µs FP4-quant trap). Can the MoE be made compute-throughput-bound instead of latency-bound at bs=1?

### 5. Algorithmic / systems black swans (non-obvious step-changes)
Anything of speculative-decoding magnitude we haven't considered: prefix/KV caching & reuse, quantized/compressed KV, weight sparsity (2:4 structured, activation sparsity, expert pruning at inference), Medusa-style multi-head, lookahead-Jacobi, retrieval/ngram fusion, output-token batching across a stream, continuous/chunked strategies at bs=1, distillation of a better drafter, layer-skip/early-exit/depth-adaptive, FlashNorm and norm-fusion, the theoretical decode-speed ceiling analysis for a 4B-active MoE at 273 GB/s. What's the single highest-EV unexplored idea?

### 6. Theoretical ceiling & measurement
Derive the HBM-roofline decode ceiling for 4B-active NVFP4 params/token at 273 GB/s (× tau). Where is the real ceiling — is 157 conservative? What's the gap between us and physics?

## DELIVERABLE
A PRIORITIZED optimization surface: each idea with (a) expected % decode gain, (b) implementation cost/risk, (c) exact code/paper reference, (d) whether it's incremental-grind or black-swan, (e) does it preserve bit-exact output + the tau moat. Rank by EV (gain × probability ÷ cost). Flag the top 3 "must-try" and the top 1 black-swan. Everything must be compatible with: pure-CUDA, Thor sm_110a, W4A16 correctness, bit-exact output, bf16 draft.
