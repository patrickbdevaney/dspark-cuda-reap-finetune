# Reverse-engineering vLLM DFlash -> pure-CUDA hybrid (2026-07-01)

## Ground truth (measured this session)
- **vLLM DFlash = 100-105 tok/s** decode (gemma-4-26B-A4B-it-NVFP4, greedy primes, single-stream,
  image vllm/vllm-openai:gemma-aarch64-cu130). The "~110" target is REAL.
- **My pure-CUDA server = 82 tok/s** (DFlash) / 44 base. Gap ~22%.

## vLLM's architecture (from serve_reference.sh + load log + checkpoint inspection)
- MoE/dense: **flashinfer/cutlass FP4 tensor-core GEMM** (fp4_gemm autotuner runs at load).
- Attention: **TRITON_ATTN** (FA2/FA3 incompatible w/ gemma-4 heterogeneous head dims).
- KV cache: **BF16** (DFlash REJECTS quantized KV, vLLM issue #41559 — so fp8-KV is NOT part of the fast path).
- **Full CUDA graphs**: PIECEWISE (mixed prefill-decode) + FULL (decode) — captures the whole step incl. draft.
- **lm_head = BF16** (checkpoint embed is bf16, tied; vLLM does NOT quantize it). num_spec=15 (k=15, block 16).

## The two asymmetric advantages
- **MINE that vLLM lacks:** NVFP4 lm_head (4x lighter, my +28% base / +35% DFlash win). vLLM eats the full 1.5GB bf16.
- **vLLM's that I lack:** full-step CUDA graph + tuned attention/MoE + scheduler overlap. Collectively beats my
  hand-written kernels by ~22% DESPITE its bf16 lm_head.

## Key finding that reframes the plan
- tcgen05 (TC FP4) COMPILES on this box (CUDA 13.0, sm_110a) — a hand-written TC kernel is *possible*.
- BUT flashinfer's TC FP4 at M=128 measured only ~1.5 GFLOP/us = **overhead-bound at small M** (matches research:
  TC below its M-crossover). So a pure-CUDA TC MoE is NOT a guaranteed win at M=15 — the vLLM gap is more likely
  SYSTEM-level (full-step graph + attention + scheduler) than a single missing kernel.

## Paths to >110 (ranked by effort/confidence)
- **(B, highest confidence, fastest):** add the NVFP4 lm_head to the vLLM stack. vLLM 100 + lm_head byte-reduction
  (~+15-25% since bf16 lm_head is ~1.5GB/step) -> likely 115-125. Requires quantizing the tied embed in the
  checkpoint (compressed-tensors nvfp4) or a small vLLM lm_head patch. NOT pure-CUDA.
- **(A, pure-CUDA, high effort):** add full-DFlash-step CUDA graph (draft has GROWING ctx -> device-seqlen refactor)
  + optionally a tcgen05 TC MoE, keeping the FP4 lm_head. Matches vLLM's system wins by hand. Multi-day, uncertain.

## CORRECTION (controlled-ish comparison) — the gap is bigger + it's kernel efficiency
- My server is WORKLOAD-DEPENDENT: 82 tok/s on short/easy primes (80 tok, tau 13.33) but **58 tok/s on longer/
  harder primes (300 tok, tau 10.0)**. The 82 was an EASY-workload number.
- vLLM = 100 tok/s on the 300-tok workload, tau **7.84** (per-position acceptance decays to 0 by pos 12).
- Per-step: mine ~172ms, vLLM ~78ms -> **vLLM's kernels are ~2.2x faster per step**, only partly offset by MY
  higher draft acceptance (10-13 vs 7.84). Net vLLM ~1.7x faster on comparable work.
- ROOT CAUSE = vLLM's mature stack: flashinfer/cutlass FP4 TC GEMM + FULL CUDA graphs (whole step) + tuned
  TRITON_ATTN + scheduler overlap. My hand-written half2 + partial-graph is ~2x behind per-step.
- **My genuine edge that survives: draft acceptance is competitive-to-better, AND the FP4 lm_head (vLLM lacks it).**

## Honest verdict + recommended path
- Matching vLLM's per-step kernels in hand-written CUDA ≈ re-implementing flashinfer+cutlass+vLLM's graph system.
  Not realistic in a reasonable timeframe. My 58-82 is a strong single-author pure-CUDA result but ~1.7x behind
  a mature production stack.
- **Fastest route to >110: add the NVFP4 lm_head to the vLLM stack** (its one untapped lever — bf16 lm_head is
  ~1.5GB/step). vLLM 100 + lm_head byte-reduction -> plausibly 110-120. Leverages vLLM's mature kernels + my
  proven lm_head win. This is path (b), inverted: bring MY win TO vLLM, not vLLM's deps to my server.
