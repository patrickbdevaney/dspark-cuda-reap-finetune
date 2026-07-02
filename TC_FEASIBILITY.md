# CUTLASS NVFP4 tensor-core on Thor sm_110a / CUDA 13.0 — GO/NO-GO = GREEN (2026-07-01)

## Result: FP4 tensor cores WORK on this box. Blocker cleared.
- Cloned CUTLASS 4.6 (has sm_110 support since 4.2). Compiled example 72a (NVFP4->BF16 GEMM)
  with `nvcc -std=c++17 --expt-relaxed-constexpr -I cutlass/include -I cutlass/tools/util/include
  -gencode arch=compute_110a,code=sm_110a`. COMPILES on CUDA 13.0 (the tcgen05 FP4 MMA assembles).
- The example's host guard whitelists only sm_100/101/103 (major==10); patched line 514 to allow
  major==11 (Thor). Then it RUNS + PASSES correctness on Thor sm_110a.
- **True kernel GPU time (ncu, isolated): 41us at M=16,N=2816,K=2816** (14% SM / 14% mem = latency-bound
  at small M, but a real 41us — the example's 220us "avg runtime" was host measurement overhead).

## Benchmark (CUTLASS example wall-clock, overhead-inflated; ncu true=41us at M16):
    M=16   N=2816 K=704  -> 289 GFLOPS | M=128 -> 2305 | (flat ~220us wall = overhead)
    M=16   N=2816 K=2816 -> 1123 GFLOPS| M=128 -> 8848 | M=256 -> 18487 (ncu true kernel M16=41us)

## Verdict for the hybrid
- TC is a REAL win for the GROUPED verify MoE (my half2 down = 787us/layer @0.6 TFLOP/s; CUTLASS grouped
  MoE ex.92 amortizes 8 experts -> ~5x). NOT a clear win for tiny single dense GEMMs (41us overhead-bound
  vs my ~20us half2).
- Path to beat vLLM (100): CUTLASS grouped-MoE (ex.92) for verify MoE + full-step CUDA graph + keep our
  draft (tau 10-13 vs vLLM 7.84) + keep our NVFP4 lm_head (vLLM eats bf16). All proven-feasible on 13.0.
- Build flags that work: -gencode arch=compute_110a,code=sm_110a, -I cutlass/include + tools/util/include.
  Callable from C++ via flashinfer CutlassFp4GemmRunner (torch-free header) OR direct CUTLASS CollectiveBuilder.

## STEP 3 BENCHMARK (2026-07-01) — the honest per-kernel numbers
- True CUTLASS NVFP4 GEMM kernel time (nsys, validated wrapper): **~11us** at verify shapes
  (8us down M16xN2816xK704, 11us square M128xN2816xK2816).
- DENSE verify GEMMs (M=15 single): CUTLASS 11us vs my half2 ~20us = **~2x faster** -> ~+10% (dense is modest %).
- GROUPED MoE (verify: ~2 tokens/expert): M=2 padded to 128-tile = ~64x compute WASTE, which roughly cancels
  the TC throughput win -> **~break-even with half2's 787us**, NOT the 5x I projected. Padding is the killer
  at tiny per-expert M.
- HONEST VERDICT: CUTLASS TC is a ~+10-15% lever (dense speedup), NOT the 2x needed to reach vLLM's 100.
  The remaining gap is the FULL PIPELINE (vLLM's whole-step graph + tuned attention + every kernel), i.e. the
  "re-implement the stack" the research warned about. Matching vLLM 100 in hand-written CUDA is a much larger
  effort than a single kernel swap.
- REVISED priority: (1) FULL-STEP CUDA GRAPH (pure CUDA, vLLM's key advantage, no CUTLASS needed) is likely the
  bigger single lever; (2) TC dense GEMMs (+10%, primitive validated & ready in cutlass_moe.cu); (3) TC grouped
  MoE only if per-expert token count rises (not at k=15 verify).

## W4A4 ACCURACY TENSION (2026-07-01) — key integration finding
- The CUTLASS TC GEMM is FP4xFP4 (W4A4). Validated full pipeline: quantize bf16 -> E2M1+e4m3 -> swizzle -> TC
  vs fp32 reference = **5.5% rel-L2 error** (the inherent NVFP4 W4A4 error; model is W4A4-native so it's "expected").
- BUT my server uses W4A16 (FP4 weight x FP16 activation) — LESS error (weight-only). My acceptance edge
  (tau 13.3 vs vLLM 7.84) may partly derive from this accuracy. Switching verify dense to W4A4 for TC speed
  risks DROPPING acceptance toward vLLM's -> could negate the ~2x kernel speedup. MUST test acceptance end-to-end.
- IDEAL alternative (keeps W4A16 accuracy + TC speed): MARLIN-style mixed GEMM = dequant FP4 weight -> bf16 in
  SMEM, then bf16xbf16 tcgen05 MMA with the bf16 activation. No activation quant error. Need to check if CUTLASS
  has an FP4-weight x bf16-act mixed-input Blackwell GEMM (ex.55-style mixed dtype), or hand-roll the dequant+MMA.
  (research agent investigating.) This is the RIGHT target, not naive W4A4.
- Foundational integration DONE: cutlass_moe.o linked into hybrid server (callable, gate PASS, DFlash 82).
