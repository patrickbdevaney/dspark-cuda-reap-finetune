# FP4_COMPUTE_NOTE.md — Thor FP4 compute strategy (shared across dspark-cuda-reap-finetune + gemma-cuda-hybrid)

**Standing decision for both repos' Thor `sm_110a` kernels.** Applies to base-model decode AND the draft-head
decode (DSpark / DFlash) — the compute-bound regimes (prefill, batched capture, spec-decode block verify).

## The fact (verified July 2026, CUDA 13.0)
- **Jetson Thor = 2070 FP4 TFLOPS (sparse) / ~1035 FP8 (dense) — FP4 is its STRONGEST mode** (96 5th-gen Tensor
  Cores, native FP4; ~2× FP8, ~4× fp16). Compute capability = `sm_110` (was `sm_101` in CUDA 12.8/12.9).
- **But CUDA 13.0 `ptxas` does NOT expose the FP4 mma to hand-written PTX for `sm_110/sm_110a`:** every form —
  `mma.sync.kind::f8f6f4`, `tcgen05.*`, block-scaled `mma…kind::mxf4.block_scale` — fails "not supported on
  .target sm_110". NVIDIA's own CUTLASS SM110 FP4 kernel is reported non-functional (dev-forum). FP8 mma
  (`m16n8k32.e4m3`) and fp16 mma ARE supported for hand-PTX.

## The strategy (ranked)
1. **Hand-PTX FP4 on `sm_110a` — the moment `ptxas` exposes it** (imminent; the silicon is there). Keeps the
   pure-hand-rolled ethos and hits 2070 TFLOPS. RE-TEST each CUDA toolkit release: compile a block-scaled
   `mma…kind::mxf4/nvf4.block_scale` for `sm_110a`; when it passes, build the hand kernel. (Our fp4 experts are
   per-32 e8m0 = **MXFP4** block32; the draft could use per-16 e4m3 = NVFP4.)
2. **Until then — the LIGHTEST-WEIGHT library FP4 GEMM: `cuBLASLt`** (`cublasLtMatmul` with `CUDA_R_4F_E2M1`
   operands + block-scale descriptor). It's a single matmul call, minimal footprint — preferred over **cuDNN**
   (heavier, graph/fusion-oriented) and **TensorRT** (whole-graph, too heavy). Use cuBLASLt FP4 for the routed-
   expert + dense GEMMs to actually reach the 2070-TFLOPS tensor cores. Verify cuBLASLt's FP4 path runs on
   `sm_110` (the CUTLASS-SM110-FP4 issue suggests testing before committing).
3. **Fallback / correctness oracle — our hand-rolled kernels** (fp4-storage + FP8/fp16 mma). fp4 *weight storage*
   is always correct (decode is bandwidth-bound; 4-bit weights = min traffic). FP8 mma (`m16n8k32`) is the
   hand-rolled compute ceiling (2× our current fp16 TC) and the near-term win while FP4-compute is library-only.

## Guardrails
- Any FP4-library GEMM must pass the **bit-exact/cosine gate vs our hand-rolled `fp4_gemm` oracle** before it
  ships (correctness gates never loosen — Constitution). A library speed win with wrong numbers is a loss.
- Log the A/B (tok/s, which roofline bound) in `OPTIMIZATION_LEDGER.md` and the rationale in `GATE_LOG.md`.
- **fp4-STORAGE = decode-bandwidth lever (done). fp4-COMPUTE = compute-bound-regime lever (this note).** Keep
  W4A8 for the target (accuracy-preserving); the draft head may push W4A4/NVFP4 aggressively (draft numerics
  only affect τ, not verified output — the target verifies).

## Lineage
The `sm_110a` arch-specific-feature path was pushed in the gemma-cuda-hybrid server work too; this note unifies
the FP4-compute decision so both repos converge on the same lightweight, correct approach as CUDA matures.
