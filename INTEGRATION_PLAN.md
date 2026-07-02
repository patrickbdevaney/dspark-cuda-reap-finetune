# Hybrid integration plan — pure-CUDA server + CUTLASS FP4 TC (toward 110-140 tok/s)

## Baseline & target
- Pure-CUDA server (banked v1.0-pure-cuda in gemma-cuda-server): base 44, DFlash 82(easy)/58(hard).
- vLLM DFlash reference: 100-105 tok/s. Our draft (tau 10-13) BEATS vLLM (7.84); our gap = per-step kernels.
- Target: 110-140 (research + DGX-Spark 108 proxy with a worse draft prove it's reachable).

## VALIDATED (2026-07-01) — all green
- CUTLASS 4.6 FP4 tensor-core GEMM COMPILES + RUNS CORRECT on Thor sm_110a / CUDA 13.0 (ex72a Passed).
  Build: nvcc -std=c++17 --expt-relaxed-constexpr -I ~/cutlass/include -I ~/cutlass/tools/util/include
  -gencode arch=compute_110a,code=sm_110a  (patch the host arch guard: allow props.major==11).
- True single-GEMM kernel time (ncu): 41us @ M=16,N=2816,K=2816 (latency-bound at small M but real).
- FP4 grouped MoE (ex.92 92_..._fp4_grouped.cu) compiles + launches on sm_110a (heavy host reference in the
  example makes wall-clock profiling slow; use the KERNEL not the example harness in the integration).

## Where TC helps (Nsight-measured on the pure server)
- verify DOWN: half2 = 787us/layer @0.6 TFLOP/s (compute-bound, per-element FP4-decode overhead) -> CUTLASS
  grouped FP4 MoE should be several-x faster. BIGGEST single win (~24ms/step -> commit target).
- verify GATEUP grouped: similar.
- verify lm_head: already FP4 + memory-bound 60.9% -> KEEP our kernel (TC won't help).
- small dense linears (M=15 single GEMM): TC 41us vs half2 ~20us -> KEEP half2 (TC overhead-bound at tiny M).

## Integration steps (next work session)
1. Wrapper: extract CUTLASS grouped-FP4-GEMM into a callable `cutlass_nvfp4_grouped_moe(D, A, B, sfa, sfb,
   gscale, per_expert_MNK, num_experts, stream)` from ex.92's CollectiveBuilder (KernelPtrArrayTmaWarpSpecialized,
   OpClassBlockScaledTensorOp, ElementA/B=nv_float4_t<float_e2m1_t>, C/D=bf16, Sm100 ArchTag, 1SM tile 128x128x256).
   Compile into forward.o; link. flashinfer's CutlassFp4GemmRunner (torch-free header) is an alt path.
2. Layout: convert our E2M1 codes + linear E4M3 group-16 scales -> CUTLASS swizzle via
   Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB (header cutlass/detail/sm100_blockscaled_layout.hpp).
   One-time at weight load (like our embed FP4 quant).
3. Wire into moe(): for the verify (seq<=16), route gateup+down through the CUTLASS grouped MoE (build the
   per-expert token lists we already have from k_moe_invert). Keep half2 for base M=1 decode.
4. Full-step CUDA graph: make the eager draft graph-capturable (device-seqlen ptr, pre-alloc max draft-KV,
   fixed topology) so draft+verify+accept are one graph. Recovers the ~5% host tax + more on the ARM host.
5. Keep our NVFP4 lm_head + our draft unchanged (both are advantages over vLLM).

## Expected trajectory
- TC grouped MoE: verify MoE ~5x -> DFlash 58 -> ~70.
- + full-step graph + TC on remaining compute-bound verify GEMMs -> approach vLLM's ~78ms/step.
- Then our tau 10-13 draft does the rest: tau/0.078s = 128-170 tok/s -> BEAT vLLM, hit 110-140.

## Gotchas
- Thor page-cache: kill vLLM with `docker kill` never `docker stop`. Free RAM: sync + drop_caches.
- CUTLASS example host guards whitelist sm_100 only -> patch major==11 for any example run.
- vLLM DFlash uses BF16 KV (rejects fp8 KV, issue #41559) -> fp8 KV is NOT part of the fast path; skip it.

## TC GROUPED MoE (the clarified real lever, 2026-07-01) — scoped from ex.92
Profile proved the MoE down is COMPUTE-bound (dedup gave ~4%); TC's HW decode+MMA is the right tool.
Build (substantial, multi-session):
1. Grouped wrapper: extend cutlass_moe.cu to kGrouped. ProblemShape = MoEProblemShape<Shape<int,int,int>>
   (device array of per-expert <M,N,K>). Mainloop takes contiguous block_A/B/SFA/SFB (grouped by expert).
   Disable the block-scale OUTPUT epilogue (we want bf16 out): set IsBlockScaleSupported path off / plain LinComb.
2. Permutation: gather the ~120 (token,expert) assignments contiguously by expert (we have ecount/elist from
   k_moe_invert) into a [total, K] activation buffer; per-expert weight ptr = ep->dp[e] (already per-expert).
3. W4A4 activation quant: quantize the grouped activation (hbuf for down / x2_16 for gateup) to E2M1+e4m3,
   swizzle SFA (cutlass_swizzle_sfa per group). Weights: swizzle each expert's linear e4m3 scale ONCE at load.
4. Output = down partials -> reuse k_moe_finalize (already built). gateup similar (-> hbuf).
5. VALIDATE acceptance end-to-end (W4A4 5.5% GEMM error MUST NOT drop tau 13.3 much) + gate.
Expected: down 752us -> ~300-500us (single-GEMM was 8us/expert; 79 grouped concurrent). ~+8-12% DFlash if
acceptance holds. This is the ~2x-MoE lever the "TC is wrong" verdict wrongly deferred (it assumed BW-bound).

## Grind status (this session): bandwidth-down banked (+2.8%, DFlash 84.3); profile clarified MoE is compute-bound
## -> TC grouped is the next build. Single-GEMM TC wrapper + swizzle already validated (cutlass_moe.cu).
