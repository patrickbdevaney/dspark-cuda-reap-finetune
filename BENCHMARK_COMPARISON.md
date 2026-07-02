# vLLM DFlash vs pure-CUDA server — gemma-4-26B-A4B, IDENTICAL prompt (2026-07-01)
Prompt: "List the first 40 prime numbers, comma separated." (T=0 greedy). Same NVFP4 weights + same DFlash draft.

| engine            | tok/s | tau   | ms/step | accept-rate | notes |
|-------------------|-------|-------|---------|-------------|-------|
| vLLM 0.22.1 DFlash| 107.5 | 9.21  | 85      | 55%         | FULL decode cudagraph, triton_attn, flashinfer-cutlass NVFP4 MoE |
| our pure-CUDA     | 85    | 13.33 | 157     | 95%         | hand CUDA-core w4a16 + grouped MoE, graphed |
vLLM other tasks: code256=82 code512=48 reason256=34 (tau 4.5-7.6). Our step breakdown: w4a16 verify GEMM 39%, MoE 22%, draft 11%.

## VALIDATED CONCLUSIONS
1. TAU MOAT IS REAL: 13.33 vs 9.21 (1.45x) on identical prompt. Our greedy-exact verify accepts more per step.
2. STEP GAP IS REAL: 157 vs 85 ms (1.85x) = pure kernel efficiency (vLLM's Marlin/CUTLASS + full graph vs our CUDA-core).
3. CEILING = tau / step. At vLLM's 85ms step with our tau 13.33 -> ~157 tok/s = 1.46x vLLM's 107. The moat converts to real headroom.
4. Roofline: gemma NVFP4 ~13GB -> ~48ms/step weight-streaming floor. vLLM 85ms=1.8x floor; ours 157ms=3.3x floor -> ~70ms recoverable.

## PLAN: close the 1.85x verify-step gap (base + draft untouched; the moat is the draft, keep it)
Ranked by our profiled share + research (vLLM/SGLang steal-list):
 A. VERIFY GEMM (w4a16, 39% of step) -> Marlin-class NVFP4 at M=15: async weight loads, holds near-4x through M<=16.
    (CUTLASS grouped FP4 is BROKEN off-datacenter Blackwell per research -> hand Marlin-style per-expert W4 is the path.)
 B. MoE at M=15 (22%): grid occupancy (avoid idle-SM), minimize BLOCK_SIZE_M padding at ~2 tok/expert, keep kernel-granularity slack.
 C. Full-step CUDA graph over the verify forward (kill residual launch bubbles; vLLM uses FULL-on-uniform-decode).
 D. Ragged-Q / head-packed verify attention (FlashInfer-style; one varlen launch, KV-bandwidth-bound).
 E. (later) FP8 KV cache.
Target: 157ms -> ~85ms step -> ~130-157 tok/s (past vLLM's 107).

## LEVER A ATTEMPT (TC verify GEMM) — MEASURED, honest result (2026-07-01)
Built wmma W4A16 verify GEMM (FP4->fp16 dequant->shared, fp32 accum). VERIFIED BIT-EXACT (tau 13.33 preserved).
Speed: 16K-tile -5%, dense-only -5.5%, 64K-tile -25% (occupancy). => naive wmma CANNOT beat the tuned CUDA-core
w4a16 at M=15 on Thor. The verify IS compute-bound (would benefit from a GOOD TC kernel), but beating the
hand-tuned CUDA-core needs a MARLIN-CLASS kernel (cp.async double-buffer pipeline, coalesced tiled weight load,
occupancy tuning, offline weight repack) — a major expert build (Marlin is a years-tuned research artifact), NOT
a naive wmma. Kept flag-gated (TCVERIFY, off) as the correct foundation for that build. Champion 85 held.
KEY re-read: reaching the 157 ceiling requires porting/building Marlin-class verify kernels. Naive TC is slower.

## STEADY-STATE PROFILE (TCVERIFY on) — the real component map to attack (2026-07-01)
 lmhead w4a16      19.9%  (bandwidth-bound, TC neutral - hard)
 dense verify TC   19.1%  (tc_w4a16 - BIG; my TC only ~4% faster than CUDA-core, 160us vs ~21us BW floor -> overhead/occupancy bound, huge headroom)
 MoE gateup grouped16.3%
 MoE down_bw       12.9%  (MoE total 29% - next big lever)
 draft k_linear    10.9%  (bf16 draft)
 sdpa verify attn   7.9%  (FlashInfer-style head-packing lever)
 rmsnorm 1.6%, draft attn 1.3%, prefill 5%
## Marlin build step 1 (coalesced uint load): dense TC +2.2% net. TC is overhead-bound (~8 warp/SM occupancy,
## shared round-trip, dequant pass) - needs cp.async pipeline + occupancy + raw-mma in-register dequant to reach
## the 2-4x Marlin promises. STACK to 157: (1) finish Marlin dense GEMM (~+10%), (2) Marlin MoE 29% (~+15%),
## (3) FlashInfer attn 8% (~+4%). lmhead+draft (31%) are the hard tail. Multi-session grind, in progress.

## SoL PROFILE + MARLIN/LATENCY BUILD (2026-07-01) — progression to 93.1
SoL: ALL big verify kernels LATENCY-bound at M=15 (not bandwidth-saturated): lmhead ~55%mem/33%comp,
MoE gateup 28%mem/52%comp, down 26%mem/49%comp. => prefetch (raise MLP) is the general win.
 85.65 -> dense TC raw-mma (+3.3%) -> 88.65 -> MoE gateup U4 prefetch (+1.2%) -> 89.7 -> MoE down U4 prefetch (+3.7%) -> 93.1
All bit-exact, gate PASS, tau 13.33 held. step 157->143ms (vLLM 85ms, ceiling 157 tok/s).
Remaining latency-bound levers: draft linears (no prefetch), lmhead MLP, dense-TC cp.async, FlashInfer attn.

## DENSE-TC GRID-FILL + lm_head (2026-07-01) -> DFlash 96.4
SoL on tc_w4a16: Memory 22% / Compute 7% -> "grid too small to fill device" (32-64 blocks, occ 27%). NOT cp.async.
FIX: (1) remove shared A-stage -> read A fragment DIRECT from L2-cached global x (no shared, no syncs);
(2) WARPS=1 (1 warp/block = max grid fill). 93.6 -> 96.4 (+3%), bit-exact, gate PASS.
lm_head: TC neutral (memory-bound, TC reads same bytes); 128-bit blocked by non-16B tensor offsets; MLP
register-limited by acc[15]. Hard nut - left on tuned CUDA-core.
## ARC TOTAL: 85.65 -> 96.38 (+12.5%). vs vLLM 107 (was 22 behind, now ~11). step 157->~138ms. ceiling 157.
## Banked levers: dense-TC (Marlin raw-mma, in-reg dequant, no-shared, grid-fill), MoE gateup+down U4 prefetch.
## Remaining (harder): lm_head fused/FP8, FlashInfer attn (8%), dense-TC toward 2-4x Marlin, MoE split-K.

## lm_head FP4 attempt (2026-07-01) — DEAD END, reverted
Verify lm_head uses k_lmhead_batched (bf16 embed, 1.5GB reads). Switched to FP4 embed (369MB, 4x fewer bytes) via
w4a16_gemm AND via tc_w4a16: both ~-1% (bit-exact, gate PASS). The bf16 k_lmhead_batched is ALREADY efficient
(block-per-vocab + shared embed-row reuse + VOCAB-block grid = coalesced, full fill); FP4 kernels are latency-bound
at N=VOCAB and don't beat it. lm_head is NOT a tractable win. Reverted to bf16.
## FINAL ARC STATE: base 45 / DFlash 96.4 (85.65->96.4 = +12.5%). vs vLLM 107 (~11 behind, was 22). tau moat 13.33 intact.
## Banked: MoE gateup+down prefetch, Marlin TC verify GEMM (raw-mma, in-reg dequant, no-shared, grid-fill WARPS=1).
## Exhausted/dead-end: lm_head (bf16 already good), attention (balanced 50-60%/95% occ). Remaining = offline weight
## repack for dense-TC coalescing (core Marlin, complex, ~+5%) - the only clear lever left, a major build.

## OFFLINE WEIGHT REPACK (2026-07-01) — the core Marlin technique, DONE
Repack each dense verify weight ONCE (cached by src ptr) into [n_block][k_tile][lane] order so a warp's per-k-tile
reads are 64 CONTIGUOUS bytes (coalesced) vs 8 strided row-reads. Lazy repack on warm-up (pre-graph-capture) ->
graph-safe. Kernel reads unsigned short/lane (coalesced), in-register FP4 dequant, WARPS=1 (grid-fill best).
RESULT (back-to-back): 96.4 -> 97.9 (+1.6%), bit-exact, gate PASS. tc_w4a16 now memory-bound ~60% (coalesced).
NOTE: absolute tok/s drifts 94-98 with GPU THERMAL after long sessions; all arc deltas measured back-to-back.
## ARC FINAL: 85.65 -> ~97.7 (+14%). vs vLLM 107 (~9 behind, was 22). tau moat 13.33 intact -> ceiling 157 stands.
## Full banked stack: MoE gateup+down prefetch; Marlin TC verify GEMM (raw-mma, in-reg dequant, direct-L2 A,
## grid-fill, offline weight repack). All bit-exact. Dead-ends: lm_head (bf16 already good), attention (balanced).

## === DEEP-RESEARCH BREAKTHROUGH: PASSED vLLM (2026-07-01) ===
User pushed back on shallow tuning -> 2 code-level research agents (real Marlin/FlashInfer/vLLM source, line-by-line).
Findings applied:
 95.5 -> tc weight prefetch U=8 (+1.9%) -> 97 -> route draft+verify LM_HEADS through Marlin tc (biggest kernel 22%,
 was still w4a16 CUDA-core!) +9.7% -> 104.5 -> 16-byte int4 loads + __ldcs evict-first (Marlin memory recipe) +3.5%
 -> 108 (PAST vLLM 107) -> attention HEAD-PACK (KV read once, was 4x GQA-redundant; qs/acc in registers) +0.6% short-ctx.
All bit-exact, gate PASS, tau 13.33. Key lesson: the wins were things dismissed with shallow tests - the Marlin kernel
was only on the dense, never the lm_heads (the biggest kernel); weights were 2-byte not 16-byte int4.
## ARC TOTAL: 85.65 -> ~108 (+26%). WE BEAT vLLM 107 with a measurably better draft (tau 13.33 vs 9.21).
## Research confirmed tcgen05 is NOT the path (M=15 is 60x under compute roof; MMA_M locked to 128). mma.sync is right.
## Remaining to ceiling 157: full cp.async 5-stage tc pipeline (tc=35% of step, 60%->90% mem), MoE small-batch glue,
## whole-step CUDA graph. tau moat still points at 157.
