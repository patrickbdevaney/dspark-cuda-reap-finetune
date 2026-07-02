# Autonomous zenith loop — base decode + draft (2026-07-01), targeting >100 tok/s
Baseline: base 44 tok/s (~29% of ~152 roofline), DFlash 84.3. Champion metric per mode; correctness gate MUST pass.
## Levers already closed (see RESEARCH_SYNTHESIS.md): TC (wrong hammer), full-graph (7%), draft->FP4 (hurts accept),
## MoE grouped (padding). Base-decode M=1 MoE is latency/compute-bound at 13.6% BW = the biggest headroom (3.2x).
## Research running: M=1 FP4 MoE zenith kernel (hit the HBM floor).

## iter1: fp4_gemv M=1 K-unroll prefetch U=4 -> LOST (44.5->43.9). Single-accumulator GEMV is register-sensitive;
## large-N (qkv N=4096, lmhead N=262144) already has enough warps for MLP -> prefetch just adds regs. Reverted.
## Base decode profile: fp4_gemv(dense+lmhead) 32.7%, MoE gateup 19.3%+down 12% =31%, rmsnorm 9% (launch-inflated,
## hidden in graph). Await M=1-GEMV-zenith research for the specific floor-hitting technique.

## iter2: base rmsnorm = ~10 gemma-4 double-norms/layer, each 1-CTA/1-SM at M=1 (latency-bound, ~2.3ms/token).
## Structural fusion won't fix it (sequential, different data) — needs a MEGAKERNEL (fuse the layer). Research pending.
## LOOP STATE: 2 research streams running (M=1-GEMV-zenith + draft/megakernel-zenith). Kernel-guessing regresses
## (fp4_gemv prefetch, draft-FP4 both LOST) -> awaiting PROVEN techniques from research before next grind.
## base 44.5 / DFlash 84.3 intact. Next: grind per research findings (M=1 GEMV floor kernel + megakernel MVP).

## iter3: MoE C_LUT -> HW cvt.e4m3 (research #1, kill divergent constant lookup). base 44.35->44.6, DFlash 84.3->84.7
## (marginal +0.5%; my constant-cache lookup wasn't the big bottleneck the research assumed, but it's cleaner). KEPT.
## RESEARCH DELIVERED (2 streams): CRITICAL - Thor = 20 SMs not 96 (SM-idle 5x smaller; launch/megakernel is #1).
## NVFP4 roofline = ~137 tok/s (89 @65%); base 44 = ~50% -> ~2x headroom. Ranked levers:
##  1. CUDA graphs (base done; DFlash draft eager but only ~7% measured). 2. FP8 DRAFT weights (not FP4 - keeps
##  acceptance, halves draft bytes, ~2x draft). 3. Megakernel (draft first, hand-writable; base via MPK). 4. FlashNorm
##  fold RMSNorm into GEMM (12-35% norms). 5. Fuse gate+up+SiLU register-intermediate (Cursor warp-decode 1.84x).
## Kernel micro-opts near limit (marginal). Next: FP8 draft (real ~2x draft potential) or fuse-gate+up (Cursor).

## LOOP STATE after 3 iters + 2 research streams (2026-07-01): kernel micro-opts NEAR LIMIT (HW-cvt +0.5%,
## prefetch/draft-FP4/split-K/TC all lost or marginal). The real zenith needs SUBSTANTIAL builds, ranked:
##  A. FUSED MoE gate+up+SiLU+down one persistent kernel (Cursor warp-decode 1.84x; hbuf in SMEM not DRAM;
##     one block/layer keeps M=1 work on-chip). Base MoE=31% -> biggest single base lever.
##  B. cp.async DOUBLE-BUFFER the FP4 weight stream (research #2): moves in-flight bytes OFF registers -> breaks
##     my U=8/48-reg wall -> gateup 13.6%->~40% BW. The identified path to the M=1 HBM floor.
##  C. FP8 DRAFT weights (E4M3, not FP4 - keeps acceptance) via a small-N-optimized FP8 k_linear (NOT w4a16);
##     + the big FC[16896->2816] is bandwidth-bound -> FP8 halves it. Addresses the draft-model target.
##  D. DRAFT megakernel (5 layers, hand-writable per Hazy 7-instruction model) - the draft zenith.
##  E. FlashNorm: fold RMSNorm scale into next GEMM + fuse q/k-norm+RoPE (14.7% of layer per TRT-LLM). base norms 9%.
## Each is a multi-hour build. Champion: base 44.6 / DFlash 84.7. Roofline 137 (89 @65%) -> ~2x base headroom real.

## iter4: lever A reconsidered - base is GRAPHED so fusion's launch-win is already captured (research: MoE DRAM
## round-trip only ~0.5% of weight traffic) -> lever A NOT worth it for graphed base. Pivoted to kernel-BW tuning:
## gateup 128-bit uint4 loads (research #5) -> CRASHED (misaligned: per-expert FP4 ptrs not 16B-aligned). Reverted.
## => base gateup micro-tuning EXHAUSTED (prefetch neutral, C_LUT neutral, widen faults). warp-per-output at its
## structural limit. Remaining base levers require RESTRUCTURE: cp.async producer/consumer (lever B, same alignment
## care needed) or megakernel. Next: lever C (FP8 draft) - the user's draft-model focus, untried at FP8 (FP4 failed
## on acceptance; FP8/E4M3 has less error -> may keep tau 13.3 while halving the bandwidth-bound FC bytes).

## iter5: lever C FP8 draft (E4M3, proper k_linear_fp8, half bytes) -> LOST. DFlash 84.7->71.1, accept 13.33->11.14
## (IDENTICAL drop to FP4). DEFINITIVE: the DFlash draft is precision-sensitive - ANY weight quant (FP4 or FP8)
## collapses acceptance to 11.14. The draft MUST stay bf16 (research's "FP8 keeps acceptance" holds for typical
## EAGLE drafts, NOT this block-diffusion DFlash draft which shares embed+lmhead and injects target K/V). Reverted.
## => CLOSED: base micro-tuning (exhausted), draft quant (FP4+FP8 both lost). Remaining = RESTRUCTURES only:
##   cp.async base GEMV (lever B, uncertain per hackathon) | megakernel (huge) | fused-MoE (graphed=minimal).
## Champion HELD: base 44.6 / DFlash 84.7. The draft's bf16 + high tau 13.3 IS the moat - don't touch it.

## iter6: lever B cp.async double-buffer gateup (depth-4, weight stream off-registers via SMEM). -> LOST -5% (44.6->42.4).
## Gate PASS (correct) but slower: SMEM round-trip + __pipeline commit/wait/syncwarp overhead > the register-freeing
## gain, because per-pass compute (8 FP4 decode + FMA) is too small to hide the staging latency. CONFIRMS the
## hackathon finding (cp.async loses vs direct streaming here). Reverted. Deeper D just adds SMEM/cuts occupancy.
## === DEFINITIVE: ALL tractable kernel levers now CLOSED by measurement (7 iters). base 44.6 / DFlash 85 is the
## practical limit of the current per-kernel warp-per-output architecture. The ONLY remaining lever to the ~137
## roofline is the MEGAKERNEL (persistent single-kernel, on-GPU instruction interpreter, counter-sync, fused
## norm+GEMM per Hazy) - a multi-SESSION engineering build, not a loop iteration. Documented as the next major
## undertaking. The bf16 draft (tau 13.3) + FP4 lm_head remain the moat that makes DFlash 85 competitive.

## iter7 (megakernel M1 scoping): examined the real attention layer = ~15 sequential kernels (rmsnorm, q/k/v proj,
## q/k/v norm, rope x2, store_kv x2, sdpa, o_proj, post_norm, add) + the MoE (gateup/down) + more norms. A megakernel
## must fuse ALL of these into one persistent grid with counter-sync + on-GPU interpreter + FlashNorm-folded norms.
## HONEST FINDING: M1 (persistent skeleton + counter-sync + input_rmsnorm+QKV fused, correctness-first) is genuinely
## the FIRST STEP of a multi-WEEK ground-up rewrite with (a) NO speed payoff until ~M4 (all 30 layers fused), (b) high
## correctness risk per milestone, (c) uncertain ~1.3-1.7x final payoff (MPK over already-fused vLLM; NOT a base 3x),
## (d) the base is already GRAPHED so the launch-overhead portion of the megakernel win is partly captured already.
## Building a persistent cooperative kernel safely needs a DEDICATED focused effort, not autonomous loop iterations in
## a large accumulated context (high risk of a broken half-built state). MEGAKERNEL_PLAN.md documents the M1-M5 path.
## === LOOP TERMINUS: the tractable optimization space is EXHAUSTIVELY mapped + closed by measurement. Champion held:
## base 44.9 / DFlash 85. The megakernel is real but a distinct major project. Honest recommendation: dedicated build,
## explicit go-ahead, not autonomous grinding. The bf16-draft moat (tau 13.3 vs vLLM 7.84) + FP4 lm_head keep DFlash 85
## genuinely competitive with vLLM's ~100 despite fewer per-kernel opts.

## === MEGAKERNEL M1 ACHIEVED (2026-07-01) — infrastructure proven correct ===
## Built src/megakernel.cu: persistent 64-block kernel, global-counter sentinel-sync (NOT grid.sync, graph-safe),
## fuses input_rmsnorm + Q/K/V proj into ONE kernel. Flag-gated MEGAKERNEL=1 (champion path default/untouched).
## VERIFIED BIT-EXACT: MEGAKERNEL=1 tokens == champion tokens (GEN=4 & GEN=8 identical). Gate PASS both flag states.
## Speed 43.3 vs 44.6 (-3%) - EXPECTED for the skeleton: per-layer counter-memset + single-block Phase-A reduction +
## spin, only 4 of ~15 kernels/layer fused, no cross-op weight prefetch yet. The pattern + correctness are the M1 win.
## NEXT (M2): extend the SAME persistent kernel to also fold the q/k-norm+RoPE (warp-per-head restructure) and/or
## chain o_proj+post_norm+residual, amortizing the memset/spin over more fused work toward break-even, then M4
## cross-op prefetch for the actual speed win. Champion base 44.9 / DFlash 85 HELD (flag off).

## M1-optimized (grid-sizing fix): mega_qkv now launches nblocks=(qd+2kd+7)/8 (~1 warp/output) instead of fixed 64
## -> matches champion's per-GEMV concurrency/latency-hiding. RESULT: 44.5 vs champion 44.4 = BREAK-EVEN, bit-exact.
## VALIDATES the megakernel approach: fusion overhead fully amortizes when grid is sized right. => each additional
## fused op (M2+) should push POSITIVE (savings accumulate, no coordination penalty). Was -2% purely from block-count
## under-provisioning, NOT fundamental. Trajectory is GO. Champion (flag off) 44.9 held.

## === MEGAKERNEL VIABILITY — measured finding (from building M1) ===
## Two fusion classes, measured:
##  (1) PRODUCER->CONSUMER (M1: block0 does rmsnorm-reduction -> many consumers do QKV): consumers need NO barrier
##      among themselves -> can launch ANY block count (768) -> full concurrency -> BREAK-EVEN. VIABLE.
##  (2) GRID-BARRIER (o_proj+post_norm+residual, MoE gateup->down: all-produce-THEN-reduce/consume): needs a grid-wide
##      arrival barrier -> all blocks must be RESIDENT -> capped at ~40 blocks -> the -2% concurrency penalty measured
##      at 64 blocks (vs 768 break-even). These fusions LOSE the GEMV concurrency they'd need to win.
## Most useful fusions (attention tail, MoE) are class (2). => the FULL-step megakernel for THIS M=1 + already-GRAPHED
## base is likely MARGINAL (~1.0-1.15x), NOT 1.5-2x: grid-barrier concurrency penalties offset the fusion savings,
## and the launch-overhead win is already captured by the graph. Aligns with research (MPK MoE = only 1.16x). A
## multi-week full build would plausibly reach base ~48-52 / DFlash ~90-98 - close to but NOT clearly past 100, and
## NOT the base 3x that >100-on-base would need. M1 (break-even, bit-exact, flag-gated) stands as the proven foundation.

## === M2 BUILT + MEASURED (o_proj+post_norm+residual, grid-barrier, NB=100=5blk/SM) ===
## VERIFIED BIT-EXACT (M1+M2 tokens == champion). Speed: M1+M2 = 44.07 vs champion 44.75 = -1.5%.
## => M1 alone break-even, M2 adds slight regression. Grid-barrier penalty is REAL but SMALLER than predicted
## (NB=100 not 40; o_proj at 100 blocks < champion's 352 -> mild concurrency loss + arrival-barrier + block0-reduce).
## DECISIVE FINDING: per-unit megakernels (SEPARATE launches, just internally fused) do NOT win - M1 0%, M2 -1.5%,
## trend negative. The research's win (Hazy 2.5x) is the FULL-STEP SINGLE persistent kernel with CROSS-OP WEIGHT
## PREFETCH (load op N+1 weights during op N compute) - which per-unit separate kernels STRUCTURALLY CANNOT have.
## To win, ALL ops must live in ONE persistent kernel (weeks-long instruction-interpreter build). Measured expected
## payoff still ~1.0-1.2x for M=1-graphed -> base ~48-52 / DFlash ~90-98, NOT >100 on base. M1+M2 stand as the proven,
## bit-exact foundation + the measured evidence that only the full single-kernel could win (uncertain, weeks).

## === FULL-STEP Stage-1 (grid-cooperative rmsnorm+QKV) BUILT + MEASURED — pivotal finding ===
## mega_af_kernel: grid-cooperative input_rmsnorm (grid-reduce sum(h^2) via atomicAdd + 2 grid_bar barriers) + QKV.
## VERIFIED BIT-EXACT. Speed: 42.7 vs champion 44.75 = -4.5% (WORSE than M1's block-0 producer-consumer at 0%).
## => The GRID-BARRIER (the mechanism the full-step layer-loop REQUIRES for op-sequencing across the persistent grid)
## costs measurably at M=1: ~60 barriers (2/layer x30) = -4.5% vs block-0. The full-step needs ~390 op-sequencing
## barriers/step (13 ops x 30 layers) -> extrapolated overhead is LARGE. This is the WRONG regime for a megakernel:
## Hazy's 2.5x was Llama-1B with BIG ops (barrier cost small vs op time); our M=1 MoE has TINY ops (barrier cost
## dominates). MEASURED trajectory: M1 block-0 0%, M2 grid-barrier -1.5%, grid-coop reduce -4.5% - consistently
## NEGATIVE and worsening with more barriers. Reverted wiring to M1's mega_qkv (best, -1.5% w/ M2). mega_af retained
## as the measured evidence. CONCLUSION (now measured, not estimated): the full-step megakernel LOSES for M=1 Thor.
