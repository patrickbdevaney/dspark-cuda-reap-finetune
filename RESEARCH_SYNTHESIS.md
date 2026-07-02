# Decisive synthesis (2 deep agents, 2026-07-01): the 2× is NOT tensor cores

## Verdict: decode is BANDWIDTH-BOUND. TC is the wrong hammer (confirmed by our own break-even benchmark).
Both agents converge. The 2× gap to vLLM = (a) host/launch tax [huge on ARM] + (b) memory-BW bubbles.
Our validated CUTLASS TC GEMM broke even with half2 because BOTH wait on HBM — TC is ~2500x over-provisioned
for decode's ~1.5 FLOP/byte arithmetic intensity. STOP chasing tensor cores for decode.

## The real levers, ranked for Thor (ARM, 273 GB/s):
1. **FULL-STEP CUDA GRAPH (#1, biggest on ARM).** ARM host is launch-bound 4x longer than x86 (Grace data:
   2.8x bs=1 latency from single-thread CPU). Our DRAFT is eager (~50 launches/step). Collapse draft+verify+accept
   into ONE graph: WHILE conditional node for the draft-K loop, IF/ELSE for accept/reject (cudaGraphSetConditional,
   CUDA 12.4+ conditional nodes), device-resident seqlen tensor + pre-allocated max KV (like our base-decode g_base).
   vLLM/SGLang/TRT-LLM all do device-seqlen + static buffers. Pure CUDA, no precision risk.
2. **BANDWIDTH-OPTIMAL MoE (#2).** Target = HBM roofline ~286us/layer (down); our half2 = 787us = 2.7x ABOVE floor
   because it re-reads each expert weight PER-TOKEN. Fix: reuse each weight byte across ALL 15 tokens routed to that
   expert (weight-resident, activations in regs/SMEM), fully-coalesced + double-buffered (cp.async) FP4 weight stream.
   CUDA-core dequant+FMA is fine (bandwidth-bound). Measure GB/s not TFLOPS; done at ~250 GB/s. NOT tensor cores.
   Layout: vLLM moe_align_block_size (BLOCK_M=16) or DeepGEMM masked (graph-capturable, no CPU per-expert counts).
3. **FP8/FP4 KV cache (#3).** Attention is memory-bound on KV; FP8 KV ~up to 2x. GQA head-stacking on M axis,
   SWA tile-skip, fused gemma soft-cap. No split-KV at short ctx.
4. Fused RMSNorm+residual+quant (#5): 1-4% cleanup (weight_bias=1 for gemma +1 centering).
Megakernel: deferred (1.0-1.7x vs tuned; MoE forces dynamic scheduling; draft+verify-in-one-megakernel unsolved).

## Our earlier TC work (steps 1-2) = a validated NEGATIVE result: TC proven to work on Thor but break-even for
## decode. Kept as a primitive (cutlass_moe.cu) for any future compute-bound (prefill/large-batch) path. NOT the
## decode lever. PIVOT the integration to #1 (graph) + #2 (bandwidth MoE).

## gemma-4 verification TODO before MoE build: confirm top-k (8?) + renormalization from HF transformers module.

## MEASURED on-device (2026-07-01) — overturns the graph priority for OUR setup
- DFlash is **93% GPU-busy** (462ms GPU kernel / 498ms wall, GEN=40). Host/launch tax = only ~7%.
- => The FULL-STEP GRAPH is a ~7% lever for us (our long 15-tok x 30-layer kernels hide launches), NOT the
  ARM-scale win agent 2 assumed for naive many-short-kernel pipelines. Deprioritize the graph.
- => The gap to vLLM (58-82 vs 100) is GPU KERNEL TIME = memory-bandwidth EFFICIENCY. Agent 1's roofline:
  our MoE is 2.7x above the HBM floor (down 787us vs ~286us). vLLM realizes ~55-60% BW; we realize ~37%.

## THE lever (revised, measured): bandwidth-optimal CUDA-core kernels. Per-kernel work to hit the HBM floor:
  1. DOWN/GATEUP MoE: reuse each expert weight across ALL routed verify tokens (weight-resident, not per-token
     re-read ~1.5x); fully-coalesced + double-buffered (cp.async) FP4 stream; minimize C_LUT/decode overhead.
     Target ~286us/layer (down). ~2.7x headroom.
  2. verify DENSE (w4a16 M=15): same — weight-once, coalesced, overlap decode. (already reuses weight across M.)
  3. attention: FP8 KV (halve KV bytes), GQA head-stacking. memory-bound.
  This is per-kernel bandwidth tuning (multi-session), the honest path to ~2x -> matching/beating vLLM given our
  BETTER draft (tau 13 vs 7.84) + FP4 lm_head. NOT graph, NOT TC.

## BUILD: bandwidth-optimal down (2026-07-01) — first bandwidth-MoE win
- k_moe_down_bw: warp-per-(e,d-pair), read Wd_e[d] ONCE, reuse across ALL tokens routing to e (register-block 4),
  write per-assignment partials dpart[(t*8+j)*H+d] (NO atomics) + k_moe_finalize (8-way ws-weighted sum per token).
  Reuses gateup's invert map (ecount/elist). Fixes the ~1.5x per-token weight re-read of the old warp-per-(t,d) down.
- RESULT: DFlash 82 -> 84.3 (+2.8%), gate PASS. Smaller than the 2.7x roofline -> the down is PARTLY compute-bound
  (per-element dec_fp4x2 + half2 FMA competes with the weight stream), not purely weight-read-bound. The dedup
  (weight-once) captured ~1.5x on the READS but compute limits the net. gateup already reuses weight (grouped).
- NEXT bandwidth levers: (a) reduce per-element decode overhead / overlap loads (cp.async double-buffer — conflicting
  research on batch-1); (b) apply weight-once to any remaining per-token kernels; (c) FP8 KV for attention bytes.
  Each is ~+2-5%; stacking across kernels is the path to ~2x. This build is the validated template.

## PROFILE FINDING (2026-07-01) — the MoE is COMPUTE-bound, which re-opens TC
- k_moe_down_bw = 752us (vs old down 787us) — the weight dedup gained ~4%, NOT the 2.7x roofline. => the down is
  COMPUTE-bound (per-element dec_fp4x2 + half2 FMA tree), not weight-read-bound. Agent 1's "bandwidth-bound" roofline
  assumed an already-efficient kernel; my half2 decode compute is the actual bottleneck.
- IMPLICATION (redirect): for a COMPUTE-bound FP4 MoE, TENSOR CORES (HW decode + MMA) are the RIGHT tool — they do
  in hardware exactly the compute that bottlenecks my half2. My earlier "TC break-even" verdict assumed the kernel
  was bandwidth-bound; it's not. Re-open the CUTLASS grouped TC down: single-GEMM measured ~8us at the down shape;
  79 experts grouped (concurrent across SMs) plausibly < my 752us -> the real ~2x lever after all, IF it clears the
  W4A4 accuracy bar (5.5% GEMM error -> must verify acceptance holds).
- NEXT: build the CUTLASS GROUPED (kGrouped/MoEProblemShape) TC down + gateup (W4A4), validate acceptance end-to-end.
  This is the substantial build the earlier "TC is wrong" verdict wrongly deferred. bandwidth-down stays (+2.8%, harmless).

## GROUPED-BUILD INVESTIGATION (2026-07-01) — decisive: TC grouped is the WRONG tool, half2 is right
- ex.92 CUTLASS FP4 grouped GEMM = masked layout BUT **M-tile floor = 128** (no sub-128 for block-scaled FP4).
  At ~2 tokens/expert each active expert still pads to a 128-row tile = ~64-84x COMPUTE WASTE.
- For the COMPUTE-bound down: TC grouped ≈ 79 experts x 128-padded x N x K = ~20G MACs → ~625us vs half2 752us
  = **only ~17%, not 2x**. The 128-tile padding EATS the TC speed at small tokens/expert. Huge build NOT worth it.
- WHY vLLM's MoE is faster: its Triton fused_moe uses **BLOCK_M=16** (small tile → 8x padding, not 64x). CUTLASS
  FP4 forces 128. => the right structure for small-batch MoE is a SMALL-TILE kernel with NO 128-padding — which is
  exactly what my half2 (warp-per-output, processes real tokens) ALREADY IS. My half2 is structurally correct.
- => The MoE down at 752us is COMPUTE-bound with the right (no-padding) structure. The only remaining lever is
  OVERLAPPING the FP4 decode with the weight stream (cp.async double-buffer) to hit the ~286us bandwidth floor —
  but the first research cascade measured cp.async REGRESSING at batch-1. High-uncertainty, likely-marginal.
- HONEST MoE CONCLUSION: my half2 MoE is near its practical structure limit for ~2 tokens/expert. TC (padding),
  dedup (compute-bound), grouped (padding) all investigated -> none gives 2x. The MoE is NOT where the win is.
  Redirect remaining effort AWAY from the MoE to: lmhead BW (60.9%->higher), draft (bf16->FP4?), FP8 KV.

## DRAFT->FP4 WEIGHTS (2026-07-01) — LOST, reverted (hurts our key advantage)
- Quantized draft bf16 linear weights -> FP4 (W4A16 via w4a16_gemm, keeping fp16 acts). Reused k_quant infra.
- RESULT: DFlash 84.3->75.1 AND accept 13.33->11.14 (both regressed). The FP4 draft weights make the draft
  LESS accurate -> proposals shift -> acceptance drops back to pre-FP4-lmhead level (loses the consistency win);
  and w4a16_gemm overhead hurts the small-N draft linears (qkv/o). Reverted. Draft weights MUST stay bf16.
- KEY: our acceptance edge (tau 13.3) DEPENDS on the accurate bf16 draft. Don't quantize the draft body.
  (only the draft lm_head benefits from FP4 — big N=VOCAB byte-bound + consistency with the FP4 verify lmhead.)
- MoE (near limit), draft->FP4 (hurts acceptance), TC (padding), graph (7%) all now closed by measurement.
  Remaining small/uncertain: lmhead BW push (60.9->higher), FP8 KV (attention ~1%). Hybrid banked at 84.3.
