# DECODE_HORIZON.md — the exhaustive grind to the CUDA decode peak (38–50 tok/s)

**Target: 38–50 tok/s** on Thor `sm_110a` for the 180B-REAP, via **base decode + DSpark speculative decoding**.
This is the standing horizon. Every lever is A/B'd, gated bit-exact vs oracle, and logged (`OPTIMIZATION_LEDGER.md`
measured deltas, `GATE_LOG.md` rationale). We grind all the levers we know AND periodically search the literature
for **black-swan / step-change** techniques so we don't thrash on diminishing returns.

## HARD CONSTRAINT (learned the painful way)
Thor's 122.8 GiB unified RAM is SHARED with the host/OS/Claude Code. The forward already uses ~108-113 GiB
(~90%). A +5.5 GiB dequant-cache starved the system and forced a power-cycle. **Every future lever MUST be
memory-neutral** (in-place / fixed-arena reuse, never on-top). This rules out naive caching; favors kernels
that read quantized data directly (no fp32 dequant) and in-place transforms.

## Where we are
- Unoptimized prefill baseline: **687 ms/tok (~1.46 tok/s)** at s=8 (correctness-first: per-token host loops,
  warp-per-output GEMMs). Target 38–50 tok/s = **20–26 ms/tok** → a ~26–36× gap. Big, but the baseline is
  deliberately naive; most of the gap is launch/host overhead + un-TC'd GEMMs, not fundamental.
- Banked kernels: **tc_fp8** dense (17.9×, cosine 1.0), **tc_fp4** MoE (19.7×, cosine 1.0), **repack-at-load**
  (zero extra mem), batched dispatch (cosine 1.0). End-to-end so far: dense+batched = **559.8 ms/tok (1.23×)**.

## The lever ladder (ranked by expected structural gain; grind top-down, A/B each)
1. **TC GEMMs everywhere** — tc_fp8 (dense) ✅ + tc_fp4 pp (MoE) [in progress]. Puts the dominant compute on
   tensor cores. *This is compute; the bigger decode wins below are about OVERHEAD.*
2. **Kill launch + host overhead (the big M=1 decode lever).** At batch-1 decode, the 43 layers × many small
   kernels + the MoE's host-side expert loop (copy hidx to host, loop, per-expert launches) DOMINATE — not the
   GEMM. Fixes: (a) **device-side MoE routing/grouping** (no host round-trip), (b) **CUDA-graph capture** of the
   whole decode step (one graph launch vs hundreds of kernel launches), (c) **kernel fusion** (RMSNorm+GEMM,
   act_quant into the GEMM epilogue). Expected: the largest single multiplier for base decode.
3. **DSpark speculative decoding — the throughput multiplier.** Draft head proposes a block of block_size=5
   tokens; target verifies the block in ONE forward. At τ≈0.8 acceptance that's ~2.5–4× decode throughput on
   top of the base. This is *the* reason for the draft head; base-decode speed multiplies through it.
4. **Overlap draft ∥ target** — run the draft head concurrently with the target verify on one GPU (streams),
   hiding the draft cost.
5. **Attention kernels** — MLA (1 KV head) + sliding-window + DSA top-k sparse: fast sparse-attn kernels,
   fused. Smaller than the MoE but real.
6. **Aligned fast path for pp weights** (loader 16B-align expert tensors → uint4 __ldcs instead of byte loads).
7. **FP4 COMPUTE (2070 TFLOPS, 4×)** — BLOCKED on Thor CUDA 13 (see `FP4_COMPUTE_NOTE.md`); armed re-test probe.
   The single biggest latent lever the moment NVIDIA exposes it.

## Discipline
- **Champion–survivor A/B:** each candidate measured vs the current champion; keep only measured wins; log deltas.
- **Stop-rule:** when a lever's returns go sub-~10% and cost rises, stop tuning it and move to the next
  structural lever (don't thrash on diminishing percentages).
- **Black-swan search (periodic):** when the ladder's near-term rungs plateau, run a literature/impl scan for
  step-change techniques (persistent megakernels, speculative-decode scheduling, new mma paths, quant-compute
  co-design). Preserve each run to `RESEARCH_*.md`. Prior scan: 23 sources, megakernel REFUTED vs CUDA graphs.
- **Never loosen a gate:** a speed win with wrong numbers is a loss. Bit-exact/cosine vs oracle before it ships.

## Scoreboard (update per grind)
| Lever | Status | Measured |
|---|---|---|
| tc_fp8 dense (17.9×) + batched | ✅ wired | 559.8 ms/tok (1.23×) |
| tc_fp4 pp MoE (repack-at-load, byte-load) | ✅ correct, but ~neutral | 555.9 ms/tok (byte-loads negate coalescing) |
| **tc_fp4 pp ALIGNED load** (funnel-shift OR loader 16B-align) | **NEXT (unlocks MoE 19.7×)** | — |
| DEQUANT — memory-NEUTRAL scheme (cache REVERTED: starved RAM) | blocked-on-memory | ~30% of warm fwd; needs in-place/arena, not on-top |
| **2x warmup (steady-state, memory-neutral)** | ✅ done | **pass1 WARM 451.2 ms/tok (2.22 tok/s), 1.52× base** |
| **TC-ify attn-out (fp16 mma per group, cosine 1.0)** | ✅ done | **388.5 ms/tok (2.57 tok/s, 1.77× base), −14%** |
| **aligned MoE load (funnel-shift, cosine 1.0)** | ✅ gated, measuring | pending warm |
| TC-ify gemm_fp32 (compressor/indexer/head, ~14% warm) | TODO | — |
| CUDA graphs + device MoE routing + fusion | TODO | — |
| DSpark spec decode (block verify) | TODO (τ≈0.815 proven) | — |
| FP4 compute (4×) | BLOCKED (CUDA 13) | re-test probe armed |
