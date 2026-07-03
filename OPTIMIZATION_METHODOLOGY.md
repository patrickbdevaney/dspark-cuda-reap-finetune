# OPTIMIZATION_METHODOLOGY.md — how we optimize decode (kernels + draft head + fine-tune)

Inherited from the `gemma-cuda-hybrid` campaign (which beat vLLM via Marlin-class kernels + rigorous
empirical evolution). This governs Phase C (decode-kernel optimization) AND the draft-head fine-tune. The
goal is **maximum tokens/sec of the target+DSpark-head system**, reached by *structural step-changes*, not
by grinding marginal percentages.

## The core loop: champion–survivor A/B evolution
1. **Benchmark harness first.** Before optimizing anything, build a repeatable decode benchmark (tok/s,
   ms/tok, per-component timing) on a fixed real workload. No change is "an improvement" without a measured
   A/B delta on that harness. Keep a **ledger** (variant → measured tok/s → keep/kill).
2. **Champion = current best measured survivor.** Each candidate is A/B'd against the champion on the same
   harness. Win → it becomes champion. Lose/tie → killed. Mutate the champion (tiling, vectorization,
   pipelining, fusion, occupancy) and repeat. Evolution, not guesswork.
3. **Everything measured, nothing assumed.** Roofline predicts; the harness decides. Record the number.

## The stop rule: detect diminishing returns, DON'T thrash
- Track **marginal gain per iteration**. When a component's incremental tuning yields **< ~2–3% per real
  effort-unit** for a couple of rounds, **STOP grinding it.** Marginal percentage-chasing is waste.
- Diminishing returns on a component is a **signal to change altitude**, not to try harder at the same level.

## The altitude change: black-swan / saltatory search
When incremental gains flatten, **do not keep tuning — search for a modular step-change.** "Saltatory" =
evolutionary leap, not gradualism. A black-swan method delivers a *large, structural* decode gain as a
drop-in module, dwarfing the marginal tuning we stopped.
- **Literature + landscape scan FIRST (before grinding).** Survey the current SOTA optimization-implementation
  landscape so we implement the known big wins *before* micro-tuning, and so we know what leaps exist. Run the
  deep-research workflow on "fastest {MoE decode / MLA attention / spec-decode verify} kernels on Blackwell
  sm_110a" and mine it for structural methods, not 2% tweaks.
- **Examples of the class of leap we hunt** (illustrative, not a checklist): persistent / megakernel fusion
  (kill launch + HBM round-trips), CUDA-graph capture of the decode step, Marlin-style `mma.sync` W4A4/W4A8
  GEMM with async copy + double-buffering, tensor-core exploitation for the MoE expert GEMMs, KV/scale layout
  changes that remove dequant, fused RMSNorm+GEMM+act-quant, warp-specialized producer/consumer pipelines,
  block-sparse attention fast paths, a fundamentally different spec-decode verify schedule. The point is
  **category jumps**, each an A/B'd modular swap.
- After a leap lands as champion, resume incremental evolution on the *new* champion until the next stop rule.

## Joint objective: optimize the target+head SYSTEM, not one kernel
The base (unpruned-trained) DSpark head is already good with our REAP (Gate 2 τ≈0.8-class). So the win is a
**co-optimized pipeline**, three coupled fronts — evolve them together, measure the *system* tok/s:
1. **Base-model decode kernels** — the target's per-step forward (verify path). MoE expert GEMMs, MLA/DSA
   attention, the fp8/fp4 GEMMs. This is where most cycles go.
2. **DSpark draft-head decode** — the head runs every step to propose the block; its kernels (block-diffusion,
   Markov bias, confidence, AR block sampling) must be fast and overlap the target where possible.
3. **The fine-tune itself is an optimization target** — τ acceptance, block-diffusion quality, and *decode
   efficiency* of the resulting head. A/B the training recipe (data mix/coverage, objective weighting,
   block-acceptance) on measured **E[accepted block length] and system tok/s**, not proxy loss alone. A
   higher-τ head that's slower per step can lose; optimize the product (acceptance × step-speed).

## Guardrails
- **Correctness gates never loosen** (CONSTITUTION): an optimized kernel must still pass its bit-exact/cosine
  gate before it can be champion. Speed never buys wrong numbers.
- **Log what you dropped.** If a leap is deferred or a path abandoned, record why (so we don't re-thrash it).
- **Ledger is durable** — variants, measured tok/s, and the stop-rule decisions live in-repo so the campaign
  is auditable and resumable.
