# Agentic Optimization Methodology — how to consistently surface & implement BIG decode-speed wins
For this project (CUDA kernel tuning for LLM decode on fixed hardware) and transferable to any single-scalar, hardware-bound optimization under an agentic coding loop. Derived from the empirical record of gemma-cuda-server + gemma-cuda-hybrid (0 → 118 tok/s).

## 0. The objective, stated correctly
- We optimize a scalar (decode tok/s) — BUT it is a scalar over a **workload distribution** (short/long context, code, chat), not one prompt. Optimizing only `primes` blinds us to long-context wins (the head-pack read ~0 on primes yet is a real long-context lever). **Maintain a representative workload basket; report the vector, optimize the weighted sum.**
- Mental model (user's, correct): each technique/lever is a DIMENSION; start = global min; regressions = local minima; each banked win = displacement toward the current global max; the TRUE global max is the composition of all compatible levers.
- **Critical correction: dimensions are NOT independent.** Levers interact and the bottleneck MOVES (Amdahl): fixing the lm_head made the MoE the new #1; making the tc kernel fast is *why* cp.async then failed. So "sum of improvements" overcounts, ORDER matters, and you must **re-profile after every win**.

## 1. The two epistemic sources — MAP vs TERRITORY (use BOTH, interleaved)
- **RESEARCH = the MAP.** What's possible, what others measured, the exact recipes (Marlin, FlashInfer, roofline math, the dead-end refutations). Embarrassingly parallel, GPU-free.
- **PROFILER/BENCHMARK = the TERRITORY.** Which of OUR kernels is the actual bottleneck; whether a lever works on OUR machine; our tested dead-ends. Serial, GPU-bound, scarce.
- **Evidence they're both required:** our two biggest wins (+9.7% lm_heads, +9.3% draft-linears) needed the PROFILE to say *where* and RESEARCH to say *how*. **Research alone produced confident ERRORS** (an agent's #1 pick "FP8-draft" was a tested dead-end; "draft-depth-bound" mismodeled our block-diffusion draft). **Measurement alone produced shallow local tuning** (the 2%-scraping trap). Neither alone is sufficient.

## 2. The resource model (why the flow must be shaped this way)
- **The scarce resource is SERIAL GPU wall-clock** (build → load weights → run → A/B), not agent-tokens. Weight-load + inference overhead per test is real.
- **Speed-tests CANNOT be parallelized on this hardware** — concurrent model instances contend for the same 273 GB/s + compute, corrupting timings. So:
  - **Parallelism budget → RESEARCH & analysis** (GPU-free): fan out N web-search agents for breadth/depth/specificity.
  - **Serial GPU budget → only the high-EV experiments research has already vetted.** Never spend scarce serial time on experiments a good research pass would have deprioritized. (We violated this on the megakernel & cp.async arcs — weeks-equivalent of serial grinding the research later reframed in minutes.)
  - Sub-agents CAN run ancillary non-speed tests (correctness, compile checks, static analysis) in parallel.

## 3. THE LOOP (Research-Grounded Optimization Loop)
```
Phase 0  GROUND TRUTH (living docs): CONSTITUTION (won/lost/neutral ledger, dead-ends, profile, roofline).
Phase 1  PROFILE (cheap serial): ncu SoL + nsys per-kernel. The current gradient = the biggest time-share kernel.
Phase 2  DEEP RESEARCH CASCADE (parallel, GPU-free), GROUNDED in the constitution:
           - N agents, each an axis (engines / spec-decode / low-level-CUDA / ceiling+black-swans).
           - INJECT field data (dead-ends, profile) so they don't re-suggest tested losers & they target the ACTUAL bottleneck.
           - Each returns an EV-ranked surface WITH CITATIONS (real code/PTX/papers/numbers).
           - SYNTHESIS pass: reconcile agents, CORRECT against field data, drop dead-ends, rank by EV = gain×P(works)÷cost.
Phase 3  ORDER: research PRE-FILTERS what earns scarce serial GPU time. Interleave (a) cheap bit-exact grinds (bank them)
           with (b) exactly ONE black-swan bet (high-variance, roofline-breaking).
Phase 4  IMPLEMENT TO SATURATION: back-to-back A/B (thermal!), bit-exact gate, commit champions, REVERT regressions/neutrals
           immediately. Log LEADS noticed while building ("saw X while doing Y").
Phase 5  DETECT SATURATION (diminishing A/B, dead-ends piling): formulate the NEXT deep-research question from
           (accumulated leads + moved bottleneck + reframed goal) → GOTO Phase 2.
```

## 4. Disciplines (non-negotiable, each earned from a mistake)
- **Profile FIRST, minimally, then research — do NOT exhaustively shallow-tune before the deep research.** Just enough field data to (a) know the bottleneck and (b) have a dead-end ledger to ground the research. Beyond that, shallow tuning burns scarce serial GPU time on low-EV paths. (We over-explored shallow before the research that unlocked the big wins.)
- **Inject the constitution INTO the research prompt.** The model knows the MAP (Marlin, roofline) but not our TERRITORY (dead-ends, profile). Ungrounded agents re-suggest tested losers (FP8-draft). This is the single highest-leverage prompt move.
- **Research is a hypothesis generator, the benchmark is the judge.** NEVER ship a research rec unmeasured; NEVER tune without research vetting. cp.async LOOKED right (SoL said L1TEX-stall) but MEASUREMENT on our structure killed it.
- **Re-profile after every win** — the bottleneck moves (Amdahl).
- **Back-to-back A/B only** (thermal drift makes absolute numbers lie). Median-of-N, stash-baseline-rebuild-compare.
- **Bit-exact gate every change**; the spec-decode moat (bf16 draft) is sacred; verify guarantees output.
- **Reserve a black-swan budget.** Don't let the loop degenerate into only 2% grinds. Each cycle funds ONE high-variance roofline-breaker (this project: activation sparsity TEAL/CATS).
- **Adaptive goal evolution via synthesis, not drift.** The goal legitimately evolved beat-vLLM(108) → zenith(157/physics). Good evolution comes FROM the research reframe (we're acceptance-saturated → stop chasing tau-alignment; MoE-verify is the cap), not from aimless local wandering.

## 5. What "a good deep-research prompt" contains (the cascade quality bar)
1. Full context of what's already built + measured numbers (so it doesn't re-derive).
2. The explicit dead-end ledger (so it doesn't re-suggest losers).
3. The exact current bottleneck from the profile (so it targets the gradient).
4. Hardware constraints (Thor sm_110a, unified LPDDR5x, bit-exact, bf16-draft).
5. A demand for BOTH incremental grinds AND black-swan/saltatory leaps, each with expected gain × P(works) ÷ cost, citations to real code/PTX/papers, and a bit-exact/tau-preservation flag.
6. Multiple agents across axes (breadth) each going deep + specific, then a synthesis that CORRECTS against field data.

## 6. Anti-patterns (what starved us of big wins)
- Scraping 2% A/B tunes from only the field data in front of us + shallow web queries, instead of a structured research cascade. ← the biggest historical inefficiency.
- Spending serial GPU time BEFORE research pre-filtering.
- Trusting a research rec without measuring (or a measurement without research context).
- Optimizing one workload point as if it were the whole objective.
- Summing "composable" wins without accounting for the moving bottleneck.

## 7. The honest limit of "Claude already has the answer"
Largely TRUE for the map — the winning recipes (Marlin lm_head routing, the roofline, activation sparsity) were latent and a good research prompt surfaced them fast. FALSE for the territory — only OUR measurements know our bottleneck and dead-ends. So the meta-skill is **prompt-engineering the research cascade to fuse the model's latent map-knowledge WITH our injected field data**, then letting the benchmark arbitrate. That fusion, looped, is the method.
