# CONSTITUTION — DSpark→REAP draft-head fine-tune & inference server

The governing charter for this project: fine-tune the DSpark speculative-decoding draft head onto
`0xSero/DeepSeek-V4-Flash-180B` (REAP K160, NVFP4/FP8) and serve the pair at maximum decode on Jetson
Thor — such that the **artifact** (REAP model + fine-tuned head) is correct, portable, and beats DFlash,
while the **tooling** (our Thor CUDA — a preserved, first-class repo artifact) is fast and never
contaminates the artifact.

Ethos: **formal correctness is front-loaded and scarce.** Every step earns the next by producing a
go/no-go signal against a fixed reference. We spend correctness before we spend speed, and we spend
minutes on gates to protect days on the fine-tune. Silent success is the enemy; loud, early, cheap
failure is the goal. Kernel-craft is delegated to `CUDA_ENGINEERING_CONSTITUTION.md`; this document
governs the whole arc — data → training → head → server → portability.

---

## Article 0 — The shape of the problem: two artifacts, two lifetimes

There are exactly two things we produce, and they must never be confused:

1. **The PRODUCT — portable artifacts.** The REAP-180B target (given) and the fine-tuned DSpark draft
   head (what we make). They are `safetensors` + a standard `deepseek_v4` MTP semantic contract. Their
   correctness is a **hardware-independent mathematical property**: *given the REAP target's hidden
   states and input ids, the head reproduces the target's next-token distribution over a block well
   enough to be accepted.* This property is identical on Thor, a DGX Spark, or a B200, under vLLM,
   SGLang, or our binary. The product outlives every line of our CUDA.

2. **The MEANS — hardware-specialized tooling (a first-class, preserved artifact).** Our Thor `sm_110a`
   CUDA: the model-architecture primitives, the draft-head **fine-tuning** kernels, and the **inference
   server**. Speciated to this die's roofline, smem, NVFP4/FP8 intrinsics. They exist to (a) capture
   training data from the real target fast, (b) train the head, (c) serve it at peak decode locally.
   **"Speciated" does NOT mean "disposable."** Every line of this CUDA is a deliverable in its own right —
   the hand-rolled low-level exposition of this model's architecture, its draft head, the fine-tuning
   process, and the serving engine — and **all of it lives in this repo, version-controlled, never
   throwaway scratch.** It is *portable-in-principle* (re-tunable for the next chip via intrinsic
   substitution + retuning, per the CUDA_ENGINEERING_CONSTITUTION), but it is *retained-in-practice*: the
   repo is the canonical home of both products — the portable weights **and** the Thor CUDA that runs them.
   The distinction from item 1 is only about *portability of the artifact* (weights run anywhere; these
   kernels are Thor-tuned), never about whether the CUDA is worth keeping. It is. Keep it, in the repo.

**The bridge between them is the only reason this works:** a hardware-locked tool that emits a
hardware-independent artifact is valid **only if the tool's numerics equal the reference's numerics.**
That is why every kernel is gated bit-exact/tolerance against a portable PyTorch oracle
(`ref/`). We are allowed to be fast and Thor-specific *because* we prove we compute the same thing the
reference does.

**Success, precisely.** The fine-tuned head, serving the REAP target, achieves acceptance length τ and
decode tok/s that **beat DFlash's proven bar** (on Gemma-4 DFlash we measured τ 13.33 @ ~118 tok/s;
the DeepSeek analog target is τ ≈ 4–5.7 → ~2.2–2.95× → ~41–56 tok/s over the 18.946 no-MTP baseline),
with correct outputs at production context length and KV capacity, **reproducible on foreign hardware.**

---

## Article I — Axioms (non-negotiable)

1. **Correctness precedes speed.** No kernel is optimized before it passes its gate. A fast wrong kernel
   is worth less than no kernel.
2. **No silent failures.** Every stage emits an explicit go/no-go. Assertions over comments; measured
   numbers over assumed ones; loud crashes over quiet drift.
3. **Determinism.** Seeded, reproducible, gated against a fixed golden. Same input → same output, or we
   find out why before proceeding.
4. **Train and serve must match.** Precision (NVFP4 experts / FP8 linears / FP8 KV), data distribution
   (on-policy from the *modified* REAP+NVFP4 target), tokenizer, and RoPE/YaRN are **identical** across
   training and serving. The head is trained on the exact distribution it will serve.
5. **Portability separation.** No Thor constant, kernel assumption, or non-standard repack may enter the
   artifact. Kernels are speciated; weights are portable. If the head only works with our kernels, we
   failed.
6. **The reference is truth.** `ref/` (pure-torch oracle) *defines* correctness; our CUDA *approximates*
   it within a declared tolerance and never redefines it. When they disagree, the CUDA is wrong until
   proven otherwise.
7. **Scarcity of trust.** Every published number is a hypothesis to reproduce, not a fact to lean on
   (96.66 GiB → measured 96.02; 18.946/24.378 tok/s; τ figures; 537K KV). We proceed on *our* measured
   number.
8. **The weights are single-tenant.** The ~96 GiB target fills most of Thor's 122.8 GiB unified pool;
   **only one process may hold the full model at a time** (torch reference *xor* CUDA serving *xor*
   data capture — never two). Full-model steps are sequenced, never overlapped. Per-op / per-block
   goldens (a few GB) are exempt. Loading two full copies is an instant OOM, not a slow-down.

---

## Article II — The correctness ladder (gates earn spend)

Each gate is a **stop condition**. Minutes here protect the 1–2 day fine-tune. (Full detail in
`ADAPTATION_PLAN.md`.)

| Gate | Proves | Stop if |
|---|---|---|
| **0** Thor/ARM64 compat | the stack runs at all | any core dependency won't run *(passed: pure-CUDA path chosen; tilelang/SGLang avoided)* |
| **1** Baseline serving | REAP loads & decodes near published numbers | footprint or decode wildly off *(memory passed: 96.02 GiB measured)* |
| **2** Unfine-tuned head | official head gives **non-zero** τ on REAP | τ ≈ 0 → pruning shifted distribution too far; stop, reconsider 162B |
| **3** Data sanity | generated self-distillation data is not degenerate | spot-check reveals garbage |
| **K** Kernel numeric | each CUDA kernel matches `ref/` within tolerance | any kernel exceeds its declared max-abs-rel / cosine bound |
| **P** Portability | fine-tuned head reproduces τ on a *foreign* stack (vLLM on DGX Spark / B200) | works only on our Thor kernels |

**Gate 2 is the cheapest, most decisive test in the program** — it alone says whether the fine-tune is
worth running. Record its pre-training τ precisely; the fine-tune's entire job is to move that number.

---

## Article III — Training / fine-tune discipline

**Objective (research-locked; supersedes the original directive — see `ADAPTATION_PLAN.md §4`):**
per-depth **token cross-entropy** (decay-weighted, β≈0.6) **+ logit-KD/KL** against the REAP target's
captured top-k logits. **Not** hidden-state feature regression (EAGLE-3 abandoned it; DeepSeek MTP is
natively token-CE).

**Regimen:** warm-start from the official DSpark head (never scratch); **LR 5e-5 cosine** (drop to 2e-5
if unstable at 180B scale); **1–3 epochs, watch the loss curve, not a fixed step count**; AdamW(0.9,0.95),
small global batch, backbone frozen, only the ~3% head params trainable.

**Data — the single highest-leverage decision:** **on-policy self-distillation from the modified
(REAP+NVFP4) target itself** (measured 1.81× vs 1.67× for fixed corpora). 68k–390k examples spanning
general + math + code + your agentic/tool traces. Captured through *our* Thor engine because it runs the
real quantized target fast; the captured tensors are the training set.

**The deliverable is head weights in the checkpoint's exact `mtp.*` tensor schema** — so they load in
vLLM/SGLang unchanged (Axiom 5).

**Anti-silent-failure invariants for training:**
- NaN/Inf guard on loss and every activation dump; abort on first non-finite.
- τ on a held-out set logged every N steps — the *real* objective, not just loss.
- Gradient-norm and update-ratio sanity bounds; warn on collapse or explosion.
- Seeded, logged data order; every run reproducible from (seed, data manifest, config hash).
- Checkpoint every epoch **and immediately re-load + re-validate** it (a checkpoint that won't reload is
  a silent failure caught late).
- Training kernels (Thor CUDA forward *and* backward) gated against the `ref/` oracle before any run.
- ETA from a rolling window after warmup (never step-1 extrapolation).

---

## Article IV — Inference-server discipline

- **Every kernel is gated** (Gate K) against `ref/` golden activations before it is trusted or tuned.
  The oracle is stood up (`ref/gen_golden.py`); the boundary goldens are the contract.
- **Decode is roofline-aware.** KV headroom is **measured** (DSA `index_topk=512` + MLA single latent
  govern real read cost, not naive per-token math), never assumed. Establish the empirical KB/token and
  the practical context ceiling at production load, then serve under it.
- **Speculative acceptance is verified, never trusted.** The draft proposes a block; the target verifies;
  accept the longest valid prefix via lossless sampling / sample-match (temp=0 ⇒ exact greedy). A draft
  token is accepted only when the math says the target would have produced it. τ is *measured*, not
  reported by the drafter.
- **The production contract is standard even though the kernels are not:** OpenAI-compatible streaming,
  FP8 KV, prefix caching, reasoning/tool parsing. Behavior is portable; implementation is speciated.

---

## Article V — Portability guarantee (the artifact contract)

1. The fine-tuned head ships as `safetensors` in the exact `mtp.*` schema with a `deepseek_v4` config
   that vLLM/SGLang already understand. No custom loader required to *use* it (only to make it fast here).
2. **Head correctness is defined hardware-independently:** given REAP hidden states + input ids, the head
   reproduces the reference logits within tolerance and yields τ ≥ target. This definition names no
   device.
3. **Portability gate (P):** load head + REAP on a foreign stack (vLLM on DGX Spark / B200) and reproduce
   τ within tolerance. Passing only on our Thor kernels is a **failure**, not a success.
4. Nothing irreversible or non-standard (a Thor-specific repack, a fused constant, a kernel-shape
   assumption) may enter the artifact. Our repacks for speed are applied at *load*, never *saved*.

---

## Article VI — Anti-silent-failure catalog (mode → guard)

| Silent failure mode | The guard that makes it loud |
|---|---|
| Quant scale/block mismatch (fp8 128×128, fp4 32) | per-block scale-shape assertions + Gate-K golden compare |
| Shape broadcast hiding a wrong dim | explicit shape asserts at every module boundary (the loader already does 1383 of them) |
| dtype coercion (fp8↔bf16↔fp32) silently degrading | explicit dtype checks; reference computes the intended path |
| Draft "accepts" tokens the target wouldn't | sample-match / lossless verify; acceptance is proven per token |
| Tolerance creep (loosening bounds to pass) | tolerances fixed in-repo; loosening requires a written amendment |
| Train/serve distribution drift | on-policy regeneration from the *served* quantized target |
| τ overfit to training distribution | ABBA eval on held-out **and** deliberately OOD prompts |
| KV under-provision at long context | measured capacity gate at production length, not the 200K benchmark point |
| Wrong MTP head attached (REAP built-in vs DSpark) | Gate 2 benchmarks both; they are structurally different (confirmed at tensor level) |
| Reference itself wrong | reference cross-checked op-by-op against the documented spec (`reference/DEEPSEEK_V4_MODELING_NOTES.md`, file:line-cited) |

---

## Article VI·5 — Explainability is permanent (the "why" is a first-class artifact)

CUDA is complex; the transferable value is not just the fast kernel but **how and why it got fast, what
iterative steps produced the speed, and the machine-learning fundamentals of every component.** This is a
standing obligation, not a one-time doc:

1. **Every gate/iteration logs finding → root cause → fix → why-correct** in `GATE_LOG.md`. When a gate catches
   something or an iteration lands a measured change, add the entry. The commit message states the same.
2. **Every optimization records the measured A/B delta + the mechanism** (why it's faster, on which roofline
   bound) in `OPTIMIZATION_LEDGER.md`. No "it's faster" without the number and the reason.
3. **Every component carries its ML fundamentals** — *why the architecture does this* (e.g. MLA = latent KV to
   cut KV bandwidth; DSA = O(Lk) sparse attention via a lightning indexer; Hyper-Connections = learned residual
   mixing; fp4 experts = W4 memory for the bulk params; block-diffusion draft = parallel multi-token proposal;
   Markov head = bigram correction of the AR block). Captured in `reference/DEEPSEEK_V4_MODELING_NOTES.md`
   (numeric spec) + `GATE_LOG.md` (build rationale). A reader must be able to learn *why*, not just *what*.
4. **Research is preserved verbatim** (`RESEARCH_*.md`) so the evidence behind decisions is auditable.
5. The trail is durable and ordered: `git log` (per-commit rationale) → the docs above. Losing the "why" is a
   defect, same as losing the code.

## Article VII — Definition of done

1. Gates 0, 1, 2, 3 green with **recorded, measured** numbers.
2. Every CUDA kernel (training + inference) passes Gate K against `ref/`.
3. Head fine-tuned per Article III; loss converged, τ tracked throughout.
4. ABBA eval: fine-tuned head vs unfine-tuned (Gate-2 baseline) vs no-spec, on held-out **and** OOD.
5. Measured decode tok/s and τ **beat the DFlash bar** at production context + KV.
6. **Portability gate P green** on a foreign device — the head is the product, and the product travels.

---

## Article VIII — Amendments

Kernel-craft discipline (profile-first, the won/lost/neutral ledger, roofline tuning) is delegated to the
inherited `CUDA_ENGINEERING_CONSTITUTION.md`. This constitution governs the arc above it. Tolerances,
gates, and the objective are amendable only in writing, with the reason recorded — never loosened
silently to make a red gate green. Amending a gate to pass is the one failure this document exists to
prevent.
