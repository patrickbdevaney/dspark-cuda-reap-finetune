# START_HERE.md — session entry point (read this first, then execute STRUCTURAL_PLAN.md)

**New session? Read this file top-to-bottom. It is the authoritative orientation.** It tells you what this
project is, the full arc, exactly where we are, the hard constraints, how to build/run/gate, and the precise
next build. Then execute `STRUCTURAL_PLAN.md`. Do not re-derive — the detailed "why" is in the linked docs.

---

## 1. What this project is
Pure-CUDA, hand-rolled inference + offline draft-head fine-tune for **DeepSeek-V4-Flash-180B-REAP** (K160,
NVFP4/FP8 MoE, 160 experts top-6) on **one Jetson Thor** (aarch64, Blackwell **`sm_110a`**, CUDA 13, **122.8 GiB
unified memory shared with the host/OS**). Repo: `/home/patrickd/dspark-cuda-reap-finetune/` (all git-committed).
Every kernel is gated **bit-exact / cosine-1.0 vs a PyTorch oracle** before it is trusted. All CUDA is a
preserved first-class repo artifact.

## 2. The arc (the bigger picture — hold this)
1. **Peak base + DSpark decode** — fastest kernels + the structural multipliers. **← WE ARE HERE.** Kernel phase
   DONE (1.81× banked). Next: the **decode engine** (see §6).
2. **OpenAI-compatible, feature-rich inference server** — vLLM/SGLang parity (streaming, tool-calling schema,
   think-block/reasoning delineation, terminal + WebUI clients, configurable KV, prefix cache), adapted for
   DeepSeek-V4-Flash. Memory-lean, quick-start, faster decode than vLLM/SGLang on Thor. **The banked product win.**
   Inherit the gemma-cuda-hybrid server abstractions (see `reference/GEMMA_ENGINE_README.md`).
3. **Capture + fine-tune the DSpark draft head** — representative on-policy capture (NOT domain-specialized: the
   head must predict what the base model says) + block-acceptance training for max τ / decode throughput.
   See `CAPTURE_TRAIN_PLAN.md`.
KEY INSIGHT (why 1→2 merge): the **decode engine** (static M=1 KV-cache step + grouped-GEMM + CUDA graphs +
spec-decode) IS the core of the server. Build it once; it delivers the multipliers AND backs the API.

## 3. Where we are RIGHT NOW (measured, gated, correct)
- **Gates 0/K/1/1.5/2 all PASSED.** Full 180B runs on Thor, numerically correct ("Paris"), and the **unfine-tuned
  DSpark draft head transfers to REAP at τ@0 = 0.815** (Gate 2 GO — light fine-tune should suffice).
- **Warm (2× warmup, steady-state, s=8 prefill): 319.8 ms/tok (3.13 tok/s) = 2.15× over the 687 baseline.**
  (was 378.8; STRUCTURAL_PLAN **Step 1b zero-sync grouped-GEMM MoE** landed −12.5% same-session A/B: 365.7→319.8,
  argmax=270, memory-neutral — and it removed the last per-layer host sync so the MoE is now graph-capturable.)
- **Banked champions (all gated cosine 1.0, memory-safe):** `tc_fp8_gemm` (dense/attn W8A8, 17.9×) · `tc_fp4_gemm`
  (MoE W4A8, 19.7×) + **repack-at-load** (in-place, zero extra memory) + **funnel-shift** aligned coalesced load ·
  **batched** MoE dispatch · **TC attention-output** (fp16 mma per group) · **device-side MoE routing** (GPU
  counting-sort, Step 1). Flags in `forward.cu`: `g_tc_fp8`, `g_tc_ogroup`, `m.use_tc_pp/batched/device_route`.
- **Kernel-compute is TAPPED**: the last kernel lever (funnel-shift) gave only −1.3%; profile shows compute is a
  small slice now. Remaining ~14× to the 38–50 tok/s target is **structural + the M=1 decode regime**, not more GEMM.

## 4. HARD CONSTRAINTS (violating these has already cost a power-cycle — do not repeat)
- **MEMORY: the forward uses ~90% of the 122.8 GiB unified RAM, which is SHARED with the host/OS/Claude Code.**
  A +5.5 GiB dequant-cache starved the system and forced a physical power-off. **NEVER add persistent memory.**
  Every optimization must be **memory-neutral** (in-place / fixed reused arena, never on-top). Prefer kernels that
  read quantized data directly over fp32 dequant caches.
- **ALWAYS run the forward detached-to-file** so it survives SSH loss and never locks the user out:
  `setsid nohup ./build/forward <model> <ids> > /home/patrickd/OUT.log 2>&1 < /dev/null &` — then monitor the file.
- **NEVER loosen a gate.** A speed win with wrong numbers is a loss. Gate cosine-1.0 vs the prior/oracle path first.
- **CUDA compiles + runs on the HOST** (`nvcc -arch=sm_110a`). Goldens are in `ref/goldens/` (generated in a
  CPU-torch container). Never `--runtime nvidia` (wedges the device).
- **FP4 tensor-core COMPUTE is blocked on Thor in CUDA 13.0** (all paths: ptxas, cuBLASLt=0 algos, CUTLASS, SASS).
  fp4 is used for STORAGE (bandwidth) only; FP8 mma is the compute ceiling. Re-test with
  `tools/cublas_fp4_probe.cu` after each CUDA update. See `FP4_COMPUTE_NOTE.md`.

## 5. How to build / run / gate
- **Build forward:** `bash scripts/build_forward.sh` → `build/forward`
- **Build + run gates:** `bash scripts/build_gate.sh` → `./build/gate_units ref/goldens` (must end `Gate K: PASS`)
- **Run forward (DETACHED — required):**
  `cd /home/patrickd/dspark-cuda-reap-finetune && setsid nohup ./build/forward /home/patrickd/models/DeepSeek-V4-Flash-180B "671,6102,294,8760,344,270,106523,294" > /home/patrickd/run.log 2>&1 < /dev/null &`
  then watch `/home/patrickd/run.log`. **Correctness check: last-token `argmax=270`.** Warm ms/tok is `[pass 1]`.
- Model weights: `/home/patrickd/models/DeepSeek-V4-Flash-180B` (~96 GiB, 46 shards). Reference (unpruned) DSpark
  head + `model.py`: `/home/patrickd/models/DeepSeek-V4-Flash-DSpark-head/`.
- A single forward run monopolizes the device for ~10–90 s at ~108 GiB — batch measurements; warn before running.

## 6. THE NEXT BUILD — execute STRUCTURAL_PLAN.md (the decode engine = the multiplier phase = the server core)
Do this as ONE focused build, gated + detached, memory-neutral. Order (details + rationale in `STRUCTURAL_PLAN.md`):
1. **Static M=1 KV-cache decode step** (Step 4 there) — the real decode regime (we've only measured 8-tok prefill).
   MLA/attention over cached latent KV (append new token's KV, attend over history). KV is tiny (MLA+SWA+DSA).
   **Execution-ready design in `DECODE_STEP4_DESIGN.md`** (append-only KV proven tractable incl. the compressor;
   equivalence gate = decode logits == prefill logits[s-1]). This is the NEXT build to execute.
2. **Zero-sync grouped-GEMM MoE** (Step 1b) — **DONE + gated + WIN (−12.5%, 365.7→319.8 ms/tok, argmax=270).**
   One grouped W4A8 launch per stage over all experts; tile→expert map built on-device from `off[]` → the last
   per-layer host sync is gone → MoE is graph-capturable. `g_moe_grouped` default-on. Next levers are 3/4/5 below.
3. **Pre-allocate every buffer + static launch sequence** (Step 2) — remove all mid-forward `cudaMalloc/Free`
   (Loader per-layer dequant, pp `x16`, funnel temps). Memory-neutral (same peak, allocated once).
4. **CUDA-graph capture** of the M=1 step (Step 3) — `cudaStreamBeginCapture`/`EndCapture`, instantiate,
   `cudaGraphLaunch` per token. Kills the hundreds of launch overheads that dominate at M=1. Gate: identical logits.
5. **DSpark spec decode** (Step 5) — draft head proposes block_size=5, target verifies in one graph-launched
   forward, accept longest matching prefix (τ≈0.75–0.8), overlap draft∥target. ~2.5–4× throughput multiplier.
Then this same engine becomes the server's core (arc step 2).

## 7. Discipline (Constitution — non-negotiable)
Correctness-first; gate every kernel cosine-1.0 vs oracle; **log finding→root-cause→fix→why in `GATE_LOG.md`
and measured A/B + mechanism in `OPTIMIZATION_LEDGER.md` for EVERY gate/iteration** (Art VI·5, explainability is
permanent); memory-neutral only; detached runs; champion–survivor A/B with a stop-rule (don't thrash sub-10%);
periodic black-swan literature search when rungs plateau (preserve to `RESEARCH_*.md`). Commit messages carry the
rationale. Co-Author trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## 8. Document map (where the detail lives)
- `CONSTITUTION.md` — rules/axioms (incl. Art VI·5 explainability, memory discipline).
- `ROADMAP.md` — full phase state & sequence.
- **`STRUCTURAL_PLAN.md` — the 5-step decode-engine/multiplier plan to EXECUTE next.**
- `DECODE_HORIZON.md` — grind ladder to 38–50 tok/s + scoreboard + hard memory constraint.
- `GATE_LOG.md` — finding+rationale per gate/iteration (the "why", incl. the memory power-cycle lesson).
- `OPTIMIZATION_LEDGER.md` — measured A/B deltas per lever.
- `FP4_COMPUTE_NOTE.md` — Thor FP4 status (blocked, re-test probe) — shared with gemma-cuda-hybrid.
- `CAPTURE_TRAIN_PLAN.md` — draft-head capture + fine-tune plan (arc step 3).
- `reference/` — `model.py` (REAP), modeling notes, `GEMMA_ENGINE_README.md` (server abstractions to inherit).
- `git log` — every commit states the finding + rationale.

**TL;DR for a fresh session:** the 180B runs correct on Thor at 378.8 ms/tok warm (1.81×, kernels tapped &
gated). Memory is the binding constraint (shared unified RAM — no persistent additions, run detached). Next:
build the **static M=1 KV-cache decode engine** per `STRUCTURAL_PLAN.md` (zero-sync grouped-GEMM + pre-alloc +
CUDA graphs + spec-decode) — it's where the multipliers pay off AND it's the core of the OpenAI server that comes
after, before the draft-head capture+fine-tune.
