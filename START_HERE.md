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
1. **Peak base + DSpark decode** — fastest kernels + the structural multipliers. **← WE ARE HERE.**
   **DECODE ENGINE BUILT + OPTIMIZED + spec-decode + CUDA graphs ALL DONE.** Full 43-layer M=1 KV-cache decode
   runs correct on the real 180B (argmax=270), every attention flavor bit-exact-gated. Decode optimized
   **0.50 → 7.89 tok/s (15.9×)** across 9 gated wins (structs-once, native e8m0/wo_a, arena, tc_fp8, M=1 fp8 GEMV,
   fp4 grouped MoE, fused ogroup, **M=1 ogroup GEMV −15%**, determinized MoE). Full **43-layer CUDA graph captured
   bit-exact** (device-pos) — measured PARITY (not launch-bound). **Spec-decode built**: M=K verify + DSpark head +
   accept-longest-prefix (currently ~parity — verify is 2.6× a decode; see below).
   **RESEARCH VERDICT (`DECODE_GAP_RESEARCH.md`): the gap to vLLM is BANDWIDTH EFFICIENCY, not algorithm.** We run
   at **~25% of peak (69 of 273 GB/s)**; well-written kernels hit 70–80%. vLLM's 24 tok/s used MTP2 (~2×); its
   no-spec rate is ~12–14 tok/s, so the real kernel gap is ~1.5–1.8×. **Next = §6.**
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
- **Full 43-layer M=1 KV-cache decode: 7.89 tok/s (126.7 ms/tok), argmax=270, all attention flavors bit-exact.**
  9 gated optimizations, 0.50 → 7.89 tok/s (15.9×) — see `OPTIMIZATION_LEDGER.md`. Latest win: **M=1 ogroup GEMV
  −15%** (the ogroup was scalar-byte reads in an m16 mma at ~40 GB/s; a warp-per-output GEMV fixed it).
- **CUDA graphs: full 43-layer step captured bit-exact** (device-pos, all 3 attention flavors + device-conditional
  compressor emit; gates `gate_{mla,compressed,indexer}_graph` cosine 1.0). Measured **PARITY (0.99×) → GPU-bound,
  NOT launch-bound.**
- **Spec-decode built**: M=K verify (weights read once for K) + DSpark draft head + accept-longest-prefix.
  Currently **~parity**: M=5 verify = 334 ms = 2.6× the M=1 decode, ~2.5 tokens accepted ⇒ S≈0.95. Determinized
  MoE raised acceptance 1.9→2.5.
- **ROOT CAUSE of the vLLM gap (measured, `DECODE_GAP_RESEARCH.md`):** active weights = 8.77 GB/tok → 32 ms floor
  at 273 GB/s; we run 126.7 ms = **~25% of peak bandwidth**. The gap is kernel bandwidth efficiency, not algorithm.
- **sm_110a hardware facts (empirically tested, CUDA 13.0):** `tcgen05` (SM100/DeepGEMM path) **NOT supported** →
  DeepGEMM = rewrite not port; `cp.async` **OK**; `__nv_cvt_fp4x2_to_halfraw2` (HW FP4×2 unpack) **OK**.

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

## 6. THE NEXT BUILD — execute `DECODE_GAP_RESEARCH.md` (close the bandwidth gap, then the spec win)
**STRUCTURAL_PLAN.md is fully DONE** (M=1 decode, zero-sync grouped MoE, arena, CUDA graphs, spec-decode — all
built, gated, committed). The forward plan is now the deep-research roadmap in `DECODE_GAP_RESEARCH.md`
(+ `RESEARCH_INVENTORY.md` = dedup key). Do these gated + detached + memory-neutral, in order:
0. **Unblock ncu** (`sudo ncu --set full -k regex:'fp8_gemv_m1|ogroup_gemv|k_grouped' --launch-count 20 ./build/decode …`)
   — confirm Memory% vs Compute% per kernel (proves the software-dequant compute-bound hypothesis). 30-min de-risk.
1. **T1.1 (⭐ top lever): rebuild the FP4 MoE GEMV with HARDWARE x2 unpack.** Our *rejected* fp4 GEMV used SCALAR
   nibble decode — the state-of-the-art path uses `cvt.f16x2.e2m1x2` HW unpack (confirmed on sm_110a) + `cp.async`
   streaming + L1 cache hints (no_allocate weights / evict_last activation) → ~50% BW. MoE is the largest kernel.
2. **T1.2: fuse the attention/indexer/compressor glue** (RoPE+norm+quant+KV-write) — vLLM reports 2–20× on these;
   we run each as a separate kernel with an arena DRAM round-trip.
3. **T2.1: fix the M=K verify expert-union dilation** — read each activated expert ONCE (group verify tokens by
   expert). 2.6× → ~1.3× flips spec-decode parity → ~1.9× at unchanged acceptance. FIRST audit whether
   `k_grouped_w4a8` verify already dedups by expert.
4. **T2.2: DSpark draft-head fine-tune** (accept 2.5→~4, compounds T2.1 to ~3×). Training-gated (`CAPTURE_TRAIN_PLAN.md`).
**Deprioritized (evidence in the research doc):** tcgen05/DeepGEMM (arch-blocked on sm_110), PDL + in-graph-metadata
MTP (subsumed by our CUDA-graph parity), DeepEP (N/A single-node), cluster/DSMEM kernels (porting risk).
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
