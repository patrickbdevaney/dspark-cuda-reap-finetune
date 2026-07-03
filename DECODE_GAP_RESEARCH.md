# Closing the Decode Gap — Deep Research Findings (Step 4 report)

## Executive summary — the gap is half what it looked like, and the cause is bandwidth efficiency

- **The 2.5–3× "gap" is really ~1.5–1.8× of kernel gap + ~2× of MTP.** The vLLM 24.4 tok/s figure was measured **with MTP2 speculative decode**; the same stack **without spec does ~12–14 tok/s** ([0xSero card](https://huggingface.co/0xSero/DeepSeek-V4-Flash-180B), [lmxxf DGX-Spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark)). And it was on **GB10/sm_121 (DGX Spark)**, not Thor — same ~273 GB/s class, different chip. Real pure-kernel gap vs generic vLLM is **12–14 → 7.89 ≈ 1.5–1.8×**; the rest is MTP (our Phase 3).
- **Step-2 profiling (analytical, hard number):** active weights/token = **8.77 GB** → floor 32 ms at 273 GB/s; we run 126.7 ms = **~69 GB/s = 25% of peak**. Well-written decode kernels reach **70–80%** (Flash Compressor 80% on H200; L4 batch-1 81% — [arXiv 2605.30571](https://arxiv.org/html/2605.30571)). **Right algorithm, kernels at ~1/3 the bandwidth they should get.** (ncu SpeedOfLight blocked by `ERR_NVGPUCTRPERM` — needs `sudo ncu` to confirm per-kernel compute-vs-memory-bound; software-dequant compute-bound strongly suspected.)
- **Hardware ground truth on this box (empirically tested, CUDA 13.0, sm_110a):**
  - `tcgen05.*` (5th-gen TC / UMMA, the SM100 DeepGEMM path) → **NOT supported** (`Instruction 'tcgen05.fence' not supported on .target 'sm_110'`). **⇒ DeepGEMM's SM100 fp8×fp4 kernels are a REWRITE, not a port.**
  - `cp.async.cg.shared.global` → **OK.** Async weight streaming / software pipelining available.
  - `__nv_cvt_fp4x2_to_halfraw2` (hardware FP4×2→half2 unpack) → **OK.** The fast NVFP4 GEMV primitive works.

---

## Tier 1 — implementable now, no arch gate, attacks the 25%→70% bandwidth gap (largest self-contained levers)

### T1.1 — Rebuild the FP4 MoE GEMV with HARDWARE x2 unpack + streaming (⭐ top lever)
- **Category 3a/3c. Sources:** [amandeepsp "Twelve Attempts at an FP4 Kernel"](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/) + [code](https://github.com/amandeepsp/cuda) · [mufeezamjad NVFP4 grouped-GEMM](https://mufeezamjad.com/blog/nvfp4-group-gemm) · [LiquidGEMM arXiv 2509.01229](https://arxiv.org/html/2509.01229v1).
- **What:** our fp4 GEMV was **rejected — but it used SCALAR nibble decode** (~a dozen ops/elt = ~21% warp stalls, per LiquidGEMM). The community M=1 NVFP4 path is different: **`cvt.rn.f16x2.e2m1x2` hardware unpack (2 fp4→2 half in one op, confirmed on sm_110a)** + `cp.async` weight streaming + **L1 cache hints** (`no_allocate`/`evict_first` for streamed weights, `evict_last` for the reused activation) + 128–256-bit vectorized loads + register-capped occupancy. Reaches **~2× off speed-of-light (~50% BW)**.
- **Applicability:** HIGH — arch-portable, intrinsics compile on sm_110a, no tcgen05/DSMEM. **Adaptation risk: LOW.**
- **Impact:** **Step-change.** MoE ≈ 44% of per-token bytes and `k_grouped_w4a8_e8m0` is the single largest kernel (507 ms across the run). ~25%→~50% BW ≈ **~1.5–2× on the MoE path.**
- **Status vs inventory:** **NEW** — our "fp4 GEMV rejected" was the *scalar-decode* version; the hardware-x2-unpack version is untested and is the state of the art.

### T1.2 — Fuse the attention / indexer / compressor glue kernels
- **Category 3a/3c. Sources:** [vLLM DeepSeek-V4 blog](https://vllm.ai/blog/2026-04-24-deepseek-v4) · [LMSYS "Flash Compressor"](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/).
- **What:** vLLM measured — **Q-norm+KV-RoPE+K-insert = 10–20×**; **inverse-RoPE+fp8-quant = 2–3×**; **compressor+RMSNorm+RoPE+cache-write = 1.4–3×**. We run each of RoPE / rmsnorm / act_quant / KV-write / compressor-pool as a **separate kernel with an arena DRAM round-trip**. Flash Compressor fuses the 5-stage chain to one on-chip pass (5→2 HBM round-trips, ~80% BW).
- **Applicability:** HIGH — pure fusion on HW we control, no arch gate. **Risk: LOW–MED** (gate each fusion cosine 1.0).
- **Impact:** **Incremental-to-large, stackable.** Smaller than T1.1 at M=1 (tiny activations) but the profile shows rmsnorm/act_quant/k_deq/rope/hc collectively non-trivial, each paying launch + DRAM round-trip.
- **Status:** **NEW.**

### T1.3 — Fuse MoE gather+GEMV+combine; adaptive grid sizing at low M
- **Category 3c. Sources:** [SonicMoE arXiv 2512.14080](https://arxiv.org/abs/2512.14080) (+28.7% over DeepGEMM via gather-into-load fusion) · [SGLang FP4 MoE 1.78× @ M=1](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison) (adaptive grid sizing; 99.3% SMs idle at M=1).
- **What:** don't run gather→pad→GEMM as separate launches; **fuse the top-6-of-160 expert-weight gather into the GEMV load stage** (on unified memory it's strided reads, hideable under the bandwidth-bound stream). **Adaptive grid sizing** at M=1 so SMs aren't idle.
- **Applicability:** HIGH — algorithmic, portable. **Risk: LOW.** **Impact:** **Step-change vs multi-launch baseline.** Folds into T1.1. **Status:** NEW.

---

## Tier 2 — Speculative decode (the other ~half of the gap; Phase-3 direction)

### T2.1 — Fix the M=K verify cost: read each activated expert ONCE (⭐ central spec finding)
- **Category 3b. Sources:** [Utility-Driven SpecDec for MoE, MICRO'25, arXiv 2506.20675](https://arxiv.org/pdf/2506.20675) · [DISCO arXiv 2405.04304](https://arxiv.org/html/2405.04304v1).
- **What:** on a bandwidth-bound engine an M=5 verify reading each weight once should cost **~1.0–1.4× the M=1 decode**, not our **2.6×**. Leading suspect: **MoE expert-union dilation** — the 5 speculative tokens activate the *union* of expert sets; if the verify MoE reads per-(token,expert) instead of once-per-activated-expert, traffic balloons. **Fix:** group the K verify tokens by expert, one grouped GEMM per activated expert (|union| ≪ 5·top-k, nearby tokens share experts). **ACTION: audit whether `k_grouped_w4a8` verify dedups by expert or re-reads per token.**
- **Applicability:** DIRECT. **Impact:** **Step-change** — c_v 2.6→~1.3 flips spec **parity → ~1.9× at unchanged acceptance** (S=a/c_v=2.5/1.3).
- **Status:** **NEW** — logged spec work optimized *acceptance* + *launch*, never *verify weight-traffic*.

### T2.2 — DSpark draft-head fine-tune (acceptance 2.5 → ~4)
- **Category 3b. Sources:** [EAGLE-3 arXiv 2503.01840](https://arxiv.org/html/2503.01840v1) · [DSpark (marktechpost)](https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/) (vendor-reported, unreproduced).
- **What:** to WIN 2× at our 2.6× verify needs a=5.3 = **~2× the MTP-1 ceiling (~2.9 at α=0.70)** — unreachable by tuning. A DSpark-class head → a≈4 → S≈1.5× alone, **compounds with T2.1** to a=4,c_v=1.3 ⇒ **S≈3×**. **Gated by training, not kernels.** **Status:** NEW (quantifies the ceiling).

### T2.3 — Static-optimal K + light dynamic length
- **Category 3b. Sources:** [DISCO](https://arxiv.org/html/2405.04304v1) (+10.3%), [DSDE arXiv 2509.01083](https://arxiv.org/html/2509.01083v1). High verify cost ⇒ **small optimal K**; draft is nearly free. **Impact:** incremental (≤~13%). **Status:** NEW.

---

## Tier 3 — Deprioritize / negative results (evidence-backed; don't spend effort)

- **PDL:** ≤10–15%, second-order — **CUDA-graph parity already closed the launch-overhead hole.** Arch-gated. **Covered.**
- **In-graph-metadata MTP:** fixes **Python/host launch overhead** we don't have (M=1 single-stream, graph parity). ~0%. **Covered.**
- **tcgen05 / FP4 tensor cores for M=1:** **not on sm_110** (empirical) *and* wrong tool — M=1 bandwidth-bound, 99.3% SMs idle. A trap. **Rejected (reinforced).**
- **DeepGEMM drop-in:** blocked (SM100/tcgen05 absent). Reimplement the *ideas* (masked grouped MoE GEMM, gather fusion → T1.3/T2.1); the *library* isn't portable. ([PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304), [vLLM #41063](https://github.com/vllm-project/vllm/issues/41063))
- **DeepEP / single-node EP:** N/A — one unified address space. Descendant idea (overlapped gather) → T1.3. **Rejected (clean negative).**
- **Lightning TopK / cluster+DSMEM:** public algorithms but lean on **thread-block clusters/DSMEM** (broke on SM120/121) — validate on sm_110a first; indexer topk likely not our bottleneck. **Risk HIGH.**
- **Hierarchical multi-stream overlap:** portable but **a single memory bus can't overlap past the bandwidth wall.** Incremental once one stream saturates DRAM.

---

## Recommended execution order

1. **Unblock ncu** (`sudo ncu --set full -k regex:'fp8_gemv_m1|ogroup_gemv|k_grouped' --launch-count 20 ./build/decode ...`) → confirm Memory% vs Compute% per kernel, proving the software-dequant compute-bound hypothesis. De-risks all below.
2. **T1.1 — hardware-x2-unpack FP4 MoE GEMV** (+ T1.3 gather fusion, adaptive grid). Biggest self-contained lever.
3. **T1.2 — fuse attention/indexer/compressor glue.** Stackable, no arch gate.
4. **T2.1 — audit + fix verify MoE to read each expert once.** Flips spec parity → ~2× free.
5. **T2.2 — DSpark draft-head fine-tune** (compounds T2.1 to ~3×). Training-gated.

**Realistic ceiling:** the ~273 GB/s wall sets the *no-spec* target at ~12–14 tok/s (vLLM's no-spec). T1 targets 7.89 → ~12–14 (bandwidth efficiency); T2 targets ~2–3× on top → the ~20–24 regime. **Chasing datacenter-Blackwell TC kernels (tcgen05/DeepGEMM) is the wrong direction for this hardware + batch size.**
