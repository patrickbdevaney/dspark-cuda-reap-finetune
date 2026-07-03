# MEMORY.md — first-principles memory plan (122.8 GiB unified LPDDR5x, one Thor)

**Verdict: the model fits and runs, with headroom, once we stop counting reclaimable page cache. KV is a
non-issue by architecture; the only real pressure is the 96 GiB of weights, which is unavoidable when the
full model must be resident (forward / capture / serve) and is ABSENT during training (offline cache).**

## What actually consumed 120.7 GiB in the Gate-1 run (profiled)
| Component | Size | Nature |
|---|---|---|
| Weights, pinned `cudaHostAlloc(Mapped)` | **96.0 GiB** | irreducible when full model resident |
| Page cache from `pread`'ing the 96 GB file | ~10–14 GiB | **reclaimable**; Tegra `cudaMalloc` won't auto-evict → looked "used" |
| OS + GNOME + RustDesk + docker | ~9 GiB | GUI itself only ~1–2 GiB RSS |
| fp32 dequant (head 2.1 GB persistent + ~0.65 GB/layer, scoped) + activations + logits | ~3–4 GiB | reducible |
| **KV cache** | **<0.1 GiB** | negligible — see below |

**Fix applied:** `posix_fadvise(POSIX_FADV_DONTNEED)` after each shard's `pread` (weight_store.h) drops the
file pages we already copied → reclaims the ~14 GiB → true free ≈ 20 GiB. Headless SSH (no GNOME/RustDesk)
adds a couple more GiB.

## Why KV is a non-issue (architecture, not luck)
DeepSeek-V4 is memory-frugal by design: **MLA** (single latent KV head, not 64) + **sliding window 128**
(only the last 128 tokens' KV kept per layer, ring buffer) + **DSA compression** (older context kept as a
handful of pooled/compressed slots). Per layer the resident KV is O(window · head_dim) ≈ 128·512·2 B ≈
128 KB, ×43 ≈ **~5 MiB**, plus tiny compressed state — even at 64K context. So KV never competes with the
weights. (The exact opposite of a dense-MHA model, where KV would dominate.)

## Two memory levers (both also speed wins)
1. **Drop page cache** — done (`fadvise`). +~14 GiB now.
2. **Native-dtype kernels (kill the fp32 dequant)** — Phase E. Make GEMMs consume the checkpoint dtypes
   directly: read **e8m0 scale bytes in-kernel** (`exp2(byte-127)`) instead of pre-dequanting to fp32; add an
   **fp8 path to `ogroup_gemm`** so `wo_a` stays fp8 (saves 2.1 GB persistent + 134 MB/layer); do the
   **lm_head GEMM in bf16**. Removes ~3–4 GiB and the dequant compute. Not needed to fit — a cleanliness/speed
   win for the server.

## Per-phase budget (headless ≈ 113 GiB usable)
| Phase | Resident | Fits? |
|---|---|---|
| **Base forward / inference** | 96 (weights) + <1 (KV) + ~2 (activations, native-dtype) ≈ **~99** | ✅ ~14 GiB spare |
| **+ DSpark draft head** | + ~2–3 GiB (`mtp.0.*`, ≈ one layer) ≈ **~102** | ✅ |
| **Capture** (on-policy gen) | full model + head + activations; **top-k logits/tokens stream to DISK, not RAM** | ✅ memory is not the bottleneck (compute/IO is); modest batching fits the ~11 GiB spare |
| **Training** | **96 GB model NOT resident** — cache-once-to-disk, train only the head: params ~3 + grads ~3 + optimizer (Adam 2× = 6, Muon less) + batch ≈ **~15–25 GiB** | ✅ huge headroom — *this is why offline/cached training is memory-necessary, not just faster* |
| **Inference server** | 96 (native) + 3 (head) + <1 (KV) + spec-decode draft buffers + few-seq activations ≈ **~102–106** | ✅ headless |

## First-principles guarantees
1. **The full model is single-tenant** (Axiom 8): never load two 96 GB things at once. Capture/serve keep the
   target resident; training does not need it (reads the disk cache). So we never need >~106 GiB at once.
2. **KV scales with window+compression, not sequence length** → bounded and tiny at any context.
3. **Training is decoupled** from the target's 96 GiB by the offline cache — the single most important
   memory design decision. Online/on-policy-resident training would OOM (96 + optimizer states); offline does not.
4. **Reclaimable ≠ used**: always read `MemAvailable`/`free available`, not `cudaMemGetInfo used`, which
   counts page cache. After `fadvise`, the two converge.

## Action checklist
- [x] `posix_fadvise(DONTNEED)` after shard load (weight_store.h). **CONFIRMED: 120.7 -> 107.6 GiB used, ~15 GiB headroom.**
- [ ] Run headless (SSH+tmux); stop GNOME/RustDesk during long runs.
- [ ] (Phase E) native-dtype kernels: e8m0-in-kernel scales, fp8 `ogroup_gemm`, bf16 lm_head → drop fp32 dequant.
- [ ] (Training) enforce cache-once-to-disk; never co-resident with the 96 GB target.
- [ ] Watch true headroom with `MemAvailable`, not `cudaMemGetInfo`.
