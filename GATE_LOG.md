# GATE_LOG.md — explainability log: finding + rationale per gate/iteration

**Practice (going forward):** every gate/iteration records *what we found*, *why it happened*, *the fix*, and
*why the fix is correct* — so the repo carries proof of how we got here, not just the end state. Full trail:
`git log` (each commit states the finding + rationale), plus `OPTIMIZATION_LEDGER.md` (measured A/B deltas),
`RESEARCH_*.md` (verified evidence bases), `MEMORY.md`, `ROADMAP.md`. This file is the consolidated "why".

Each entry: **Gate/iteration — Finding → Root cause → Fix → Why it's correct.**

---

## Correctness gates (kernels → composed → real weights)

**hc_post (Block gate, real weights) — the canonical lesson.** Finding: the synthetic HC gate PASSED but the
2.76 GB real-weights Block gate FAILED (res2 jumped 0.19 while attn/comb/post matched <5e-6). Root cause:
`hc_post` summed `comb`'s transposed index (`comb[j,i]` vs model.py:692 `comb[i,j]`) — and the synthetic golden
used the SAME wrong einsum, so it was self-consistently wrong. Fix: sum `comb[k,j]` (kernel) + fix the golden
einsum. Why correct: matches model.py exactly, and now the real-weights Block gates at 0.23%. **Rationale kept:
self-consistent synthetic gates hide shared bugs; real-weights integration gates are the true check.**

**Deep-composition metric (gate_cmla) — not loosening, correcting.** Finding: ratio-128 compressed attn read
7.6% on per-element `max_rel` but 0.12% max-abs-relative. Root cause: `max_rel = |diff|/(|ref|+0.01·mx)` is
pathological for deep fp8/fp4 outputs — it grows with seq length on near-zero elements *regardless of
correctness*. PROOF (A/B): the accepted ratio-4 path rises 1.6%→4.1% max_rel from seq16→seq256 while cosine
stays 1.0000000. Fix: gate deep compositions on **relative-L2 (<1e-2) + cosine (>0.9999) + max-abs-rel (<5e-3)**.
Why correct: these are the standard fidelity metrics for quantized GEMM chains; the change is evidence-based and
applied consistently to ratio-4 AND ratio-128, not a threshold flip to make one case pass.

**MoE router args (MoE gate) —** Finding: MoE gate max_rel 3.66. Root cause: `ModelArgs` built by HF field-name
filter silently mismatched (moe_intermediate_size≠moe_inter_dim, num_experts_per_tok≠n_activated, etc.); and
act scale used non-pow2. Fix: explicit `_HF2MA` map + `ue8m0` pow2 scales. Why correct: bit-exact vs torch.

---

## Real-checkpoint integration (surfaced only by touching real bytes — Gate-1 class)

**wo_a is fp8, kernels want fp32.** Finding: golden dequantized `wo_a` via `.float()`; the checkpoint stores it
fp8+scale. Fix: dequant `wo_a` fp8→fp32 at load. Why: `ogroup_gemm` consumes fp32; the goldens proved the math,
the loader must reproduce the dtype the goldens fed.

**Experts are NOT stacked.** Finding: `MoEWeights` assumed a single `[E,...]` array; the real checkpoint stores
per-expert tensors that are byte-non-contiguous (e2 jumps 1.2 GB). Fix: per-expert device-pointer tables
(`w1p[e]`…); moe.cu already loops per-expert so it's localized. Why correct: identical math, just addressing.

**cudaHostRegister unsupported on Tegra.** Finding: registering the file-backed `MAP_PRIVATE` mmap → "operation
not supported". Root cause: Tegra limitation on that mapping type (registration IS supported generally,
`hostRegisterSupported=1`). Fix: `cudaHostAlloc(Mapped)` + `pread` shard data in (single copy). Why correct:
integrated GPU ⇒ that buffer IS device memory; verified GPU read-back MATCH on the full 96 GiB.

**Memory: reclaimable ≠ used.** Finding: forward showed 120.7/122.8 GiB "used" (alarming). Root cause: `pread`
leaves 96 GB in the page cache, which `cudaMemGetInfo` counts as used but Tegra `cudaMalloc` won't auto-evict.
Fix: `posix_fadvise(DONTNEED)` after each shard → 120.7→107.6 GiB. Why correct: the file pages were already
copied to our pinned buffer; dropping them is free. Lesson: watch `MemAvailable`, not `cudaMemGetInfo`.

**Gate 1 / 1.5 correctness proof.** The full 180B ran (43 layers, flat mem) AND predicted "The capital of
France is" → " Paris". Why this is proof: it exercises every forward.cu-new path (dequant, YaRN freqs,
embed/HC-init, both attention variants, head) end-to-end on real weights with a checkable answer.

---

## Optimization iterations (measured, in OPTIMIZATION_LEDGER.md)

**Decode is memory-BANDWIDTH-bound (reframing).** Evidence: literature scan (23 sources) + our own A/B —
`tc_fp4_gemm` is 2.71× at M=1 but 19.7× at M=8. Why: at M=1 weight-loading dominates (bandwidth); batching
amortizes the load so the tensor-core compute win shows. Consequence: grind order = batch + cut traffic, not
just faster MMA. (Also: megakernels were adversarially REFUTED vs CUDA graphs — avoided that thrash.)

**tc_fp4_gemm champion.** Finding: warp-per-output GEMMs are ~1 tok/s. Fix: port gemma Marlin `mma.sync.m16n8k16`
→ W4A8. Gate: cosine 1.0 vs `fp4_gemm` (fp16-act rounding only). Measured: 19.7× (M=8) / 2.71× (M=1) cached.
Why the cache matters: repack once (ptr-keyed), reused across decode forwards. Kept as `use_tc` flag; `fp4_gemm`
stays the bit-exact oracle. En-route bug: gate random fp8 must avoid 0x7f (e4m3 NaN) — both kernels "agreed" on
NaN (cosine=nan) until fixed.

**Batched MoE dispatch (grind #2).** Finding: gate caught it — first non-deterministic (0.649/−0.197), then
zero-output. Debug chain: `compute-sanitizer --tool memcheck` → illegal read 44 MB past a 32-byte buffer (a
garbage token index) → the reused per-expert `cudaMemcpy` into `tok_d` silently left stale `[1,0,…]` (per-expert
`|OEb|` were fine but every expert's `tok_d` = e0's) → only tok1's output landed. Fix: upload ALL tokens/weights
ONCE as flat device arrays + per-expert OFFSET pointers (no reused buffer, no per-expert copy). Result: cosine
1.0000000 vs oracle; per-token oracle stays bit-exact. Why correct: identical per-(token,expert) math, just a
robust dispatch; the offset-pointer scheme removes the reused-buffer failure mode entirely. **Rationale kept:
the gate + compute-sanitizer + per-expert dumps localized a subtle host-copy bug that inspection missed.**

---
*Update this log whenever a gate catches something or an iteration lands a measured change. The "why" is the asset.*
