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

**Compounded fast path (batched + TC).** Finding: need to confirm the two wins COMPOSE. Gate: batched=true +
use_tc=true vs per-token oracle -> cosine 1.0000000. Why correct: batched only reorders the accumulation
(atomicAdd, tiny fp32) and TC only rounds acts to fp16 -> both cosine-preserving; together still 1.0. ML note:
MoE compute per token is top-6 experts + 1 shared SwiGLU; batching groups tokens by expert so each expert's
weight is loaded ONCE for all its tokens (amortizes the bandwidth-bound weight load), and TC does the GEMM on
tensor cores instead of one-warp-per-output.

**HARDWARE: Thor sm_110 tensor-core dtype support — CORRECTED (I was wrong; user caught it).**
FIRST (WRONG) claim: 'Thor has no native FP4 compute.' CORRECTION (web-verified + re-probed): **Thor HAS 2070
FP4 TFLOPS** (96 5th-gen Tensor Cores, native FP4; 2x its 1035 FP8 TFLOPS, ~4x fp16) — FP4 is its STRONGEST
mode. My error: I tested only the legacy `mma.sync.kind::f8f6f4` and concluded 'no FP4'. NUANCE (re-probed in
CUDA 13.0): the FP4 mma is NOT exposed to hand-written PTX for sm_110 — `mma.sync.kind::f8f6f4`, `tcgen05.*`,
AND block-scaled `mma...kind::mxf4.block_scale` ALL fail 'not supported on .target sm_110'; NVIDIA's CUTLASS
SM110 FP4 is reported non-functional (forum). So Thor's 2070 FP4 TFLOPS is currently reachable ONLY via NVIDIA
LIBRARY kernels (cuBLASLt/cuDNN/TensorRT/Transformer-Engine), not our hand-PTX path (as of CUDA 13.0).
CONSEQUENCES: (a) hand-rolled compute ceiling = **FP8 mma m16n8k32 (verified ✅, 2x our current fp16)** — the
near-term grind; (b) FP4 (4x) needs cuBLASLt/library today (A/B it — breaks pure hand-roll but hits 2070 TFLOPS)
OR hand-PTX once ptxas exposes fp4 for sm_110. LESSON: verify HW capability from specs+multiple probes before
concluding from one instruction form; and log corrections openly. ML/HW note: fp4-STORAGE (weights) was always
the right decode-bandwidth lever; fp4-COMPUTE is the compute-bound-regime lever (prefill/capture/spec-block).

**(SUPERSEDED) HARDWARE: Thor sm_110a tensor-core dtype support (probed via ptxas).** Finding: FP8 mma `m16n8k32.e4m3`
✅ supported; FP16 mma ✅; **FP4/NVFP4 mma (`kind::f8f6f4`, e2m1) ❌ NOT supported** ('not supported on .target
sm_110'). Why it matters: native FP4 *compute* is a datacenter-Blackwell (B200/sm_100) feature, ABSENT on the
Jetson sm_110 die — so there is no fp4 tensor-core to leverage. We leverage fp4 for MEMORY (4-bit weight storage
= min bandwidth = the decode-bound lever) which is correct and the only fp4 lever here. BUT our tc_fp4_gemm
upconverts fp4→fp16 + FP16 mma, leaving 2× on the table: Thor's FP8 mma is 2× fp16, and our acts are already
fp8. NEXT GRIND: dequant fp4-weight→fp8 (not fp16) + FP8 tensor core (m16n8k32) — 2× compute + drops the
fp8→fp16 act upconvert. ML/HW note: decode bandwidth-bound so this helps compute-bound regimes most (prefill,
batched capture, spec-block verify); single-token decode already rides fp4-memory.

**tc_fp8_gemm champion (dense/attn W8A8).** Finding: fp8_block_gemm is warp-per-output (~1 tok/s class),
used on every MLA q/kv/o proj + shared experts. Fix: native FP8 tensor-core GEMM `mma.sync.m16n8k32.e4m3` —
fp8 acts + fp8 weights feed the TC directly (no fp16 upconvert), per-128 act + per-128x128 wt scale applied
per K-block (matching fp8_block_gemm's block-scale math). Gate: cosine 1.000000, max_rel 2e-5 vs oracle.
Measured: **17.88x** (0.413->0.023 ms, M=48/N=256/K=512). Why correct: fp8 mma computes fp8xfp8->f32 = the
same values as the oracle's dec_e4m3 fp32 dot; only accumulation-order rounding differs (2e-5). ML/HW: this is
the FP8 tensor-core lever (Thor's 1035 TFLOPS FP8, 2x its fp16); FP4 (2070) stays blocked in CUDA 13. Both
dominant kernel classes now TC-accelerated: MoE experts (tc_fp4_gemm 19.7x) + dense/attn (tc_fp8_gemm 17.9x).

**Wiring champions into forward.cu — real-model OOB LOCALIZED + FIXED (compute-sanitizer).** RESOLVED (was 'IN PROGRESS'): Finding: tc_fp8_gemm
(cosine 1.0, 17.9x) + batched MoE (cosine 1.0) each PASS their unit gates, but enabling them in the full 180B
forward crashes with an illegal memory access in the batched-MoE path (moe.cu:215 sync catches it; also seen at
tc_moe_gemm.cu:104 with tc_fp4 on). The PLAIN forward (champions OFF) still runs correctly (Gate 1 Paris).
Root cause: NOT found by inspection — every batched kernel (k_gather_x/swiglu_wrow/k_scatter_add/act_quant) and
GEMM shape (incl. the w2 N=4096/K=2048 the unit gates never covered) checks in-bounds on paper; weight indexing
is identical to the working per-token path (w1p[e]). This is the class of bug tiny-shape gates + inspection
miss — needs `compute-sanitizer` on the forward to localize (the recurring 'real weights catch what synthetic
gates can't' lesson). ACTION: forward reverted to working config (champions wired but flagged OFF: g_tc_fp8=false,
use_tc=false, batched=false); tc_fp8_gemm banked as a validated KERNEL; end-to-end measurement BLOCKED pending
the sanitizer localization. Why logged: honesty > a green checkmark; must say so.

RESOLUTION: `compute-sanitizer --tool memcheck` (with -lineinfo) named it in one shot: **k_gather_x invalid
write, moe.cu:138** — block(132) existed, so the launch had me*dim > 132*256 threads → me≥9, but the batched
scratch (Xe…) was sized bs*dim=32768. ROOT CAUSE: a token can route to the SAME expert in multiple of its na
slots (hash layer 0 especially), so per-expert me can EXCEED bs — up to bs*na total assignments; the write
`Xe[i], i<me*dim` overran the bs-sized buffer. The read `x[tok[r]*dim]` was fine (tok<bs); only the grouped
WRITE overflowed. FIX: size all routed scratch for **maxm=bs*na** (the max any expert can receive). Why the
unit gate missed it: synthetic routing gave me≤bs (no duplicate-slot collisions); the real hash layer has
them. Gates still cosine 1.0 after the fix. LESSON (again): compute-sanitizer localizes in minutes what
inspection can't — and 'me≤bs' was an unstated assumption the real router violates.

**End-to-end wiring result (both champions measured on the full 180B).** tc_fp8 dense + batched -> **559.8
ms/tok vs 687 baseline = 1.23x**, forward RAN, argmax stable (correct). Adding tc_fp4 MoE -> **773 ms/tok
(SLOWER)**. Root cause: tc_fp4's cache is cleared per-layer to avoid the ~82GB repack-doubling OOM, so every
expert re-repacks every layer; that repack cost (k_repack_w/s + malloc/free per expert per layer) EXCEEDS the
GEMM win at s=8. Lesson: tc_fp4's 19.7x REQUIRES the repacked layout to PERSIST (repack-at-load, storing
repacked in place + separate fp16 scale ~7GB), not per-forward. Banked config: **tc_fp8 dense + batched
(1.23x, no repack, no OOM)**; tc_fp4 MoE deferred to repack-at-load. The MoE is the dominant cost, so
repack-at-load is the next big lever (unlocks 19.7x on the bulk).

**Repack-at-load pp MoE — correct + zero-mem, but byte-loads negate the win (measured).** Repack-at-load
solved the OOM (in-place, same byte-size) AND the per-layer churn (no malloc, read original scale). Kernel
cosine 1.0. BUT the in-place weight sits at arbitrary safetensors offsets, so the 16B uint4 __ldcs faulted →
switched to byte loads (alignment-safe) → **555.9 ms/tok (~neutral vs 559.8 dense-only)**. Root cause of the
non-win: decode is BANDWIDTH-bound; tc_fp4's 19.7x came from COALESCED uint4 weight loads, and byte loads are
uncoalesced (≈ fp4_gemm's load). So the TC compute win is masked by the slow load. FIX (next): aligned loads —
either (a) funnel-shift (2 aligned uint4 straddling each 16B, combine by the constant per-weight offset; L2
absorbs the overlap) or (b) loader places routed-expert tensors 16B-aligned so uint4 __ldcs works in place.
LESSON: a correct kernel isn't a fast kernel — on a bandwidth-bound path, load coalescing IS the win; measure
end-to-end, don't assume the unit-gate speedup transfers when the memory access pattern changed.

---
*Update this log whenever a gate catches something or an iteration lands a measured change. The "why" is the asset.*
