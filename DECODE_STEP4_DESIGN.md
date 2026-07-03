# DECODE_STEP4_DESIGN.md — the M=1 KV-cache decode spine (STRUCTURAL_PLAN Step 4)

**Status:** IN PROGRESS — the two hardest primitives are built & gated BIT-EXACT; remaining = mechanical wiring.
Prereqs landed: Step 1b zero-sync grouped-GEMM MoE (graph-capturable, `g_moe_grouped`). This is the spine — the
first path where **decode tok/s** is actually measured (everything so far is s=8 PREFILL).

## PROGRESS
- ✅ **Milestone 1 — sliding-window M=1 decode** (`mla_cache_kv` + `mla_decode_step`, `kernels/mla_decode.cu`).
  Equivalence gate `tests/gate_mla_decode.cu`: prefill vs cache+decode = **cosine 1.0, rms 0, maxabs 0**.
- ✅ **Milestone 2 crux — incremental compressor emit** (`compressor_emit_group`, `kernels/compressor.cu`).
  Gate `tests/gate_compressor_emit.cu`: **bit-exact** for ratio-128 non-overlap, ratio-4 overlap, and the
  indexer's own hadamard+fp4 compressor (d=128). The append-only compressed KV cache is proven correct.
- ✅ **Milestone 2a — strided (ratio-128) compressed decode step** (`compressed_decode_step_strided`,
  `kernels/compressed_decode.cu`). Gate `tests/gate_compressed_decode.cu`: **cosine 1.0, rms 0, maxabs 0**.
- ✅ **Milestone 2b — ratio-4 DSA-indexer decode step** (`compressed_decode_step_indexer` + `..._cache_r4`).
  Gate `tests/gate_indexer_decode.cu`: **cosine 1.0, rms 0, maxabs 0**. **All three attention flavors decode
  bit-exact** (sliding / strided-128 / indexer-4).
- ✅ **Milestone 3 — full 43-layer decode loop + head** (`src/decode.cu` + `kernels/block_decode.cu`). GATE PASS
  on the real 180B: first decoded token argmax=270 (== gated prefill logits[s-1]). **First measured 0.50 tok/s.**

## DECODE OPTIMIZATION TRAJECTORY (measured, warm M=1, argmax=270 throughout) — 0.50 → 4.67 tok/s (9.3×)
1. native-e8m0 expert scales (no per-token dequant): 0.50 → 1.47 (3.0×)
2. per-step bump-arena scratch (Step 2, graph-ready): 1.47 → 1.55
3. hc/ogroup → arena (kill per-call syncs): 1.55 → 1.72
4. native wo_a (fp8→f16 one-pass): 1.72 → 2.21
5. build per-layer structs ONCE (persistent): 2.21 → 3.02
6. fused fp8 wo_a in TC ogroup (no per-token wo16 conv): 3.02 → 4.27
7. warp-per-expert MoE router: 4.27 → 4.67
   (all committed + pushed; A/B + mechanism in OPTIMIZATION_LEDGER, nsys-guided.)

## REMAINING PATH TO ~50 tok/s (bandwidth bound ~273 GB/s) — nsys per-token: tc_fp8 attn ~91ms, MoE grouped ~78ms, tc_ogroup ~30ms
- ⬜ **M=1 GEMV for fp8 (attn) + fp4 (MoE)** — the m16-tile mma wastes 15/16 at M=1; a bandwidth-optimal
  M=1 GEMV (vectorized uint4 weight loads, warp-reduce) should cut tc_fp8 (91→~25ms) + MoE (78→~25ms).
  (NOTE: the naive warp-per-output oracle was A/B'd SLOWER than TC at M=1 — needs a proper vectorized GEMV.)
- 🟡 **CUDA-graph capture (Step 3) — ANALYZED, device-pos rewrite scoped.** Collapse ~2500 launches/token into
  one `cudaGraphLaunch`. **Finding (verified): the M=1 decode path has NO host syncs** (no D2H, no
  cudaStreamSynchronize — dsync is an arena no-op) — so it is nearly capturable. Prep done: `run_layer` threads
  a capture stream. Remaining blockers are exactly the **pos-baked / host-resident** bits; each has a bounded fix:
  1. **Host `comb` idx buffers** (compressed_decode_step_strided/indexer build a `std::vector` then H2D) — freed
     after the step, so they don't survive replay. FIX: build comb in a DEVICE buffer via a kernel from `d_pos`,`d_T`.
  2. **`pos` baked into pointer math** — rope offset `cosT+pos*half`, KV append `kvcache+pos*HEAD_DIM`,
     `k_win_idx(base,width)`. FIX: a `d_pos` device int; device-pos rope variant (kernel reads `d_pos`); write new
     KV to a fixed scratch then a `d_pos`-indexed copy kernel appends it; `k_win_idx`/comb kernels read `d_pos`.
  3. **Compressor emit conditionality + host `T` counter** (`if((pos+1)%ratio==0){...; ++*T}`). FIX: an
     always-launched emit kernel that self-masks on `d_pos` and advances a DEVICE `T`; the attention reads device `T`.
  Once all decode-step addressing is device-`d_pos`-driven, `cudaStreamBeginCapture`→instantiate→`cudaGraphLaunch`
  per token (bump `d_pos` with a 1-thread kernel between launches). Gate: identical logits vs the un-captured path.
  This is the "rebuild the forward as a static step" — invasive across mla_decode/compressed_decode + a rope
  variant, but bounded (~300-500 lines) and the M=K verify path generalizes the same way. Est. saves the
  ~100 ms/token host launch overhead AND brings the M=5 verify toward the weight-bound floor -> spec-decode >1x.
- 🟡 **DSpark spec-decode (Step 5) — IN PROGRESS. The M=K VERIFY primitive is DONE + gated (the hard part).**
  - ✅ **M=K verify forward** (`mla_verify_step`, `compressed_verify_step_{strided,indexer}`, `block_verify_step`,
    `cblock_verify_step`): K tokens in ONE forward, GEMMs at M=K read weights ONCE. `gate_mla_verify` cosine 1.0;
    full 180B M=5 verify = **67.9 ms/tok if accepted vs 149.3 M=1 = 2.2×** (`decode.cu` spec-verify gate).
  - ⬜ **DSpark block-head draft + accept loop.** Block draft REQUIRES the separate block-diffusion DSpark-head
    model (`~/models/DeepSeek-V4-Flash-DSpark-head/`, mtp.* shards) — the REAP built-in `mtp.0` is only 1-ahead,
    not chainable. Head forward EXISTS (`kernels/dspark_real.cu`: `dspark_main_x`, `dspark_main_kv`,
    `dspark_block_forward`, `dspark_forward_head`) and is proven τ≈0.8 (Gate-2-real in `forward.cu` — copy that
    wiring). Loop: (1) verify emits the L40/41/42 main-hidden tap for the accepted position (add the tap to the
    verify path, as `forward.cu` does with `dspark_tap_pool`); (2) DSpark head drafts block=5 from that tap;
    (3) M=K verify the drafts (built above) → target argmax + fresh taps; (4) accept longest matching prefix,
    advance. Effective ms/tok = verify_ms / accepted_len. With ~4 accepted → ~85 ms/tok now; once base nears the
    bandwidth floor (CUDA graphs) → ~40ms verify → ~50 tok/s. **Determinize the MoE scatter (sorted, not atomic)
    for a clean accept gate** — the current run-to-run near-tie flips are benign but muddy strict-match checks.

## MILESTONE 3 (original)
- Per-layer `KVCache` arena (pre-alloc → forces Step 2);
  L0-1 use `mla_decode_step`, L2-42 the compressed steps; HC + `moe_forward(bs=1)` (grouped path shines at M=1);
  hc_head→norm→lm_head→argmax. Gate: decode logits == prefill logits[s-1] (argmax + cosine), then multi-step
  KV==recompute. **Measure decode tok/s** (the point of Step 4). Detached, memory-neutral.

## Why it's tractable (the key realization)
The forward's three attention flavors all have **append-only** KV state under autoregression:
- **Sliding window (L0-1):** cache the last `WINDOW=128` tokens' `kv_win = fp8sim(rope(kv_norm(wkv·x)))`. Append
  per step, evict >128. A new query attends over the cached window (last `min(pos+1,128)` rows) + itself.
- **Compressor (ratio-4 overlap, ratio-128 non-overlap):** a compressed row `g` pools tokens
  `[(g-1)·ratio, g·ratio+ratio)` (overlap) / `[g·ratio, g·ratio+ratio)` (non-overlap). It **finalizes** the moment
  its last constituent token exists and never changes. So the compressed KV cache is **append-only: +1 row every
  `ratio` decode steps.** Keep a tiny rolling buffer of the last `2·ratio` tokens' compressor inputs; when a
  group completes, emit its one row (pool→rmsnorm→rope→quant) and append.
- **DSA indexer (ratio-4):** for the single query, score it against all `T=pos/ratio` existing compressed rows,
  take top-`min(512,T)`. **Strided (ratio-128):** deterministic idxs = compressed rows `t < (pos+1)/ratio`.

KV footprint (maxseq≈4096): window 128·512·4=256KB/layer; compressed ratio-4 ≤1024·512·4=2MB/layer,
ratio-128 ≤32 rows; + indexer-compressor same order. Total ≪0.1 GiB across 43 layers — **memory-safe**
(matches the ROADMAP "KV is tiny by MLA+SWA+DSA design").

## Data structures (pre-allocated ONCE — memory-neutral, and required for graph capture Step 3)
Per layer `L` a `KVCache`:
- `kv_win[WINDOW, HEAD_DIM]` fp32 ring + `win_len` (write pos = pos % WINDOW; logical order handled in idx build).
- If compressed: `kv_comp[Tmax, HEAD_DIM]` fp32 + `T` (count); rolling `xhist[2·ratio, DIM]` of recent hiddens
  (for the next group's pool) — or recompute kv/score for those tokens on the fly (cheap, per-token linear).
- If indexer: `kv_idxcomp[Tmax, INDEX_HEAD_DIM-ish]` append buffer for the indexer's own compressor rows.
- Scalar `pos` (global position).
All caches for all 43 layers allocated up front in a fixed arena → the decode step is a fixed kernel sequence on
fixed pointers → capturable (Step 3).

## Control flow
1. **Prefill (populate caches).** Run the prompt `[id0..id_{s-1}]` through a prefill that ALSO writes caches:
   for each layer, store `kv_win` for the last ≤128 positions, compute every COMPLETE compressed row into
   `kv_comp`, seed `xhist`/`T`/`win_len`/`pos=s-1`. (Extend the existing `compressed_attn_forward` /
   `mla_forward` to emit their kv into the cache instead of a scratch that's freed. Keep the current prefill as
   the golden.) Head → logits[s-1] → first decoded token.
2. **Decode step (M=1), per new token id_t at pos:**
   - `embed(id_t)` → HC-expand → `h[1,hc,d]`.
   - per layer L:
     - q: `wq_a→q_norm→wq_b→per-head rms→rope(pos)` at M=1.
     - kv_win_new: `wkv·x→kv_norm→rope(pos)→fp8sim`; append to ring; `win_len=min(win_len+1,128)`.
     - if compressed and `(pos+1)%ratio==0`: emit the just-completed compressed row from `xhist`+new token
       (pool→rmsnorm→rope→quant), append to `kv_comp`, `T++`. Update `xhist`.
     - build idxs for THIS query: window rows (last `win_len`) ⊕ selected compressed rows
       (indexer top-k over `kv_comp[0:T]`, or strided `t<(pos+1)/ratio`).
     - `sparse_attn(M=1)` over `[kv_win_ring ⊕ kv_comp[0:T]]` with those idxs → o[1,Kd].
     - de-rotate → `ogroup_gemm` → `wo_b` → attn out; HC; `moe_forward(bs=1)` (grouped GEMM at M=1 = its best
       regime); HC → h.
   - head: hc_head→norm→lm_head → logits[1,vocab] → argmax → next id. `pos++`.
3. **Loop** to generate N tokens; measure ms/tok over the steady-state (warm) region.

## Kernels to add/adapt (reuse gated primitives; add M=1 + cache-append variants)
- `mla_forward` / `compressed_attn_forward`: split into (a) prefill-that-writes-cache, (b) `*_decode_step`
  (M=1, reads cache, appends new kv). Reuse `sparse_attn` (already takes arbitrary `n`, `topk` idxs, `m=1`),
  `rope_interleaved`, `rmsnorm`, `act_quant_fp8sim`, `ogroup_gemm`, `fp8_block_gemm`, `compressor_pool*`.
- `k_win_append`, `k_comp_append` (trivial copy kernels). Indexer M=1: reuse `indexer_forward` scoring for 1 query.
- MoE at bs=1: `moe_forward` already handles bs=1; grouped path (`g_moe_grouped`) is ideal (tiny tiles, one launch).

## GATE (equivalence — the honest correctness check, no golden needed)
Feed `[id0..id_{s-2}]` as prefill, then DECODE the one token at position `s-1`; its logits must match the
current PREFILL's `logits[s-1]` (which is already gated argmax=270). **Gate: decode argmax == prefill argmax AND
cosine(decode_logits, prefill_logits[s-1]) > 0.9999.** Then multi-step: decode several tokens, re-run a fresh
prefill on the grown sequence each step, confirm identical argmax (KV-cache ≡ recompute). Run detached-to-file;
memory-neutral; log GATE_LOG + OPTIMIZATION_LEDGER; commit.

## After Step 4
- **Step 2** (pre-alloc every buffer + static launch) is largely FORCED by the cache arena above → then
- **Step 3** (CUDA-graph capture of the M=1 step: `cudaStreamBeginCapture`→instantiate→`cudaGraphLaunch`/token;
  gate identical logits) — kills the hundreds of per-kernel launch overheads that dominate M=1.
- **Step 5** (DSpark spec-decode: draft proposes block=5, target verifies in one graph launch, accept longest
  prefix τ≈0.75–0.8, overlap draft∥target) — the ~2.5–4× throughput multiplier. `src/dspark_real.cu` + the
  Gate-2-real block-acceptance harness already prove the head; wire it into the graph-launched decode.
