# DECODE_STEP4_DESIGN.md — the M=1 KV-cache decode spine (STRUCTURAL_PLAN Step 4)

**Status:** DESIGNED (execution-ready), not yet built. Prereqs landed: Step 1b zero-sync grouped-GEMM MoE
(graph-capturable, `g_moe_grouped`). This is the spine — the first path where **decode tok/s** is actually
measured (everything so far is s=8 PREFILL). Build it as ONE focused, gated, detached increment.

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
