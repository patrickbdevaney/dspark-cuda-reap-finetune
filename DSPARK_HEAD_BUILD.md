# DSPARK_HEAD_BUILD.md — exact build spec (decoded from DSpark-head repo model.py:744-874)

The real DSpark block-diffusion head. Source: `~/models/DeepSeek-V4-Flash-DSpark-head/inference/model.py`.
Config: `dspark_block_size=5`, `dspark_noise_token_id=128799`, `dspark_target_layer_ids=[40,41,42]`,
`dspark_markov_rank=256`. Shares main `embed`+`head`. `n_mtp_layers` stages (REAP N_MTP=1 → single stage that
is BOTH first AND last → has main_proj/main_norm AND norm/markov/confidence/hc_head). Repo has 3 stages;
**CONFIRM `n_mtp_layers` in the DSpark-head config.json** — build for the config's value (chain stages).

## Tap (main model, model.py:920-921)
Main forward appends `h.mean(dim=2)` (HC 4→mean, [b,s,d]) at layers 40,41,42 → `main_hidden = cat = [b,s,3d]`.
→ In our `forward.cu`: during the 43-layer loop, at Lyr∈{40,41,42} mean-pool `h` over hc (h is [s,hc,d]) and
   stash [s,d]; concat the 3 → main_hidden [s,3d]. (Add a small mean-pool kernel; stash 3 buffers.)

## forward_embed (model.py:851-858)
```
main_x = main_norm( main_proj(main_hidden) )              # main_proj: Linear(3d -> d) FP8+scale; main_norm RMSNorm
draft_input_ids = full([s, block_size], noise_token_id);  draft_input_ids[:,0] = input_ids   # [real, noise,noise,noise,noise]
x = embed(draft_input_ids)                                # [s, block, d]  bf16 lookup
x = x.unsqueeze(2).repeat(1,1,hc,1)                        # [s, block, hc, d]
return x, main_x
```

## DSparkBlock.forward (model.py:846-850) + DSparkAttention (model.py:750-793)  ← the NEW attention
- `start_pos==0` (prefill): **only build KV** from `main_x` into a sliding-128 cache → `return self.attn(x,0,main_x)`.
- `start_pos>0` (decode): full Block.forward but attention is DSparkAttention over the block:
  - q/kv from the block x [bsz, block, ...]; RoPE with `freqs_cis[start_pos+seqlen : +block]`.
  - `topk_idxs = get_dspark_topk_idxs(win, bsz, block, start_pos)` = `cat([arange(min(win,start_pos+1)), win+arange(block)])`
    → attend `[sliding main-KV (up to win) ⊕ the block's own KV]`. `sparse_attn` over that. (M:782-788)
  - Then the normal Block MoE + HC.
- **CUDA:** a `dspark_attn_forward` = mla_forward-like but KV = [main-KV window ⊕ block-KV], idxs from get_dspark_topk_idxs.
  Reuse sparse_attn + the MLA q/kv projections; the main-KV was built at prefill from main_x (sliding-128 ring).

## DSparkMarkovHead (model.py:795-804)  ← small, build first
```
embed  = markov_w1(token_ids)      # ParallelEmbedding(vocab, 256) → [.,256]  (bf16 lookup)
logits = markov_w2(embed)          # ParallelHead(vocab, 256) used as Linear 256→vocab, full_logits → [.,vocab]
return logits, embed
```
CUDA: `k_markov` = per token: gather markov_w1[token] (256, bf16→f32) then GEMM 256→vocab (markov_w2, bf16).
Returns logits_bias[vocab] + markov_embed[256].

## DSparkConfidenceHead (model.py:807-815)
`proj: Linear(d+256 → 1)` fp32. `confidence = proj( cat([hidden, markov_embed]) )`. Per draft token.

## forward_head (model.py:860-874)  ← AR block sampling, host-orchestrated
```
x = hc_head(x, hc_head_fn/scale/base)                    # [s, d]  (collapse hc 4→1)
logits = head( norm(x) )                                  # [s, block, vocab]  main lm_head; norm = mtp norm
output_ids[:,0] = input_ids
for i in 0..block_size-1:
    logits_bias, markov_embed_i = markov_head(output_ids[:,i])
    logits[:,i] += logits_bias                            # Markov bigram correction
    output_ids[:,i+1] = sample(logits[:,i], temperature) # greedy for tau: argmax
confidence = confidence_head(x, stack(markov_embeds))
return output_ids[block+1], logits, confidence
```
NOTE: `head` produces logits for ALL block positions from the block hidden x[s,block,...]; the AR loop only
adds the Markov bias + samples. So x must be [s, block, hc, d] through the block, and hc_head/norm/head map
[s,block,hc,d]→[s,block,vocab].

## Verify / accept loop (NOT in reference — build in our harness; modeling-notes §10)
Spec-decode step: draft proposes `output_ids[1:block+1]` (block_size tokens). Run the TARGET on those block
positions (one target forward) → target greedy tokens. **Accept the longest matching prefix**; `confidence`
can early-exit. **Block-τ (Gate 2-real) = E[accepted prefix length per target forward]** over a representative
prompt set. This is the number that anchors decode throughput.

## Build order (tractable → hard)   [1,2,3 DONE — compile; 4,5 remain]
1. `DSparkMarkovHead` kernel (`k_markov`) — small, self-contained. Gate vs a torch golden.
2. `main_x` = fp8_gemm(main_proj, 3d→d) + rmsnorm(main_norm). Tap-pool kernel in forward.cu (layers 40/41/42).
3. `forward_head` AR loop (reuse hc_head+rmsnorm+lm_head gemm + markov + sample) — host-orchestrated.
4. `dspark_attn_forward` (main-KV window ⊕ block) — the new attention; reuse sparse_attn + MLA projs.
5. Verify/accept loop in harness → **block-τ on REAP** (real Gate 2). Then fine-tune to lift it.

## Piece 4 — EXACT DSparkAttention (model.py:750-793), turnkey for CUDA build
```
# PREFILL (start_pos==0): build main-KV from main_x, store in sliding-128 kv_cache; return x (no attn output).
main_kv = kv_norm(wkv(main_x)); rope(main_kv[...,-rd:], main_freqs=freqs_cis[start_pos:+seqlen]); act_quant(main_kv[...,:-rd],64)
kv_cache[:seqlen] = main_kv   (or ring split if seqlen>win)
# DECODE (start_pos>0): x = block [bsz, block_size, .]
q  = wq_b(q_norm(wq_a(x))).unflatten(heads,head_dim); q *= rsqrt(mean(q^2)+eps); rope(q[...,-rd:], freqs_cis[start_pos+seqlen:+block])
kv = kv_norm(wkv(x)); rope(kv[...,-rd:], same freqs); act_quant(kv[...,:-rd],64)
topk_idxs = get_dspark_topk_idxs(win,bsz,block,start_pos) = cat([arange(min(win,start_pos+1)), win+arange(block)])
kv_cache[start_pos % win] = main_kv.squeeze(1)          # append this step's main context token
kv = cat([kv_cache[:bsz], kv], dim=1)                   # [main-KV window ⊕ block-KV]
o = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale); rope(o[...,-rd:], freqs, inverse=True)
o = einsum("bsgd,grd->bsgr", o.view(bsz,block,n_groups,-1), wo_a); x = wo_b(o.flatten)
```
CUDA: identical primitives to `mla_forward` (wq_a/wq_b/wkv fp8 gemms, q_norm/kv_norm rmsnorm, rope, act_quant,
sparse_attn, ogroup_gemm wo_a, wo_b) — ONLY difference is KV = [main-KV window (from main_x) ⊕ block-KV] and the
get_dspark_topk_idxs index set. For the block-τ MEASUREMENT: build main-KV from main_x[0..t] per anchor t (sliding
win=128), block[0]=x_{t+1}, predict x_{t+2..t+block+1}; loop anchors t∈[0, s-block-1] (each its own window).

## MEASUREMENT (piece 5) — real block-τ
For anchors t: draft proposes block via forward_embed→dspark_attn(decode)→block MoE/HC→forward_head → output_ids[1:].
Ground truth = prompt tokens x_{t+2..} (teacher-forced) OR target greedy. **block-τ = E[longest matching prefix len]**.
Report per-anchor + mean. This is the number that anchors throughput; then fine-tune to lift it.

## STATUS: pieces 1,2,3 built+compile (dspark_real.cu). Pieces 4,5 = the new per-anchor block attention +
## the anchor-loop harness — need a full run to validate (fresh-context build; all interfaces + exact code above).

## Weights (mtp.* from DSpark-head repo, ~11GB shards 46-48; run on REAP taps)
Per stage: full Block (attn wq_a/wq_b/wkv/wo_a/wo_b+scale, q_norm/kv_norm, attn_sink, attn_norm/ffn_norm,
hc_attn/ffn, ffn gate+bias+experts[256]+shared). Stage0: `main_proj.{weight,scale}` `main_norm.weight`.
Last stage: `norm.weight` `markov_head.markov_w1.weight` `markov_head.markov_w2.weight` `confidence_head.proj.weight`
`hc_head_{fn,scale,base}`. Head's MoE is 256-expert (its own capacity) — per-expert ptr tables like the main MoE.

## Piece 5 — measurement integration (the ONE thing left; produces the block-τ number)
All compute kernels exist + compile (pieces 1-4). Piece 5 = orchestrate them on real DSpark-head weights:
1. **Selective DSpark-head loader.** The DSpark-head repo (`~/models/DeepSeek-V4-Flash-DSpark-head`) has only
   shards 46-48 present (the mtp.* head, ~11GB); its index references shards 1-48. `WeightStore`/`ShardedSafeTensors`
   opens ALL referenced shards → fails on missing 1-45. FIX: add a filtered load — accept a name prefix ("mtp.")
   and skip shards that don't exist / hold no matching tensor. Small, bounded addition to `safetensors.h`+`weight_store.h`.
2. **Tap L40/41/42 in forward.cu.** In the 43-layer loop, at Lyr∈{40,41,42} call `dspark_tap_pool(main_hidden, h,
   s, hc, DIM, slot, 3)` (slot 0/1/2). After the loop: `dspark_main_x(main_x, main_hidden, mtp.0.main_proj, ...,
   mtp.0.main_norm)` → `dspark_main_kv(main_kv, main_x, mtpBlock.attn, s)`.
3. **Anchor loop** (t ∈ [0, s-block-1]): draft_ids=[x_{t+1}, noise×4]; embed(bf16)+HC-expand → x_block[block,hc,d];
   `dspark_block_forward(xb, x_block, draft_ids, main_kv, t, mtpBlockW, blkCos[t+1..], blkSin[..], block, win)`;
   `dspark_forward_head(out_ids, xb, x_{t+1}, hc_head/norm/lm_head/markov, ...)`; compare out_ids[1..block] vs
   ground-truth x_{t+2..t+block+1} → longest matching prefix.
4. **block-τ = mean(prefix_len) over anchors** on a representative prompt set (real Gate-2). Then fine-tune to lift.
Freqs: DSpark block has compress_ratio=0 → sliding rope_theta (yarn off). Build a `[s+block, rd/2]` table; main_kv
uses [0..s-1], the block at anchor t uses [t+1..t+block]. Head's MoE is 256-expert → per-expert ptr tables (mtp).
