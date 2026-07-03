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

## Build order (tractable → hard)
1. `DSparkMarkovHead` kernel (`k_markov`) — small, self-contained. Gate vs a torch golden.
2. `main_x` = fp8_gemm(main_proj, 3d→d) + rmsnorm(main_norm). Tap-pool kernel in forward.cu (layers 40/41/42).
3. `forward_head` AR loop (reuse hc_head+rmsnorm+lm_head gemm + markov + sample) — host-orchestrated.
4. `dspark_attn_forward` (main-KV window ⊕ block) — the new attention; reuse sparse_attn + MLA projs.
5. Verify/accept loop in harness → **block-τ on REAP** (real Gate 2). Then fine-tune to lift it.

## Weights (mtp.* from DSpark-head repo, ~11GB shards 46-48; run on REAP taps)
Per stage: full Block (attn wq_a/wq_b/wkv/wo_a/wo_b+scale, q_norm/kv_norm, attn_sink, attn_norm/ffn_norm,
hc_attn/ffn, ffn gate+bias+experts[256]+shared). Stage0: `main_proj.{weight,scale}` `main_norm.weight`.
Last stage: `norm.weight` `markov_head.markov_w1.weight` `markov_head.markov_w2.weight` `confidence_head.proj.weight`
`hc_head_{fn,scale,base}`. Head's MoE is 256-expert (its own capacity) — per-expert ptr tables like the main MoE.
