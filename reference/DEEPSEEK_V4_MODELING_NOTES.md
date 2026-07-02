# DeepSeek-V4-Flash (deepseek_v4) — Exact Numerical Spec (ground truth for CUDA kernels)

Source of truth: the DSpark repo's own reference impl, downloaded to
`~/models/DeepSeek-V4-Flash-DSpark-head/inference/{model.py,kernel.py,generate.py,convert.py}`
and `config.json`. All `M:line` cite `inference/model.py`; `K:line` cite `inference/kernel.py`;
`cvt:line` cite `inference/convert.py`; `[cfg]` = `config.json`. Read from the actual checkpoint code,
not assumed. Where this differs from the original directive, this doc wins.

> Target for the fork is **`0xSero/DeepSeek-V4-Flash-180B` (REAP K160)**. Only difference from the
> unpruned DSpark config below: `n_routed_experts 256 → 160`. Everything else identical.

---

## 0. Config (deepseek_v4) — [cfg] + ModelArgs (M:34-87)

```
hidden_size / dim            = 4096
num_hidden_layers            = 43        (n_layers)          + 1 MTP (n_mtp_layers, DSpark)
vocab_size                   = 129280
num_attention_heads (n_heads)= 64
head_dim                     = 512       (per head; MLA — NOT hidden/heads)
qk_rope_head_dim / rope_head = 64        (last 64 of 512 rotate; first 448 are NoPE)
q_lora_rank                  = 1024      (MLA query down-projection rank)
o_lora_rank                  = 1024      (MLA grouped output down-projection rank)
o_groups                     = 8         (n_groups for the grouped O projection)
num_key_value_heads          = 1         (MLA: single latent KV per position, dim=head_dim=512)
window_size / sliding_window = 128
n_routed_experts             = 256  (REAP target: 160)
num_experts_per_tok          = 6         (n_activated_experts)
n_shared_experts             = 1
moe_intermediate_size        = 2048      (per-expert & shared-expert FFN width; NOTE ModelArgs default moe_inter_dim mislabeled 4096 — use [cfg] 2048)
num_hash_layers              = 3         (first 3 layers route by token-id hash, not score)
score_func                   = sqrtsoftplus
topk_method                  = noaux_tc  (bias added for SELECTION only, not weights)
routed_scaling_factor        = 1.5       (route_scale)
norm_topk_prob               = True      (renormalize top-k weights to sum 1)
swiglu_limit                 = 10.0      (clamp gate/up before SiLU)
rms_norm_eps                 = 1e-6
rope_theta                   = 10000     (sliding-only layers, YaRN OFF)
compress_rope_theta          = 160000    (compressed layers, YaRN ON)
rope_scaling (YaRN)          = factor 16, original_max_pos 65536, beta_fast 32, beta_slow 1
max_position_embeddings      = 1048576
tie_word_embeddings          = False     (separate lm_head; but MTP shares main embed+head)
# Hyper-Connections (HC)
hc_mult                      = 4         (maintain 4 hidden-state copies instead of 1 residual)
hc_sinkhorn_iters            = 20
hc_eps                       = 1e-6
# DSA indexer
index_n_heads                = 64
index_head_dim               = 128
index_topk                   = 512
# per-layer KV compression (compress_ratios, [cfg], one per layer, 46 entries):
#   layers 0,1 = 0 (pure sliding, NO compression); layers 2..42 alternate 4,128,4,128,...; tail 43,44,45 = 0
#   ratio 4  -> layer ALSO builds a DSA Indexer (overlap compression)   (M:474-475)
#   ratio 128-> compression only, deterministic strided topk, no indexer (M:476-477)
# DSpark draft
dspark_block_size            = 5         (draft proposes 5 tokens/block)
dspark_noise_token_id        = 128799
dspark_target_layer_ids      = [40,41,42](draft taps mean-pooled HC hidden of these 3 layers)
dspark_markov_rank           = 256
```

Quant (quantization_config [cfg]): `quant_method fp8`, `fmt e4m3`, `scale_fmt ue8m0`,
`weight_block_size [128,128]`; `expert_dtype fp4`. i.e. **dense/attn linears = FP8 e4m3 (128×128 block);
experts = NVFP4 e2m1 (32-block)**. See §1.

---

## 1. Quantization & the three GEMMs (kernel.py — tilelang; MUST reimplement in CUDA)

Weight dtype per class (set at module construction):
- **FP8 e4m3** (`default_dtype` when args.dtype="fp8", M:884): `wq_a, wq_b, wkv, wo_b`, expert path acts. Weight `[out,in]` fp8; scale `[ceil(out/128), ceil(in/128)]` e8m0 (M:144-148).
- **FP4 e2m1** (experts, M:628-629): `experts.*.w1/w2/w3`. Weight stored `[out,in//2]` float4_e2m1fn_x2; scale `[out,in//32]` e8m0 (M:137-143). E2M1 codebook (cvt:11-14): `{0,.5,1,1.5,2,3,4,6}` ± via sign bit.
- **BF16** (explicit): `wo_a` (M:468 — and convert dequants it to bf16, cvt:123-127), indexer `weights_proj` (M:400), `embed` table.
- **FP32** (explicit): all `RMSNorm.weight` (M:195), `gate.weight` (used as `linear(x.float(), w.float())`, M:570), Compressor `wkv/wgate/ape` (M:300-304), `confidence.proj` (M:811), `lm_head/head.weight` (M:729).

`linear()` dispatch (M:114-126): if weight fp4 → `act_quant(x)`→`fp4_gemm`; if fp8 → `act_quant(x)`→`fp8_gemm`; else `F.linear`.

**act_quant (K:40-125):** block-wise FP8. Per row-block (blk_m=32) × K-group (block_size): amax over group,
`amax=max(amax,1e-4)`, `scale = round_pow2(amax/448)` if ue8m0 else `amax/448`; `y=clamp(x/scale,-448,448)`.
`inplace=True` (K:84-91) = fused **quant→dequant back to bf16** = QAT simulation (used on KV non-rope dims, block 64, M:512/761). fp8_max=448.
**fp4_act_quant (K:128-200):** same but fp4_max=6.0, block 32, e8m0 scale, `amax=max(amax,6·2⁻¹²⁶)`. Used (inplace) on indexer q & compressed-kv (M:376,422).
**fp8_gemm (K:203-273):** `C[M,N]=A_fp8[M,K]@B_fp8[N,K]ᵀ`, per-128 block scales both sides, fp32 accum with per-block rescale (blk_M32/N128/K128).
**fp4_gemm (K:441-536):** `A_fp8[M,K]@B_fp4[N,K]ᵀ`; act per-128-K fp8 scale, weight per-32-K e8m0 scale; FP4→FP8 via fp32 then fp8 mma; blk_K=32.

**RoPE precision:** rope (last 64) dims stay bf16 for positional precision; NoPE (first 448) dims fp8-simulated (M:511-512 comment).

---

## 2. Embedding + Hyper-Connections (HC) — THE big structural difference vs Gemma

No PLE, no embed-scale, no logit softcap. Instead **Hyper-Connections**: the hidden state is kept as
`hc_mult=4` parallel copies; each block mixes them via a Sinkhorn-normalized combination.

**Model forward (M:912-926):**
```
h = embed(input_ids)                       # [b,s,d] bf16, NO sqrt(d) scaling
h = h.unsqueeze(2).repeat(1,1,hc_mult,1)   # [b,s,4,d]  -- expand to 4 HC copies (M:916)
for i,layer in layers: h = layer(h, start_pos, input_ids)      # each Block mixes+updates the 4 copies
    if i in target_layer_ids[40,41,42]: main_hiddens.append(h.mean(dim=2))   # tap for DSpark (M:920-921)
h = hc_head(h, hc_head_fn, hc_head_scale, hc_head_base)         # 4 copies -> 1  (M:922, M:709-716)
logits = head(norm(h))                     # final RMSNorm then lm_head (M:923)
```

**Block.forward (M:695-707)** — attention and FFN each wrapped in hc_pre/hc_post (NOT plain residual):
```
residual = x                                # [b,s,4,d]
x,post,comb = hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)   # 4->1 mix  (M:697)
x = attn(attn_norm(x), start_pos, *args)                          # [b,s,d]
x = hc_post(x, residual, post, comb)        # 1->4 expand + combine (M:700)
residual = x
x,post,comb = hc_pre(x, hc_ffn_fn, ...)
x = ffn(ffn_norm(x), input_ids)
x = hc_post(x, residual, post, comb)
```
- **hc_pre (M:680-688):** flatten 4 copies→`[b,s,4d]` fp32; `rsqrt(mean(x²)+eps)`; `mixes = (x @ hc_fn) * rsqrt`
  where `hc_fn:[mix_hc,4d]`, `mix_hc=(2+4)*4=24`. Then `pre,post,comb = hc_split_sinkhorn(mixes, scale, base)`.
  `y = Σ_hc pre·x` (weighted sum of the 4 copies → 1) → attn/ffn input.
- **hc_post (M:690-693):** `y = post·x_new + Σ_hc comb·residual` → back to `[b,s,4,d]`.
- **hc_split_sinkhorn (K:371-438):** from the 24 mixes per token: `pre[4]=sigmoid(m[0:4]·s0+base)+eps`;
  `post[4]=2·sigmoid(m[4:8]·s1+base)`; `comb[4,4]=m[8:24]·s2+base` → row-softmax+eps → col-normalize →
  **Sinkhorn: 19 more iters** of alternating row/col normalization (doubly-stochastic combine matrix).
- **hc_head (M:709-716):** collapses 4→1 with `pre=sigmoid(mixes·scale+base)+eps` (no Sinkhorn), weighted sum.

Per-block HC params (fp32, M:672-678): `hc_attn_fn/hc_ffn_fn [24,4d]`, `hc_attn_base/hc_ffn_base [24]`,
`hc_attn_scale/hc_ffn_scale [3]`. Model-level `hc_head_fn [4,4d]`, `hc_head_base[4]`, `hc_head_scale[1]` (M:908-910).

## 3. RMSNorm (M:189-202)
fp32: `x*rsqrt(mean(x²)+1e-6)*weight`. weight stored bf16 in ckpt, param fp32. Plain weight (init ones), **no `1+`**.

---

## 4. MLA Attention (M:442-548) — replaces Gemma GQA entirely

Single latent KV (`num_key_value_heads=1`), low-rank Q, grouped low-rank O, learnable per-head sink,
sliding-window + optional compressed KV, sparse (top-k) attention.

**Q path (M:502-505):** `qr = q_norm(wq_a(x))` (down to q_lora_rank=1024, RMSNorm) → `wq_b` → `[b,s,64,512]`
→ **per-head RMS** `q *= rsqrt(mean(q²,-1)+eps)` (M:504, extra norm, NO weight) → RoPE on last 64 dims (M:505).
`qr` (the 1024-d pre-wq_b) is ALSO fed to the Indexer (M:517).
**KV path (M:508-512):** `kv = kv_norm(wkv(x))` → `[b,s,512]` (single latent, shared across heads) →
RoPE last 64 (M:510) → fp8-simulate dims `[:-64]` (M:512).
**Sink:** `attn_sink [n_heads]` fp32 (M:462) → adds `exp(sink-max)` to softmax denom (K:345-346): an always-available "attend to nothing" slot.
**O path (M:541-547):** `o:[b,s,64,512]` → view `[b,s,n_groups=8, 64*512/8]` → grouped einsum with
`wo_a` (bf16, viewed `[8, o_lora_rank=1024, -1]`) → `bsgr` → flatten → `wo_b` (fp8) → `[b,s,d]`.
`softmax_scale = head_dim**-0.5 = 512**-0.5` (M:470).

**KV cache (M:479-480):** size = `window_size + (max_seq//ratio if compressed else 0)`. Sliding window ring
buffer of 128 latent-512 vectors + compressed entries. Single latent per pos (MLA is very KV-light).
**topk_idxs assembly (M:513-520):** window idxs (`get_window_topk_idxs` M:260-271) `⊕` compressed idxs
(from Indexer if ratio==4, else strided `get_compress_topk_idxs` M:274-282). `sparse_attn` gathers exactly these.

**sparse_attn (K:276-368):** FlashAttention-style online-softmax over gathered top-k KV; `-1` idx → masked
(zero KV, `-inf` score). Heads padded to 16 for kernel efficiency (K:359-362). q,kv,o all bf16.

---

## 5. RoPE / YaRN (M:205-250)
`precompute_freqs_cis(dim=64, seqlen, original_seq_len, base, factor=16, beta_fast=32, beta_slow=1)`.
- **Compressed layers** (ratio≠0): `original_seq_len=65536`, `base=compress_rope_theta=160000` → **YaRN ON**
  (freq interpolation with linear ramp between correction dims, M:227-230).
- **Pure sliding layers** (ratio==0): `original_seq_len=0`, `base=rope_theta=10000` → **YaRN OFF** (M:482-487).
`apply_rotary_emb` (M:238-250): complex-mult on interleaved pairs (`view_as_complex` of (…,2) unflatten —
**interleaved**, not rotate_half). `inverse=True` conjugates → used to DE-rotate `o` after attention (M:539).

## 6. KV Compressor (M:285-383) — learned gated pooling
For compressed layers, KV is pooled over `ratio` (4 or 128) consecutive tokens:
`kv=wkv(x)`, `score=wgate(x)` (both fp32) + learned APE `ape[ratio, coff·d]`; `kv=Σ (kv·softmax(score))` over
the window (M:348). `ratio==4` uses **overlapping** windows (coff=2, overlap_transform M:313-320). Result →
RMSNorm → RoPE last-64 → (indexer path) Hadamard-rotate + fp4-quant, else fp8-simulate non-rope (M:368-378).
Decode path keeps `kv_state`/`score_state` ring buffers, compresses every `ratio` steps (M:349-365).

## 7. DSA Indexer (M:386-439) — the "lightning indexer", only on ratio==4 layers
Own `Compressor(rotate=True)` builds a Hadamard-rotated, fp4 compressed-KV cache `[b, max_seq//4, index_head_dim=128]`.
`q = wq_b(qr)` → `[b,s,64,128]`, RoPE last-64, Hadamard-rotate (M:420), fp4-quant (M:422). 
`weights = weights_proj(x)·(softmax_scale·64**-0.5)` (bf16, M:424).
`index_score = relu(einsum("bshd,btd->bsht", q, kv_cache)) · weights` summed over heads (M:426-427);
causal-mask at prefill; `topk_idxs = topk(index_score, min(512, end//4))` (M:433) → these compressed positions
are appended to the window idxs for `sparse_attn`. **This top-512 selection is what governs real KV read cost.**

---

## 8. MoE (M:551-649)
**Gate (M:551-589):** `scores = linear(x.float(), weight.float())` → `[*,n_routed]`.
- score_func `sqrtsoftplus`: `scores = sqrt(softplus(scores))` (M:576). (softmax/sigmoid also supported.)
- **noaux_tc:** `bias` (fp32, per-expert) added to scores for **top-k SELECTION only** (M:579-580); routing
  **weights** gathered from the *original* (pre-bias) scores (M:585).
- **hash layers** (layer_id < num_hash_layers=3): indices come from `tid2eid[input_ids]` lookup (int32 param),
  NOT from scores (M:581-582). First 3 layers are hash-routed.
- weights renormalized to sum 1 (norm_topk_prob, M:586-587) then `× route_scale=1.5` (M:588).
**Expert (M:592-611):** SwiGLU `w2(silu(clamp(w1 x, max=10)) · clamp(w3 x, ±10))`; compute in fp32; ×routing weight.
**MoE (M:634-649):** top-6 routed (loop/bincount, M:639-645) + **always-on shared expert** (M:648). No dense-MLP-parallel branch (unlike Gemma).

## 9. Final head (M:719-740)
`hc_head` (4→1) → `RMSNorm(model.norm)` → `lm_head` (fp32 weight, **NOT tied**, vocab 129280). No softcap.
`sample` (M:939-946): Gumbel-max (temp>0) or argmax (temp=0).

---

## 10. DSpark draft (M:743-874) — block-diffusion MTP, replaces DFlash

Stored under `mtp.*` namespace; **shares** the main model's `embed` + `head` (M:903-904). One `DSparkBlock`
(a `Block` subclass with `DSparkAttention`, `compress_ratio=0` → pure sliding, no compression/indexer).

**Tap (M:920-921):** main model appends `h.mean(dim=2)` (HC 4→mean) at layers 40,41,42 →
`main_hidden = cat(3×[b,s,d]) = [b,s,3d]`.
**forward_embed (M:851-858):** `main_x = main_norm(main_proj(main_hidden))` (3d→d). Draft input = `[input_ids, noise,noise,noise,noise]` (block_size=5, first slot real, rest `noise_token_id=128799`) → embed → expand to 4 HC copies.
**DSparkAttention (M:750-792):** at start_pos==0 only builds KV from `main_x` into a sliding-128 cache (M:763-769).
At decode: draft q/kv over the block, `sparse_attn` over `[sliding main-KV ⊕ block]` (`get_dspark_topk_idxs` M:743-747).
**forward_head (M:860-874):** `hc_head` → norm → `head` → per-block **autoregressive** sampling of block_size draft
tokens; each step adds a **Markov bias** `markov_head(prev_id)` (bigram: embed rank-256 → head, M:795-804) to the logits,
samples next (M:867-871). A **confidence head** (M:807-815) scores each draft token from `[hidden ⊕ markov_embed]`.
Returns `(output_ids[block+1], logits, confidence)`.

**forward_spec (M:928-936):** prefill (start_pos==0) only builds draft KV (returns None); decode returns the block.
**IMPORTANT:** `generate.py` runs **pure AR** (`model.forward` only, M:42) — the reference ships **no accept/verify
loop**. The verify (compare draft `output_ids` vs target, accept longest matching prefix, use `confidence` for
early-exit) must be implemented in our harness (the DeepSpec repo / SGLang holds the production version). The
`__main__` in model.py (M:949-961) shows only the call *pattern*, not acceptance.

---

## 11. Checkpoint layout & key mapping (convert.py) — for the CUDA loader
`convert.py` transforms raw HF shards → `model{rank}-mp{world}.safetensors`. Single-GPU Thor = **mp=1**
(`--model-parallel 1`, `--n-experts 160` for REAP), no sharding, one output file. Key rewrites (cvt:89-105):
strip `model.`; **skip** `mtp.*` emb/head weights (shared from main, cvt:91); `self_attn→attn`; `mlp→ffn`;
`weight_scale_inv→scale`; `e_score_correction_bias→bias`. `wo_a`: fp8→**dequantized to bf16** at convert (cvt:123-127).
Experts int8→`view(float4_e2m1fn_x2)` for fp4 (cvt:135) OR `cast_e2m1fn_to_e4m3fn` for fp8 (cvt:17-52).
Our loader can either consume converted mp=1 files OR read raw HF shards + replicate this layout in `safetensors.h`.

## 12. Driver (generate.py)
Left-padded batch; prefill `[prev:cur]` then 1-tok decode (M-gen:41-48); prompt tokens override predictions.
Interactive: max_seq_len 64K, batch 1. Loads via `load_model(..., strict=False)`. Chat encoding via
`encoding/encoding_dsv4.py` (`encode_messages(..., thinking_mode="chat")`).

---

## 13. Gemma-4 engine → DeepSeek-V4 adaptation impact (what carries / what's new)

CARRIES OVER (from gemma-cuda-hybrid): safetensors mmap loader (extend for the key map above), NVFP4 e2m1
dequant + fp4 GEMM (`fp4_gemm.cu` — experts), Marlin `mma.sync` tc GEMM, KV ring-buffer mgmt, sampler
(Gumbel-max already matches M:939), server/OpenAI API, prefix cache, bench gates.

NET-NEW KERNELS (none exist in the Gemma engine):
1. **HC + Sinkhorn** (`hc_split_sinkhorn`, hc_pre/post/head) — pervasive, every block. Highest-risk novelty.
2. **MLA attention** — q/kv-LoRA, per-head q-RMS, nope/rope split, grouped o-LoRA einsum, attn_sink.
3. **sparse_attn** — index-gather FlashAttention over window⊕compressed top-k (K:276-368).
4. **KV Compressor** — gated-pooling, overlap, APE, decode ring-state.
5. **DSA Indexer** — own Hadamard-rotated fp4 compressed cache + top-512 scoring.
6. **FP8 128×128-block GEMM** (`fp8_gemm`) — dense/attn linears (Gemma was NVFP4-only).
7. **Hadamard transform** (`fast_hadamard_transform`) — indexer/compressor rotation.
8. **YaRN** freq precompute (interleaved-pair RoPE, not rotate_half — differs from Gemma).
9. **noaux_tc / sqrtsoftplus / hash router** — new routing math; +shared expert (no dense-parallel).
10. **DSpark draft** — replaces DFlash entirely (block-diffusion MTP + Markov + confidence heads).

REMOVE (Gemma-specific): PLE, layer_scalar, logit softcap, dual dense∥MoE branch, embed sqrt(d) scale, k_eq_v.

## 14. Open questions / verify-at-implementation
1. `compress_ratios` has 46 entries for 43 layers + tail — confirm exact per-layer ratio array from the REAP
   checkpoint's `config.json` (indexer only on ratio==4 layers).
2. Exact `tid2eid` hash table contents (first 3 layers) — read from checkpoint.
3. Whether REAP ships its own trained `mtp.*` (config `num_nextn_predict_layers:1`) and how it compares to the
   DSpark head — decide at Gate 2 which to attach.
4. Hadamard transform size/normalization (`scale = dim**-0.5`, M:257) and whether Thor needs a custom kernel vs
   the `fast_hadamard_transform` CUDA extension building on sm_110a.
5. tilelang `sparse_attn` block=64/heads-padded-16 tuning → retune for Thor's 20 SMs / 228KB smem.
