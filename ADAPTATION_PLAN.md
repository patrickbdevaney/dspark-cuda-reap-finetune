# dspark-cuda-reap-finetune — Adaptation Plan

Fork of `gemma-cuda-hybrid` (bit-exact pure-CUDA NVFP4 engine for Jetson Thor sm_110a,
~118 tok/s on Gemma-4-26B). Goal: pure-CUDA inference **and** offline draft-head fine-tune
of the **DSpark MTP head** onto **`0xSero/DeepSeek-V4-Flash-180B`** (REAP K160, NVFP4/FP8).

Why pure-CUDA over vLLM/SGLang here: model sits at the edge of Thor's 122.8 GiB unified pool
(weights ~96.66 GiB). vLLM stages weights host-side then copies → transient ~2× spike → OOM
risk on unified memory. A binary that mmaps NVFP4/FP8 shards straight to device, frees
instantly, and exposes hidden-state + logit tensors directly is the right tool for a fixed
model on a fixed edge chip — and gives the draft-head trainer exactly the tensors it needs.

---

## 1. Architecture delta: Gemma-4-26B (have) → DeepSeek-V4-Flash-180B-REAP (want)

| Axis | Gemma-4-26B (current engine) | DeepSeek-V4-Flash-180B-REAP | Kernel impact |
|---|---|---|---|
| hidden H | 2816 | **4096** | constants only |
| layers | 30 | **43** | constants only |
| vocab | 262144 | **129280** | constants + tokenizer |
| attention | GQA (hd256/kv8 sliding + hd512/kv2 full) | **MLA** (q/kv-LoRA rank 1024, hd 512, qk_rope 64, kv-heads 1) | **NEW attention kernel** |
| sparse attn | none | **DSA lightning indexer** (index_n_heads 64, index_topk 512, index_head_dim 128) | **NEW indexer kernel** — drives real KV footprint |
| MoE experts | 128 routed, top-8, +1 dense-MLP | **160 routed, top-6, +1 shared** (REAP-pruned from 256) | router + expert loop (constants + routing math) |
| MoE routing | softmax→topk→renorm→per_expert_scale | **noaux_tc / sqrtsoftplus, routed_scaling_factor 1.5** | **NEW router math** |
| quant | NVFP4 W4A16 everywhere (router/lm_head bf16) | **mixed: FP8 e4m3 block-128×128 for linear/attn + NVFP4 for experts** | **NEW fp8-block GEMM** alongside existing `fp4_gemm.cu` |
| RoPE | default θ1e4 + proportional θ1e6 | θ 10000, **compress_rope_theta 160000**, qk_rope_head_dim 64 | rope table variant |
| draft | DFlash (qwen3 block-diffusion, gemma-specific) | **DSpark MTP** (1 nextn layer = MLA attn + 256-expert MoE + enorm/hnorm/eh_proj) | **replace `draft.cu`** |
| PLE / layer_scalar / logit-softcap | present (gemma-specific) | absent in deepseek_v4 | **remove** |

**Carries over unchanged:** safetensors mmap loader (`include/safetensors.h`), Marlin-class
`mma.sync` tc GEMM (`kernels/tc_verify_gemm.cu`), NVFP4 dequant (`kernels/fp4_gemm.cu`),
KV-cache mgmt, prefix caching, FP8 KV, sampling/acceptance, server + OpenAI API, bench gates.

## 2. Code anchors (where each change lands)

- `src/forward.cu:54-55` — model constants block → deepseek_v4 values.
- `src/forward.cu:585 attention_cached()` → MLA + DSA indexer (biggest single piece).
- `src/forward.cu:651 moe()` / `:641 expert_ffn()` → 160/top-6/shared, noaux_tc router.
- `src/forward.cu:539 linear()` → dispatch FP8-block GEMM (new) vs NVFP4 (experts) by tensor.
- `kernels/` → add `fp8_block_gemm.cu` (128×128 blockwise e4m3 dequant GEMM).
- `src/draft.cu` → replace DFlash with DSpark MTP nextn-layer (reuses target embed+lm_head; REAP prunes experts only, so embed/lm_head identical to unpruned → correct to share).
- `include/tokenizer.h` → DeepSeek BPE (vocab 129280) + DeepSeek chat/tool grammar.

## 3. Milestones & gates (cheap gates before the expensive train)

- **M0** repo forked from gemma-cuda-hybrid. ✅ (this commit)
- **M1 — inference port.** Adapt forward.cu (MLA, DSA, FP8+FP4 mixed, router, tokenizer).
  Bit-exact gate vs DeepSeek reference (`inference/generate.py` from the DSpark repo, or a
  transformers ref for a few tokens). → **Gate 1**: memory footprint ≈ published 96.66 GiB;
  ground-truth KV capacity at full load (validate ~37.7 KB/token / ~537K tokens claim — but
  DSA+MLA make this non-naive, MEASURE it); decode tok/s vs published no-MTP 18.9 / MTP2 24.4.
- **M2 — draft attach (unfine-tuned).** Load DSpark head (shards 46-48), wire MTP spec-decode
  through the existing verify path. → **Gate 2 (go/no-go)**: unfine-tuned accept rate / τ and
  decode speedup on the REAP target. Accept-rate ≈ 0 → stop. Non-trivial → fine-tune warranted.
  *This is the single cheapest, most decisive test. Record the pre-training τ precisely.*
- **M3 — offline capture.** Use the working CUDA target to self-generate responses (T≈0.8) and
  cache per-token {final hidden state h∈ℝ^4096, next-token input embedding, top-k target logits}.
- **M4 — head fine-tune** (see §4). Warm-start DSpark head, ~1-3 epochs on cached data.
- **M5 — eval.** ABBA interleaved: fine-tuned vs unfine-tuned (M2) vs no-spec. Held-out + OOD.
  Target τ ~4-5.7, speedup ~2.2-2.95× (optimistic ceiling from full-model analog) → ~41-56 tok/s.

## 4. Training recipe — CORRECTED by deep-research (supersedes the original directive)

Research (LK-Losses, FastMTP, EAGLE-3, SpecForge, SpecMQuant; 22/25 claims 3-0 verified):

| Param | Original directive | **Corrected (research-backed)** | Why |
|---|---|---|---|
| Loss | L1 feature (0.9) + argmax CE (0.1) | **per-depth token CE, decay-weighted (β=0.6, K=3)** + optional **logit-KD/KL** | DeepSeek-V3 native MTP loss is token-CE; EAGLE-3 *abandoned* feature regression. L1/feature is the wrong core objective for an MTP head. |
| LR | 6e-4 cosine | **5e-5 cosine** (FastMTP), consider **2e-5** (EAGLE-3 for ≥20B) | 6e-4 is ~10-100× too hot for a warm-started 180B head. |
| Steps | fixed 4500-6000 | **1-3 epochs**, warm-started (watch loss, not step count) | Warm-start converges in 1 epoch (LK-Losses) to 3 (FastMTP). |
| Init | official DSpark weights | ✅ warm-start confirmed correct | Scratch needs ~10 epochs; warm-start needs 1-3. |
| Data source | on-policy from REAP target | ✅ **CONFIRMED single most important decision** | Self-distillation from the *modified* target recovers the shift (FastMTP 1.81× vs 1.67×; +9.5 acc pts for pruned models). |
| Data size | 15-50k examples | **68k-390k** demonstrated sweet spot (15-50k is low end) | Token count + on-policy freshness > raw example count. |
| Data mix | tool/agentic + general + code | ✅ keep; add math | Acceptance is domain-sensitive. |
| Optimizer | — | AdamW(0.9,0.95), global batch ~64, only ~3% MTP params trainable, backbone frozen | FastMTP recipe, closest arch+objective match to DSpark. |

**Key caveat (de-risks the whole project):** quantization-transfer evidence covers weight-only
W4A16/W8A8 (near-lossless, no retrain). **NVFP4 = 4-bit weights AND activations (W4A4)** is more
aggressive than any measured case (W4A8 degraded most). And no source measures REAP *expert*-pruning's
hidden-state shift. So the shift is real and unmeasured → **the fine-tune is genuinely warranted**,
and Gate 2 is the true empirical unknown. Self-distillation-on-the-modified-target is what repairs it.

## 5. Open design decision (defer until Gate 2 passes)
Head training touches only ~200-600M MTP params on *cached* tensors — the 180B is NOT resident
during the gradient step. Options for the backward pass: (a) hand-CUDA autograd for the MTP
layer (MLA+MoE backward — high effort, maximal leanness), or (b) a minimal trainer on the cached
{hidden, target-token, top-k-logit} tuples. Decide after Gate 2 confirms it's worth building.
