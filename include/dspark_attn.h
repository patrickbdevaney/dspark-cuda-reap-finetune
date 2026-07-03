// dspark_attn.h — DSparkAttention (model.py:750-793): block queries attend to [main-KV window ⊕ block-KV].
// main-KV is built from main_x (the projected layer-40/41/42 taps). Same MLA projections as mla_forward.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "mla_forward.h"   // MLAWeights

// Precompute main-KV from main_x[s,dim]: kv_norm(wkv(main_x)) + rope(per-position) + act_quant. -> main_kv[s, HEAD_DIM].
void dspark_main_kv(float* main_kv, const float* main_x, const MLAWeights& w, int s, float eps, cudaStream_t stream = 0);

// One anchor t: xin[block, dim] (draft block, post attn_norm). main_kv[s, HEAD_DIM] precomputed.
// window = main_kv[t+1-nwin .. t] (nwin=min(win,t+1)); block queries attend [window ⊕ block-KV] (bidirectional
// within block, per get_dspark_topk_idxs). cosB/sinB[block, ROPE_DIM/2] = block-position freqs. -> out[block, dim].
void dspark_attn_forward(float* out, const float* xin, const float* main_kv, int t,
                         const MLAWeights& w, const float* cosB, const float* sinB,
                         int block, int win, float eps, cudaStream_t stream = 0);

#include "block.h"   // BlockWeights
// Full DSparkBlock forward for one anchor: hc_pre->attn_norm->dspark_attn->hc_post->hc_pre->ffn_norm->moe->hc_post.
// x:[block, hc, dim] draft block. input_ids:[block]. main_kv:[s,HEAD_DIM]. t=anchor. -> out:[block, hc, dim].
void dspark_block_forward(float* out, const float* x, const int* input_ids, const float* main_kv, int t,
                          const BlockWeights& w, const float* cosB, const float* sinB, int block, int win,
                          int iters, float eps, cudaStream_t stream = 0);
