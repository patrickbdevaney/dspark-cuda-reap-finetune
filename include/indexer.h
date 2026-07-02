// indexer.h — DSA Indexer primitives (deepseek_v4, model.py:386-439).
// hadamard: randomized Hadamard rotation (rotate_activation, model.py:253-257) = (x @ H_D) * D^-0.5.
// index_score: DSA scoring index_score[s,t] = Σ_h relu(q[s,h]·kv[t]) * weights[s,h] (model.py:426-427).
#pragma once
#include <cuda_runtime.h>

// y[rows,D] = (x[rows,D] @ H_D) * D^-0.5, H_D[i,j] = (-1)^popcount(i&j). D must be a power of two.
void hadamard(float* y, const float* x, int rows, int D, cudaStream_t stream = 0);

// index_score[s,t] = Σ_h relu(Σ_d q[s,h,d]*kv[t,d]) * weights[s,h].
// q:[S,H,d]  kv:[T,d]  weights:[S,H]  -> score:[S,T].
void index_score(float* score, const float* q, const float* kv, const float* weights,
                 int S, int T, int H, int d, cudaStream_t stream = 0);

// Full DSA Indexer forward (prefill, b=1). wq_b:[n_heads*idx_hd, q_lora] fp8 + scale; weights_proj:[n_heads,dim];
// rotate-compressor weights (c_*); q_cos/q_sin:[s,rd/2] (query freqs), c_cos/c_sin:[s/ratio,rd/2] (compressed).
// Outputs: index_score:[s, s/ratio] (post causal-mask, for gating) and topk_idxs:[s, min(index_topk,s/ratio)]
// (offset-applied, -1 where masked). offset = position base for the compressed idxs in the KV.
void indexer_forward(float* index_score_out, int* topk_idxs, const float* x, const float* qr,
                     const unsigned char* wq_b, const float* wq_b_s, const float* weights_proj,
                     const float* c_wkv, const float* c_wgate, const float* c_ape, const float* c_norm,
                     const float* q_cos, const float* q_sin, const float* c_cos, const float* c_sin,
                     int s, int dim, int q_lora, int n_heads, int idx_hd, int rd, int ratio,
                     int index_topk, int offset, float eps, cudaStream_t stream = 0);
