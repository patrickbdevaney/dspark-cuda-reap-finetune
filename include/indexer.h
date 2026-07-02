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
