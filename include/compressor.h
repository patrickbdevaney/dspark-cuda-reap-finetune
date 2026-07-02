// compressor.h — KV Compressor primitives (deepseek_v4, model.py:285-383).
// Core = learned gated-softmax pooling over `ratio` consecutive tokens. wkv/wgate are fp32 linears.
// This header covers the non-overlap pooling core (ratio!=4). Overlap (ratio==4) + DSA indexer come next.
#pragma once
#include <cuda_runtime.h>

// C[M,N] = A[M,K] @ B[N,K]^T, all fp32.
void gemm_fp32(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream = 0);

// Gated pooling: pooled[g,e] = Σ_p softmax_p(score[g*ratio+p,e] + ape[p,e]) * kv[g*ratio+p,e].
// kv,score:[groups*ratio, d]; ape:[ratio,d]; pooled:[groups,d].
void compressor_pool(float* pooled, const float* kv, const float* score, const float* ape,
                     int groups, int ratio, int d, cudaStream_t stream = 0);
