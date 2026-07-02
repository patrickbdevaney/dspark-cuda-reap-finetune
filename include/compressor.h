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

// Overlap pooling (ratio==4, model.py overlap_transform + softmax). kv,score:[groups*ratio, 2d];
// ape:[ratio,2d]; pooled:[groups,d]. Each group softmaxes over 2*ratio slots: current group (dims [d:2d])
// + previous group (dims [0:d], masked for g=0).
void compressor_pool_overlap(float* pooled, const float* kv, const float* score, const float* ape,
                             int groups, int ratio, int d, cudaStream_t stream = 0);

// Full Compressor forward (prefill, remainder-free): gemm(wkv/wgate) -> pool -> norm -> RoPE(last 64) ->
// [rotate ? hadamard(full d) + fp4-sim(full d) : fp8-sim(NoPE)]. x:[s,dim] -> out:[s/ratio, d].
// cos/sin:[s/ratio, 64/2] (compressed-position freqs). rotate=True is the DSA indexer's compressor.
void compressor_forward(float* out, const float* x, const float* wkv, const float* wgate,
                        const float* ape, const float* norm_w, const float* cosT, const float* sinT,
                        int s, int dim, int d, int ratio, bool overlap, int rope_dim, float eps,
                        bool rotate, cudaStream_t stream = 0);
