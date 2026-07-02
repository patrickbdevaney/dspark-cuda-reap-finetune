// mla_attn.h — MLA attention primitives (deepseek_v4). Correctness-first, fp32 I/O for Gate K.
// sparse_attn: gathered top-k attention w/ learnable sink (kernel.py:276-368).
// rope_interleaved: interleaved-pair RoPE fwd/inverse (model.py:238-250).
// rmsnorm: fp32 RMSNorm, optional weight (model.py:189-202; per-head q-norm has no weight).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// q:[b,m,h,d] kv:[b,n,d] attn_sink:[h] topk_idxs:[b,m,topk] (-1=masked) -> o:[b,m,h,d]. All fp32; sink fp32.
void sparse_attn(float* o, const float* q, const float* kv, const float* attn_sink,
                 const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                 float scale, cudaStream_t stream = 0);

// In-place interleaved-pair RoPE on x[rows, rope_dim] using cos/sin[rows, rope_dim/2]. inverse => conj.
void rope_interleaved(float* x, const float* cosT, const float* sinT,
                      int rows, int rope_dim, bool inverse, cudaStream_t stream = 0);

// y[rows,dim] = x * rsqrt(mean(x^2)+eps) * (weight if has_weight). fp32.
void rmsnorm(float* y, const float* x, const float* weight, int rows, int dim,
             float eps, bool has_weight, cudaStream_t stream = 0);

// Fused FP8 QAT-sim: per `block`-group along dim, quant->e4m3 (pow2/ue8m0 scale)->dequant, in-place. fp32.
void act_quant_fp8sim(float* x, int rows, int dim, int block, cudaStream_t stream = 0);

// Grouped o-LoRA GEMM: out[bs,G,R] = sum_d o[bs,G,d] * wo_a[G,R,d]. fp32.
void ogroup_gemm(float* out, const float* o, const float* wo_a,
                 int bs, int G, int R, int Kd, cudaStream_t stream = 0);
