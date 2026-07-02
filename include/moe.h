// moe.h — DeepSeek-V4 MoE primitives (deepseek_v4). Correctness-first, fp32 I/O for Gate K.
// fp4_gemm: FP8-act x FP4-weight GEMM (kernel.py fp4_gemm) — the routed-expert GEMM.
// moe_router_score: sqrtsoftplus + noaux_tc (bias-for-selection) topk + renorm*route_scale.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T. a_s[M,K/128] (per-128 act, f32). B_fp4 packed [N,K/2] (2 nibbles/byte).
// b_s[N,K/32] (per-32 weight, f32 pow2). E2M1 nibble = sign(bit3)|magnitude-index.
void fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s,
              const uint8_t* B_fp4, const float* b_s, int M, int N, int K, cudaStream_t stream = 0);

// Score router. x[n,dim], gate_w[n_routed,dim] (f32). -> weights[n,topk], indices[n,topk].
// scores = sqrtsoftplus(x@gate_w^T); select top-k of (scores+bias); weights = gather(scores)/sum*route_scale.
void moe_router_score(float* weights, int* indices, const float* x, const float* gate_w,
                      const float* bias, int n, int dim, int n_routed, int topk,
                      float route_scale, cudaStream_t stream = 0);

// Full MoE forward (model.py:551-649). Routed experts fp4 SwiGLU (top-k) + always-on fp8 shared expert.
// Weights: routed w1/w3 packed [E,inter,dim/2] + scale [E,inter,dim/32]; w2 [E,dim,inter/2] + [E,dim,inter/32].
// Shared sw1/sw3 fp8 [inter,dim] + scale [ceil(inter/128),ceil(dim/128)]; sw2 [dim,inter] + [.,.].
struct MoEWeights {
    const float* gate_w;                       // [n_routed, dim] f32
    const float* gate_bias;                    // [n_routed] f32 (null for hash layers)
    const long*  tid2eid;                      // [vocab, n_act] i64 (hash layers) else null
    bool is_hash;
    const uint8_t *w1, *w2, *w3;  const float *w1s, *w2s, *w3s;     // routed experts (fp4)
    const uint8_t *sw1, *sw2, *sw3; const float *sw1s, *sw2s, *sw3s; // shared expert (fp8)
    int n_routed, n_act, dim, inter, vocab;
    float route_scale, swiglu_limit;
};
// x:[bs,dim] fp32, input_ids:[bs] i32 (for hash routing) -> out:[bs,dim] fp32.
void moe_forward(float* out, const float* x, const int* input_ids, const MoEWeights& w,
                 int bs, cudaStream_t stream = 0);
