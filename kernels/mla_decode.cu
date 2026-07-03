// mla_decode.cu — M=1 KV-cache decode for a pure-sliding MLA layer. See mla_decode.h.
#include "mla_decode.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "deepseek_v4.h"
#include <vector>
#include <cmath>
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// Build the sliding-window key indices for a single query at position `pos`: [base .. pos], base=max(0,pos-W+1).
__global__ void k_win_idx(int* idx, int base, int width){
    int k = blockIdx.x*blockDim.x + threadIdx.x; if(k>=width) return; idx[k] = base + k;
}

// ---- prefill: fill window-KV cache for x[0..s-1] (identical to mla_forward's internal kv) ----
void mla_cache_kv(float* kvcache, const float* x, const MLAWeights& w, int s, cudaStream_t stream){
    uint8_t* xq; float* xs;
    CU(cudaMalloc(&xq,(size_t)s*DIM)); CU(cudaMalloc(&xs,(size_t)s*(DIM/128)*4));
    act_quant_fp8(xq, xs, x, s, DIM, 128, stream);
    fp8_block_gemm(kvcache, xq, xs, w.wkv, w.wkv_s, s, HEAD_DIM, DIM, stream);
    rmsnorm(kvcache, kvcache, w.kv_norm, s, HEAD_DIM, EPS, true, stream);
    rope_interleaved(kvcache + NOPE_DIM, w.cosT, w.sinT, s, ROPE_DIM, false, HEAD_DIM, 1, stream);   // 1 cos row per token
    act_quant_fp8sim(kvcache, s, NOPE_DIM, 64, HEAD_DIM, stream);
    CU(cudaStreamSynchronize(stream)); cudaFree(xq); cudaFree(xs);
}

// ---- decode step: one token at position `pos` ----
void mla_decode_step(float* out, const float* x, const MLAWeights& w, float* kvcache, int pos, cudaStream_t stream){
    const int half = ROPE_DIM/2, Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float *cosP = w.cosT + (size_t)pos*half, *sinP = w.sinT + (size_t)pos*half;   // this position's RoPE row

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og;
    CU(cudaMalloc(&xq,DIM)); CU(cudaMalloc(&xs,(DIM/128)*4));
    CU(cudaMalloc(&qr,Q_LORA*4)); CU(cudaMalloc(&qrq,Q_LORA)); CU(cudaMalloc(&qrs,(Q_LORA/128)*4));
    CU(cudaMalloc(&q,Kd*4)); CU(cudaMalloc(&o,Kd*4)); CU(cudaMalloc(&og,OB*4));
    CU(cudaMalloc(&ogq,OB)); CU(cudaMalloc(&ogs,(OB/128)*4));

    // 1. quantize x once (shared by wq_a and wkv)
    act_quant_fp8(xq, xs, x, 1, DIM, 128, stream);

    // 2. q = rope( per-head-rms( wq_b( q_norm( wq_a(x) ) ) ) )
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, 1, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, 1, Q_LORA, EPS, true, stream);
    act_quant_fp8(qrq, qrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, 1, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, N_HEADS, HEAD_DIM, EPS, false, stream);                          // per-head, 1 token
    rope_interleaved(q + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);  // all heads share pos row

    // 3. new token's KV -> append to cache[pos]
    float* kvn = kvcache + (size_t)pos*HEAD_DIM;
    fp8_block_gemm(kvn, xq, xs, w.wkv, w.wkv_s, 1, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, w.kv_norm, 1, HEAD_DIM, EPS, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, 1, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(kvn, 1, NOPE_DIM, 64, HEAD_DIM, stream);

    // 4. sparse attention over the sliding window [base..pos]
    int base = pos - WINDOW + 1; if(base < 0) base = 0; int width = pos + 1 - base;
    int* didx; CU(cudaMalloc(&didx, width*4));
    k_win_idx<<<(width+63)/64,64,0,stream>>>(didx, base, width);
    sparse_attn(o, q, kvcache, w.attn_sink, didx, 1, 1, N_HEADS, HEAD_DIM, pos+1, width, scale, stream);

    // 5. de-rotate o, grouped o-LoRA, wo_b
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, w.wo_a, 1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, 1, DIM, OB, stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);
    cudaFree(o);cudaFree(og);cudaFree(ogq);cudaFree(ogs);cudaFree(didx);
}
