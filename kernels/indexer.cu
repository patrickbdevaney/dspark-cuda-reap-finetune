// indexer.cu — DSA Indexer primitives, correctness-first (Gate K: ref/gen_units gen_hadamard/gen_index_score).
#include "indexer.h"

// Hadamard: y[r,j] = D^-0.5 * Σ_i x[r,i] * (-1)^popcount(i&j). One thread per (row, j).
__global__ void hadamard_kernel(float* __restrict__ y, const float* __restrict__ x, int rows, int D, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= rows * D) return;
    int r = idx / D, j = idx % D;
    const float* xr = x + (size_t)r * D;
    float acc = 0.f;
    for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -xr[i] : xr[i];
    y[idx] = acc * scale;
}
void hadamard(float* y, const float* x, int rows, int D, cudaStream_t stream) {
    float scale = rsqrtf((float)D);
    hadamard_kernel<<<(rows * D + 255) / 256, 256, 0, stream>>>(y, x, rows, D, scale);
}

// index_score[s,t] = Σ_h relu(Σ_d q[s,h,d]*kv[t,d]) * weights[s,h]. One thread per (s,t).
__global__ void index_score_kernel(float* __restrict__ score, const float* __restrict__ q,
                                   const float* __restrict__ kv, const float* __restrict__ weights,
                                   int S, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= S * T) return;
    int s = idx / T, t = idx % T;
    const float* kvt = kv + (size_t)t * d;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (((size_t)s * H + h) * d);
        float dot = 0.f; for (int e = 0; e < d; ++e) dot += qh[e] * kvt[e];
        acc += fmaxf(dot, 0.f) * weights[(size_t)s * H + h];      // relu * head weight
    }
    score[(size_t)s * T + t] = acc;
}
void index_score(float* score, const float* q, const float* kv, const float* weights,
                 int S, int T, int H, int d, cudaStream_t stream) {
    index_score_kernel<<<(S * T + 255) / 256, 256, 0, stream>>>(score, q, kv, weights, S, T, H, d);
}

// ================= DSA Indexer forward =================
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // act_quant_fp8, rope_interleaved, act_quant_fp4sim
#include "compressor.h"    // gemm_fp32, compressor_forward
#include <cstdio>
#define CUI(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void k_scale(float* y, float sc, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=sc; }

// causal mask: score[si,t] = -inf where t >= (si+1)/ratio.
__global__ void k_causal_mask(float* score, int s, int T, int ratio){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s*T) return; int si=i/T, t=i%T;
    if (t >= (si+1)/ratio) score[i] = -1e30f;
}
// per query: descending top-k of score[si,:T]; then idx = (t >= (si+1)/ratio) ? -1 : t+offset.
__global__ void k_topk_offset(int* out, const float* score, int s, int T, int topk, int ratio, int offset){
    int si=blockIdx.x; if(si>=s || threadIdx.x) return;
    extern __shared__ float sh[];
    for(int t=0;t<T;++t) sh[t]=score[(size_t)si*T+t];
    int thr=(si+1)/ratio;
    for(int k=0;k<topk;++k){
        float best=-1e30f; int bi=-1;
        for(int t=0;t<T;++t) if(sh[t]>best){best=sh[t];bi=t;}
        if(bi>=0) sh[bi]=-1e30f;
        out[(size_t)si*topk+k] = (bi<0 || bi>=thr) ? -1 : bi+offset;
    }
}

void indexer_forward(float* index_score_out, int* topk_idxs, const float* x, const float* qr,
                     const unsigned char* wq_b, const float* wq_b_s, const float* weights_proj,
                     const float* c_wkv, const float* c_wgate, const float* c_ape, const float* c_norm,
                     const float* q_cos, const float* q_sin, const float* c_cos, const float* c_sin,
                     int s, int dim, int q_lora, int n_heads, int idx_hd, int rd, int ratio,
                     int index_topk, int offset, float eps, cudaStream_t stream) {
    int T = s / ratio, QD = n_heads * idx_hd;
    float softmax_scale = rsqrtf((float)idx_hd), wscale = softmax_scale * rsqrtf((float)n_heads);
    unsigned char* qrq; float *qrs, *q, *qtmp, *ckv, *weights;
    CUI(cudaMalloc(&qrq,(size_t)s*q_lora)); CUI(cudaMalloc(&qrs,(size_t)s*(q_lora/128)*4));
    CUI(cudaMalloc(&q,(size_t)s*QD*4)); CUI(cudaMalloc(&qtmp,(size_t)s*QD*4));
    CUI(cudaMalloc(&ckv,(size_t)T*idx_hd*4)); CUI(cudaMalloc(&weights,(size_t)s*n_heads*4));

    act_quant_fp8(qrq, qrs, qr, s, q_lora, 128, stream);
    fp8_block_gemm(q, qrq, qrs, wq_b, wq_b_s, s, QD, q_lora, stream);              // [s, n_heads*idx_hd]
    rope_interleaved(q + (idx_hd - rd), q_cos, q_sin, s*n_heads, rd, false, idx_hd, n_heads, stream);
    hadamard(qtmp, q, s*n_heads, idx_hd, stream);                                 // out!=in
    act_quant_fp4sim(qtmp, s*n_heads, idx_hd, 32, idx_hd, stream);                // fp4-sim
    compressor_forward(ckv, x, c_wkv, c_wgate, c_ape, c_norm, c_cos, c_sin, s, dim, idx_hd, ratio, true, rd, eps, true, stream);
    gemm_fp32(weights, x, weights_proj, s, n_heads, dim, stream);
    k_scale<<<(s*n_heads+255)/256,256,0,stream>>>(weights, wscale, s*n_heads);
    index_score(index_score_out, qtmp, ckv, weights, s, T, n_heads, idx_hd, stream);
    k_causal_mask<<<(s*T+255)/256,256,0,stream>>>(index_score_out, s, T, ratio);
    int topk = index_topk < T ? index_topk : T;
    k_topk_offset<<<s, 32, T*sizeof(float), stream>>>(topk_idxs, index_score_out, s, T, topk, ratio, offset);
    CUI(cudaStreamSynchronize(stream));
    cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(qtmp);cudaFree(ckv);cudaFree(weights);
}
