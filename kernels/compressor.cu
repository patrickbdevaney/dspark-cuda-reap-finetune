// compressor.cu — KV Compressor gated-pooling core, correctness-first (Gate K: ref/gen_units gen_compressor).
#include "compressor.h"

// C[M,N] = A[M,K] @ B[N,K]^T. One warp per (m,n).
__global__ void gemm_fp32_kernel(float* __restrict__ C, const float* __restrict__ A,
                                 const float* __restrict__ B, int M, int N, int K) {
    int n = blockIdx.x, m = blockIdx.y; if (m >= M || n >= N) return;
    int lane = threadIdx.x & 31;
    const float* a = A + (size_t)m * K; const float* b = B + (size_t)n * K;
    float acc = 0.f; for (int k = lane; k < K; k += 32) acc += a[k] * b[k];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
void gemm_fp32(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    dim3 grid(N, M); gemm_fp32_kernel<<<grid, 32, 0, stream>>>(C, A, B, M, N, K);
}

// pooled[g,e] = Σ_p softmax_p(score[g*ratio+p,e]+ape[p,e]) * kv[g*ratio+p,e]. One thread per (g,e).
__global__ void compressor_pool_kernel(float* __restrict__ pooled, const float* __restrict__ kv,
                                       const float* __restrict__ score, const float* __restrict__ ape,
                                       int groups, int ratio, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= groups * d) return;
    int g = i / d, e = i % d;
    float mx = -1e30f;
    for (int p = 0; p < ratio; ++p) { float s = score[((size_t)(g * ratio + p)) * d + e] + ape[(size_t)p * d + e]; mx = fmaxf(mx, s); }
    float sum = 0.f, acc = 0.f;
    for (int p = 0; p < ratio; ++p) {
        float s = score[((size_t)(g * ratio + p)) * d + e] + ape[(size_t)p * d + e];
        float w = expf(s - mx); sum += w;
        acc += w * kv[((size_t)(g * ratio + p)) * d + e];
    }
    pooled[i] = acc / sum;
}
void compressor_pool(float* pooled, const float* kv, const float* score, const float* ape,
                     int groups, int ratio, int d, cudaStream_t stream) {
    compressor_pool_kernel<<<(groups * d + 255) / 256, 256, 0, stream>>>(pooled, kv, score, ape, groups, ratio, d);
}

// ---------------- overlap pooling (ratio==4) ----------------
// kv,score:[s,2d] (s=groups*ratio); ape:[ratio,2d]. Slot q in [0,2*ratio): q>=ratio -> current group token
// (q-ratio), dims [d:2d]; q<ratio -> previous group (g-1) token q, dims [0:d] (masked for g=0).
__global__ void compressor_pool_overlap_kernel(float* __restrict__ pooled, const float* __restrict__ kv,
                                               const float* __restrict__ score, const float* __restrict__ ape,
                                               int groups, int ratio, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= groups * d) return;
    int g = i / d, e = i % d; int twod = 2 * d, nslot = 2 * ratio;
    float mx = -1e30f;
    for (int q = 0; q < nslot; ++q) {
        float sc;
        if (q >= ratio) { int tok = g * ratio + (q - ratio); sc = score[(size_t)tok * twod + d + e] + ape[(size_t)(q - ratio) * twod + d + e]; }
        else if (g >= 1) { int tok = (g - 1) * ratio + q; sc = score[(size_t)tok * twod + e] + ape[(size_t)q * twod + e]; }
        else sc = -1e30f;
        mx = fmaxf(mx, sc);
    }
    float sum = 0.f, acc = 0.f;
    for (int q = 0; q < nslot; ++q) {
        float sc, kvv;
        if (q >= ratio) { int tok = g * ratio + (q - ratio); sc = score[(size_t)tok * twod + d + e] + ape[(size_t)(q - ratio) * twod + d + e]; kvv = kv[(size_t)tok * twod + d + e]; }
        else if (g >= 1) { int tok = (g - 1) * ratio + q; sc = score[(size_t)tok * twod + e] + ape[(size_t)q * twod + e]; kvv = kv[(size_t)tok * twod + e]; }
        else continue;
        float w = expf(sc - mx); sum += w; acc += w * kvv;
    }
    pooled[i] = acc / sum;
}
void compressor_pool_overlap(float* pooled, const float* kv, const float* score, const float* ape,
                             int groups, int ratio, int d, cudaStream_t stream) {
    compressor_pool_overlap_kernel<<<(groups * d + 255) / 256, 256, 0, stream>>>(pooled, kv, score, ape, groups, ratio, d);
}

// ---------------- full Compressor forward ----------------
#include "mla_attn.h"     // rmsnorm, rope_interleaved, act_quant_fp8sim/fp4sim
#include "indexer.h"      // hadamard
#include <cstdio>
#define CU2(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
void compressor_forward(float* out, const float* x, const float* wkv, const float* wgate,
                        const float* ape, const float* norm_w, const float* cosT, const float* sinT,
                        int s, int dim, int d, int ratio, bool overlap, int rope_dim, float eps,
                        bool rotate, cudaStream_t stream) {
    int coff = overlap ? 2 : 1, groups = s / ratio, od = coff * d;
    float *kv, *score;
    CU2(cudaMalloc(&kv, (size_t)s * od * 4)); CU2(cudaMalloc(&score, (size_t)s * od * 4));
    gemm_fp32(kv, x, wkv, s, od, dim, stream);
    gemm_fp32(score, x, wgate, s, od, dim, stream);
    if (overlap) compressor_pool_overlap(out, kv, score, ape, groups, ratio, d, stream);
    else         compressor_pool(out, kv, score, ape, groups, ratio, d, stream);
    rmsnorm(out, out, norm_w, groups, d, eps, true, stream);
    rope_interleaved(out + (d - rope_dim), cosT, sinT, groups, rope_dim, false, d, 1, stream);
    if (rotate) { hadamard(out, out, groups, d, stream); act_quant_fp4sim(out, groups, d, 32, d, stream); }  // indexer compressor
    else        { act_quant_fp8sim(out, groups, d - rope_dim, 64, d, stream); }                             // main compressor NoPE
    CU2(cudaStreamSynchronize(stream));
    cudaFree(kv); cudaFree(score);
}

// ---- incremental single-group emit (STRUCTURAL_PLAN Step 4 decode) ----
// Emit ONE compressed row = compressor_forward's out[g], from just this group's tokens (append-only KV: a
// compressed row finalizes once its `ratio` tokens exist and never changes). Non-overlap (ratio!=4): pools
// x[g*ratio .. g*ratio+ratio-1]. Overlap (ratio==4): pools the 2 local groups [(g-1)*ratio .. g*ratio+ratio-1]
// (prev half masked for g==0) and takes the current one. Bit-exact vs compressor_forward (same per-group math).
void compressor_emit_group(float* out_row, const float* x, int g, int ratio, const float* wkv,
                           const float* wgate, const float* ape, const float* norm_w,
                           const float* cc_cos, const float* cc_sin, int dim, int d, bool overlap,
                           int rope_dim, float eps, bool rotate, cudaStream_t stream){
    int coff = overlap ? 2 : 1, od = coff * d;
    int ntok, tok0, localg;
    if(overlap){ tok0 = (g>=1) ? (g-1)*ratio : 0; ntok = (g>=1) ? 2*ratio : ratio; localg = (g>=1) ? 1 : 0; }
    else       { tok0 = g*ratio; ntok = ratio; localg = 0; }
    const float* xg = x + (size_t)tok0*dim;
    float *kv,*score,*pooled;
    CU2(cudaMalloc(&kv,(size_t)ntok*od*4)); CU2(cudaMalloc(&score,(size_t)ntok*od*4));
    CU2(cudaMalloc(&pooled,(size_t)(localg+1)*d*4));
    gemm_fp32(kv, xg, wkv, ntok, od, dim, stream);
    gemm_fp32(score, xg, wgate, ntok, od, dim, stream);
    if(overlap) compressor_pool_overlap(pooled, kv, score, ape, localg+1, ratio, d, stream);
    else        compressor_pool(pooled, kv, score, ape, 1, ratio, d, stream);
    float* prow = pooled + (size_t)localg*d;                 // the target group's pooled row
    rmsnorm(out_row, prow, norm_w, 1, d, eps, true, stream);
    rope_interleaved(out_row + (d - rope_dim), cc_cos + (size_t)g*(rope_dim/2), cc_sin + (size_t)g*(rope_dim/2),
                     1, rope_dim, false, d, 1, stream);
    if(rotate){ hadamard(out_row, out_row, 1, d, stream); act_quant_fp4sim(out_row, 1, d, 32, d, stream); }
    else      { act_quant_fp8sim(out_row, 1, d - rope_dim, 64, d, stream); }
    CU2(cudaStreamSynchronize(stream));
    cudaFree(kv); cudaFree(score); cudaFree(pooled);
}
