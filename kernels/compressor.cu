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
