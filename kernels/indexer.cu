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
