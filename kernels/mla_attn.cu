// mla_attn.cu — MLA attention primitives, correctness-first (Gate K oracle: ref/gen_units.py).
// Optimization (mma, smem KV staging, bf16) comes AFTER these pass their gate (CONSTITUTION Art. I).
#include "mla_attn.h"

// ---------------- sparse_attn ----------------
// One warp per (b, m, head): stream over the top-k gathered KV with online (running max/sum) softmax,
// then fold in the learnable sink (denominator-only). MLA => single latent KV shared across heads.
// Lanes hold d/32 elements of the accumulator and of q; scores are warp-reduced dot products.
__global__ void sparse_attn_kernel(float* __restrict__ o, const float* __restrict__ q,
                                   const float* __restrict__ kv, const float* __restrict__ attn_sink,
                                   const int* __restrict__ topk_idxs,
                                   int b, int m, int h, int d, int n, int topk, float scale) {
    int gid = blockIdx.x;                       // (b*m*h) index
    int head = gid % h; int bm = gid / h; int mi = bm % m; int bi = bm / m;
    int lane = threadIdx.x & 31;
    int per = (d + 31) / 32;                    // elems per lane along d
    const float* qp = q + (((size_t)(bi * m + mi) * h + head) * d);
    const int*   ip = topk_idxs + ((size_t)(bi * m + mi) * topk);

    float qreg[32];                             // per<=32 for d<=1024
    #pragma unroll
    for (int r = 0; r < 32; ++r) { int j = lane + r * 32; qreg[r] = (r < per && j < d) ? qp[j] : 0.f; }

    float acc[32]; for (int r = 0; r < 32; ++r) acc[r] = 0.f;
    float run_max = -1e30f, run_sum = 0.f;

    for (int t = 0; t < topk; ++t) {
        int idx = ip[t];
        if (idx < 0) continue;                  // masked slot
        const float* kp = kv + (((size_t)bi * n + idx) * d);
        // score = scale * dot(q, kv[idx])
        float part = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) part += qreg[r] * kp[j]; }
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) part += __shfl_down_sync(0xffffffff, part, o2);
        float score = __shfl_sync(0xffffffff, part, 0) * scale;   // broadcast to all lanes
        // online softmax update
        float new_max = fmaxf(run_max, score);
        float corr = expf(run_max - new_max);
        float p = expf(score - new_max);
        run_sum = run_sum * corr + p;
        #pragma unroll
        for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) acc[r] = acc[r] * corr + p * kp[j]; }
        run_max = new_max;
    }
    // sink: contributes exp(sink-max) to denominator only
    run_sum += expf(attn_sink[head] - run_max);
    float inv = (run_sum > 0.f) ? 1.f / run_sum : 0.f;
    float* op = o + (((size_t)(bi * m + mi) * h + head) * d);
    #pragma unroll
    for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) op[j] = acc[r] * inv; }
}

void sparse_attn(float* o, const float* q, const float* kv, const float* attn_sink,
                 const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                 float scale, cudaStream_t stream) {
    int blocks = b * m * h;
    sparse_attn_kernel<<<blocks, 32, 0, stream>>>(o, q, kv, attn_sink, topk_idxs, b, m, h, d, n, topk, scale);
}

// ---------------- rope_interleaved ----------------
// x[rows, rope_dim]; pairs (2j,2j+1) rotated by (cos_j, sin_j). inverse => sin -> -sin.
__global__ void rope_kernel(float* __restrict__ x, const float* __restrict__ cosT,
                            const float* __restrict__ sinT, int rows, int rope_dim, int inv) {
    int row = blockIdx.x; if (row >= rows) return;
    int half = rope_dim / 2;
    float* xr = x + (size_t)row * rope_dim;
    const float* c = cosT + (size_t)row * half;
    const float* s = sinT + (size_t)row * half;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float a = xr[2 * j], bb = xr[2 * j + 1];
        float sj = inv ? -s[j] : s[j], cj = c[j];
        xr[2 * j]     = a * cj - bb * sj;
        xr[2 * j + 1] = a * sj + bb * cj;
    }
}

void rope_interleaved(float* x, const float* cosT, const float* sinT,
                      int rows, int rope_dim, bool inverse, cudaStream_t stream) {
    rope_kernel<<<rows, 64, 0, stream>>>(x, cosT, sinT, rows, rope_dim, inverse ? 1 : 0);
}

// ---------------- rmsnorm ----------------
__global__ void rmsnorm_kernel(float* __restrict__ y, const float* __restrict__ x,
                               const float* __restrict__ w, int rows, int dim, float eps, int has_w) {
    int row = blockIdx.x; if (row >= rows) return;
    const float* xr = x + (size_t)row * dim; float* yr = y + (size_t)row * dim;
    __shared__ float red[256];
    float ss = 0.f; for (int j = threadIdx.x; j < dim; j += blockDim.x) ss += xr[j] * xr[j];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) yr[j] = xr[j] * inv * (has_w ? w[j] : 1.f);
}

void rmsnorm(float* y, const float* x, const float* weight, int rows, int dim,
             float eps, bool has_weight, cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(y, x, weight, rows, dim, eps, has_weight ? 1 : 0);
}
