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
                            const float* __restrict__ sinT, int rows, int rope_dim, int inv,
                            int row_stride, int cos_stride_rows) {
    int row = blockIdx.x; if (row >= rows) return;
    int half = rope_dim / 2;
    float* xr = x + (size_t)row * row_stride;
    int crow = row / cos_stride_rows;
    const float* c = cosT + (size_t)crow * half;
    const float* s = sinT + (size_t)crow * half;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float a = xr[2 * j], bb = xr[2 * j + 1];
        float sj = inv ? -s[j] : s[j], cj = c[j];
        xr[2 * j]     = a * cj - bb * sj;
        xr[2 * j + 1] = a * sj + bb * cj;
    }
}

void rope_interleaved(float* x, const float* cosT, const float* sinT,
                      int rows, int rope_dim, bool inverse, int row_stride, int cos_stride_rows,
                      cudaStream_t stream) {
    if (row_stride < 0) row_stride = rope_dim;
    rope_kernel<<<rows, 64, 0, stream>>>(x, cosT, sinT, rows, rope_dim, inverse ? 1 : 0,
                                         row_stride, cos_stride_rows);
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

// ---------------- act_quant_fp8sim ----------------
// Per (row, block-of-`block`): amax -> pow2 scale (ue8m0) -> clamp(x/scale) to e4m3 -> dequant*scale.
// One block per (row, group); threads cover the group. Matches kernel.py act_quant(inplace, round_scale).
#include <cuda_fp8.h>
__global__ void act_quant_fp8sim_kernel(float* __restrict__ x, int rows, int active_dim, int block, int row_stride) {
    int ng = active_dim / block; int gid = blockIdx.x; if (gid >= rows * ng) return;
    int row = gid / ng, g = gid % ng;
    float* xr = x + (size_t)row * row_stride + (size_t)g * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 1e-4f);
    float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));      // pow2 (ue8m0)
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -448.f), 448.f);
        __nv_fp8_e4m3 e = __nv_fp8_e4m3(q);
        __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)e.__x, __NV_E4M3);
        xr[threadIdx.x] = __half2float(*reinterpret_cast<__half*>(&hr)) * scale;
    }
}
void act_quant_fp8sim(float* x, int rows, int active_dim, int block, int row_stride, cudaStream_t stream) {
    if (row_stride < 0) row_stride = active_dim;
    int threads = block < 32 ? 32 : block;
    act_quant_fp8sim_kernel<<<rows * (active_dim / block), threads, threads * sizeof(float), stream>>>(x, rows, active_dim, block, row_stride);
}

// Real activation quant -> fp8 bytes + f32 pow2 scale (the activation half of an fp8 linear).
__global__ void act_quant_fp8_kernel(uint8_t* __restrict__ a, float* __restrict__ as,
                                     const float* __restrict__ x, int rows, int K, int block) {
    int nb = K / block; int gid = blockIdx.x; if (gid >= rows * nb) return;
    int row = gid / nb, b = gid % nb;
    const float* xr = x + (size_t)row * K + (size_t)b * block;
    uint8_t* ar = a + (size_t)row * K + (size_t)b * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 1e-4f);
    float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));
    if (threadIdx.x == 0) as[(size_t)row * nb + b] = scale;
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -448.f), 448.f);
        ar[threadIdx.x] = __nv_fp8_e4m3(q).__x;
    }
}
void act_quant_fp8(uint8_t* a_fp8, float* a_s, const float* x, int rows, int K, int block, cudaStream_t stream) {
    int threads = block < 32 ? 32 : block;
    act_quant_fp8_kernel<<<rows * (K / block), threads, threads * sizeof(float), stream>>>(a_fp8, a_s, x, rows, K, block);
}

// ---------------- ogroup_gemm ----------------
// out[bs,G,R] = sum_d o[bs,G,d] * wo_a[G,R,d]. One warp per (bs,G,R) reducing over d.
__global__ void ogroup_gemm_kernel(float* __restrict__ out, const float* __restrict__ o,
                                   const float* __restrict__ wo_a, int bs, int G, int R, int Kd) {
    int gid = blockIdx.x; int r = gid % R; int bg = gid / R; int gg = bg % G; int bb = bg / G;
    int lane = threadIdx.x & 31;
    const float* op = o + (((size_t)bb * G + gg) * Kd);
    const float* wp = wo_a + (((size_t)gg * R + r) * Kd);
    float acc = 0.f;
    for (int d = lane; d < Kd; d += 32) acc += op[d] * wp[d];
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
    if (lane == 0) out[((size_t)bb * G + gg) * R + r] = acc;
}
void ogroup_gemm(float* out, const float* o, const float* wo_a,
                 int bs, int G, int R, int Kd, cudaStream_t stream) {
    ogroup_gemm_kernel<<<bs * G * R, 32, 0, stream>>>(out, o, wo_a, bs, G, R, Kd);
}

// ---------------- act_quant_fp4sim ----------------
// FP4 e2m1 QAT-sim (quant->dequant), pow2 scale, fp4_max=6, block=32. Matches kernel.py fp4_act_quant inplace.
__device__ __forceinline__ float round_e2m1(float v) {   // nearest signed E2M1 grid value {0,.5,1,1.5,2,3,4,6}
    float a = fabsf(v), m;
    if (a < 0.25f) m = 0.f; else if (a < 0.75f) m = 0.5f; else if (a < 1.25f) m = 1.f;
    else if (a < 1.75f) m = 1.5f; else if (a < 2.5f) m = 2.f; else if (a < 3.5f) m = 3.f;
    else if (a < 5.f) m = 4.f; else m = 6.f;
    return (v < 0.f) ? -m : m;
}
__global__ void act_quant_fp4sim_kernel(float* __restrict__ x, int rows, int active_dim, int block, int row_stride) {
    int ng = active_dim / block; int gid = blockIdx.x; if (gid >= rows * ng) return;
    int row = gid / ng, g = gid % ng;
    float* xr = x + (size_t)row * row_stride + (size_t)g * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 6.f * 7.5231631e-37f);              // 6*2^-126
    float scale = exp2f(ceilf(log2f(amax * (1.f / 6.f))));
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -6.f), 6.f);
        xr[threadIdx.x] = round_e2m1(q) * scale;
    }
}
void act_quant_fp4sim(float* x, int rows, int active_dim, int block, int row_stride, cudaStream_t stream) {
    if (row_stride < 0) row_stride = active_dim;
    int threads = block < 32 ? 32 : block;
    act_quant_fp4sim_kernel<<<rows * (active_dim / block), threads, threads * sizeof(float), stream>>>(x, rows, active_dim, block, row_stride);
}
