// moe.cu — MoE primitives, correctness-first (Gate K oracle: ref/gen_units.py).
#include "moe.h"
#include <cuda_fp8.h>

__constant__ float E2M1_MAG[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float dec_e4m3(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}
__device__ __forceinline__ float dec_fp4(uint8_t nib) {   // sign(bit3) | mag-index(bits0-2)
    float m = E2M1_MAG[nib & 7];
    return (nib & 8) ? -m : m;
}

// ---------------- fp4_gemm (fp8 act x fp4 weight) ----------------
// One warp per (m,n). Walk K; per K-block accumulate raw dot then apply act(per-128) & weight(per-32) scales.
__global__ void fp4_gemm_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                const uint8_t* __restrict__ B, const float* __restrict__ bs,
                                int M, int N, int K) {
    int n = blockIdx.x, m = blockIdx.y; if (m >= M || n >= N) return;
    int lane = threadIdx.x & 31;
    int KBa = K / 128, KBw = K / 32;
    const uint8_t* Arow = A + (size_t)m * K;
    const uint8_t* Bpack = B + (size_t)n * (K / 2);          // packed nibbles
    const float* asr = as + (size_t)m * KBa;
    const float* bsr = bs + (size_t)n * KBw;
    float acc = 0.f;
    for (int kb = 0; kb < KBw; ++kb) {                       // per 32-weight-block (constant weight scale)
        float sub = 0.f; int base = kb * 32;
        for (int j = lane; j < 32; j += 32) {                // 1 iter (32 lanes cover 32)
            int k = base + j;
            float av = dec_e4m3(Arow[k]) * asr[k / 128];
            uint8_t byte = Bpack[k >> 1];
            uint8_t nib = (k & 1) ? (byte >> 4) & 0xF : byte & 0xF;
            sub += av * dec_fp4(nib);
        }
        acc += sub * bsr[kb];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
void fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
              int M, int N, int K, cudaStream_t stream) {
    dim3 grid(N, M); fp4_gemm_kernel<<<grid, 32, 0, stream>>>(C, A_fp8, a_s, B_fp4, b_s, M, N, K);
}

// ---------------- moe_router_score ----------------
// One block per token. Compute n_routed scores (sqrtsoftplus), pick top-k of (score+bias) by iterative
// max, gather the PRE-bias scores, renormalize, scale.
__global__ void router_kernel(float* __restrict__ weights, int* __restrict__ indices,
                              const float* __restrict__ x, const float* __restrict__ gate_w,
                              const float* __restrict__ bias, int n, int dim, int n_routed, int topk,
                              float route_scale) {
    int tok = blockIdx.x; if (tok >= n) return;
    extern __shared__ float sh[];                 // [n_routed] orig scores + [n_routed] sel scores
    float* orig = sh; float* sel = sh + n_routed;
    const float* xr = x + (size_t)tok * dim;
    for (int e = threadIdx.x; e < n_routed; e += blockDim.x) {
        const float* gw = gate_w + (size_t)e * dim;
        float d = 0.f; for (int j = 0; j < dim; ++j) d += xr[j] * gw[j];
        float sp = (d > 20.f) ? d : log1pf(expf(d));          // softplus (stable)
        float s = sqrtf(sp);
        orig[e] = s; sel[e] = s + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float wsum = 0.f;
        for (int t = 0; t < topk; ++t) {
            float best = -1e30f; int bi = -1;
            for (int e = 0; e < n_routed; ++e) if (sel[e] > best) { best = sel[e]; bi = e; }
            sel[bi] = -1e30f;                                  // remove
            indices[(size_t)tok * topk + t] = bi;
            weights[(size_t)tok * topk + t] = orig[bi];
            wsum += orig[bi];
        }
        for (int t = 0; t < topk; ++t)
            weights[(size_t)tok * topk + t] = weights[(size_t)tok * topk + t] / wsum * route_scale;
    }
}
void moe_router_score(float* weights, int* indices, const float* x, const float* gate_w,
                      const float* bias, int n, int dim, int n_routed, int topk,
                      float route_scale, cudaStream_t stream) {
    router_kernel<<<n, 64, 2 * n_routed * sizeof(float), stream>>>(weights, indices, x, gate_w, bias,
                                                                   n, dim, n_routed, topk, route_scale);
}
