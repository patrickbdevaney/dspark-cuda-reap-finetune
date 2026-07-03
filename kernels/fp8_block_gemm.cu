// fp8_block_gemm.cu — FP8 e4m3 block-scaled GEMM, correctness-first (Gate K oracle: ref/gen_units.py).
// One warp per output element C[m,n]; the warp reduces over K, decoding e4m3 via the HW intrinsic and
// applying per-128-block activation and weight scales. Matches kernel.py fp8_gemm math (fp32 accumulate).
// Optimization (mma.sync tiling) comes AFTER this passes its gate — never before (CONSTITUTION Art. I).
#include "fp8_block_gemm.h"
#include <cuda_fp8.h>

#define BLK 128   // scale block along K (weight_block_size)

__device__ __forceinline__ float dec_e4m3(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}

// grid.x = N, grid.y = M ; blockDim.x = 32 (one warp -> one output C[by, bx]).
__global__ void fp8_block_gemm_kernel(float* __restrict__ C,
                                      const uint8_t* __restrict__ A, const float* __restrict__ a_s,
                                      const uint8_t* __restrict__ B, const float* __restrict__ b_s,
                                      int M, int N, int K) {
    int n = blockIdx.x, m = blockIdx.y;
    if (m >= M || n >= N) return;
    int lane = threadIdx.x & 31;
    int KB = K / BLK;                                  // #K-blocks
    const uint8_t* Arow = A + (size_t)m * K;
    const uint8_t* Brow = B + (size_t)n * K;
    const float*  asr = a_s + (size_t)m * KB;
    const float*  bsr = b_s + (size_t)(n / BLK) * KB;

    float acc = 0.f;
    // walk K-blocks; within a block the two scales are constant, so accumulate the raw dot then scale once.
    for (int kb = 0; kb < KB; ++kb) {
        float sub = 0.f;
        int base = kb * BLK;
        for (int j = lane; j < BLK; j += 32)
            sub += dec_e4m3(Arow[base + j]) * dec_e4m3(Brow[base + j]);
        acc += sub * asr[kb] * bsr[kb];
    }
    // warp reduce
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}

// Global toggle: route dense fp8 GEMMs through the native FP8 tensor core (tc_fp8_gemm, ~18x). Default OFF so
// gates use this warp-per-output oracle (bit-exact). forward.cu sets it true for decode. All our fp8 GEMM
// shapes satisfy N%8==0 && K%128==0 (checked); fall back to the oracle otherwise.
bool g_tc_fp8 = false;
void tc_fp8_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t);
void fp8_block_gemm(float* C, const uint8_t* A_fp8, const float* a_s,
                    const uint8_t* B_fp8, const float* b_s,
                    int M, int N, int K, cudaStream_t stream) {
    // (A/B result: the m16-tile TC BEATS the warp-per-output oracle even at M=1 — its coalesced weight layout
    // wins. Kept TC for all M. The M=1 attn GEMM cost is closer to bandwidth than expected.)
    if (g_tc_fp8 && (N % 8 == 0) && (K % 128 == 0)) { tc_fp8_gemm(C, A_fp8, a_s, B_fp8, b_s, M, N, K, stream); return; }
    dim3 grid(N, M);
    fp8_block_gemm_kernel<<<grid, 32, 0, stream>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
