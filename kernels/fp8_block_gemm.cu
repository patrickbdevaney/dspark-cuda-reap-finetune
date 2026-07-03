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

// M=1 GEMV: one warp per output n. Lanes read the B[n] row uint-vectorized (4 fp8/load), coalesced (32 lanes
// = 128 contiguous bytes); per-128-block scales applied per element; single warp-reduce. Bandwidth-bound —
// beats the m16-tile TC at M=1 (which is mma-latency bound). Gated cosine vs fp8_block_gemm (tests/gate_fp8_gemv).
__global__ void fp8_gemv_m1_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                   const uint8_t* __restrict__ B, const float* __restrict__ bs, int N, int K){
    int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; if (warp >= N) return; int n = warp;
    int lane = threadIdx.x & 31; int KB = K/128;
    const uint8_t* Brow = B + (size_t)n*K; const float* bsr = bs + (size_t)(n/128)*KB;
    float acc = 0.f;
    for (int kb = 0; kb < KB; ++kb){
        int base = kb*128 + lane*4;                       // 32 lanes * 4 = 128 contiguous bytes
        unsigned av = *(const unsigned*)(A + base);        // 4 fp8 activations
        unsigned bv = *(const unsigned*)(Brow + base);     // 4 fp8 weights
        float sub = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) sub += dec_e4m3((av>>(i*8))&0xff) * dec_e4m3((bv>>(i*8))&0xff);
        acc += sub * as[kb] * bsr[kb];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[n] = acc;
}
// M=K GEMV (small M, e.g. spec-decode block-verify M=5): one warp per output n reads the B[n] weight row ONCE
// (uint-vectorized) and dots it against ALL M activation rows -> the weight bandwidth is amortized M× with no
// wasted rows (vs the m16-tile TC which wastes 16-M rows). Bandwidth-bound. Gated cosine vs fp8_block_gemm.
#define GEMV_MK_MAXM 16
__global__ void fp8_gemv_mk_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                   const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; if(warp>=N) return; int n=warp; int lane=threadIdx.x&31; int KB=K/128;
    const uint8_t* Brow=B+(size_t)n*K; const float* bsr=bs+(size_t)(n/128)*KB;
    float acc[GEMV_MK_MAXM];
    #pragma unroll
    for(int m=0;m<GEMV_MK_MAXM;++m) acc[m]=0.f;
    for(int kb=0; kb<KB; ++kb){
        int base=kb*128+lane*4; unsigned bv=*(const unsigned*)(Brow+base); float wsc=bsr[kb];
        float b0=dec_e4m3(bv&0xff)*wsc, b1=dec_e4m3((bv>>8)&0xff)*wsc, b2=dec_e4m3((bv>>16)&0xff)*wsc, b3=dec_e4m3((bv>>24)&0xff)*wsc;
        for(int m=0; m<M; ++m){
            unsigned av=*(const unsigned*)(A+(size_t)m*K+base); float asc=as[(size_t)m*KB+kb];
            acc[m]+=(dec_e4m3(av&0xff)*b0 + dec_e4m3((av>>8)&0xff)*b1 + dec_e4m3((av>>16)&0xff)*b2 + dec_e4m3((av>>24)&0xff)*b3)*asc;
        }
    }
    for(int m=0; m<M; ++m){ float a=acc[m];
        #pragma unroll
        for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffff,a,o);
        if(lane==0) C[(size_t)m*N+n]=a; }
}
// Global toggle: route dense fp8 GEMMs through the native FP8 tensor core (tc_fp8_gemm, ~18x). Default OFF so
// gates use this warp-per-output oracle (bit-exact). forward.cu sets it true for decode. All our fp8 GEMM
// shapes satisfy N%8==0 && K%128==0 (checked); fall back to the oracle otherwise.
bool g_tc_fp8 = false;
void tc_fp8_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t);
void fp8_block_gemm(float* C, const uint8_t* A_fp8, const float* a_s,
                    const uint8_t* B_fp8, const float* b_s,
                    int M, int N, int K, cudaStream_t stream) {
    // M=1 decode: vectorized GEMV (bandwidth-bound, beats the mma-latency-bound TC at M=1). K%128==0 always here.
    if (g_tc_fp8 && M == 1 && (K % 128 == 0) && getenv("NO_GEMV")==nullptr) {
        int threads=256; fp8_gemv_m1_kernel<<<(N*32+threads-1)/threads, threads, 0, stream>>>(C, A_fp8, a_s, B_fp8, b_s, N, K); return; }
    // small-M M=K GEMV (env GEMV_MK=1): A/B'd SLOWER than TC at M>=2 (334->362 ms verify) — TC reads the weight
    // ONCE *and* does the M×N compute via mma, while this GEMV does M scalar dots/weight-read. Kept as a gated
    // reference/negative result; the M=1 GEMV above still wins (trivial compute at M=1). Default OFF for M>=2.
    if (g_tc_fp8 && M >= 2 && M <= GEMV_MK_MAXM && (K % 128 == 0) && getenv("GEMV_MK")!=nullptr) {
        int threads=256; fp8_gemv_mk_kernel<<<(N*32+threads-1)/threads, threads, 0, stream>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); return; }
    if (g_tc_fp8 && (N % 8 == 0) && (K % 128 == 0)) { tc_fp8_gemm(C, A_fp8, a_s, B_fp8, b_s, M, N, K, stream); return; }
    dim3 grid(N, M);
    fp8_block_gemm_kernel<<<grid, 32, 0, stream>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
