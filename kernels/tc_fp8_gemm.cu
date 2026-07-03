// tc_fp8_gemm.cu — native FP8 tensor-core GEMM (W8A8) via mma.sync.m16n8k32.e4m3 (2x fp16 on Thor sm_110a).
// C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T, per-128 act scale a_s[M,K/128], per-128x128 wt scale b_s[N/128,K/128].
// Drop-in for fp8_block_gemm; fp8 acts/weights feed the tensor core directly (no fp16 upconvert).
// *** UNGATED until it passes cosine vs fp8_block_gemm (tests: gate_units [fp8_block_gemm+TC]). ***
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

// mma D[16,8] += A[16,32] @ B[8,32]^T ; A/B = e4m3 (4 fp8 per reg), C/D = f32 (4 per lane).
__device__ __forceinline__ void mma_m16n8k32_e4m3(float* c, const unsigned* a, const unsigned* b){
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}

// grid = (ceil(N/8), ceil(M/16)); one warp -> one [16-row, 8-col] tile. Accumulate raw per 128-K-block, then
// scale (a_s per-row, b_s per-128-N-block) and add — matches fp8_block_gemm's per-block scale application.
__global__ void tc_fp8_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                              const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    int lane=threadIdx.x&31, gid=lane>>2, tid4=lane&3;
    int nb=blockIdx.x, mt=blockIdx.y; int n0=nb*8, m0=mt*16; if(n0>=N) return;
    int r0=m0+gid, r1=m0+gid+8, nn=n0+gid; int KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int kblk=0; kblk<KB; ++kblk){
        float cb[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int kt=0; kt<4; ++kt){ int k0=kblk*128+kt*32;
            unsigned a[4], b[2];
            a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)      : 0u;
            a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)      : 0u;
            a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16)   : 0u;
            a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16)   : 0u;
            b[0]=(nn<N)? *(const unsigned*)(B+(size_t)nn*K+k0+tid4*4)      : 0u;
            b[1]=(nn<N)? *(const unsigned*)(B+(size_t)nn*K+k0+tid4*4+16)   : 0u;
            mma_m16n8k32_e4m3(cb, a, b);
        }
        float bsc = bs[(size_t)(n0/128)*KB + kblk];
        float as0 = (r0<M)? as[(size_t)r0*KB + kblk] : 0.f;
        float as1 = (r1<M)? as[(size_t)r1*KB + kblk] : 0.f;
        acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
    }
    int cn=tid4*2;
    if(r0<M && n0+cn  <N) C[(size_t)r0*N + n0+cn  ]=acc[0];
    if(r0<M && n0+cn+1<N) C[(size_t)r0*N + n0+cn+1]=acc[1];
    if(r1<M && n0+cn  <N) C[(size_t)r1*N + n0+cn  ]=acc[2];
    if(r1<M && n0+cn+1<N) C[(size_t)r1*N + n0+cn+1]=acc[3];
}

void tc_fp8_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp8, const float* b_s,
                 int M, int N, int K, cudaStream_t s){
    dim3 grid((N+7)/8, (M+15)/16);
    tc_fp8_kernel<<<grid, 32, 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
