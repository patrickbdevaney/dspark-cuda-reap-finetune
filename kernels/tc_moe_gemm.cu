// tc_moe_gemm.cu — Marlin-class tensor-core GEMM for fp8-act × fp4-weight (W4A8), our MoE experts.
// Adapted from gemma-cuda-hybrid/kernels/tc_verify_gemm.cu (raw mma.sync.m16n8k16, 1 warp = 8 N-cols,
// weight-repack + __ldcs coalesced loads + in-register FP4→fp16 dequant).
// OUR adaptation: (a) fp8-e4m3 act → fp16 with per-128 act scale folded in; (b) fp4 weight path unchanged;
// (c) gemma's per-k-tile fp8 weight-scale → our per-32 e8m0 (exp2(byte-127)), pre-expanded to fp16 per-k-tile.
// *** UNGATED — must pass bit-exact/cosine vs fp4_gemm (tests/gate_tc_moe.cu) before it is trusted / used. ***
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include "moe.h"   // fp4_gemm signature parity

__device__ __forceinline__ __half2 tcm_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }
__device__ __forceinline__ float tcm_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
__device__ __forceinline__ void mma_m16n8k16(float* c, const unsigned* a, const unsigned* b){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}

// A fp8[M,K] + a_s[M,K/128] (f32) -> x16[M,K] fp16, with act scale folded (matches fp4_gemm's dec_e4m3(A)*as).
__global__ void k_a_to_fp16(__half* x16, const uint8_t* A, const float* a_s, int M, int K){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)M*K) return; int m=i/K, k=i%K;
    x16[i]=__float2half(tcm_e4m3(A[i]) * a_s[(long)m*(K/128)+k/128]);
}
// weight repack (fp4 [N,K/2] -> [N/8][K/128][32 lane][16B]), same as gemma k_tc_repack_w.
__global__ void k_repack_w(uint8_t* wpr, const uint8_t* wp, int N, int K){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int kg8=K/128; long tot=(long)(N/8)*kg8*32*8; if(idx>=tot)return;
    int kl=idx&7; long r=idx>>3; int lane=r&31; long r2=r>>5; int g=r2%kg8, n_block=r2/kg8, gid=lane>>2, t4=lane&3;
    int k_tile=g*8+kl; long src=(long)(n_block*8+gid)*(K/2) + (long)k_tile*8;
    long dst=((long)n_block*kg8 + g)*512 + (long)lane*16 + 2*kl;
    wpr[dst]=wp[src+t4]; wpr[dst+1]=wp[src+t4+4];
}
// weight scale b_s[N,K/32] (f32, already-dequantized pow2 — parity with fp4_gemm) -> wsr[N/8][K/16][8] fp16
// (per-k-tile, per-n): one 32-block scale covers 2 k-tiles (32/16).
__global__ void k_repack_s(__half* wsr, const float* bs, int N, int K){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int kt=K/16; long tot=(long)(N/8)*kt*8; if(idx>=tot)return;
    int g=idx&7; long r=idx>>3; int k_tile=r%kt, n_block=r/kt; int n=n_block*8+g;
    wsr[((long)n_block*kt + k_tile)*8 + g] = __float2half(bs[(long)n*(K/32) + k_tile/2]);
}
#ifndef TCM_WARPS
#define TCM_WARPS 1
#endif
__global__ void tc_w4a8_kernel(float* out, const uint8_t* wpr, const __half* wsr, const __half* x16, int M, int N, int K){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int warp=threadIdx.x>>5; int n_block=blockIdx.x*TCM_WARPS+warp; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, kt=K/16;
    const uint8_t* wb = wpr + (long)n_block*kg8*512;
    const __half*  sb = wsr + (long)n_block*kt*8;
    const __half* xg0 = x16 + (size_t)gid*K, *xg8 = x16 + (size_t)(gid+8)*K;
    bool m0=gid<M, m8=(gid+8)<M;
    for(int g=0; g<kg8; ++g){
        uint4 w16 = __ldcs((const uint4*)(wb + (long)g*512 + lane*16));
        const uint8_t* wby=(const uint8_t*)&w16;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(sb[(long)k_tile*8 + gid]);
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<M   && n0+cn  <N) out[(size_t)gid*N   + n0+cn  ]=c[0];
    if(gid<M   && n0+cn+1<N) out[(size_t)gid*N   + n0+cn+1]=c[1];
    if(gid+8<M && n0+cn  <N) out[(size_t)(gid+8)*N + n0+cn ]=c[2];
    if(gid+8<M && n0+cn+1<N) out[(size_t)(gid+8)*N + n0+cn+1]=c[3];
}

#define CT(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#include <unordered_map>
// weight-repack cache: keyed by (B_fp4 ptr, b_s ptr). Repacked ONCE (lazy warm-up), reused across all decode
// forwards — this is the decode win. NOTE: repacked ≈ same bytes as the original fp4 layout, so caching every
// routed expert doubles expert-weight memory (~82 GB -> OOM on the full model). For the full model, repack at
// LOAD storing repacked in place of the original (loader change), or scope per-layer. Fine for gates/tests here.
struct TcW { uint8_t* wpr; __half* wsr; };
static std::unordered_map<const void*, TcW> g_tc_cache;
static TcW tc_get_weight(const uint8_t* B_fp4, const float* b_s, int N, int K, cudaStream_t s){
    auto it=g_tc_cache.find(B_fp4); if(it!=g_tc_cache.end()) return it->second;
    TcW w; CT(cudaMalloc(&w.wpr,(size_t)(N/8)*(K/128)*512)); CT(cudaMalloc(&w.wsr,(size_t)(N/8)*(K/16)*8*2));
    long tw=(long)(N/8)*(K/128)*32*8, ts=(long)(N/8)*(K/16)*8;
    k_repack_w<<<(tw+255)/256,256,0,s>>>(w.wpr, B_fp4, N, K);
    k_repack_s<<<(ts+255)/256,256,0,s>>>(w.wsr, b_s, N, K);
    g_tc_cache.emplace(B_fp4, w); return w;
}
// Same signature as fp4_gemm (drop-in). Weight repack cached by ptr; only the fp8->fp16 act convert is per-call.
void tc_fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
                 int M, int N, int K, cudaStream_t s){
    __half* x16; CT(cudaMalloc(&x16,(size_t)M*K*2));
    k_a_to_fp16<<<((long)M*K+255)/256,256,0,s>>>(x16, A_fp8, a_s, M, K);
    TcW w = tc_get_weight(B_fp4, b_s, N, K, s);
    dim3 grid((N/8 + TCM_WARPS-1)/TCM_WARPS); tc_w4a8_kernel<<<grid, TCM_WARPS*32, 0, s>>>(C, w.wpr, w.wsr, x16, M, N, K);
    CT(cudaStreamSynchronize(s)); cudaFree(x16);
}

// Free the repack cache — call per-layer in forward.cu so the full model doesn't accumulate ~82GB of repacked
// expert weights (the cache is a decode-across-forwards optimization; per single forward it's per-layer scoped).
void tc_moe_clear_cache(){ for(auto& kv : g_tc_cache){ cudaFree(kv.second.wpr); cudaFree(kv.second.wsr); } g_tc_cache.clear(); }
