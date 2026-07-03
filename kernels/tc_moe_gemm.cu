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

// ===================== REPACK-AT-LOAD (zero extra memory) =====================
// The repacked layout is the SAME byte-size as the fp4 weight (N*K/2), so we repack IN PLACE at load and the
// kernel reads the ORIGINAL scale b_s[N,K/32] directly (no wsr). Result: no 82GB doubling, no per-layer repack.
// Funnel-combine two 16B-aligned uint4 loads straddling an unaligned 16B read. off = base&15 (uniform across the
// kernel: every g*512/lane*16 is a multiple of 16). k0=off>>2, sh=(off&3)*8. Register-based, uniform branch on k0.
__device__ __forceinline__ uint4 tcm_funnel16(uint4 A, uint4 B, int k0, unsigned sh){
    uint4 r;
    if(k0==0){r.x=__funnelshift_r(A.x,A.y,sh);r.y=__funnelshift_r(A.y,A.z,sh);r.z=__funnelshift_r(A.z,A.w,sh);r.w=__funnelshift_r(A.w,B.x,sh);}
    else if(k0==1){r.x=__funnelshift_r(A.y,A.z,sh);r.y=__funnelshift_r(A.z,A.w,sh);r.z=__funnelshift_r(A.w,B.x,sh);r.w=__funnelshift_r(B.x,B.y,sh);}
    else if(k0==2){r.x=__funnelshift_r(A.z,A.w,sh);r.y=__funnelshift_r(A.w,B.x,sh);r.z=__funnelshift_r(B.x,B.y,sh);r.w=__funnelshift_r(B.y,B.z,sh);}
    else          {r.x=__funnelshift_r(A.w,B.x,sh);r.y=__funnelshift_r(B.x,B.y,sh);r.z=__funnelshift_r(B.y,B.z,sh);r.w=__funnelshift_r(B.z,B.w,sh);}
    return r;   // off==0 (k0=0,sh=0) -> r=A (identity)
}
// Pre-packed kernel: weight already in wpr layout; scale read from original b_s (per-32, one 32-block=2 k-tiles).
__global__ void tc_w4a8_pp_kernel(float* out, const uint8_t* wpr, const float* b_s, const __half* x16, int M, int N, int K, int off){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int warp=threadIdx.x>>5; int n_block=blockIdx.x*TCM_WARPS+warp; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, Ks32=K/32; int k0f=off>>2; unsigned shf=(off&3)*8;
    const uint8_t* wb = wpr + (long)n_block*kg8*512;
    const __half* xg0 = x16 + (size_t)gid*K, *xg8 = x16 + (size_t)(gid+8)*K;
    const float* bsr = b_s + (long)(n0+gid)*Ks32;   // original per-32 scale for this lane's weight row n0+gid
    bool m0=gid<M, m8=(gid+8)<M;
    for(int g=0; g<kg8; ++g){
        // ALIGNED coalesced load via funnel-shift: the in-place repacked weight is at an arbitrary WeightStore byte
        // offset. Load the two 16B-aligned uint4 straddling this lane's 16B, funnel-combine by the constant `off`.
        // Recovers coalescing (vs the byte-load fallback) while staying alignment-correct.
        const uint8_t* wa = wb + (long)g*512 + lane*16 - off;
        uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
        uint4 W=tcm_funnel16(A,B,k0f,shf); const uint8_t* wby=(const uint8_t*)&W;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(__float2half(bsr[k_tile/2]));  // 32-block = k_tile/2 (16 per k-tile)
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
// Repack one fp4 weight IN PLACE (via a reused temp of size N*K/2). Call once per routed expert weight at load.
void tc_repack_weight_inplace(uint8_t* w_fp4, int N, int K, uint8_t* temp, cudaStream_t s){
    long tw=(long)(N/8)*(K/128)*32*8; size_t bytes=(size_t)(N/8)*(K/128)*512;   // == N*K/2
    k_repack_w<<<(tw+255)/256,256,0,s>>>(temp, w_fp4, N, K);
    CT(cudaMemcpyAsync(w_fp4, temp, bytes, cudaMemcpyDeviceToDevice, s));
}
// Drop-in for fp4_gemm, but B_fp4 is ALREADY repacked (via tc_repack_weight_inplace) and b_s is the ORIGINAL scale.
void tc_fp4_gemm_pp(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* Bpacked, const float* b_s,
                    int M, int N, int K, cudaStream_t s){
    __half* x16; CT(cudaMalloc(&x16,(size_t)M*K*2));
    k_a_to_fp16<<<((long)M*K+255)/256,256,0,s>>>(x16, A_fp8, a_s, M, K);
    int off = (int)((uintptr_t)Bpacked & 15);   // constant per weight; funnel-shift makes loads aligned+coalesced
    dim3 grid((N/8 + TCM_WARPS-1)/TCM_WARPS); tc_w4a8_pp_kernel<<<grid, TCM_WARPS*32, 0, s>>>(C, Bpacked, b_s, x16, M, N, K, off);
    CT(cudaStreamSynchronize(s)); cudaFree(x16);
}
// Auto variant (drop-in for tc_fp4_gemm): repack B IN PLACE the first time it's seen (zero extra mem), then run
// the pre-packed GEMM. First forward warms up (repacks); every later forward is pure pp GEMM. Reused temp buffer.
#include <unordered_set>
static std::unordered_set<const void*> g_pp_done;
static uint8_t* g_pp_tmp=nullptr; static size_t g_pp_tmpsz=0;
void tc_fp4_gemm_pp_auto(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
                         int M, int N, int K, cudaStream_t s){
    if(g_pp_done.find(B_fp4)==g_pp_done.end()){
        size_t bytes=(size_t)(N/8)*(K/128)*512;                 // == N*K/2
        if(bytes>g_pp_tmpsz){ if(g_pp_tmp) cudaFree(g_pp_tmp); CT(cudaMalloc(&g_pp_tmp,bytes)); g_pp_tmpsz=bytes; }
        tc_repack_weight_inplace((uint8_t*)B_fp4, N, K, g_pp_tmp, s);   // overwrite the fp4 weight with its repacked layout
        g_pp_done.insert(B_fp4);
    }
    tc_fp4_gemm_pp(C, A_fp8, a_s, B_fp4, b_s, M, N, K, s);
}

// Idempotently repack one expert weight IN PLACE (shared g_pp_done set with pp_auto, so a weight already warmed
// by either path is never re-repacked -> bytes stay identical -> grouped path is cosine-1.0 with the pp path).
void tc_ensure_repacked(uint8_t* B_fp4, int N, int K, cudaStream_t s){
    if(g_pp_done.find(B_fp4)!=g_pp_done.end()) return;
    size_t bytes=(size_t)(N/8)*(K/128)*512;
    if(bytes>g_pp_tmpsz){ if(g_pp_tmp) cudaFree(g_pp_tmp); CT(cudaMalloc(&g_pp_tmp,bytes)); g_pp_tmpsz=bytes; }
    tc_repack_weight_inplace(B_fp4, N, K, g_pp_tmp, s);
    g_pp_done.insert(B_fp4);
}
// fp8[M,K]+a_s -> fp16[M,K] (act-scale folded), exposed so the grouped path converts ALL gathered rows once.
void tc_a_to_fp16(__half* x16, const uint8_t* A_fp8, const float* a_s, int M, int K, cudaStream_t s){
    k_a_to_fp16<<<((long)M*K+255)/256,256,0,s>>>(x16, A_fp8, a_s, M, K);
}

// ===================== GROUPED (zero-sync) W4A8 GEMM — STRUCTURAL_PLAN Step 1b =====================
// ONE launch over ALL experts. A "tile" = up to 16 rows of ONE expert (its own repacked weight+scale). The
// tile->expert map is built ON DEVICE from off[] (k_build_tiles) so the host never needs off[] -> removes the
// last per-layer host sync (the off[] D2H copy) that blocked CUDA-graph capture. Per-expert byte alignment of
// the in-place repacked weight is handled by funnel-shift, computed per tile (uniform across the block).
// Weights & scales are the SAME bytes the pp path uses -> identical mma -> cosine 1.0 vs the per-expert loop.

// For each expert e, emit ceil(me/16) tiles at rows off[e], off[e]+16, ...  (single thread; nr<=~160).
__global__ void k_build_tiles(int* tile_e, int* tile_row0, int* ntiles, const int* off, int nr){
    if(threadIdx.x||blockIdx.x) return;
    int nt=0;
    for(int e=0;e<nr;++e){ int r0=off[e], r1=off[e+1];
        for(int r=r0;r<r1;r+=16){ tile_e[nt]=e; tile_row0[nt]=r; ++nt; } }
    *ntiles=nt;
}

// gridDim.x = N/8 (n-blocks); gridDim.y = maxtiles (host UPPER BOUND = total gathered rows; extra tiles early-exit).
__global__ void k_grouped_w4a8_kernel(float* out, const uint8_t* const* wptr, const float* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const __half* x16all, int N, int K){
    int tile = blockIdx.y; if(tile >= *ntiles) return;
    int e = tile_e[tile]; int row0 = tile_row0[tile];
    int me = off[e+1]-row0; if(me>16) me=16;                    // rows this tile owns (<=16)
    const uint8_t* wprE = wptr[e]; const float* b_s = sptr[e];
    int off_b=(int)((uintptr_t)wprE & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;   // per-expert alignment
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int n_block=blockIdx.x; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, Ks32=K/32;
    const uint8_t* wb = wprE + (long)n_block*kg8*512;
    const __half* xg0 = x16all + (size_t)(row0+gid)*K, *xg8 = x16all + (size_t)(row0+gid+8)*K;
    const float* bsr = b_s + (long)(n0+gid)*Ks32;
    bool m0=gid<me, m8=(gid+8)<me;
    for(int g=0; g<kg8; ++g){
        const uint8_t* wa = wb + (long)g*512 + lane*16 - off_b;             // funnel-aligned coalesced load
        uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
        uint4 W=tcm_funnel16(A,B,k0f,shf); const uint8_t* wby=(const uint8_t*)&W;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(__float2half(bsr[k_tile/2]));
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<me   && n0+cn  <N) out[(size_t)(row0+gid)*N   + n0+cn  ]=c[0];
    if(gid<me   && n0+cn+1<N) out[(size_t)(row0+gid)*N   + n0+cn+1]=c[1];
    if(gid+8<me && n0+cn  <N) out[(size_t)(row0+gid+8)*N + n0+cn ]=c[2];
    if(gid+8<me && n0+cn+1<N) out[(size_t)(row0+gid+8)*N + n0+cn+1]=c[3];
}
// Build tile descriptors on device from off[] (no host sync).
void tc_build_tiles(int* tile_e, int* tile_row0, int* ntiles_d, const int* off_d, int nr, cudaStream_t s){
    k_build_tiles<<<1,1,0,s>>>(tile_e, tile_row0, ntiles_d, off_d, nr);
}

// ===================== M=1 fp4 GEMV (small-M decode, ORIGINAL fp4 layout, no repack) =====================
// One warp per (output n, tile); loops the tile's <=16 rows. Reads the ORIGINAL packed fp4 weight row
// (uint4-vectorized, coalesced) + fp8 act (uint4) + e8m0 per-32 scale in-register. Bandwidth-bound — beats the
// m16-tile mma at small M (which is mma-latency bound). Gated cosine vs fp4_gemm (tests/gate_fp4_gemv).
__constant__ float GEMV_E2M1[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
__device__ __forceinline__ float gv_fp4(uint8_t nib){ float m=GEMV_E2M1[nib&7]; return (nib&8)?-m:m; }
__device__ __forceinline__ float gv_e4m3(uint8_t b){ __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
__global__ void k_grouped_fp4_gemv_e8m0(float* out, const uint8_t* const* wptr, const uint8_t* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const uint8_t* Xq, const float* Xs, int N, int K){
    int tile=blockIdx.y; if(tile>=*ntiles) return;
    int e=tile_e[tile], row0=tile_row0[tile]; int me=off[e+1]-row0; if(me>16)me=16;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int n=warp; if(n>=N) return; int lane=threadIdx.x&31;
    const uint8_t* Wn = wptr[e] + (size_t)n*(K/2);       // packed fp4 row (2 nibbles/byte) — arbitrary alignment
    const uint8_t* Sn = sptr[e] + (size_t)n*(K/32);      // e8m0 per-32 scale row
    int nb32 = K/32;                                     // 32-weight blocks = 16 bytes each
    int off_b=(int)((uintptr_t)Wn & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;   // funnel-align the weight loads
    for(int r=0;r<me;++r){
        const uint8_t* Aq = Xq + (size_t)(row0+r)*K;
        const float*  As = Xs + (size_t)(row0+r)*(K/128);
        float acc=0.f;
        for(int kb=lane; kb<nb32; kb+=32){              // lane -> whole 32-weight block (16B), coalesced across warp
            float ws=exp2f((float)Sn[kb]-127.f); float asc=As[kb/4];   // act scale per-128 = per 4 of the 32-blocks
            const uint8_t* wa=Wn+(size_t)kb*16-off_b;
            uint4 WA=__ldcs((const uint4*)wa), WB=__ldcs((const uint4*)(wa+16));
            uint4 w16=tcm_funnel16(WA,WB,k0f,shf);
            uint4 a0 =*(const uint4*)(Aq+(size_t)kb*32);
            uint4 a1 =*(const uint4*)(Aq+(size_t)kb*32+16);
            const uint8_t* wb=(const uint8_t*)&w16; const uint8_t* ab0=(const uint8_t*)&a0; const uint8_t* ab1=(const uint8_t*)&a1;
            float sub=0.f;
            #pragma unroll
            for(int j=0;j<16;++j){ uint8_t byte=wb[j]; uint8_t a2j=(j<8)?ab0[2*j]:ab1[2*(j-8)]; uint8_t a2j1=(j<8)?ab0[2*j+1]:ab1[2*(j-8)+1];
                sub += gv_e4m3(a2j)  * gv_fp4(byte&0xF);
                sub += gv_e4m3(a2j1) * gv_fp4((byte>>4)&0xF);
            }
            acc += sub * asc * ws;
        }
        #pragma unroll
        for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
        if(lane==0) out[(size_t)(row0+r)*N + n]=acc;
    }
}
void tc_fp4_grouped_gemv_e8m0(float* out, const uint8_t* Xq, const float* Xs, const uint8_t* const* wptr_d,
        const uint8_t* const* sptr_d, const int* off_d, const int* tile_e, const int* tile_row0, const int* ntiles_d,
        int maxtiles, int N, int K, cudaStream_t s){
    int threads=128; dim3 grid((N*32+threads-1)/threads, maxtiles);
    k_grouped_fp4_gemv_e8m0<<<grid, threads, 0, s>>>(out, wptr_d, sptr_d, tile_e, tile_row0, ntiles_d, off_d, Xq, Xs, N, K);
}

// NATIVE-e8m0 grouped GEMM: scale ptrs point to the ORIGINAL e8m0 scale BYTES (F8_E8M0) in the WeightStore —
// exp2f(byte-127) is computed in-register (bit-identical to the pre-dequanted f32 pow2). This removes the
// per-layer-per-token scale dequant (160x3 mallocs+kernels/layer) AND keeps the scale pointers persistent
// (no dequant buffer) -> the dominant decode cost. Only the scale read differs from k_grouped_w4a8_kernel.
__global__ void k_grouped_w4a8_e8m0_kernel(float* out, const uint8_t* const* wptr, const uint8_t* const* sptr,
        const int* __restrict__ tile_e, const int* __restrict__ tile_row0, const int* __restrict__ ntiles,
        const int* __restrict__ off, const __half* x16all, int N, int K){
    int tile = blockIdx.y; if(tile >= *ntiles) return;
    int e = tile_e[tile]; int row0 = tile_row0[tile];
    int me = off[e+1]-row0; if(me>16) me=16;
    const uint8_t* wprE = wptr[e]; const uint8_t* b_s = sptr[e];       // b_s = e8m0 bytes [N, K/32]
    int off_b=(int)((uintptr_t)wprE & 15); int k0f=off_b>>2; unsigned shf=(off_b&3)*8;
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int n_block=blockIdx.x; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0,0,0,0}; int kg8=K/128, Ks32=K/32;
    const uint8_t* wb = wprE + (long)n_block*kg8*512;
    const __half* xg0 = x16all + (size_t)(row0+gid)*K, *xg8 = x16all + (size_t)(row0+gid+8)*K;
    const uint8_t* bsr = b_s + (long)(n0+gid)*Ks32;
    bool m0=gid<me, m8=(gid+8)<me;
    for(int g=0; g<kg8; ++g){
        const uint8_t* wa = wb + (long)g*512 + lane*16 - off_b;
        uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
        uint4 W=tcm_funnel16(A,B,k0f,shf); const uint8_t* wby=(const uint8_t*)&W;
        #pragma unroll
        for(int kl=0; kl<8; ++kl){ int k_tile=g*8+kl, k0=k_tile*16;
            unsigned a[4];
            a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
            a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
            a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
            a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
            __half2 sc2 = __half2half2(__float2half(exp2f((float)bsr[k_tile/2]-127.f)));  // e8m0 -> pow2 in-register
            __half2 b0 = __hmul2(tcm_fp4x2(wby[2*kl]),   sc2);
            __half2 b1 = __hmul2(tcm_fp4x2(wby[2*kl+1]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
    }
    int cn=2*t4;
    if(gid<me   && n0+cn  <N) out[(size_t)(row0+gid)*N   + n0+cn  ]=c[0];
    if(gid<me   && n0+cn+1<N) out[(size_t)(row0+gid)*N   + n0+cn+1]=c[1];
    if(gid+8<me && n0+cn  <N) out[(size_t)(row0+gid+8)*N + n0+cn ]=c[2];
    if(gid+8<me && n0+cn+1<N) out[(size_t)(row0+gid+8)*N + n0+cn+1]=c[2+1];
}
void tc_fp4_grouped_gemm_e8m0(float* out, const __half* x16all, const uint8_t* const* wptr_d, const uint8_t* const* sptr_d,
        const int* off_d, const int* tile_e, const int* tile_row0, const int* ntiles_d,
        int maxtiles, int N, int K, cudaStream_t s){
    dim3 grid(N/8, maxtiles);
    k_grouped_w4a8_e8m0_kernel<<<grid, 32, 0, s>>>(out, wptr_d, sptr_d, tile_e, tile_row0, ntiles_d, off_d, x16all, N, K);
}
// Grouped W4A8: out[total,N] = per-tile (expert wptr[e]) mma over x16all rows. maxtiles = host upper bound on tiles.
void tc_fp4_grouped_gemm(float* out, const __half* x16all, const uint8_t* const* wptr_d, const float* const* sptr_d,
        const int* off_d, const int* tile_e, const int* tile_row0, const int* ntiles_d,
        int maxtiles, int N, int K, cudaStream_t s){
    dim3 grid(N/8, maxtiles);
    k_grouped_w4a8_kernel<<<grid, 32, 0, s>>>(out, wptr_d, sptr_d, tile_e, tile_row0, ntiles_d, off_d, x16all, N, K);
}
