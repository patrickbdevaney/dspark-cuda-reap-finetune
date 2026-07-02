// forward.cu — Gemma-4-26B-A4B NVFP4 single-sequence forward (W4A4), correctness-first.
// Reuses verified kernels: nvfp4_gemm, nvfp4_quantize_activations, rmsnorm, rope, sdpa, gelu, softcap.
// Produces next-token logits for a token-id sequence (prefill); gate compares to the reference server.
#include "safetensors.h"
#include "fp4_gemm.h"
#include "nvfp4_quant.h"
#include "elementwise.h"
#include "attention.h"
#include "draft.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <mutex>
#include <random>
#include "tokenizer.h"
#include "third_party/httplib.h"
#include "webui.h"
// TEAL/CATS activation-sparsity probe (SPARSITY=1): per-row retained-L2-norm at each magnitude-threshold sparsity.
static void measure_sparsity(const __half* dev, int rows, int dim, const char* name){
    std::vector<__half> h((size_t)rows*dim);
    cudaMemcpy(h.data(), dev, (size_t)rows*dim*sizeof(__half), cudaMemcpyDeviceToHost);
    const double sp[5]={0.3,0.4,0.5,0.6,0.7}; double ret[5]={0,0,0,0,0}, zf=0; int used=0;
    std::vector<float> mag(dim);
    for(int r=0;r<rows;++r){
        double total=0; int nz=0;
        for(int d=0;d<dim;++d){ float v=__half2float(h[(size_t)r*dim+d]); mag[d]=fabsf(v); total+=(double)v*v; if(mag[d]<1e-4f)nz++; }
        if(total<=0) continue;   // skip empty rows (unrouted hbuf slots)
        used++; zf+=(double)nz/dim;
        std::sort(mag.begin(), mag.end());
        for(int s=0;s<5;++s){ int cut=(int)(sp[s]*dim); double disc=0; for(int d=0;d<cut;++d) disc+=(double)mag[d]*mag[d];
            ret[s]+=sqrt(1.0-disc/total); }
    }
    if(!used) used=1;
    printf("[SPARSITY %s] rows=%d dim=%d  near-zero(<1e-4)=%.1f%%  retained-L2 @ sparsity  30%%=%.4f 40%%=%.4f 50%%=%.4f 60%%=%.4f 70%%=%.4f\n",
        name, used, dim, 100.0*zf/used, ret[0]/used, ret[1]/used, ret[2]/used, ret[3]/used, ret[4]/used);
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "megakernel.h"
#include <unordered_map>
#include <algorithm>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---- Gemma-4-26B-A4B config (verified) ----
static const int H=2816, NLAYER=30, NHEAD=16, HD_S=256, NKV_S=8, HD_F=512, NKV_F=2;
static const int VOCAB=262144, NEXP=128, TOPK=8, MOE_INT=704, MLP_INT=2112, SWIN=1024;
static const float EPS=1e-6f, SOFTCAP=30.0f;
static const float EMB_SCALE=53.06599664f;             // sqrt(2816)
static inline bool is_full(int L){ return L==5||L==11||L==17||L==23||L==29; }
__device__ int g_base=0;  // decode position read by base-dependent kernels (enables CUDA-graph replay)
static uint8_t *g_ewp=nullptr,*g_ews=nullptr; static float g_egs=0.f;  // NVFP4-quantized embed (lm_head): codes + e4m3 scales + global scale

// ---- helper kernels ----
__device__ __forceinline__ float bf2f(uint16_t h){ unsigned u=(unsigned)h<<16; float f; memcpy(&f,&u,4); return f; }
__global__ void k_transpose(float* out,const float* in,int M,int N,int ld){ // in col-major (ld) -> out row-major[M,N]
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M*N)return; int m=i/N,n=i%N; out[(size_t)m*N+n]=in[(size_t)m+(size_t)n*ld]; }
__global__ void k_embed(float* out,const uint16_t* emb,const int* ids,int seq,int H,float scale){
    int t=blockIdx.x,i=threadIdx.x; for(int j=i;j<H;j+=blockDim.x) out[(size_t)t*H+j]=bf2f(emb[(size_t)ids[t]*H+j])*scale; }
__global__ void k_mul(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)a[i]*=b[i]; }
__global__ void k_scale_rows(float* x,const float* s,int rows,int dim){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*dim)x[i]*=s[i/dim]; }
__global__ void k_gather(float* out,const float* in,const int* idx,int n,int dim){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n*dim)out[i]=in[(size_t)idx[i/dim]*dim + i%dim]; }
__global__ void k_scatter_add(float* out,const float* in,const int* idx,const float* w,int n,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*dim)return; int r=i/dim; atomicAdd(&out[(size_t)idx[r]*dim + i%dim], in[i]*w[r]); }
// lm_head for the LAST token: logits[v]=softcap(sum_h hlast[h]*bf16(emb[v,h])); one block per vocab
__global__ void k_lmhead(float* logits,const float* hlast,const uint16_t* emb,int H,int V,float cap){
    int v=blockIdx.x; if(v>=V)return; __shared__ float red[256];
    const uint16_t* ev=emb+(size_t)v*H; float acc=0;  // uint2 = 4 bf16/load (8B-aligned; embed not 16B-aligned), streaming
    for(int h=threadIdx.x*4; h<H; h+=blockDim.x*4){ uint2 w=__ldcs((const uint2*)(ev+h)); const uint16_t* b=(const uint16_t*)&w;
        acc += hlast[h]*bf2f(b[0])+hlast[h+1]*bf2f(b[1])+hlast[h+2]*bf2f(b[2])+hlast[h+3]*bf2f(b[3]); }
    red[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0){ float x=red[0]; logits[v]=cap*tanhf(x/cap); } }

// ---- NVFP4 lm_head: quantize the bf16 embed to E2M1 codes + E4M3 group-16 scales (w4a16 linear layout) ----
__device__ __forceinline__ uint8_t q_fp4(float x){ float a=fabsf(x); uint8_t idx;
    if(a<0.25f)idx=0; else if(a<0.75f)idx=1; else if(a<1.25f)idx=2; else if(a<1.75f)idx=3;
    else if(a<2.5f)idx=4; else if(a<3.5f)idx=5; else if(a<5.0f)idx=6; else idx=7;
    return idx|((x<0.0f)?0x8:0x0); }
__global__ void k_embed_amax(const uint16_t* emb,long total,float* gamax){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; float a=0.f;
    for(long i=idx;i<total;i+=(long)gridDim.x*blockDim.x) a=fmaxf(a,fabsf(bf2f(emb[i])));
    __shared__ float sm[256]; sm[threadIdx.x]=a; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s)sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]); __syncthreads(); }
    if(threadIdx.x==0) atomicMax((int*)gamax,__float_as_int(sm[0])); }
__global__ void k_quant_embed_fp4(const uint16_t* emb,uint8_t* wp,uint8_t* ws,int H,int V,float gscale){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int KB=H/16; long total=(long)V*KB; if(idx>=total)return;
    int n=idx/KB, b=idx%KB; const uint16_t* xb=emb+(size_t)n*H+(size_t)b*16;
    float amax=0.f;
    #pragma unroll
    for(int i=0;i<16;++i) amax=fmaxf(amax,fabsf(bf2f(xb[i])));
    float e4r=gscale*(amax*(1.f/6.f)); __nv_fp8_e4m3 e4=__nv_fp8_e4m3(e4r); float e4v=float(e4);
    float inv_local=(e4v>0.f)?(gscale/e4v):0.f;   // 1/local, local=e4v/gscale
    ws[(size_t)n*KB+b]=e4.__x;
    uint8_t* po=wp+(size_t)n*(H/2)+(size_t)b*8;
    #pragma unroll
    for(int i=0;i<8;++i){ uint8_t lo=q_fp4(bf2f(xb[2*i])*inv_local); uint8_t hi=q_fp4(bf2f(xb[2*i+1])*inv_local); po[i]=lo|(hi<<4); } }

// DFlash: capture residual after a tapped target layer into taps[mtok,6,H] at tap slot j
__global__ void k_tap(float* taps, const float* h, int mtok, int j, int H){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=mtok*H)return; int t=i/H,d=i%H;
    taps[((size_t)t*6 + j)*H + d] = h[i];
}
static const int TAP_LAYERS[6]={1,6,11,17,22,27};
static inline int tap_slot(int L){ for(int j=0;j<6;++j) if(TAP_LAYERS[j]==L) return j; return -1; }

// ---- KV-cache decode kernels ----
// store projected K/V (fp32 [m,nkv,hd]) into BF16 cache at positions [base, base+m)
__device__ __forceinline__ float hw_e4m3(uint8_t b){ __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
__global__ void k_store_kv(void* cache, const float* src, int m, int nkv, int hd, bool fp8){
    int i=blockIdx.x*blockDim.x+threadIdx.x; int n=m*nkv*hd; if(i>=n)return;
    int t=i/(nkv*hd), rest=i%(nkv*hd);
    float v=src[i]; size_t idx=(size_t)(g_base+t)*nkv*hd + rest;
    if(fp8) ((uint8_t*)cache)[idx]=(uint8_t)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);   // FP8 e4m3 KV (server): half the bytes
    else { unsigned u; memcpy(&u,&v,4); ((uint16_t*)cache)[idx]=(uint16_t)(u>>16); }              // BF16 KV (bit-exact gate path)
}
// single-query-block attention over a BF16 KV cache. query i (abs pos base+i) attends keys [lo, base+i].
// out[m,nh,hd], Q[m,nh,hd] fp32; Kc/Vc[CAP,nkv,hd] bf16. window=0 => full causal.
// HEAD-PACKED: one block per (query i, kv_head kvh) processes ALL G=nh/nkv sibling query heads; KV read ONCE
// (was 4x redundant per query-head). Batched reduction over the G heads keeps syncs low.
__global__ void sdpa_cache_kernel(float* out, const float* Q, const void* Kc, const void* Vc,
                                  int m, int nkv, int nh, int hd, int window, float scaling, bool fp8){
    int i=blockIdx.x, kvh=blockIdx.y; if(i>=m||kvh>=nkv) return;
    int G=nh/nkv, p=g_base+i, lo=(window>0)?max(0,p-window+1):0, d=threadIdx.x;
    extern __shared__ float red[];   // [G*hd] only; Qs/acc in registers -> shmem small even for hd=512,G=8
    __shared__ float mr[16],lr[16],cb[16],pb[16];
    float qs[8], acc[8];
    #pragma unroll
    for(int g=0;g<8;++g){ if(g<G){ int h=kvh*G+g; qs[g]=Q[((size_t)i*nh+h)*hd+d]; } acc[g]=0.f; }
    if(d<G){ mr[d]=-1e30f; lr[d]=0.f; }
    __syncthreads();
    for(int j=lo;j<=p;++j){
        size_t kidx=((size_t)j*nkv+kvh)*hd+d; float kf,vf;   // KV read ONCE for all G heads
        if(fp8){ kf=hw_e4m3(((const uint8_t*)Kc)[kidx]); vf=hw_e4m3(((const uint8_t*)Vc)[kidx]); }
        else { unsigned uk=(unsigned)((const uint16_t*)Kc)[kidx]<<16; memcpy(&kf,&uk,4); unsigned uv=(unsigned)((const uint16_t*)Vc)[kidx]<<16; memcpy(&vf,&uv,4); }
        #pragma unroll
        for(int g=0;g<8;++g) if(g<G) red[g*hd+d]=qs[g]*kf;
        __syncthreads();
        for(int s=hd/2;s>0;s>>=1){ if(d<s){
            #pragma unroll
            for(int g=0;g<8;++g) if(g<G) red[g*hd+d]+=red[g*hd+d+s]; } __syncthreads(); }
        if(d<G){ int g=d; float score=red[g*hd]*scaling; float nm=fmaxf(mr[g],score);
            cb[g]=__expf(mr[g]-nm); pb[g]=__expf(score-nm); lr[g]=lr[g]*cb[g]+pb[g]; mr[g]=nm; }
        __syncthreads();
        #pragma unroll
        for(int g=0;g<8;++g) if(g<G) acc[g]=acc[g]*cb[g]+pb[g]*vf;
    }
    #pragma unroll
    for(int g=0;g<8;++g) if(g<G){ int h=kvh*G+g; out[((size_t)i*nh+h)*hd+d]=acc[g]/lr[g]; }
}

// batched lm_head over mtok positions: logits[mtok,V]; each block loads one embed row, reused across positions
__global__ void k_lmhead_batched(float* out,const float* hn,const uint16_t* emb,int H,int V,int mtok){
    int v=blockIdx.x; if(v>=V)return; extern __shared__ float se[]; __shared__ float red[256];
    const uint16_t* ev=emb+(size_t)v*H;
    for(int h=threadIdx.x;h<H;h+=256) se[h]=bf2f(ev[h]);  // embed row once, reused across all mtok positions
    __syncthreads();
    for(int p=0;p<mtok;++p){
        const float* hp=hn+(size_t)p*H; float acc=0; for(int h=threadIdx.x;h<H;h+=256) acc+=hp[h]*se[h];
        red[threadIdx.x]=acc; __syncthreads();
        for(int s=128;s>0;s>>=1){ if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
        if(threadIdx.x==0) out[(size_t)p*V+v]=red[0]; __syncthreads();
    }
}
__global__ void k_scale_const(float* x,float s,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=s; }
__global__ void k_f32_to_f16(__half* out,const float* in,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2half(in[i]); }
// batched lm_head, HALF2 warp-per-vocab: embed row read ONCE, mtok dots via __hfma2 (3x faster than scalar block-per-vocab)
__global__ void k_lmhead_batched_h2(float* out,const __half* hn16,const uint16_t* emb,int H,int V,int mtok){
    int lane=threadIdx.x&31; long v=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5); if(v>=V)return;
    const uint2* ev=(const uint2*)(emb+(size_t)v*H);
    float acc[16]; for(int m=0;m<mtok;++m)acc[m]=0.f; int nv=H/4;
    for(int vi=lane;vi<nv;vi+=32){ uint2 ww=__ldcs(&ev[vi]); const uint16_t* b=(const uint16_t*)&ww; int h=vi*4;
        __half2 w01=__floats2half2_rn(bf2f(b[0]),bf2f(b[1])), w23=__floats2half2_rn(bf2f(b[2]),bf2f(b[3]));
        for(int m=0;m<mtok;++m){ const __half2* xh=(const __half2*)(hn16+(size_t)m*H+h);
            __half2 a=__hfma2(w23,xh[1],__hmul2(w01,xh[0])); acc[m]+=__half2float(__low2half(a))+__half2float(__high2half(a)); } }
    for(int m=0;m<mtok;++m){
        #pragma unroll
        for(int s=16;s>0;s>>=1) acc[m]+=__shfl_down_sync(0xffffffffu,acc[m],s);
        if(lane==0) out[(size_t)m*V+v]=acc[m]; }
}
// device argmax over V for each of mtok rows
__global__ void k_argmax(int* out,const float* lg,int V){
    int p=blockIdx.x; __shared__ float sv[256]; __shared__ int si[256];
    float bv=-1e30f; int bi=0;
    for(int v=threadIdx.x;v<V;v+=256){ float x=lg[(size_t)p*V+v]; if(x>bv){bv=x;bi=v;} }
    sv[threadIdx.x]=bv; si[threadIdx.x]=bi; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s && sv[threadIdx.x+s]>sv[threadIdx.x]){sv[threadIdx.x]=sv[threadIdx.x+s];si[threadIdx.x]=si[threadIdx.x+s];} __syncthreads(); }
    if(threadIdx.x==0) out[p]=si[0];
}
__device__ unsigned long long g_sample_seed=0ULL;   // per-step RNG seed for k_sample (updated like g_base for graph replay)
__device__ __forceinline__ float hashu(unsigned long long x){   // splitmix64 -> uniform (0,1)
    x+=0x9E3779B97F4A7C15ULL; x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL; x=(x^(x>>27))*0x94D049BB133111EBULL; x^=x>>31;
    return ((float)((x>>11))+0.5f)*(1.0f/9007199254740992.0f); }
// LOSSLESS temperature sampling in spec-decode: sample target[pos] via Gumbel-max (argmax(logit*invT + gumbel)).
// Accepting draft iff draft==this sample => every committed token is a genuine target sample; reduces to greedy at temp=0.
__global__ void k_sample(int* out,const float* lg,int V,float invT){
    int p=blockIdx.x; __shared__ float sv[256]; __shared__ int si[256];
    unsigned long long base=g_sample_seed ^ ((unsigned long long)p*0x100000001B3ULL);
    float bv=-1e30f; int bi=0;
    for(int v=threadIdx.x;v<V;v+=256){
        float u=hashu(base ^ ((unsigned long long)(unsigned)v*0x9E3779B1ULL));
        float g=-logf(-logf(u+1e-20f)+1e-20f);
        float s=lg[(size_t)p*V+v]*invT + g; if(s>bv){bv=s;bi=v;} }
    sv[threadIdx.x]=bv; si[threadIdx.x]=bi; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s && sv[threadIdx.x+s]>sv[threadIdx.x]){sv[threadIdx.x]=sv[threadIdx.x+s];si[threadIdx.x]=si[threadIdx.x+s];} __syncthreads(); }
    if(threadIdx.x==0) out[p]=si[0];
}

// typical acceptance: per position i, target softmax prob (at temp 1/invT) of the draft token block[i+1].
__global__ void k_typical_prob(float* prob,const float* logits,const int* block,int V,float invT){
    int i=blockIdx.x; const float* lg=logits+(size_t)i*V; int dt=block[i+1];
    __shared__ float red[256];
    float mx=-1e30f; for(int v=threadIdx.x;v<V;v+=256) mx=fmaxf(mx,lg[v]);
    red[threadIdx.x]=mx; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s) red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+s]); __syncthreads(); }
    mx=red[0]; __syncthreads();
    float se=0.f; for(int v=threadIdx.x;v<V;v+=256) se+=__expf((lg[v]-mx)*invT);
    red[threadIdx.x]=se; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0) prob[i]=__expf((lg[dt]-mx)*invT)/red[0];
}
// CUDA-graph self-advance: feed argmax back as next input token + increment position (all on-device)
__global__ void k_advance(int* dids,const int* darg){ if(threadIdx.x==0){ dids[0]=darg[0]; g_base++; } }
// ---- device-MoE kernels (no host round-trip; indexes expert weights by device top-8 ids) ----
__device__ __forceinline__ float e2m1d(uint8_t c){ const float t[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; float v=t[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ __half2 dec_fp4x2(unsigned char b){ __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }
// W4A4 emulation: round-trip activations through FP4 (per-group-16 amax scale) to inject W4A4 precision into
// the hidden states/taps without changing the GEMM. Tests if W4A16-vs-W4A4 taps explain low code acceptance.
__global__ void k_fp4_roundtrip(float* x,long n){
    long g=(long)blockIdx.x*blockDim.x+threadIdx.x; long base=g*16; if(base>=n) return;
    float amax=0; for(int i=0;i<16;++i) amax=fmaxf(amax,fabsf(x[base+i]));
    if(amax<1e-12f){ for(int i=0;i<16;++i)x[base+i]=0.f; return; }
    float scale=amax/6.f, inv=1.f/scale;
    for(int i=0;i<16;++i){ float v=x[base+i]; float a=fabsf(v)*inv; float s=v<0?-1.f:1.f; float q;
        if(a<0.25f)q=0;else if(a<0.75f)q=.5f;else if(a<1.25f)q=1.f;else if(a<1.75f)q=1.5f;
        else if(a<2.5f)q=2.f;else if(a<3.5f)q=3.f;else if(a<5.f)q=4.f;else q=6.f;
        x[base+i]=s*q*scale; }
}
static bool W4A4_TAPS=false;
__device__ __forceinline__ float e4m3d(uint8_t b){ int s=(b>>7)&1,e=(b>>3)&0xF,man=b&7; float v=(e==0)?(man*0.125f*0.015625f):(ldexpf(1.f+man*0.125f,e-7)); return s?-v:v; }
__constant__ float C_LUT[256];   // e4m3 byte -> value, computed once (no per-block recompute)
// HW e4m3 block-scale decode (register-only) — hw_e4m3 now defined above k_store_kv (shared with FP8 KV read)
static void init_clut(){ float h[256]; for(int i=0;i<256;++i){ int s=(i>>7)&1,e=(i>>3)&0xF,man=i&7;
    float v=(e==0)?(man*0.125f*0.015625f):((1.f+man*0.125f)*ldexp(1.0,e-7)); h[i]=s?-v:v; }
    CU(cudaMemcpyToSymbol(C_LUT,h,sizeof(h))); }
// router hidden: hn[mtok,H] = rmsnorm_noscale(resid) * router_scale * H^-0.5  (one block/token)
__global__ void k_router_hn(float* hn,const float* resid,const uint16_t* rscale,int mtok,int H){
    int t=blockIdx.x; const float* x=resid+(size_t)t*H; __shared__ float red[256];
    float ls=0; for(int i=threadIdx.x;i<H;i+=blockDim.x) ls+=x[i]*x[i];
    red[threadIdx.x]=ls; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    float inv=rsqrtf(red[0]/H+1e-6f), root=rsqrtf((float)H);
    for(int i=threadIdx.x;i<H;i+=blockDim.x) hn[(size_t)t*H+i]=x[i]*inv*bf2f(rscale[i])*root;
}
// router scores[mtok,NE] = hn @ router_proj^T, warp-per-(token,expert) (parallel across all experts)
__global__ void k_router_scores(float* scores,const float* hn,const uint16_t* rproj,int mtok,int H,int NE){
    int lane=threadIdx.x&31; long o=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    if(o>=(long)mtok*NE) return; int e=o%NE,t=o/NE;
    const float* x=hn+(size_t)t*H; const uint16_t* w=rproj+(size_t)e*H; float acc=0;
    for(int i=lane;i<H;i+=32) acc+=x[i]*bf2f(w[i]);
    #pragma unroll
    for(int s=16;s>0;s>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,s);
    if(lane==0) scores[(size_t)t*NE+e]=acc;
}
// softmax over NE -> top-K (by score) -> renorm to sum 1 -> *per_expert_scale. one thread/token.
// 128 threads (one per expert): parallel max/softmax + K rounds of reduction-argmax (was 1 thread, 1024 ops).
__global__ void k_router_top8(int* ids,float* ws,const float* scores,const uint16_t* pes,int NE,int K){
    int t=blockIdx.x, e=threadIdx.x; const float* sc=scores+(size_t)t*NE;
    __shared__ float rv[128]; __shared__ int ri[128]; __shared__ bool used[128];
    __shared__ float mx,Z; __shared__ int sel[16]; __shared__ float selp[16];
    float my=(e<NE)?sc[e]:-1e30f; used[e]=(e>=NE);
    rv[e]=my; __syncthreads();
    for(int s=64;s>0;s>>=1){ if(e<s)rv[e]=fmaxf(rv[e],rv[e+s]); __syncthreads(); }
    if(e==0)mx=rv[0]; __syncthreads();
    float ex=(e<NE)?__expf(my-mx):0.f; rv[e]=ex; __syncthreads();
    for(int s=64;s>0;s>>=1){ if(e<s)rv[e]+=rv[e+s]; __syncthreads(); }
    if(e==0)Z=rv[0]; __syncthreads();
    for(int j=0;j<K;++j){
        rv[e]=used[e]?-1e30f:my; ri[e]=e; __syncthreads();
        for(int s=64;s>0;s>>=1){ if(e<s && rv[e+s]>rv[e]){rv[e]=rv[e+s];ri[e]=ri[e+s];} __syncthreads(); }
        if(e==0){ int bi=ri[0]; used[bi]=true; sel[j]=bi; selp[j]=__expf(sc[bi]-mx)/Z; } __syncthreads();
    }
    if(e==0){ float sumw=0; for(int j=0;j<K;++j)sumw+=selp[j];
        for(int j=0;j<K;++j){ ids[(size_t)t*K+j]=sel[j]; ws[(size_t)t*K+j]=(selp[j]/sumw)*bf2f(pes[sel[j]]); } }
}
// expert gate/up: hbuf[t,j,i] = gelu(Wg_e[i]·x2[t]) * (Wu_e[i]·x2[t]); one block per (t,j,i)
__global__ void k_moe_gateup(__half* hbuf,const __half* x2,const int* ids,
    const uint8_t* const* gp,const uint8_t* const* gs,const float* gg,
    const uint8_t* const* up,const uint8_t* const* us,const float* ug,int mtok,int H,int MI){
    int lane=threadIdx.x&31; long idx=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);  // warp per (t,j,i)
    if(idx>=(long)mtok*8*MI) return;
    int i=idx%MI; long rest=idx/MI; int j=rest%8,t=rest/8; int e=ids[(size_t)t*8+j];
    const __half* x=x2+(size_t)t*H;
    const unsigned* gpw=(const unsigned*)(gp[e]+(size_t)i*(H/2)); const uint8_t* gsw=gs[e]+(size_t)i*(H/16); float ginv=1.f/gg[e];
    const unsigned* upw=(const unsigned*)(up[e]+(size_t)i*(H/2)); const uint8_t* usw=us[e]+(size_t)i*(H/16); float uinv=1.f/ug[e];
    float ag=0,au=0; int nu=H/8; const int U=4;   // K-unroll: prefetch U independent weight loads before FMA -> raises MLP (fixes serial-chain latency stall)
    for(int vi=lane;vi<nu;vi+=32*U){
        unsigned wg[U],wu[U]; int vv[U];
        #pragma unroll
        for(int u=0;u<U;++u){ int v=vi+u*32; vv[u]=v; if(v<nu){ wg[u]=__ldcs(&gpw[v]); wu[u]=__ldcs(&upw[v]); } }   // U loads issued together
        #pragma unroll
        for(int u=0;u<U;++u){ int v=vv[u]; if(v>=nu) continue; int k=v*8;
            uint4 xpk=*(const uint4*)(x+k); const __half2* xh2=(const __half2*)&xpk;   // activation (L2-cached, low latency)
            float sg=hw_e4m3(__ldcs(&gsw[k>>4]))*ginv, su=hw_e4m3(__ldcs(&usw[k>>4]))*uinv;
            const unsigned char* wgb=(const unsigned char*)&wg[u]; const unsigned char* wub=(const unsigned char*)&wu[u];
            __half2 g2=__float2half2_rn(0.f),u2=__float2half2_rn(0.f);
            #pragma unroll
            for(int b=0;b<4;++b){ g2=__hfma2(dec_fp4x2(wgb[b]),xh2[b],g2); u2=__hfma2(dec_fp4x2(wub[b]),xh2[b],u2); }
            ag += sg*(__half2float(__low2half(g2))+__half2float(__high2half(g2)));
            au += su*(__half2float(__low2half(u2))+__half2float(__high2half(u2))); } }
    #pragma unroll
    for(int o=16;o>0;o>>=1){ ag+=__shfl_down_sync(0xffffffffu,ag,o); au+=__shfl_down_sync(0xffffffffu,au,o); }
    if(lane==0){ float gel=0.5f*ag*(1.f+tanhf(0.7978845608f*(ag+0.044715f*ag*ag*ag))); hbuf[idx]=__float2half(gel*au); }
}
// inverse routing: ecount[e]=#assignments to e; elist[e*EL+c]=c-th (t*8+j) routing to e.
__global__ void k_moe_invert(int* ecount,int* elist,const int* ids,int nass,int EL){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=nass)return;
    int e=ids[idx]; int c=atomicAdd(&ecount[e],1); if(c<EL) elist[e*EL+c]=idx;
}
// GROUPED gateup: warp per (e,i). Read+decode Wg_e[i]/Wu_e[i] ONCE, reuse across the tokens routing to e (<=4/pass).
__global__ void k_moe_gateup_grouped(__half* hbuf,const __half* x2,const int* ecount,const int* elist,int EL,
    const uint8_t* const* gp,const uint8_t* const* gs,const float* gg,
    const uint8_t* const* up,const uint8_t* const* us,const float* ug,int H,int MI){
    int lane=threadIdx.x&31; long idx=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    int i=idx%MI; int e=idx/MI; if(e>=128)return; int cnt=ecount[e]; if(cnt==0)return;
    const unsigned* gpw=(const unsigned*)(gp[e]+(size_t)i*(H/2)); const uint8_t* gsw=gs[e]+(size_t)i*(H/16); float ginv=1.f/gg[e];
    const unsigned* upw=(const unsigned*)(up[e]+(size_t)i*(H/2)); const uint8_t* usw=us[e]+(size_t)i*(H/16); float uinv=1.f/ug[e];
    int nu=H/8;
    for(int base=0;base<cnt;base+=4){ int nb=cnt-base; if(nb>4)nb=4;
        float ag[4]={0,0,0,0},au[4]={0,0,0,0}; const __half* xp[4];
        for(int c=0;c<nb;++c){ xp[c]=x2+(size_t)(elist[e*EL+base+c]/8)*H; }
        const int U=4;   // K-unroll prefetch: issue U gate+up weight loads before decode/FMA -> raise MLP (SoL: latency-bound)
        for(int vi=lane;vi<nu;vi+=32*U){
            unsigned wg[U],wu[U]; int vv[U];
            #pragma unroll
            for(int u=0;u<U;++u){ int v=vi+u*32; vv[u]=v; if(v<nu){ wg[u]=__ldcs(&gpw[v]); wu[u]=__ldcs(&upw[v]); } }
            #pragma unroll
            for(int u=0;u<U;++u){ int v=vv[u]; if(v>=nu) continue; int k=v*8;
                float sg=hw_e4m3(__ldcs(&gsw[k>>4]))*ginv, su=hw_e4m3(__ldcs(&usw[k>>4]))*uinv;
                const unsigned char* wgb=(const unsigned char*)&wg[u]; const unsigned char* wub=(const unsigned char*)&wu[u];
                __half2 gd0=dec_fp4x2(wgb[0]),gd1=dec_fp4x2(wgb[1]),gd2=dec_fp4x2(wgb[2]),gd3=dec_fp4x2(wgb[3]);
                __half2 ud0=dec_fp4x2(wub[0]),ud1=dec_fp4x2(wub[1]),ud2=dec_fp4x2(wub[2]),ud3=dec_fp4x2(wub[3]);
                for(int c=0;c<nb;++c){ uint4 xpk=*(const uint4*)(xp[c]+k); const __half2* x=(const __half2*)&xpk;
                    __half2 g2=__hadd2(__hadd2(__hmul2(gd0,x[0]),__hmul2(gd1,x[1])),__hadd2(__hmul2(gd2,x[2]),__hmul2(gd3,x[3])));
                    __half2 u2=__hadd2(__hadd2(__hmul2(ud0,x[0]),__hmul2(ud1,x[1])),__hadd2(__hmul2(ud2,x[2]),__hmul2(ud3,x[3])));
                    ag[c]+=sg*(__half2float(__low2half(g2))+__half2float(__high2half(g2)));
                    au[c]+=su*(__half2float(__low2half(u2))+__half2float(__high2half(u2))); } } }
        for(int c=0;c<nb;++c){
            #pragma unroll
            for(int o=16;o>0;o>>=1){ ag[c]+=__shfl_down_sync(0xffffffffu,ag[c],o); au[c]+=__shfl_down_sync(0xffffffffu,au[c],o); }
            if(lane==0){ int tj=elist[e*EL+base+c]; float g=ag[c]; float gel=0.5f*g*(1.f+tanhf(0.7978845608f*(g+0.044715f*g*g*g))); hbuf[(size_t)tj*MI+i]=__float2half(gel*au[c]); } }
    }
}
// expert down: out[t,d] = sum_j ws[t,j]*(Wd_e[d]·hbuf[t,j]). WARP per (t,d): lanes stride over MI/8,
// 8 experts FUSED into one accumulator -> ONE shfl reduce (vs 8 block reductions + 34% thread util before).
__global__ void k_moe_down(float* out,const __half* hbuf,const int* ids,const float* ws,
    const uint8_t* const* dp,const uint8_t* const* ds,const float* dg,int mtok,int H,int MI){
    int lane=threadIdx.x&31; long pidx=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);  // RB=2: warp does d0,d0+1
    if(pidx>=(long)mtok*(H/2)) return; int dp_=pidx%(H/2),t=pidx/(H/2); int d0=dp_*2;
    int E[8]; float WI[8];   // hoist ints/floats only (avoid the 32-pointer register pressure that regressed)
    #pragma unroll
    for(int j=0;j<8;++j){ E[j]=ids[(size_t)t*8+j]; WI[j]=ws[(size_t)t*8+j]/dg[E[j]]; }
    float acc0=0,acc1=0; int nu=MI/8;
    for(int vi=lane;vi<nu;vi+=32){ int k=vi*8;
        #pragma unroll
        for(int j=0;j<8;++j){ int e=E[j];
            uint4 hpk=*(const uint4*)(hbuf+((size_t)t*8+j)*MI+k); const __half2* hh2=(const __half2*)&hpk;  // shared by d0,d1
            const unsigned* dpw0=(const unsigned*)(dp[e]+(size_t)d0*(MI/2)); const uint8_t* dsw0=ds[e]+(size_t)d0*(MI/16);
            const unsigned* dpw1=(const unsigned*)(dp[e]+(size_t)(d0+1)*(MI/2)); const uint8_t* dsw1=ds[e]+(size_t)(d0+1)*(MI/16);
            unsigned wd0=__ldcs(&dpw0[vi]); float sc0=hw_e4m3(__ldcs(&dsw0[k>>4]))*WI[j];
            unsigned wd1=__ldcs(&dpw1[vi]); float sc1=hw_e4m3(__ldcs(&dsw1[k>>4]))*WI[j];
            const unsigned char* w0=(const unsigned char*)&wd0; const unsigned char* w1=(const unsigned char*)&wd1;
            __half2 s20=__float2half2_rn(0.f),s21=__float2half2_rn(0.f);
            #pragma unroll
            for(int b=0;b<4;++b){ s20=__hfma2(dec_fp4x2(w0[b]),hh2[b],s20); s21=__hfma2(dec_fp4x2(w1[b]),hh2[b],s21); }
            acc0 += sc0*(__half2float(__low2half(s20))+__half2float(__high2half(s20)));
            acc1 += sc1*(__half2float(__low2half(s21))+__half2float(__high2half(s21))); } }
    #pragma unroll
    for(int o=16;o>0;o>>=1){ acc0+=__shfl_down_sync(0xffffffffu,acc0,o); acc1+=__shfl_down_sync(0xffffffffu,acc1,o); }
    if(lane==0){ out[(size_t)t*H+d0]=acc0; out[(size_t)t*H+d0+1]=acc1; }
}

// BANDWIDTH-OPTIMAL down: warp per (expert e, d0-pair). Read Wd_e[d0],Wd_e[d0+1] ONCE, reuse across ALL tokens
// routing to e (weight-resident) -> hits the HBM floor (no ~1.5x per-token weight re-read). Writes per-assignment
// partials dpart[(t*8+j)*H + d] (NO atomics). dg dequant applied here; routing weight ws applied in finalize.
__global__ void k_moe_down_bw(float* dpart,const __half* hbuf,const int* ecount,const int* elist,int EL,
    const uint8_t* const* dp,const uint8_t* const* ds,const float* dg,int H,int MI){
    int lane=threadIdx.x&31; long widx=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    int dpr=widx%(H/2); int e=widx/(H/2); if(e>=128)return; int cnt=ecount[e]; if(cnt==0)return; int d0=dpr*2;
    const unsigned* dpw0=(const unsigned*)(dp[e]+(size_t)d0*(MI/2)); const uint8_t* dsw0=ds[e]+(size_t)d0*(MI/16);
    const unsigned* dpw1=(const unsigned*)(dp[e]+(size_t)(d0+1)*(MI/2)); const uint8_t* dsw1=ds[e]+(size_t)(d0+1)*(MI/16);
    float dinv=1.f/dg[e]; int nu=MI/8;
    for(int base=0;base<cnt;base+=4){ int nb=cnt-base; if(nb>4)nb=4;
        float a0[4]={0,0,0,0},a1[4]={0,0,0,0}; const __half* hp[4]; int tj[4];
        for(int c=0;c<nb;++c){ tj[c]=elist[e*EL+base+c]; hp[c]=hbuf+(size_t)tj[c]*MI; }
        const int U=4;   // K-unroll prefetch (SoL: down_bw latency-bound)
        for(int vi=lane;vi<nu;vi+=32*U){
            unsigned wd0[U],wd1[U]; int vv[U];
            #pragma unroll
            for(int u=0;u<U;++u){ int v=vi+u*32; vv[u]=v; if(v<nu){ wd0[u]=__ldcs(&dpw0[v]); wd1[u]=__ldcs(&dpw1[v]); } }
            #pragma unroll
            for(int u=0;u<U;++u){ int v=vv[u]; if(v>=nu) continue; int k=v*8;
                float sc0=hw_e4m3(__ldcs(&dsw0[k>>4]))*dinv, sc1=hw_e4m3(__ldcs(&dsw1[k>>4]))*dinv;
                const unsigned char* wb0=(const unsigned char*)&wd0[u]; const unsigned char* wb1=(const unsigned char*)&wd1[u];
                __half2 d0a=dec_fp4x2(wb0[0]),d0b=dec_fp4x2(wb0[1]),d0c=dec_fp4x2(wb0[2]),d0d=dec_fp4x2(wb0[3]);
                __half2 d1a=dec_fp4x2(wb1[0]),d1b=dec_fp4x2(wb1[1]),d1c=dec_fp4x2(wb1[2]),d1d=dec_fp4x2(wb1[3]);
                for(int c=0;c<nb;++c){ uint4 hpk=*(const uint4*)(hp[c]+k); const __half2* h=(const __half2*)&hpk;
                    __half2 s0=__hadd2(__hadd2(__hmul2(d0a,h[0]),__hmul2(d0b,h[1])),__hadd2(__hmul2(d0c,h[2]),__hmul2(d0d,h[3])));
                    __half2 s1=__hadd2(__hadd2(__hmul2(d1a,h[0]),__hmul2(d1b,h[1])),__hadd2(__hmul2(d1c,h[2]),__hmul2(d1d,h[3])));
                    a0[c]+=sc0*(__half2float(__low2half(s0))+__half2float(__high2half(s0)));
                    a1[c]+=sc1*(__half2float(__low2half(s1))+__half2float(__high2half(s1))); } } }
        for(int c=0;c<nb;++c){
            #pragma unroll
            for(int o=16;o>0;o>>=1){ a0[c]+=__shfl_down_sync(0xffffffffu,a0[c],o); a1[c]+=__shfl_down_sync(0xffffffffu,a1[c],o); }
            if(lane==0){ dpart[(size_t)tj[c]*H+d0]=a0[c]; dpart[(size_t)tj[c]*H+d0+1]=a1[c]; } }
    }
}
// finalize: out[t][d] = sum_{j=0..7} ws[t][j] * dpart[(t*8+j)*H + d]   (cheap 8-way reduce per output)
__global__ void k_moe_finalize(float* out,const float* dpart,const float* ws,int mtok,int H){
    long idx=(long)blockIdx.x*256+threadIdx.x; if(idx>=(long)mtok*H)return; int t=idx/H,d=idx%H;
    float acc=0;
    #pragma unroll
    for(int j=0;j<8;++j) acc+=ws[(size_t)t*8+j]*dpart[(size_t)(t*8+j)*H+d];
    out[idx]=acc;
}

// device pointer arrays for one layer's 128 experts (indexable by device top-8 id)
struct ExpertPtrs { const uint8_t **gp,**gs,**up,**us,**dp,**ds; float *gg,*ug,*dg; };

// ---- Model ----
struct Model {
    st::SafeTensors* st; uint8_t* draw; const uint8_t* h0;
    std::unordered_map<int,ExpertPtrs> ecache;
    cublasLtHandle_t lt; void* ws; size_t wsb=64u<<20;
    std::unordered_map<std::string,uint8_t*> swz;   // swizzled scales
    std::unordered_map<std::string,uint8_t*> wpk;   // aligned packed-weight copies (cublasLt needs 16B align)
    std::unordered_map<std::string,float*> norms;   // fp32 norm weights
    Model(const std::string& path){
        st=new st::SafeTensors(path); h0=st->dataStart();
        CU(cudaMalloc(&draw, st->dataBytes()));
        CU(cudaMemcpy(draw, h0, st->dataBytes(), cudaMemcpyHostToDevice));
        cublasLtCreate(&lt); CU(cudaMalloc(&ws,wsb));
        printf("uploaded %.1f GB weights to device\n", st->dataBytes()/1e9);
    }
    const st::Tensor& T(const std::string& n){ return st->get(n); }
    bool has(const std::string& n){ return st->has(n); }
    template<class P> P dptr(const std::string& n){ return (P)(draw + (st->get(n).data - h0)); }
    float scalarF32(const std::string& n){ float v; memcpy(&v, st->get(n).data, 4); return v; }
    float scalarBf16(const std::string& n){ uint16_t b; memcpy(&b, st->get(n).data,2); unsigned u=(unsigned)b<<16; float f; memcpy(&f,&u,4); return f; }
    // device pointer arrays for layer L's 128 experts (cached) — for the device-MoE kernels
    ExpertPtrs* experts(int L){
        auto it=ecache.find(L); if(it!=ecache.end()) return &it->second;
        std::string P="model.language_model.layers."+std::to_string(L)+".experts.";
        std::vector<const uint8_t*> hgp(128),hgs(128),hup(128),hus(128),hdp(128),hds(128);
        std::vector<float> hgg(128),hug(128),hdg(128);
        for(int e=0;e<128;++e){ std::string E=P+std::to_string(e)+".";
            hgp[e]=dptr<const uint8_t*>(E+"gate_proj.weight_packed"); hgs[e]=dptr<const uint8_t*>(E+"gate_proj.weight_scale"); hgg[e]=scalarF32(E+"gate_proj.weight_global_scale");
            hup[e]=dptr<const uint8_t*>(E+"up_proj.weight_packed");   hus[e]=dptr<const uint8_t*>(E+"up_proj.weight_scale");   hug[e]=scalarF32(E+"up_proj.weight_global_scale");
            hdp[e]=dptr<const uint8_t*>(E+"down_proj.weight_packed"); hds[e]=dptr<const uint8_t*>(E+"down_proj.weight_scale"); hdg[e]=scalarF32(E+"down_proj.weight_global_scale"); }
        ExpertPtrs ep;
        auto P8=[&](const std::vector<const uint8_t*>& hv,const uint8_t**& dv){ CU(cudaMalloc(&dv,128*sizeof(uint8_t*))); CU(cudaMemcpy(dv,hv.data(),128*sizeof(uint8_t*),cudaMemcpyHostToDevice)); };
        auto PF=[&](const std::vector<float>& hv,float*& dv){ CU(cudaMalloc(&dv,128*4)); CU(cudaMemcpy(dv,hv.data(),128*4,cudaMemcpyHostToDevice)); };
        P8(hgp,ep.gp);P8(hgs,ep.gs);P8(hup,ep.up);P8(hus,ep.us);P8(hdp,ep.dp);P8(hds,ep.ds); PF(hgg,ep.gg);PF(hug,ep.ug);PF(hdg,ep.dg);
        return &ecache.emplace(L,ep).first->second;
    }
    // aligned packed-weight device copy (cached) — cublasLt FP4 requires aligned operands
    uint8_t* wpacked(const std::string& wname){
        auto it=wpk.find(wname); if(it!=wpk.end())return it->second;
        const auto& t=T(wname+".weight_packed"); uint8_t* d; CU(cudaMalloc(&d,t.nbytes));
        CU(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); wpk[wname]=d; return d;
    }
    // swizzled scale device ptr (cached). weight_scale shape [N, K/16]
    uint8_t* wscale(const std::string& wname){
        auto it=swz.find(wname); if(it!=swz.end())return it->second;
        const auto& ts=T(wname+".weight_scale"); int N=ts.shape[0], K=ts.shape[1]*16;
        size_t bytes=nvfp4_scale_buffer_bytes(N,K);
        std::vector<uint8_t> host(bytes); nvfp4_swizzle_scales(ts.data, host.data(), N, K);
        uint8_t* d; CU(cudaMalloc(&d,bytes)); CU(cudaMemcpy(d,host.data(),bytes,cudaMemcpyHostToDevice));
        swz[wname]=d; return d;
    }
    // fp32 norm weight (cached) from bf16
    float* norm(const std::string& n){
        auto it=norms.find(n); if(it!=norms.end())return it->second;
        const auto& t=T(n); int d=t.numel(); std::vector<float> hf(d);
        const uint16_t* b=(const uint16_t*)t.data;
        for(int i=0;i<d;++i){ unsigned u=(unsigned)b[i]<<16; memcpy(&hf[i],&u,4);}
        float* dp; CU(cudaMalloc(&dp,d*4)); CU(cudaMemcpy(dp,hf.data(),d*4,cudaMemcpyHostToDevice));
        norms[n]=dp; return dp;
    }
};

// scratch (reused)
struct Scratch { float* in_pad; uint8_t* xp; uint8_t* xs; float* dcol; size_t cap_rows=512, cap_k=8192, cap_n=8192;
    Scratch(){ CU(cudaMalloc(&in_pad, cap_rows*cap_k*4)); CU(cudaMalloc(&xp, cap_rows*cap_k/2));
        CU(cudaMalloc(&xs, nvfp4_scale_buffer_bytes(cap_rows,cap_k))); CU(cudaMalloc(&dcol, cap_rows*cap_n*4)); } };
static Scratch* SC;

// pre-allocated decode scratch (no per-step cudaMalloc; cudaMalloc blocks + prevents graph capture)
static const int MAXM=16, QDMAX=16*512, KDMAX=16*512;
struct DScratch {
    float *hmain,*hn,*q,*k,*v,*ao,*op,*dc,*ds;
    float *resid,*mi,*g,*u,*hs1,*x2,*moe_out,*scores,*top8_w,*sc1; int* top8_ids; int* dids;
    float *hl,*dlog,*hln,*lg2,*aq,*tprob,*dpart; int* darg,*ecount,*elist; __half *xh16,*hbuf,*x2_16;
    DScratch(){ auto A=[&](float*&p,size_t n){ CU(cudaMalloc(&p,n*4)); };
        CU(cudaMalloc(&xh16,(size_t)MAXM*8192*sizeof(__half)));  // fp16 activation scratch: M=1 GEMV[K] or M<=16 GEMM[M,K]
        CU(cudaMalloc(&aq,(size_t)MAXM*8192*4));  // W4A4-emulation activation round-trip scratch
        CU(cudaMalloc(&hbuf,(size_t)MAXM*8*MOE_INT*sizeof(__half))); CU(cudaMalloc(&x2_16,(size_t)MAXM*H*sizeof(__half)));
        A(hl,H);A(dlog,VOCAB);A(hln,MAXM*H);A(lg2,(size_t)MAXM*VOCAB);CU(cudaMalloc(&darg,MAXM*4));
        A(hmain,MAXM*H);A(hn,MAXM*H);A(q,MAXM*QDMAX);A(k,MAXM*KDMAX);A(v,MAXM*KDMAX);A(ao,MAXM*QDMAX);A(op,MAXM*H);
        A(dc,MAXM*256);A(ds,MAXM*256);A(resid,MAXM*H);A(mi,MAXM*H);A(g,MAXM*MLP_INT);A(u,MAXM*MLP_INT);A(hs1,MAXM*H);
        A(x2,MAXM*H);A(moe_out,MAXM*H);A(scores,MAXM*128);A(top8_w,MAXM*8);A(sc1,MAXM);
        CU(cudaMalloc(&top8_ids,MAXM*8*4)); CU(cudaMalloc(&dids,MAXM*4));
        CU(cudaMalloc(&ecount,128*4)); CU(cudaMalloc(&elist,128*16*4)); CU(cudaMalloc(&tprob,MAXM*4));
        CU(cudaMalloc(&dpart,(size_t)MAXM*8*H*4)); }   // per-assignment down partials (bandwidth-optimal down)
};
static DScratch* DS;

// ---- single-session KV cache (BF16), fixed capacity ----
static int CAP = 8192;   // context capacity (set via CTX env); server defaults to 65536 (64K)
static bool g_fp8kv = false;   // FP8 (e4m3) KV cache: half the bytes, lossy (server opts in; gate stays bf16/bit-exact)
struct Session {
    uint8_t* Kc[NLAYER]; uint8_t* Vc[NLAYER]; int valid_len=0;   // byte buffers: bf16=2B/elem or fp8=1B/elem
    Session(){ for(int L=0;L<NLAYER;++L){ int nkv=is_full(L)?NKV_F:NKV_S, hd=is_full(L)?HD_F:HD_S;
        size_t sz=(size_t)CAP*nkv*hd*(g_fp8kv?1:2); CU(cudaMalloc(&Kc[L],sz)); CU(cudaMalloc(&Vc[L],sz)); } }
};

// W4A4 Linear: out_row[M,N] = (in_row[M,K] @ W[N,K]^T) using stored global scales. Pads M>=128.
// Unified W4A16 Linear (FP4 weight x fp32 activation): out_row[M,N] = dequant(W[N,K]) @ in_row[M,K]^T.
// Same precision for ALL M (decode, prefill, verify) -> self-consistent + accurate (matches vLLM on ties).
extern "C" void tc_w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale, const void* x16, int M, int N, int K, cudaStream_t s);
static void linear(Model& m, float* out_row, const float* in_row, const std::string& prefix, int M, int N, int K){
    uint8_t* Wp = m.dptr<uint8_t*>(prefix+".weight_packed");
    uint8_t* Ws = m.dptr<uint8_t*>(prefix+".weight_scale");
    float wg = m.scalarF32(prefix+".weight_global_scale");
    const float* xin=in_row; float* aq_big=nullptr;
    if(W4A4_TAPS){   // emulate W4A4: round-trip activation through FP4 before the (W4A16) GEMM
        float* aq; bool biga=(size_t)M*K>(size_t)MAXM*8192;
        if(biga){ CU(cudaMalloc(&aq,(size_t)M*K*4)); aq_big=aq; } else aq=DS->aq;
        CU(cudaMemcpyAsync(aq,in_row,(size_t)M*K*4,cudaMemcpyDeviceToDevice));
        k_fp4_roundtrip<<<(unsigned)(((long)M*K/16+255)/256),256>>>(aq,(long)M*K); xin=aq;
    }
    if(M==1){ k_f32_to_f16<<<(K+255)/256,256>>>(DS->xh16, xin, K);   // fp16 activation -> 128-bit loads in GEMV
        fp4_gemv(out_row, Wp, Ws, wg, DS->xh16, N, K, 0); }
    else {   // batched W4A16: HW-decode weight once, reuse across M tokens via half2 (fp16 acts)
        __half* x16; bool bigx = M>MAXM;
        if(bigx) CU(cudaMalloc(&x16,(size_t)M*K*sizeof(__half))); else x16=DS->xh16;
        k_f32_to_f16<<<((size_t)M*K+255)/256,256>>>(x16, xin, (size_t)M*K);
        static bool noTC = getenv("NOTCVERIFY")!=nullptr;
        if(!noTC && M<=16 && N<=8192) tc_w4a16_gemm(out_row, Wp, Ws, wg, x16, M, N, K, 0);   // lever A (champion): raw-mma.sync TC verify GEMM, dense only (+3.3%, bit-exact); lm_head stays CUDA-core (bandwidth-bound)
        else w4a16_gemm(out_row, Wp, Ws, wg, x16, M, N, K, 0);
        if(bigx){ CU(cudaStreamSynchronize(0)); CU(cudaFree(x16)); }
    }
    if(aq_big){ CU(cudaStreamSynchronize(0)); CU(cudaFree(aq_big)); }
}

// build rope cos/sin tables [seq, half] on host -> device
static void rope_tables(int seq,int head_dim,double theta,int rope_angles,float** dcos,float** dsin,int base=0){
    int half=head_dim/2; std::vector<float> c((size_t)seq*half), s((size_t)seq*half);
    for(int pp=0;pp<seq;++pp){ int p=base+pp; for(int j=0;j<half;++j){
        double inv = (j<rope_angles)? 1.0/pow(theta, (2.0*j)/head_dim) : 0.0;
        double a=p*inv; c[(size_t)pp*half+j]=cosf(a); s[(size_t)pp*half+j]=sinf(a);
    }}
    CU(cudaMalloc(dcos,c.size()*4)); CU(cudaMalloc(dsin,s.size()*4));
    CU(cudaMemcpy(*dcos,c.data(),c.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(*dsin,s.data(),s.size()*4,cudaMemcpyHostToDevice));
}

// device rope cos/sin tables into pre-allocated dc/ds [mtok, head_dim/2] for absolute positions base..base+mtok
__global__ void k_rope_tables(float* dc,float* ds,int head_dim,double theta,int rope_angles){
    int base=g_base;
    int p=blockIdx.x,j=threadIdx.x,half=head_dim/2; if(j>=half)return;
    double inv=(j<rope_angles)?1.0/pow(theta,(2.0*j)/head_dim):0.0; double a=(double)(base+p)*inv;
    dc[(size_t)p*half+j]=cosf(a); ds[(size_t)p*half+j]=sinf(a);
}

// ---- attention with KV cache: processes mtok new tokens at positions [base, base+mtok) ----
static void attention_cached(Model& m, Session& S, float* h, int mtok, int base, int L){
    std::string P="model.language_model.layers."+std::to_string(L)+".";
    int hd = is_full(L)?HD_F:HD_S, nkv = is_full(L)?NKV_F:NKV_S;
    int qd=NHEAD*hd, kd=nkv*hd;
    bool big=mtok>MAXM; std::vector<float*> tf;  // DS for hot path (mtok<=16); malloc fallback for prefill
    auto pick=[&](float* d,size_t n)->float*{ if(!big)return d; float* p; CU(cudaMalloc(&p,n*4)); tf.push_back(p); return p; };
    float *hn=pick(DS->hn,mtok*H),*q=pick(DS->q,mtok*qd),*k=pick(DS->k,mtok*kd),*v=pick(DS->v,mtok*kd),
          *ao=pick(DS->ao,mtok*qd),*op=pick(DS->op,mtok*H),*dc=pick(DS->dc,mtok*256),*ds=pick(DS->ds,mtok*256);
    static bool MEGA = getenv("MEGAKERNEL")!=nullptr;
    if(MEGA && mtok==1 && !W4A4_TAPS){   // M1: fused input_rmsnorm + Q/K/V proj (persistent counter-synced megakernel)
        std::string qp_=P+"self_attn.q_proj", kp_=P+"self_attn.k_proj", vp_=P+"self_attn.v_proj";
        uint8_t *vpp=nullptr,*vps=nullptr; float vwg=1.f;
        if(!is_full(L)){ vpp=m.dptr<uint8_t*>(vp_+".weight_packed"); vps=m.dptr<uint8_t*>(vp_+".weight_scale"); vwg=m.scalarF32(vp_+".weight_global_scale"); }
        mega_qkv(q,k,v,h,m.dptr<const uint16_t*>(P+"input_layernorm.weight"),
            m.dptr<uint8_t*>(qp_+".weight_packed"),m.dptr<uint8_t*>(qp_+".weight_scale"),m.scalarF32(qp_+".weight_global_scale"),
            m.dptr<uint8_t*>(kp_+".weight_packed"),m.dptr<uint8_t*>(kp_+".weight_scale"),m.scalarF32(kp_+".weight_global_scale"),
            vpp,vps,vwg, H,qd,kd,EPS,0);
    } else {
    rmsnorm(hn, h, m.dptr<const uint16_t*>(P+"input_layernorm.weight"), mtok, H, EPS, 0);
    linear(m, q, hn, P+"self_attn.q_proj", mtok, qd, H);
    linear(m, k, hn, P+"self_attn.k_proj", mtok, kd, H);
    if(is_full(L)) CU(cudaMemcpyAsync(v, k, (size_t)mtok*kd*4, cudaMemcpyDeviceToDevice)); // k_eq_v
    else linear(m, v, hn, P+"self_attn.v_proj", mtok, kd, H);
    }
    rmsnorm(q, q, m.dptr<const uint16_t*>(P+"self_attn.q_norm.weight"), mtok*NHEAD, hd, EPS, 0);
    rmsnorm(k, k, m.dptr<const uint16_t*>(P+"self_attn.k_norm.weight"), mtok*nkv, hd, EPS, 0);
    rmsnorm(v, v, (const uint16_t*)nullptr, mtok*nkv, hd, EPS, 0); // v_norm no scale
    double theta=is_full(L)?1e6:1e4; int rang=is_full(L)?64:128;
    k_rope_tables<<<mtok,hd/2>>>(dc, ds, hd, theta, rang);   // reads g_base
    rope_rotate_half(q, dc, ds, mtok, NHEAD, hd, hd, 0);
    rope_rotate_half(k, dc, ds, mtok, nkv,   hd, hd, 0);
    k_store_kv<<<(mtok*kd+255)/256,256>>>(S.Kc[L], k, mtok, nkv, hd, g_fp8kv);   // reads g_base
    k_store_kv<<<(mtok*kd+255)/256,256>>>(S.Vc[L], v, mtok, nkv, hd, g_fp8kv);
    int G=NHEAD/nkv; int shmem=G*hd*4; dim3 grid(mtok,nkv);   // head-packed: one block per (query, kv_head)
    sdpa_cache_kernel<<<grid,hd,shmem>>>(ao, q, S.Kc[L], S.Vc[L], mtok, nkv, NHEAD, hd, is_full(L)?0:SWIN, 1.0f, g_fp8kv);   // reads g_base
    if(MEGA && mtok==1 && !W4A4_TAPS){   // M2: fused o_proj + post_attn_norm + residual
        std::string o_=P+"self_attn.o_proj";
        mega_oproj(h, ao, m.dptr<const uint16_t*>(P+"post_attention_layernorm.weight"),
            m.dptr<uint8_t*>(o_+".weight_packed"), m.dptr<uint8_t*>(o_+".weight_scale"), m.scalarF32(o_+".weight_global_scale"),
            H, qd, EPS, 0);
    } else {
    linear(m, op, ao, P+"self_attn.o_proj", mtok, H, qd);
    rmsnorm(op, op, m.dptr<const uint16_t*>(P+"post_attention_layernorm.weight"), mtok, H, EPS, 0);
    add_inplace(h, op, mtok*H, 0);   // no sync: all on stream 0, ordered
    }
    if(big){ CU(cudaDeviceSynchronize()); for(auto p:tf)cudaFree(p); }
}

// host fp32 copy of a bf16 tensor
static const std::vector<float>& bf16_host(Model& m,const std::string& n){
    static std::unordered_map<std::string,std::vector<float>> cache;  // memoize (router weights reused every step)
    auto it=cache.find(n); if(it!=cache.end()) return it->second;
    const auto& t=m.T(n); int d=t.numel(); std::vector<float> o(d); const uint16_t* b=(const uint16_t*)t.data;
    for(int i=0;i<d;++i){ unsigned u=(unsigned)b[i]<<16; memcpy(&o[i],&u,4);} return cache.emplace(n,std::move(o)).first->second; }

// expert FFN for a gathered set of |T| tokens at expert e: out += down(gelu(gate(Xe))*up(Xe)) * w
static void expert_ffn(Model& m,const std::string& EP,float* Xe,int nt,float* moe_out,const int* didx,const float* dw){
    float *g,*u,*dn; CU(cudaMalloc(&g,(size_t)nt*MOE_INT*4)); CU(cudaMalloc(&u,(size_t)nt*MOE_INT*4)); CU(cudaMalloc(&dn,(size_t)nt*H*4));
    linear(m, g, Xe, EP+"gate_proj", nt, MOE_INT, H);
    linear(m, u, Xe, EP+"up_proj",   nt, MOE_INT, H);
    gelu_tanh(g, g, nt*MOE_INT, 0); k_mul<<<(nt*MOE_INT+255)/256,256>>>(g,u,nt*MOE_INT);
    linear(m, dn, g, EP+"down_proj", nt, H, MOE_INT);
    k_scatter_add<<<(nt*H+255)/256,256>>>(moe_out, dn, didx, dw, nt, H);
    CU(cudaDeviceSynchronize()); cudaFree(g);cudaFree(u);cudaFree(dn);
}

static void moe(Model& m, float* h, int seq, int L){
    std::string P="model.language_model.layers."+std::to_string(L)+".";
    bool big=seq>MAXM; std::vector<float*> tf;
    auto pick=[&](float* d,size_t n)->float*{ if(!big)return d; float* p; CU(cudaMalloc(&p,n*4)); tf.push_back(p); return p; };
    float *resid=pick(DS->resid,seq*H),*mi=pick(DS->mi,seq*H),*g=pick(DS->g,seq*MLP_INT),*u=pick(DS->u,seq*MLP_INT),
          *hs1=pick(DS->hs1,seq*H),*x2=pick(DS->x2,seq*H),*moe_out=pick(DS->moe_out,seq*H),
          *scores=pick(DS->scores,seq*128),*top8_w=pick(DS->top8_w,seq*8);
    int* top8_ids; __half *hbuf,*x2_16;
    if(big){CU(cudaMalloc(&top8_ids,seq*8*4)); CU(cudaMalloc(&hbuf,(size_t)seq*8*MOE_INT*sizeof(__half))); CU(cudaMalloc(&x2_16,(size_t)seq*H*sizeof(__half)));}
    else { top8_ids=DS->top8_ids; hbuf=DS->hbuf; x2_16=DS->x2_16; }
    CU(cudaMemcpyAsync(resid,h,(size_t)seq*H*4,cudaMemcpyDeviceToDevice));
    rmsnorm(mi, resid, m.dptr<const uint16_t*>(P+"pre_feedforward_layernorm.weight"), seq, H, EPS, 0);
    linear(m,g,mi,P+"mlp.gate_proj",seq,MLP_INT,H); linear(m,u,mi,P+"mlp.up_proj",seq,MLP_INT,H);
    gelu_tanh(g,g,seq*MLP_INT,0); k_mul<<<(seq*MLP_INT+255)/256,256>>>(g,u,seq*MLP_INT);
    linear(m,hs1,g,P+"mlp.down_proj",seq,H,MLP_INT);
    rmsnorm(hs1,hs1,m.dptr<const uint16_t*>(P+"post_feedforward_layernorm_1.weight"),seq,H,EPS,0);
    k_router_hn<<<seq,256>>>(mi, resid, m.dptr<const uint16_t*>(P+"router.scale"), seq, H);  // reuse mi as hn buffer
    k_router_scores<<<(unsigned)((seq*NEXP+7)/8),256>>>(scores, mi, m.dptr<const uint16_t*>(P+"router.proj.weight"), seq, H, NEXP);
    k_router_top8<<<seq,128>>>(top8_ids, top8_w, scores, m.dptr<const uint16_t*>(P+"router.per_expert_scale"), NEXP, TOPK);
    rmsnorm(x2,resid,m.dptr<const uint16_t*>(P+"pre_feedforward_layernorm_2.weight"),seq,H,EPS,0);
    if(W4A4_TAPS) k_fp4_roundtrip<<<(unsigned)(((long)seq*H/16+255)/256),256>>>(x2,(long)seq*H);  // W4A4 emul (MoE act)
    k_f32_to_f16<<<(seq*H+255)/256,256>>>(x2_16, x2, seq*H);  // fp16 MoE activation for HW-decode gateup
    ExpertPtrs* ep=m.experts(L);
    if(seq>1 && seq<=16){   // verify: GROUPED gateup (reuse each expert weight across tokens routing to it; EL=16 cap)
        CU(cudaMemsetAsync(DS->ecount,0,128*4,0));
        k_moe_invert<<<(seq*8+255)/256,256>>>(DS->ecount, DS->elist, top8_ids, seq*8, 16);
        k_moe_gateup_grouped<<<(unsigned)((128*MOE_INT+7)/8),256>>>(hbuf, x2_16, DS->ecount, DS->elist, 16, ep->gp,ep->gs,ep->gg, ep->up,ep->us,ep->ug, H, MOE_INT);
        { static bool spm=getenv("SPARSITY")!=nullptr; static int spc=0;
          if(spm && spc<3){ spc++; cudaDeviceSynchronize();
            measure_sparsity(x2_16, seq, H, "x2 (gate/up input)");
            measure_sparsity(hbuf, seq*8, MOE_INT, "hbuf (down input=silu*up)"); } }
    } else
    k_moe_gateup<<<(unsigned)((seq*TOPK*MOE_INT+7)/8),256>>>(hbuf, x2_16, top8_ids, ep->gp,ep->gs,ep->gg, ep->up,ep->us,ep->ug, seq, H, MOE_INT);
    if(seq>1 && seq<=16){   // BANDWIDTH-OPTIMAL down: weight-resident (reuse each expert weight across its tokens) + per-assignment partials (no atomics) + cheap finalize
        k_moe_down_bw<<<(unsigned)((128*(H/2)+7)/8),256>>>(DS->dpart, hbuf, DS->ecount, DS->elist, 16, ep->dp,ep->ds,ep->dg, H, MOE_INT);
        k_moe_finalize<<<(unsigned)((seq*H+255)/256),256>>>(moe_out, DS->dpart, top8_w, seq, H);
    } else
    k_moe_down<<<(unsigned)((seq*(H/2)+7)/8),256>>>(moe_out, hbuf, top8_ids, top8_w, ep->dp,ep->ds,ep->dg, seq, H, MOE_INT);
    rmsnorm(moe_out,moe_out,m.dptr<const uint16_t*>(P+"post_feedforward_layernorm_2.weight"),seq,H,EPS,0);
    add_inplace(hs1, moe_out, seq*H, 0);
    rmsnorm(hs1, hs1, m.dptr<const uint16_t*>(P+"post_feedforward_layernorm.weight"), seq, H, EPS, 0);
    add_inplace(resid, hs1, seq*H, 0);
    CU(cudaMemcpyAsync(h, resid, (size_t)seq*H*4, cudaMemcpyDeviceToDevice));
    k_scale_const<<<(seq*H+255)/256,256>>>(h, m.scalarBf16(P+"layer_scalar"), seq*H);  // no sync (stream-0 ordered)
    if(big){ CU(cudaDeviceSynchronize()); for(auto p:tf)cudaFree(p); cudaFree(top8_ids); cudaFree(hbuf); cudaFree(x2_16); }
}

static bool DUMP=false;
static void dump_h(float* h,int seq,int idx){
    if(!DUMP)return; std::vector<float> hh((size_t)seq*H); CU(cudaMemcpy(hh.data(),h,(size_t)seq*H*4,cudaMemcpyDeviceToHost));
    char p[128]; snprintf(p,128,"/tmp/cudahs/hs_%d.bin",idx); FILE* f=fopen(p,"wb"); fwrite(hh.data(),4,hh.size(),f); fclose(f);
}
static const char* EMBN="model.language_model.embed_tokens.weight";
// forward a block of mtok tokens at [base,base+mtok); updates KV; fills lo(logits of last)/allarg(per-pos argmax)/ftok(argmax) as requested.
// Extracted from main()'s run-lambda VERBATIM so the benchmark and the server share one code path (Phase 1b engine refactor).
static void forward_block(Model& m, Session& S, const std::vector<int>& nids,int base,std::vector<float>& lo,float* taps,std::vector<int>* allarg,int* ftok){
    int mtok=nids.size(); bool big=mtok>MAXM;
    int* dids; float* h;
    if(big){ CU(cudaMalloc(&dids,mtok*4)); CU(cudaMalloc(&h,(size_t)mtok*H*4)); } else { dids=DS->dids; h=DS->hmain; }
    CU(cudaMemcpyAsync(dids,nids.data(),mtok*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpyToSymbol(g_base,&base,sizeof(int)));
    k_embed<<<mtok,256>>>(h, m.dptr<const uint16_t*>(EMBN), dids, mtok, H, EMB_SCALE);
    static double t_layers=0,t_head=0; bool prof=getenv("PROF");
    cudaEvent_t a,b,c; if(prof){cudaEventCreate(&a);cudaEventCreate(&b);cudaEventCreate(&c);cudaDeviceSynchronize();cudaEventRecord(a);}
    for(int L=0;L<NLAYER;++L){ attention_cached(m,S,h,mtok,base,L); moe(m,h,mtok,L);
        if(taps){ int j=tap_slot(L); if(j>=0) k_tap<<<(mtok*H+255)/256,256>>>(taps,h,mtok,j,H); } }
    if(prof){cudaEventRecord(b);}
    rmsnorm(DS->hl, h+(size_t)(mtok-1)*H, m.dptr<const uint16_t*>("model.language_model.norm.weight"),1,H,EPS,0);
    k_f32_to_f16<<<(H+255)/256,256>>>(DS->xh16, DS->hl, H);
    fp4_gemv(DS->dlog, g_ewp, g_ews, g_egs, DS->xh16, VOCAB, H, 0);
    if(ftok){ k_argmax<<<1,256>>>(DS->darg, DS->dlog, VOCAB); CU(cudaMemcpy(ftok,DS->darg,4,cudaMemcpyDeviceToHost)); }
    else { CU(cudaDeviceSynchronize()); lo.resize(VOCAB); CU(cudaMemcpy(lo.data(),DS->dlog,(size_t)VOCAB*4,cudaMemcpyDeviceToHost)); }
    if(prof){cudaEventRecord(c);cudaEventSynchronize(c);float ab,bc;cudaEventElapsedTime(&ab,a,b);cudaEventElapsedTime(&bc,b,c);t_layers+=ab;t_head+=bc;
        fprintf(stderr,"[prof] layers=%.1fms head=%.1fms (cum layers=%.0f head=%.0f)\n",ab,bc,t_layers,t_head);}
    if(allarg){ allarg->resize(mtok);
        rmsnorm(DS->hln,h,m.dptr<const uint16_t*>("model.language_model.norm.weight"),mtok,H,EPS,0);
        k_lmhead_batched<<<VOCAB,256,(size_t)H*4>>>(DS->lg2,DS->hln,m.dptr<const uint16_t*>(EMBN),H,VOCAB,mtok);
        k_argmax<<<mtok,256>>>(DS->darg,DS->lg2,VOCAB); CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(allarg->data(),DS->darg,mtok*4,cudaMemcpyDeviceToHost)); }
    if(big){ cudaFree(dids);cudaFree(h); }
}
// prefill the prompt: forward + set valid_len + return first generated token (greedy). out_logits optional (for top-5).
static int engine_prefill(Model& m, Session& S, const std::vector<int>& ids, float* taps_ctx, std::vector<float>* out_logits=nullptr){
    std::vector<float> logits; forward_block(m,S,ids,0,logits,taps_ctx,nullptr,nullptr); S.valid_len=ids.size();
    int b=0; for(int i=1;i<VOCAB;++i) if(logits[i]>logits[b])b=i;
    if(out_logits)*out_logits=std::move(logits);
    return b;
}
// DFlash speculative generate. emit(token)->bool (false = caller wants to stop, e.g. client gone). Returns tokens generated.
// stats optional: {steps, accepted_sum} written back. Extracted from main()'s DFlash loop VERBATIM.
static int engine_generate_dflash(Model& m, Session& S, DraftModel* dm, float* taps_ctx, int k,
        int first_tok, int max_new, float typEps, float invT, bool do_sample, unsigned long long seed0,
        const std::function<bool(int)>& emit, int* out_steps=nullptr, int* out_acc=nullptr){
    float* taps_blk; CU(cudaMalloc(&taps_blk,(size_t)(k+1)*6*H*4));
    uint16_t* embed = m.dptr<uint16_t*>(EMBN);
    std::vector<int> draft_ids(k), allarg(k+1), ctxpos; std::vector<float> tprob(k);
    bool typical = typEps>0.f;
    int ngen=0, steps=0, accepted_sum=0, tok=first_tok;
    auto verify_step=[&](){
        k_embed<<<k+1,256>>>(DS->hmain, embed, DS->dids, k+1, H, EMB_SCALE);
        float* h=DS->hmain;
        for(int L=0;L<NLAYER;++L){ attention_cached(m,S,h,k+1,0,L); moe(m,h,k+1,L);
            int j=tap_slot(L); if(j>=0) k_tap<<<((k+1)*H+255)/256,256>>>(taps_blk,h,k+1,j,H); }
        rmsnorm(DS->hln, h, m.dptr<const uint16_t*>("model.language_model.norm.weight"), k+1, H, EPS, 0);
        k_f32_to_f16<<<((k+1)*H+255)/256,256>>>(DS->xh16, DS->hln, (k+1)*H);
        tc_w4a16_gemm(DS->lg2, g_ewp, g_ews, g_egs, DS->xh16, k+1, VOCAB, H, 0);
        if(do_sample) k_sample<<<k+1,256>>>(DS->darg, DS->lg2, VOCAB, invT);   // lossless temperature sampling (Gumbel-max); reads g_sample_seed
        else k_argmax<<<k+1,256>>>(DS->darg, DS->lg2, VOCAB);
        if(typical) k_typical_prob<<<k,256>>>(DS->tprob, DS->lg2, DS->dids, VOCAB, invT);
    };
    bool vUseGraph=!getenv("NOGRAPH"); bool vCap=false; cudaGraph_t vg; cudaGraphExec_t vexec;
    bool stop=false;
    while(ngen<max_new && !stop){
        ctxpos.resize(S.valid_len); for(int i=0;i<S.valid_len;++i) ctxpos[i]=i;
        draft_propose(dm, taps_ctx, ctxpos.data(), S.valid_len, tok, embed, embed, g_ewp, g_ews, g_egs, draft_ids.data(), k);
        std::vector<int> blk; blk.reserve(k+1); blk.push_back(tok); for(int i=0;i<k;++i) blk.push_back(draft_ids[i]);
        CU(cudaMemcpyAsync(DS->dids, blk.data(), (k+1)*4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CU(cudaMemcpyToSymbolAsync(g_base, &S.valid_len, sizeof(int), 0, cudaMemcpyHostToDevice, cudaStreamPerThread));
        if(do_sample){ unsigned long long ss=seed0 + (unsigned long long)steps*0x100000001B3ULL;
            CU(cudaMemcpyToSymbolAsync(g_sample_seed, &ss, 8, 0, cudaMemcpyHostToDevice, cudaStreamPerThread)); }
        if(vUseGraph){
            if(!vCap){ verify_step(); CU(cudaStreamSynchronize(cudaStreamPerThread));
                CU(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)); verify_step();
                CU(cudaStreamEndCapture(cudaStreamPerThread,&vg)); CU(cudaGraphInstantiate(&vexec,vg,0)); vCap=true; }
            else CU(cudaGraphLaunch(vexec,cudaStreamPerThread));
        } else verify_step();
        CU(cudaMemcpyAsync(allarg.data(), DS->darg, (k+1)*4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        if(typical) CU(cudaMemcpyAsync(tprob.data(), DS->tprob, k*4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CU(cudaStreamSynchronize(cudaStreamPerThread));
        int na=0; if(typical){ for(int i=0;i<k;++i){ if(tprob[i]>=typEps) na++; else break; } }
                  else       { for(int i=0;i<k;++i){ if(draft_ids[i]==allarg[i]) na++; else break; } }
        int newbonus = allarg[na];
        if(!emit(tok)){ stop=true; } ngen++;
        if(!stop) stop = (tok==1||tok==106);
        for(int i=0;i<na && ngen<max_new && !stop;++i){ if(!emit(draft_ids[i])){stop=true;break;} ngen++; stop=(draft_ids[i]==1||draft_ids[i]==106); }
        CU(cudaMemcpy(taps_ctx+(size_t)S.valid_len*6*H, taps_blk, (size_t)(na+1)*6*H*4, cudaMemcpyDeviceToDevice));
        S.valid_len += 1+na; tok = newbonus; steps++; accepted_sum += na;
    }
    if(vCap){ cudaGraphExecDestroy(vexec); cudaGraphDestroy(vg); }
    cudaFree(taps_blk);
    if(out_steps)*out_steps=steps; if(out_acc)*out_acc=accepted_sum;
    return ngen;
}

// ---- Phase 2: prefix caching. Persist the token-ids in the KV; reuse the longest common prefix across requests. ----
static std::vector<int> g_cached_ids;   // tokens currently valid in S->Kc/Vc (+taps_ctx); LCP-matched per request
// Prefill with prefix reuse: LCP with the cache -> reprocess only the new suffix at base=p (KV[0:p) reused, bit-exact).
static int engine_prefill_cached(Model& m, Session& S, const std::vector<int>& ids, float* taps_ctx, int* reused=nullptr){
    int p=0, cmax=std::min((int)ids.size(),(int)g_cached_ids.size());
    while(p<cmax && ids[p]==g_cached_ids[p]) ++p;
    if(p>=(int)ids.size()) p=(int)ids.size()-1;   // must reprocess >=1 token for its next-token logits
    if(p<0) p=0;
    if(reused)*reused=p;
    std::vector<int> suffix(ids.begin()+p, ids.end());
    std::vector<float> logits;
    forward_block(m,S,suffix,p,logits,taps_ctx+(size_t)p*6*H,nullptr,nullptr);   // g_base=p -> KV/taps written at [p,len)
    S.valid_len=(int)ids.size();
    g_cached_ids=ids;
    int b=0; for(int i=1;i<VOCAB;++i) if(logits[i]>logits[b])b=i; return b;
}
// ---- Phase 1c: OpenAI-compatible HTTP server (pure C++ single binary, cpp-httplib) ----
static std::mutex g_serve_mtx;   // serialize GPU access: one request at a time (single-instance engine)
static std::string sse_role(const std::string& cid){
    nlohmann::json j={{"id",cid},{"object","chat.completion.chunk"},{"model","gemma-4-26b-dflash"},
        {"choices",{{{"index",0},{"delta",{{"role","assistant"}}},{"finish_reason",nullptr}}}}};
    return "data: "+j.dump()+"\n\n"; }
static std::string sse_delta(const std::string& cid,const std::string& d){
    nlohmann::json j={{"id",cid},{"object","chat.completion.chunk"},{"model","gemma-4-26b-dflash"},
        {"choices",{{{"index",0},{"delta",{{"content",d}}},{"finish_reason",nullptr}}}}};
    return "data: "+j.dump()+"\n\n"; }
static std::string sse_reason(const std::string& cid,const std::string& d){   // reasoning_content delta (vLLM/DeepSeek convention)
    nlohmann::json j={{"id",cid},{"object","chat.completion.chunk"},{"model","gemma-4-26b-dflash"},
        {"choices",{{{"index",0},{"delta",{{"reasoning_content",d}}},{"finish_reason",nullptr}}}}};
    return "data: "+j.dump()+"\n\n"; }
// routes generated tokens to content vs reasoning via gemma-4 channel state (<|channel>thought..<channel|>answer),
// decoding each stream separately for correct UTF-8. Also strips the channel-name + control tokens.
struct ChanRouter {
    Tokenizer* tk; int state=0;   // 0=bare content, 1=reading channel-name, 2=thought(reasoning), 3=named channel (content)
    std::vector<int> cid_, rid_, tid_, nid_; std::string cprev, rprev; bool tstate=false;
    std::vector<std::string> tools_raw;   // captured "call:name{args}" tool-call strings
    bool feed(int t, std::string& cd, std::string& rd){   // false => stop token
        cd.clear(); rd.clear();
        if(t==tk->chan_start){ state=1; nid_.clear(); return true; }
        if(state==1){ if(t==107){ std::string nm=tk->decode(nid_); state=(nm.find("thought")!=std::string::npos||nm.find("analysis")!=std::string::npos)?2:3; } else nid_.push_back(t); return true; }
        if(t==tk->chan_end){ state=0; return true; }             // channel closed -> back to bare content
        if(t==tk->tcall_start){ tstate=true; tid_.clear(); return true; }   // <|tool_call>
        if(tstate){ if(t==tk->tcall_end){ tools_raw.push_back(tk->decode(tid_)); tstate=false; } else tid_.push_back(t); return true; }
        if(tk->is_stop(t)) return false;
        if(state==2){ rid_.push_back(t); std::string f=tk->decode(rid_); rd=f.substr(rprev.size()); rprev=f; }   // thought -> reasoning_content
        else        { cid_.push_back(t); std::string f=tk->decode(cid_); cd=f.substr(cprev.size()); cprev=f; }   // final/bare -> content
        return true;
    }
    std::string content(){ return tk->decode(cid_); }
    std::string reasoning(){ return tk->decode(rid_); }
    int n_content(){ return (int)cid_.size(); }
    // gemma flat args {k:v,k2:3} -> JSON {"k":"v","k2":3} (best-effort; nested values kept as string)
    static std::string args_json(const std::string& g){
        if(g.size()<2||g[0]!='{') return "{}";
        std::string b=g.substr(1,g.size()-2); std::vector<std::string> parts; std::string cur; int dep=0;
        for(char c:b){ if(c=='{'||c=='[')dep++; else if(c=='}'||c==']')dep--; if(c==','&&dep==0){parts.push_back(cur);cur.clear();} else cur+=c; }
        if(!cur.empty())parts.push_back(cur);
        auto trim=[](std::string s){ size_t a=s.find_first_not_of(" \t"); if(a==std::string::npos)return std::string(); size_t z=s.find_last_not_of(" \t"); std::string r=s.substr(a,z-a+1);
            if(r.size()>=2&&r.front()=='"'&&r.back()=='"')r=r.substr(1,r.size()-2); return r; };
        nlohmann::json o=nlohmann::json::object();
        for(auto& p:parts){ size_t co=p.find(':'); if(co==std::string::npos)continue; std::string k=trim(p.substr(0,co)),v=trim(p.substr(co+1));
            if(v=="true")o[k]=true; else if(v=="false")o[k]=false;
            else { char* e; double d=strtod(v.c_str(),&e); if(!v.empty()&&*e==0)o[k]=d; else o[k]=v; } }
        return o.dump();
    }
    nlohmann::json tool_calls(){   // gemma "call:NAME{args}" -> OpenAI tool_calls (arguments = JSON string)
        nlohmann::json arr=nlohmann::json::array(); int i=0;
        for(auto& raw : tools_raw){ size_t b=raw.find('{'); std::string pre=(b==std::string::npos)?raw:raw.substr(0,b);
            if(pre.rfind("call:",0)==0) pre=pre.substr(5);
            std::string name=pre, args=(b==std::string::npos)?"{}":args_json(raw.substr(b));
            arr.push_back({{"id","call_"+std::to_string(i++)},{"type","function"},{"function",{{"name",name},{"arguments",args}}}}); }
        return arr;
    }
};
static std::string sse_stop(const std::string& cid){
    nlohmann::json j={{"id",cid},{"object","chat.completion.chunk"},{"model","gemma-4-26b-dflash"},
        {"choices",{{{"index",0},{"delta",nlohmann::json::object()},{"finish_reason","stop"}}}}};
    return "data: "+j.dump()+"\n\ndata: [DONE]\n\n"; }
static std::vector<std::pair<std::string,std::string>> parse_msgs(const nlohmann::json& j){
    std::vector<std::pair<std::string,std::string>> msgs;
    if(j.contains("messages")) for(auto& mm:j["messages"]){ std::string role=mm.value("role","user"), content;
        if(mm.contains("content")){ if(mm["content"].is_string()) content=mm["content"].get<std::string>();
            else if(mm["content"].is_array()) for(auto& p:mm["content"]) if(p.contains("text")) content+=p["text"].get<std::string>(); }
        msgs.push_back({role,content}); }
    return msgs; }
static void run_server(const std::string& ckpt, int port){
    CAP = getenv("CTX")?atoi(getenv("CTX")):65536;                  // server default 64K context (CTX to override)
    g_fp8kv = getenv("FP8KV")?(atoi(getenv("FP8KV"))!=0):true;      // FP8 KV default ON for the server (FP8KV=0 to disable)
    Model m(ckpt);
    printf("[server] CAP(context)=%d  fp8_kv=%d\n", CAP, (int)g_fp8kv);
    SC=new Scratch(); DS=new DScratch(); init_clut(); Session* S=new Session();
    { const uint16_t* embp=m.dptr<const uint16_t*>(EMBN);   // quantize tied embed -> NVFP4 lm_head (same as benchmark setup)
      CU(cudaMalloc(&g_ewp,(size_t)VOCAB*(H/2))); CU(cudaMalloc(&g_ews,(size_t)VOCAB*(H/16)));
      float* d_amax; CU(cudaMalloc(&d_amax,4)); CU(cudaMemset(d_amax,0,4));
      k_embed_amax<<<1024,256>>>(embp,(long)VOCAB*H,d_amax);
      float h_amax; CU(cudaMemcpy(&h_amax,d_amax,4,cudaMemcpyDeviceToHost)); g_egs=2688.f/h_amax;
      k_quant_embed_fp4<<<(unsigned)(((long)VOCAB*(H/16)+255)/256),256>>>(embp,g_ewp,g_ews,H,VOCAB,g_egs);
      CU(cudaDeviceSynchronize()); CU(cudaFree(d_amax)); }
    int k=getenv("DK")?atoi(getenv("DK")):14; k=std::max(1,std::min(15,k));
    DraftModel* dm=draft_load((std::string(getenv("HOME"))+"/models/gemma-4-26B-A4B-DFlash/model.safetensors").c_str(), CAP+32);
    float* taps_ctx; CU(cudaMalloc(&taps_ctx,(size_t)(CAP+32)*6*H*4));
    Tokenizer tk; { std::string d=ckpt; auto p=d.rfind('/'); tk.load((p==std::string::npos?std::string("."):d.substr(0,p))+"/tokenizer.json"); }
    printf("[server] model+draft+tokenizer loaded (vocab=%zu). OpenAI API on http://0.0.0.0:%d/v1/chat/completions\n",tk.vocab.size(),port); fflush(stdout);
    httplib::Server svr;
    svr.Get("/",[&](const httplib::Request&,httplib::Response& r){ r.set_content(WEBUI_HTML,"text/html"); });
    svr.Get("/v1/models",[&](const httplib::Request&,httplib::Response& r){ r.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"gemma-4-26b-dflash\",\"object\":\"model\"}]}","application/json"); });
    svr.Post("/v1/chat/completions",[&](const httplib::Request& req, httplib::Response& res){
        nlohmann::json j; try{ j=nlohmann::json::parse(req.body); }catch(...){ res.status=400; res.set_content("{\"error\":\"bad json\"}","application/json"); return; }
        auto msgs=parse_msgs(j);
        bool think = j.value("enable_thinking",false) || (j.contains("chat_template_kwargs")&&j["chat_template_kwargs"].value("enable_thinking",false))
                     || (j.contains("reasoning_effort")&&j["reasoning_effort"].is_string()&&j["reasoning_effort"].get<std::string>()!="none");
        std::vector<std::string> tool_decls;   // gemma <|tool>declaration:..<tool|> (reasonable-effort format)
        if(j.contains("tools")&&j["tools"].is_array()) for(auto& t:j["tools"]){ if(!t.contains("function"))continue; auto& f=t["function"];
            std::string d="declaration:"+f.value("name",std::string())+"{description:\""+f.value("description",std::string())+"\"";
            if(f.contains("parameters")) d+=",parameters:"+f["parameters"].dump(); d+="}"; tool_decls.push_back(d); }
        auto ids=tk.chat_prompt(msgs,think,tool_decls);
        int max_tokens=j.value("max_tokens",256); bool stream=j.value("stream",false);
        float temperature=j.value("temperature",0.0f);   // top_p accepted; v1 applies temperature only (Gumbel-max)
        bool do_sample=temperature>0.f; float invT=1.f/(temperature>0.f?temperature:1.f);
        std::random_device rd; unsigned long long seed0=((unsigned long long)rd()<<32)^(unsigned long long)rd();
        std::string cid="chatcmpl-thor";
        if(stream){
            res.set_chunked_content_provider("text/event-stream",[&,ids,max_tokens,cid,do_sample,invT,seed0](size_t, httplib::DataSink& sink){
                std::lock_guard<std::mutex> lk(g_serve_mtx);
                int reused=0; int t0=engine_prefill_cached(m,*S,ids,taps_ctx,&reused);
                fprintf(stderr,"[prefix-cache] reused %d/%zu prompt tokens\n",reused,ids.size());
                std::string r0=sse_role(cid); sink.write(r0.data(),r0.size());
                ChanRouter rt{&tk}; bool alive=true;
                engine_generate_dflash(m,*S,dm,taps_ctx,k,t0,max_tokens,0.f,invT,do_sample,seed0,[&](int t)->bool{
                    std::string cd,rd; if(!rt.feed(t,cd,rd)) return false;
                    if(!rd.empty()){ std::string c=sse_reason(cid,rd); if(!sink.write(c.data(),c.size())){alive=false;return false;} }
                    if(!cd.empty()){ std::string c=sse_delta(cid,cd);  if(!sink.write(c.data(),c.size())){alive=false;return false;} }
                    return true; });
                std::string e=sse_stop(cid); sink.write(e.data(),e.size()); sink.done(); return true; });
        } else {
            std::lock_guard<std::mutex> lk(g_serve_mtx);
            int reused=0; int t0=engine_prefill_cached(m,*S,ids,taps_ctx,&reused);
            fprintf(stderr,"[prefix-cache] reused %d/%zu prompt tokens\n",reused,ids.size());
            ChanRouter rt{&tk}; int st=0,ac=0, ntok=0;
            engine_generate_dflash(m,*S,dm,taps_ctx,k,t0,max_tokens,0.f,invT,do_sample,seed0,[&](int t){ std::string cd,rd; ntok++; return rt.feed(t,cd,rd); },&st,&ac);
            nlohmann::json msg={{"role","assistant"},{"content",rt.content()}};
            std::string reason=rt.reasoning(); if(!reason.empty()) msg["reasoning_content"]=reason;
            const char* finish="stop";
            if(!rt.tools_raw.empty()){ msg["tool_calls"]=rt.tool_calls(); finish="tool_calls"; }
            nlohmann::json out={{"id",cid},{"object","chat.completion"},{"model","gemma-4-26b-dflash"},
                {"choices",{{{"index",0},{"message",msg},{"finish_reason",finish}}}},
                {"usage",{{"prompt_tokens",(int)ids.size()},{"completion_tokens",ntok},{"total_tokens",(int)ids.size()+ntok}}}};
            res.set_content(out.dump(),"application/json");
        }
    });
    svr.listen("0.0.0.0",port);
}

int main(int argc,char**argv){
    if(getenv("DUMP")) DUMP=true; system("mkdir -p /tmp/cudahs");
    std::string ckpt = argc>1?argv[1]:std::string(getenv("HOME"))+"/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors";
    std::string tokfile = argc>2?argv[2]:"/tmp/tokens.txt";
    if(getenv("CTX")) CAP=atoi(getenv("CTX"));
    W4A4_TAPS=getenv("W4A4_TAPS")!=nullptr;
    if(getenv("SERVE")){ int port=getenv("PORT")?atoi(getenv("PORT")):8080; run_server(ckpt,port); return 0; }
    Model m(ckpt); SC=new Scratch(); DS=new DScratch(); init_clut(); Session* S=new Session();
    {   // quantize the bf16 embed -> NVFP4 (E2M1 + e4m3 group-16 scales) for a 4x-lighter lm_head GEMV
        const uint16_t* embp=m.dptr<const uint16_t*>("model.language_model.embed_tokens.weight");
        CU(cudaMalloc(&g_ewp,(size_t)VOCAB*(H/2))); CU(cudaMalloc(&g_ews,(size_t)VOCAB*(H/16)));
        float* d_amax; CU(cudaMalloc(&d_amax,4)); CU(cudaMemset(d_amax,0,4));
        k_embed_amax<<<1024,256>>>(embp,(long)VOCAB*H,d_amax);
        float h_amax; CU(cudaMemcpy(&h_amax,d_amax,4,cudaMemcpyDeviceToHost)); g_egs=2688.f/h_amax;
        k_quant_embed_fp4<<<(unsigned)(((long)VOCAB*(H/16)+255)/256),256>>>(embp,g_ewp,g_ews,H,VOCAB,g_egs);
        CU(cudaDeviceSynchronize()); CU(cudaFree(d_amax)); }
    // read token ids
    std::vector<int> ids; { FILE* f=fopen(tokfile.c_str(),"r"); int t; while(fscanf(f,"%d",&t)==1)ids.push_back(t); fclose(f);}
    int seq=ids.size(); printf("seq=%d tokens, CAP=%d\n",seq,CAP);
    int NGEN = getenv("GEN")? atoi(getenv("GEN")) : 0;
    const char* embn="model.language_model.embed_tokens.weight";
    // run mtok tokens at positions [base, base+mtok); fill logits_out for the LAST token; updates KV cache
    // per-position argmax (verify): for each of the mtok positions, target's greedy next-token
    auto run=[&](const std::vector<int>& nids,int base,std::vector<float>& lo,float* taps=nullptr,std::vector<int>* allarg=nullptr,int* ftok=nullptr){
        int mtok=nids.size(); bool big=mtok>MAXM;
        int* dids; float* h;
        if(big){ CU(cudaMalloc(&dids,mtok*4)); CU(cudaMalloc(&h,(size_t)mtok*H*4)); } else { dids=DS->dids; h=DS->hmain; }
        CU(cudaMemcpyAsync(dids,nids.data(),mtok*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpyToSymbol(g_base,&base,sizeof(int)));   // set decode position read by base-dependent kernels
        k_embed<<<mtok,256>>>(h, m.dptr<const uint16_t*>(embn), dids, mtok, H, EMB_SCALE);
        static double t_layers=0,t_head=0; bool prof=getenv("PROF");
        cudaEvent_t a,b,c; if(prof){cudaEventCreate(&a);cudaEventCreate(&b);cudaEventCreate(&c);cudaDeviceSynchronize();cudaEventRecord(a);}
        for(int L=0;L<NLAYER;++L){ attention_cached(m,*S,h,mtok,base,L); moe(m,h,mtok,L);
            if(taps){ int j=tap_slot(L); if(j>=0) k_tap<<<(mtok*H+255)/256,256>>>(taps,h,mtok,j,H); } }
        if(prof){cudaEventRecord(b);}
        rmsnorm(DS->hl, h+(size_t)(mtok-1)*H, m.dptr<const uint16_t*>("model.language_model.norm.weight"),1,H,EPS,0);
        k_f32_to_f16<<<(H+255)/256,256>>>(DS->xh16, DS->hl, H);   // NVFP4 lm_head: 4x-lighter embed read via fp4_gemv (no softcap; argmax-invariant)
        fp4_gemv(DS->dlog, g_ewp, g_ews, g_egs, DS->xh16, VOCAB, H, 0);
        if(ftok){ k_argmax<<<1,256>>>(DS->darg, DS->dlog, VOCAB); CU(cudaMemcpy(ftok,DS->darg,4,cudaMemcpyDeviceToHost)); }  // device argmax, copy 1 int (no 1MB host roundtrip)
        else { CU(cudaDeviceSynchronize()); lo.resize(VOCAB); CU(cudaMemcpy(lo.data(),DS->dlog,(size_t)VOCAB*4,cudaMemcpyDeviceToHost)); }
        if(prof){cudaEventRecord(c);cudaEventSynchronize(c);float ab,bc;cudaEventElapsedTime(&ab,a,b);cudaEventElapsedTime(&bc,b,c);t_layers+=ab;t_head+=bc;
            fprintf(stderr,"[prof] layers=%.1fms head=%.1fms (cum layers=%.0f head=%.0f)\n",ab,bc,t_layers,t_head);}
        if(allarg){ allarg->resize(mtok);  // batched: norm all positions -> one batched lm_head -> device argmax
            rmsnorm(DS->hln,h,m.dptr<const uint16_t*>("model.language_model.norm.weight"),mtok,H,EPS,0);
            k_lmhead_batched<<<VOCAB,256,(size_t)H*4>>>(DS->lg2,DS->hln,m.dptr<const uint16_t*>(embn),H,VOCAB,mtok);
            k_argmax<<<mtok,256>>>(DS->darg,DS->lg2,VOCAB); CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(allarg->data(),DS->darg,mtok*4,cudaMemcpyDeviceToHost)); }
        if(big){ cudaFree(dids);cudaFree(h); }
    };
    auto argmax=[&](const std::vector<float>& lg){ int b=0; for(int i=1;i<VOCAB;++i) if(lg[i]>lg[b])b=i; return b; };
    std::vector<float> logits;
    // taps buffer for all committed positions (DFlash context); prefill captures prompt taps
    float* taps_ctx; CU(cudaMalloc(&taps_ctx,(size_t)(CAP+32)*6*H*4));
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    run(ids, 0, logits, taps_ctx); S->valid_len = seq;
    int tok = argmax(logits);
    if(NGEN==0){
        double mx=-1e30; for(float v:logits)mx=std::max(mx,(double)v); double Z=0; for(float v:logits)Z+=exp((double)v-mx);
        std::vector<int> ord(VOCAB); for(int i=0;i<VOCAB;++i)ord[i]=i;
        std::partial_sort(ord.begin(),ord.begin()+5,ord.end(),[&](int a,int b){return logits[a]>logits[b];});
        printf("\ntop-5 next-token  (id : logit : logprob):\n");
        for(int j=0;j<5;++j){ int id=ord[j]; double lp=(double)logits[id]-mx-log(Z); printf("  %6d : %8.4f : %8.4f\n",id,logits[id],lp); }
        return 0;
    }
    int ngen=0;
    if(!getenv("DFLASH")){
        // ---- base incremental decode, CUDA-graph captured (one captured step replays for all positions) ----
        uint16_t* embw=m.dptr<uint16_t*>(embn);
        auto decode_step=[&](){   // mtok=1, reads DS->dids (token) + g_base (pos), writes DS->darg, self-advances
            k_embed<<<1,256>>>(DS->hmain, embw, DS->dids, 1, H, EMB_SCALE);
            float* h=DS->hmain;
            for(int L=0;L<NLAYER;++L){ attention_cached(m,*S,h,1,0,L); moe(m,h,1,L); }
            rmsnorm(DS->hl, h, m.dptr<const uint16_t*>("model.language_model.norm.weight"), 1, H, EPS, 0);
            k_f32_to_f16<<<(H+255)/256,256>>>(DS->xh16, DS->hl, H);
            fp4_gemv(DS->dlog, g_ewp, g_ews, g_egs, DS->xh16, VOCAB, H, 0);
            k_argmax<<<1,256>>>(DS->darg, DS->dlog, VOCAB);
            k_advance<<<1,1>>>(DS->dids, DS->darg);
        };
        bool useGraph = !getenv("NOGRAPH");
        cudaEventRecord(t0);
        int g=0;
        printf("%d ",tok); fflush(stdout); ngen++;   // print tok0; warmup 1 eager step (lazy init + advance)
        if(tok==1||tok==106) useGraph=false;
        else { int nt; run(std::vector<int>{tok}, S->valid_len, logits, nullptr, nullptr, &nt); S->valid_len++; tok=nt; g=1; }
        if(useGraph){
            CU(cudaMemcpy(DS->dids,&tok,4,cudaMemcpyHostToDevice));
            CU(cudaMemcpyToSymbol(g_base,&S->valid_len,sizeof(int)));
            cudaGraph_t graph; cudaGraphExec_t exec;
            CU(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            decode_step();
            CU(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            CU(cudaGraphInstantiate(&exec,graph,0));
            for(; g<NGEN; ++g){
                printf("%d ",tok); fflush(stdout); ngen++;
                if(tok==1||tok==106) break;
                CU(cudaGraphLaunch(exec,cudaStreamPerThread));
                CU(cudaMemcpyAsync(&tok,DS->darg,4,cudaMemcpyDeviceToHost,cudaStreamPerThread));
                CU(cudaStreamSynchronize(cudaStreamPerThread));
                S->valid_len += 1;
            }
            cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
        } else {   // eager fallback (NOGRAPH=1 or EOS)
            for(; g<NGEN; ++g){
                printf("%d ",tok); fflush(stdout); ngen++;
                if(tok==1||tok==106) break;
                int nt; run(std::vector<int>{tok}, S->valid_len, logits, nullptr, nullptr, &nt); S->valid_len++; tok=nt;
            }
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        printf("\n[base decode %d tok in %.1f ms = %.2f tok/s]\n", ngen, ms, ngen*1000.0/ms);
    } else {
        // ---- DFlash speculative decode via the reusable engine_generate_dflash (shared with the server) ----
        int k = getenv("DK")?atoi(getenv("DK")):14; k=std::max(1,std::min(15,k));  // k=14 optimal (k-sweep)
        DraftModel* dm = draft_load((std::string(getenv("HOME"))+"/models/gemma-4-26B-A4B-DFlash/model.safetensors").c_str(), CAP+32);
        float typEps = getenv("TYP_EPS")?atof(getenv("TYP_EPS")):0.f;
        float tempv = getenv("TEMP")?atof(getenv("TEMP")):1.f;
        bool doSamp = getenv("SAMPLE")!=nullptr && tempv>0.f;   // greedy by default (champion); SAMPLE=1 TEMP=0.8 to sample
        float invT = 1.f/(tempv>0.f?tempv:1.f);
        unsigned long long seed0 = getenv("SEED")?strtoull(getenv("SEED"),0,10):1ULL;
        int steps=0, accepted_sum=0;
        cudaEventRecord(t0);
        ngen = engine_generate_dflash(m,*S,dm,taps_ctx,k,tok,NGEN,typEps,invT,doSamp,seed0,
                 [&](int t){ printf("%d ",t); fflush(stdout); return true; }, &steps, &accepted_sum);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        printf("\n[dflash decode %d tok in %.1f ms = %.2f tok/s | %d steps, mean accept %.2f drafts/block (k=%d), tau=%.2f]\n",
               ngen, ms, ngen*1000.0/ms, steps, steps?(double)accepted_sum/steps:0, k, steps?(double)ngen/steps:0);
    }
    return 0;
}
