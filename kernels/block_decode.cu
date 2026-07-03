// block_decode.cu — full-Block M=1 KV-cache decode wrappers. See block_decode.h.
// Prefill-cache runs the normal block for the output AND populates the per-layer KV cache from the same
// attention-input x1 (redundant recompute, one-time, small s). Decode-step is the block at bs=1 with the
// attention swapped to the gated decode step. HC scratch is alloc'd per call for now (Step 2 pre-allocates).
#include "block_decode.h"
#include "hc.h"
#include "mla_attn.h"        // rmsnorm
#include "mla_decode.h"
#include "compressed_decode.h"
#include "moe.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// ================= sliding (ratio==0) =================
void block_prefill_cache(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                         int s, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int bs=s, d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)bs*d*4); post=(decltype(post))dmalloc((size_t)bs*hc*4); comb=(decltype(comb))dmalloc((size_t)bs*hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)bs*d*4); res2=(decltype(res2))dmalloc((size_t)bs*hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,bs,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,bs,d,eps,true,stream);
    mla_cache_kv(kv.win_kv, x1, w.attn, s, stream);           // populate window-KV from the attention input
    mla_forward(sub, x1, w.attn, 1, s, stream);
    hc_post(res2,sub,x,post,comb,bs,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,bs,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,bs,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,bs,stream);
    hc_post(out,sub,res2,post,comb,bs,hc,d,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void block_decode_step(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                       int pos, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)d*4); post=(decltype(post))dmalloc((size_t)hc*4); comb=(decltype(comb))dmalloc((size_t)hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)d*4); res2=(decltype(res2))dmalloc((size_t)hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,1,d,eps,true,stream);
    mla_decode_step(sub, x1, w.attn, kv.win_kv, pos, stream);
    hc_post(res2,sub,x,post,comb,1,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,1,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,1,stream);
    hc_post(out,sub,res2,post,comb,1,hc,d,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}

// ================= compressed (ratio 4 / 128) =================
// Copy the s attention-input rows x1 into the layer's xin history (for future compressor group emits).
__global__ void k_copy(float* dst, const float* src, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) dst[i]=src[i]; }

void cblock_prefill_cache(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                          int s, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int bs=s, d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)bs*d*4); post=(decltype(post))dmalloc((size_t)bs*hc*4); comb=(decltype(comb))dmalloc((size_t)bs*hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)bs*d*4); res2=(decltype(res2))dmalloc((size_t)bs*hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,bs,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,bs,d,eps,true,stream);
    // retain attention-input history + populate KV caches from x1
    k_copy<<<((size_t)s*d+255)/256,256,0,stream>>>(kv.xin, x1, (size_t)s*d);
    if(w.ratio==4) compressed_attn_cache_r4(kv.win_kv, kv.comp_kv, kv.idx_ckv, &kv.T, x1, w.attn, s, w.ratio, eps, stream);
    else           compressed_attn_cache   (kv.win_kv, kv.comp_kv,             &kv.T, x1, w.attn, s, w.ratio, eps, stream);
    compressed_attn_forward(sub, x1, w.attn, s, w.win, w.ratio, eps, stream);
    hc_post(res2,sub,x,post,comb,bs,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,bs,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,bs,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,bs,stream);
    hc_post(out,sub,res2,post,comb,bs,hc,d,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void cblock_decode_step(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                        int pos, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(decltype(x1))dmalloc((size_t)d*4); post=(decltype(post))dmalloc((size_t)hc*4); comb=(decltype(comb))dmalloc((size_t)hc*hc*4);
    sub=(decltype(sub))dmalloc((size_t)d*4); res2=(decltype(res2))dmalloc((size_t)hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,1,d,eps,true,stream);
    k_copy<<<((size_t)d+255)/256,256,0,stream>>>(kv.xin + (size_t)pos*d, x1, (size_t)d);   // store this position's attn input
    if(w.ratio==4) compressed_decode_step_indexer(sub, kv.xin, pos, w.attn, kv.win_kv, kv.comp_kv, kv.idx_ckv, &kv.T, w.ratio, eps, stream);
    else           compressed_decode_step_strided(sub, kv.xin, pos, w.attn, kv.win_kv, kv.comp_kv,             &kv.T, w.ratio, eps, stream);
    hc_post(res2,sub,x,post,comb,1,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,1,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,1,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,1,stream);
    hc_post(out,sub,res2,post,comb,1,hc,d,stream);
    dsync(stream);
    dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}

// ================= M=K VERIFY block steps (spec-decode) =================
void block_verify_step(float* out, const float* x, const int* input_ids, const BlockWeights& w,
                       int pos, int K, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(float*)dmalloc((size_t)K*d*4); post=(float*)dmalloc((size_t)K*hc*4); comb=(float*)dmalloc((size_t)K*hc*hc*4);
    sub=(float*)dmalloc((size_t)K*d*4); res2=(float*)dmalloc((size_t)K*hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,K,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,K,d,eps,true,stream);
    mla_verify_step(sub, x1, w.attn, kv.win_kv, pos, K, stream);
    hc_post(res2,sub,x,post,comb,K,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,K,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,K,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,K,stream);
    hc_post(out,sub,res2,post,comb,K,hc,d,stream);
    dsync(stream); dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
void cblock_verify_step(float* out, const float* x, const int* input_ids, const CompressedBlockWeights& w,
                        int pos, int K, int iters, float eps, LayerKV& kv, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    x1=(float*)dmalloc((size_t)K*d*4); post=(float*)dmalloc((size_t)K*hc*4); comb=(float*)dmalloc((size_t)K*hc*hc*4);
    sub=(float*)dmalloc((size_t)K*d*4); res2=(float*)dmalloc((size_t)K*hc*d*4);
    hc_pre(x1,post,comb,x,w.hc_attn_fn,w.hc_attn_scale,w.hc_attn_base,K,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.attn_norm,K,d,eps,true,stream);
    k_copy<<<((size_t)K*d+255)/256,256,0,stream>>>(kv.xin+(size_t)pos*d, x1, (size_t)K*d);   // store attn-input history
    if(w.ratio==4) compressed_verify_step_indexer(sub, kv.xin, pos, K, w.attn, kv.win_kv, kv.comp_kv, kv.idx_ckv, &kv.T, w.ratio, eps, stream);
    else           compressed_verify_step_strided(sub, kv.xin, pos, K, w.attn, kv.win_kv, kv.comp_kv,             &kv.T, w.ratio, eps, stream);
    hc_post(res2,sub,x,post,comb,K,hc,d,stream);
    hc_pre(x1,post,comb,res2,w.hc_ffn_fn,w.hc_ffn_scale,w.hc_ffn_base,K,hc,d,iters,eps,stream);
    rmsnorm(x1,x1,w.ffn_norm,K,d,eps,true,stream);
    moe_forward(sub,x1,input_ids,w.ffn,K,stream);
    hc_post(out,sub,res2,post,comb,K,hc,d,stream);
    dsync(stream); dfree(x1);dfree(post);dfree(comb);dfree(sub);dfree(res2);
}
