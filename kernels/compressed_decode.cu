// compressed_decode.cu — M=1 KV-cache decode for a compressed (strided, ratio!=4) MLA layer. See header.
#include "compressed_decode.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "compressor.h"
#include "indexer.h"
#include "deepseek_v4.h"
#include "dscratch.h"
#include <vector>
#include <cmath>
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// single-query descending top-k over score[0..T-1]; out[k] = (k-th best t) + offset (or -1 if none left).
// Decode-time: all T cached rows are already causal-valid (t < (pos+1)/ratio == T), so no mask needed here.
__global__ void k_topk_decode(int* out, const float* score, int T, int topk, int offset){
    if(threadIdx.x||blockIdx.x) return;
    extern __shared__ float sh[];
    for(int t=0;t<T;++t) sh[t]=score[t];
    for(int k=0;k<topk;++k){ float best=-1e30f; int bi=-1;
        for(int t=0;t<T;++t) if(sh[t]>best){best=sh[t];bi=t;}
        if(bi>=0) sh[bi]=-1e30f;
        out[k] = (bi<0)? -1 : bi+offset; }
}
__global__ void k_iw_scale(float* y, float sc, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=sc; }

// ---- prefill: fill window-KV + compressed-KV caches ----
void compressed_attn_cache(float* win_kv, float* comp_kv, int* T, const float* x,
                           const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn;
    uint8_t* xq; float* xs;
    xq=(decltype(xq))dmalloc((size_t)s0*DIM); xs=(decltype(xs))dmalloc((size_t)s0*(DIM/128)*4);
    act_quant_fp8(xq, xs, x, s0, DIM, 128, stream);
    fp8_block_gemm(win_kv, xq, xs, a.wkv, a.wkv_s, s0, HEAD_DIM, DIM, stream);
    rmsnorm(win_kv, win_kv, a.kv_norm, s0, HEAD_DIM, eps, true, stream);
    rope_interleaved(win_kv + NOPE_DIM, a.cosT, a.sinT, s0, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(win_kv, s0, NOPE_DIM, 64, HEAD_DIM, stream);
    dsync(stream); dfree(xq); dfree(xs);
    // compressed rows for every COMPLETE group (group g's last token g*ratio+ratio-1 <= s0-1)
    int Tc = 0;
    for(int g=0; g*ratio + ratio - 1 <= s0 - 1; ++g){
        compressor_emit_group(comp_kv + (size_t)g*HEAD_DIM, x, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, false, ROPE_DIM, eps, false, stream);
        ++Tc;
    }
    *T = Tc;
}

// ---- decode step (strided) ----
void compressed_decode_step_strided(float* out, const float* x_full, int pos, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, int* T, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn;
    const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float *cosP = a.cosT + (size_t)pos*half, *sinP = a.sinT + (size_t)pos*half;
    const float* xt = x_full + (size_t)pos*DIM;

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og;
    xq=(decltype(xq))dmalloc(DIM); xs=(decltype(xs))dmalloc((DIM/128)*4);
    qr=(decltype(qr))dmalloc(Q_LORA*4); qrq=(decltype(qrq))dmalloc(Q_LORA); qrs=(decltype(qrs))dmalloc((Q_LORA/128)*4);
    q=(decltype(q))dmalloc(Kd*4); o=(decltype(o))dmalloc(Kd*4); og=(decltype(og))dmalloc(OB*4);
    ogq=(decltype(ogq))dmalloc(OB); ogs=(decltype(ogs))dmalloc((OB/128)*4);

    act_quant_fp8(xq, xs, xt, 1, DIM, 128, stream);
    // q = rope( per-head-rms( wq_b( q_norm( wq_a(x) ) ) ) )
    fp8_block_gemm(qr, xq, xs, a.wq_a, a.wq_a_s, 1, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, a.q_norm, 1, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, a.wq_b, a.wq_b_s, 1, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // window kv new -> win_kv[pos]
    float* kvn = win_kv + (size_t)pos*HEAD_DIM;
    fp8_block_gemm(kvn, xq, xs, a.wkv, a.wkv_s, 1, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, a.kv_norm, 1, HEAD_DIM, eps, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, 1, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(kvn, 1, NOPE_DIM, 64, HEAD_DIM, stream);
    // emit compressed row if this token completes a group
    if((pos+1) % ratio == 0){
        int g = pos / ratio;
        compressor_emit_group(comp_kv + (size_t)(*T)*HEAD_DIM, x_full, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, false, ROPE_DIM, eps, false, stream);
        ++(*T);
    }
    int Tn = *T, nwin = pos+1, ntot = nwin + Tn;
    // kv_all = [win_kv[0..pos] ; comp_kv[0..Tn-1]]  (contiguous for sparse_attn)
    float* kv_all; kv_all=(decltype(kv_all))dmalloc((size_t)ntot*HEAD_DIM*4);
    CU(cudaMemcpyAsync(kv_all, win_kv, (size_t)nwin*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)nwin*HEAD_DIM, comp_kv, (size_t)Tn*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    // combined idxs: window [base..pos] ⊕ compressed [nwin + t] for t<Tn (strided: all t<(pos+1)/ratio == Tn)
    int base = pos - WINDOW + 1; if(base<0) base=0; int wwidth = pos+1-base;
    int tot = wwidth + Tn;
    std::vector<int> comb(tot);
    for(int k=0;k<wwidth;++k) comb[k]=base+k;
    for(int t=0;t<Tn;++t) comb[wwidth+t]=nwin+t;
    int* dcomb; dcomb=(decltype(dcomb))dmalloc((size_t)tot*4);
    CU(cudaMemcpyAsync(dcomb, comb.data(), (size_t)tot*4, cudaMemcpyHostToDevice, stream));
    sparse_attn(o, q, kv_all, a.attn_sink, dcomb, 1, 1, N_HEADS, HEAD_DIM, ntot, tot, scale, stream);
    // de-rotate, grouped o-LoRA, wo_b
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og, o, a.wo_a_fp8, a.wo_a_sc, 1, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, a.wo_a,                1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, 1, DIM, OB, stream);

    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);
    dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kv_all);dfree(dcomb);
}

// ================= ratio-4 (DSA indexer) decode =================
// Prefill cache for a ratio-4 layer: window KV + MAIN compressed KV (overlap) + INDEXER compressor KV
// (overlap+rotate, for scoring). All append-only. Sets *T = complete-group count.
void compressed_attn_cache_r4(float* win_kv, float* comp_kv, float* idx_ckv, int* T, const float* x,
                              const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn; const int idx_hd = w.index_head_dim;
    uint8_t* xq; float* xs;
    xq=(decltype(xq))dmalloc((size_t)s0*DIM); xs=(decltype(xs))dmalloc((size_t)s0*(DIM/128)*4);
    act_quant_fp8(xq, xs, x, s0, DIM, 128, stream);
    fp8_block_gemm(win_kv, xq, xs, a.wkv, a.wkv_s, s0, HEAD_DIM, DIM, stream);
    rmsnorm(win_kv, win_kv, a.kv_norm, s0, HEAD_DIM, eps, true, stream);
    rope_interleaved(win_kv + NOPE_DIM, a.cosT, a.sinT, s0, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(win_kv, s0, NOPE_DIM, 64, HEAD_DIM, stream);
    dsync(stream); dfree(xq); dfree(xs);
    int Tc = 0;
    for(int g=0; g*ratio + ratio - 1 <= s0 - 1; ++g){
        compressor_emit_group(comp_kv + (size_t)g*HEAD_DIM, x, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, true, ROPE_DIM, eps, false, stream);
        compressor_emit_group(idx_ckv + (size_t)g*idx_hd, x, g, ratio, w.idx_c_wkv, w.idx_c_wgate, w.idx_c_ape,
                              w.idx_c_norm, w.cc_cos, w.cc_sin, DIM, idx_hd, true, ROPE_DIM, eps, true, stream);
        ++Tc;
    }
    *T = Tc;
}

void compressed_decode_step_indexer(float* out, const float* x_full, int pos, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, float* idx_ckv, int* T, int ratio,
                                    float eps, cudaStream_t stream){
    const auto& a = w.attn;
    const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const int nH=w.index_n_heads, idx_hd=w.index_head_dim, QD=nH*idx_hd, rd=ROPE_DIM;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    const float wscale = rsqrtf((float)idx_hd) * rsqrtf((float)nH);
    const float *cosP = a.cosT + (size_t)pos*half, *sinP = a.sinT + (size_t)pos*half;
    const float* xt = x_full + (size_t)pos*DIM;

    uint8_t *xq,*qrq,*ogq,*iqrq; float *xs,*qrs,*ogs,*iqrs,*qr,*q,*o,*og,*qidx,*qtmp,*iw,*iscore;
    xq=(decltype(xq))dmalloc(DIM); xs=(decltype(xs))dmalloc((DIM/128)*4);
    qr=(decltype(qr))dmalloc(Q_LORA*4); qrq=(decltype(qrq))dmalloc(Q_LORA); qrs=(decltype(qrs))dmalloc((Q_LORA/128)*4);
    q=(decltype(q))dmalloc(Kd*4); o=(decltype(o))dmalloc(Kd*4); og=(decltype(og))dmalloc(OB*4);
    ogq=(decltype(ogq))dmalloc(OB); ogs=(decltype(ogs))dmalloc((OB/128)*4);
    iqrq=(decltype(iqrq))dmalloc(Q_LORA); iqrs=(decltype(iqrs))dmalloc((Q_LORA/128)*4);
    qidx=(decltype(qidx))dmalloc(QD*4); qtmp=(decltype(qtmp))dmalloc(QD*4); iw=(decltype(iw))dmalloc(nH*4);

    act_quant_fp8(xq, xs, xt, 1, DIM, 128, stream);
    // qr = q_norm(wq_a(x))  (shared by main-q and indexer-q)
    fp8_block_gemm(qr, xq, xs, a.wq_a, a.wq_a_s, 1, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, a.q_norm, 1, Q_LORA, eps, true, stream);
    // main q
    act_quant_fp8(qrq, qrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, a.wq_b, a.wq_b_s, 1, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // window kv new -> win_kv[pos]
    float* kvn = win_kv + (size_t)pos*HEAD_DIM;
    fp8_block_gemm(kvn, xq, xs, a.wkv, a.wkv_s, 1, HEAD_DIM, DIM, stream);
    rmsnorm(kvn, kvn, a.kv_norm, 1, HEAD_DIM, eps, true, stream);
    rope_interleaved(kvn + NOPE_DIM, cosP, sinP, 1, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(kvn, 1, NOPE_DIM, 64, HEAD_DIM, stream);
    // emit compressed rows (main overlap + indexer overlap+rotate) if group completes
    if((pos+1) % ratio == 0){
        int g = pos / ratio; int Tn = *T;
        compressor_emit_group(comp_kv + (size_t)Tn*HEAD_DIM, x_full, g, ratio, w.mc_wkv, w.mc_wgate, w.mc_ape,
                              w.mc_norm, w.cc_cos, w.cc_sin, DIM, HEAD_DIM, true, ROPE_DIM, eps, false, stream);
        compressor_emit_group(idx_ckv + (size_t)Tn*idx_hd, x_full, g, ratio, w.idx_c_wkv, w.idx_c_wgate,
                              w.idx_c_ape, w.idx_c_norm, w.cc_cos, w.cc_sin, DIM, idx_hd, true, ROPE_DIM, eps, true, stream);
        ++(*T);
    }
    int Tn = *T, nwin = pos+1;
    // --- DSA indexer scoring for the single query -> top-k compressed idxs (mirrors indexer_forward, m=1) ---
    act_quant_fp8(iqrq, iqrs, qr, 1, Q_LORA, 128, stream);
    fp8_block_gemm(qidx, iqrq, iqrs, w.idx_wq_b, w.idx_wq_b_s, 1, QD, Q_LORA, stream);
    rope_interleaved(qidx + (idx_hd - rd), cosP, sinP, nH, rd, false, idx_hd, nH, stream);
    hadamard(qtmp, qidx, nH, idx_hd, stream);
    act_quant_fp4sim(qtmp, nH, idx_hd, 32, idx_hd, stream);
    gemm_fp32(iw, xt, w.idx_weights_proj, 1, nH, DIM, stream);
    k_iw_scale<<<(nH+63)/64,64,0,stream>>>(iw, wscale, nH);
    iscore=(decltype(iscore))dmalloc((size_t)Tn*4);
    index_score(iscore, qtmp, idx_ckv, iw, 1, Tn, nH, idx_hd, stream);
    int topk = w.index_topk < Tn ? w.index_topk : Tn;
    int* dtop; dtop=(decltype(dtop))dmalloc((size_t)topk*4);
    k_topk_decode<<<1,32,(size_t)Tn*4,stream>>>(dtop, iscore, Tn, topk, nwin);
    // --- kv_all = [win_kv[0..pos] ; comp_kv[0..Tn-1]] ---
    int ntot = nwin + Tn;
    float* kv_all; kv_all=(decltype(kv_all))dmalloc((size_t)ntot*HEAD_DIM*4);
    CU(cudaMemcpyAsync(kv_all, win_kv, (size_t)nwin*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)nwin*HEAD_DIM, comp_kv, (size_t)Tn*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    // --- combined idxs: window [base..pos] ⊕ indexer topk ---
    int base = pos - WINDOW + 1; if(base<0) base=0; int wwidth = pos+1-base;
    int tot = wwidth + topk;
    std::vector<int> hwin(wwidth); for(int k=0;k<wwidth;++k) hwin[k]=base+k;
    int* comb; comb=(decltype(comb))dmalloc((size_t)tot*4);
    CU(cudaMemcpyAsync(comb, hwin.data(), (size_t)wwidth*4, cudaMemcpyHostToDevice, stream));
    CU(cudaMemcpyAsync(comb + wwidth, dtop, (size_t)topk*4, cudaMemcpyDeviceToDevice, stream));
    sparse_attn(o, q, kv_all, a.attn_sink, comb, 1, 1, N_HEADS, HEAD_DIM, ntot, tot, scale, stream);
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og, o, a.wo_a_fp8, a.wo_a_sc, 1, O_GROUPS, O_LORA, GKd, stream);
    else              ogroup_gemm    (og, o, a.wo_a,                1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, 1, DIM, OB, stream);

    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);
    dfree(ogq);dfree(ogs);dfree(iqrq);dfree(iqrs);dfree(qidx);dfree(qtmp);dfree(iw);
    dfree(iscore);dfree(dtop);dfree(kv_all);dfree(comb);
}

// ================= M=K VERIFY steps (spec-decode) =================
// Process K tokens at [pos..pos+K-1] in ONE forward: GEMMs at M=K (weights read once), compressor emits any
// groups completing in the block, per-query combined idxs (window ⊕ compressed). ≡ K sequential decode steps.
static void build_qKV(const CompressedAttnWeights& w, const float* xK, int K, int pos, float* qOut, float* win_kv,
                      float eps, cudaStream_t stream){
    const auto& a=w.attn; const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM;
    const float *cosP=a.cosT+(size_t)pos*half, *sinP=a.sinT+(size_t)pos*half;
    uint8_t *xq,*qrq; float *xs,*qrs,*qr;
    xq=(uint8_t*)dmalloc((size_t)K*DIM); xs=(float*)dmalloc((size_t)K*(DIM/128)*4);
    qr=(float*)dmalloc((size_t)K*Q_LORA*4); qrq=(uint8_t*)dmalloc((size_t)K*Q_LORA); qrs=(float*)dmalloc((size_t)K*(Q_LORA/128)*4);
    act_quant_fp8(xq,xs,xK,K,DIM,128,stream);
    fp8_block_gemm(qr,xq,xs,a.wq_a,a.wq_a_s,K,Q_LORA,DIM,stream);
    rmsnorm(qr,qr,a.q_norm,K,Q_LORA,eps,true,stream);
    act_quant_fp8(qrq,qrs,qr,K,Q_LORA,128,stream);
    fp8_block_gemm(qOut,qrq,qrs,a.wq_b,a.wq_b_s,K,Kd,Q_LORA,stream);
    rmsnorm(qOut,qOut,nullptr,K*N_HEADS,HEAD_DIM,eps,false,stream);
    rope_interleaved(qOut+NOPE_DIM,cosP,sinP,K*N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,stream);
    float* kvn=win_kv+(size_t)pos*HEAD_DIM;
    fp8_block_gemm(kvn,xq,xs,a.wkv,a.wkv_s,K,HEAD_DIM,DIM,stream);
    rmsnorm(kvn,kvn,a.kv_norm,K,HEAD_DIM,eps,true,stream);
    rope_interleaved(kvn+NOPE_DIM,cosP,sinP,K,ROPE_DIM,false,HEAD_DIM,1,stream);
    act_quant_fp8sim(kvn,K,NOPE_DIM,64,HEAD_DIM,stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);
}
static void finish_attn(const CompressedAttnWeights& w, const float* q, const float* kv_all, const int* comb,
                        int K, int pos, int ntot, int topk, float* out, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const float *cosP=a.cosT+(size_t)pos*half, *sinP=a.sinT+(size_t)pos*half; const float scale=1.f/sqrtf((float)HEAD_DIM);
    float *o,*og; uint8_t *ogq; float *ogs;
    o=(float*)dmalloc((size_t)K*Kd*4); og=(float*)dmalloc((size_t)K*OB*4); ogq=(uint8_t*)dmalloc((size_t)K*OB); ogs=(float*)dmalloc((size_t)K*(OB/128)*4);
    sparse_attn(o,q,kv_all,a.attn_sink,comb,1,K,N_HEADS,HEAD_DIM,ntot,topk,scale,stream);
    rope_interleaved(o+NOPE_DIM,cosP,sinP,K*N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og,o,a.wo_a_fp8,a.wo_a_sc,K,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,a.wo_a,               K,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,K,OB,128,stream);
    fp8_block_gemm(out,ogq,ogs,a.wo_b,a.wo_b_s,K,DIM,OB,stream);
    dfree(o);dfree(og);dfree(ogq);dfree(ogs);
}

void compressed_verify_step_strided(float* out, const float* x_full, int pos, int K, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, int* T, int ratio, float eps, cudaStream_t stream){
    const int Kd=N_HEADS*HEAD_DIM;
    float* q; q=(float*)dmalloc((size_t)K*Kd*4);
    build_qKV(w, x_full+(size_t)pos*DIM, K, pos, q, win_kv, eps, stream);
    for(int j=pos;j<pos+K;++j) if((j+1)%ratio==0){                         // emit groups completing in the block
        compressor_emit_group(comp_kv+(size_t)(*T)*HEAD_DIM, x_full, j/ratio, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,
                              w.mc_norm,w.cc_cos,w.cc_sin, DIM,HEAD_DIM,false,ROPE_DIM,eps,false,stream); ++(*T); }
    int Tf=*T, nwin=pos+K, ntot=nwin+Tf;
    float* kv_all; kv_all=(float*)dmalloc((size_t)ntot*HEAD_DIM*4);
    CU(cudaMemcpyAsync(kv_all,win_kv,(size_t)nwin*HEAD_DIM*4,cudaMemcpyDeviceToDevice,stream));
    CU(cudaMemcpyAsync(kv_all+(size_t)nwin*HEAD_DIM,comp_kv,(size_t)Tf*HEAD_DIM*4,cudaMemcpyDeviceToDevice,stream));
    int wmax=0,tmax=0; for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;if(wid>wmax)wmax=wid;int Ti=(ig+1)/ratio;if(Ti>tmax)tmax=Ti;}
    int topk=wmax+tmax; std::vector<int> comb((size_t)K*topk,-1);
    for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;int Ti=(ig+1)/ratio;
        for(int k=0;k<wid;++k) comb[(size_t)i*topk+k]=b+k;
        for(int t=0;t<Ti;++t) comb[(size_t)i*topk+wmax+t]=nwin+t; }
    int* dcomb; dcomb=(int*)dmalloc((size_t)K*topk*4); CU(cudaMemcpyAsync(dcomb,comb.data(),(size_t)K*topk*4,cudaMemcpyHostToDevice,stream));
    finish_attn(w, q, kv_all, dcomb, K, pos, ntot, topk, out, eps, stream);
    dsync(stream); dfree(q);dfree(kv_all);dfree(dcomb);
}

void compressed_verify_step_indexer(float* out, const float* x_full, int pos, int K, const CompressedAttnWeights& w,
                                    float* win_kv, float* comp_kv, float* idx_ckv, int* T, int ratio, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int half=ROPE_DIM/2, Kd=N_HEADS*HEAD_DIM, nH=w.index_n_heads, ihd=w.index_head_dim, QD=nH*ihd, rd=ROPE_DIM;
    const float wscale=rsqrtf((float)ihd)*rsqrtf((float)nH); const float *cosP=a.cosT+(size_t)pos*half, *sinP=a.sinT+(size_t)pos*half;
    float* q; q=(float*)dmalloc((size_t)K*Kd*4);
    build_qKV(w, x_full+(size_t)pos*DIM, K, pos, q, win_kv, eps, stream);
    // emit main + indexer compressed rows for groups completing in the block
    for(int j=pos;j<pos+K;++j) if((j+1)%ratio==0){ int g=j/ratio; int t=*T;
        compressor_emit_group(comp_kv+(size_t)t*HEAD_DIM, x_full, g, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,w.mc_norm,w.cc_cos,w.cc_sin,DIM,HEAD_DIM,true,ROPE_DIM,eps,false,stream);
        compressor_emit_group(idx_ckv+(size_t)t*ihd, x_full, g, ratio, w.idx_c_wkv,w.idx_c_wgate,w.idx_c_ape,w.idx_c_norm,w.cc_cos,w.cc_sin,DIM,ihd,true,ROPE_DIM,eps,true,stream); ++(*T); }
    int Tf=*T, nwin=pos+K;
    // indexer scoring for K queries: qidx = fp4sim(hadamard(rope(idx_wq_b(qr)))) — recompute qr (cheap) at M=K
    uint8_t *iqrq; float *iqrs,*qr2,*qidx,*qtmp,*iw,*iscore;
    // qr again (needed for indexer); recompute from x
    uint8_t* xq2; float* xs2; xq2=(uint8_t*)dmalloc((size_t)K*DIM); xs2=(float*)dmalloc((size_t)K*(DIM/128)*4);
    act_quant_fp8(xq2,xs2,x_full+(size_t)pos*DIM,K,DIM,128,stream);
    qr2=(float*)dmalloc((size_t)K*Q_LORA*4); fp8_block_gemm(qr2,xq2,xs2,a.wq_a,a.wq_a_s,K,Q_LORA,DIM,stream); rmsnorm(qr2,qr2,a.q_norm,K,Q_LORA,eps,true,stream);
    iqrq=(uint8_t*)dmalloc((size_t)K*Q_LORA); iqrs=(float*)dmalloc((size_t)K*(Q_LORA/128)*4);
    qidx=(float*)dmalloc((size_t)K*QD*4); qtmp=(float*)dmalloc((size_t)K*QD*4); iw=(float*)dmalloc((size_t)K*nH*4);
    act_quant_fp8(iqrq,iqrs,qr2,K,Q_LORA,128,stream);
    fp8_block_gemm(qidx,iqrq,iqrs,w.idx_wq_b,w.idx_wq_b_s,K,QD,Q_LORA,stream);
    rope_interleaved(qidx+(ihd-rd),cosP,sinP,K*nH,rd,false,ihd,nH,stream);
    hadamard(qtmp,qidx,K*nH,ihd,stream); act_quant_fp4sim(qtmp,K*nH,ihd,32,ihd,stream);
    gemm_fp32(iw,x_full+(size_t)pos*DIM,w.idx_weights_proj,K,nH,DIM,stream);
    k_iw_scale<<<((size_t)K*nH+63)/64,64,0,stream>>>(iw,wscale,K*nH);
    iscore=(float*)dmalloc((size_t)K*Tf*4);
    index_score(iscore,qtmp,idx_ckv,iw,K,Tf,nH,ihd,stream);
    // per-query top-k with GLOBAL causal threshold (t < (pos+i+1)/ratio), offset nwin
    int topkc = (w.index_topk<Tf)?w.index_topk:Tf;
    int* dtop; dtop=(int*)dmalloc((size_t)K*topkc*4);
    // build on host from iscore
    std::vector<float> hsc((size_t)K*Tf); CU(cudaMemcpyAsync(hsc.data(),iscore,(size_t)K*Tf*4,cudaMemcpyDeviceToHost,stream)); dsync(stream);
    std::vector<int> htop((size_t)K*topkc,-1);
    for(int i=0;i<K;++i){ int ig=pos+i,thr=(ig+1)/ratio; std::vector<float> s(hsc.begin()+(size_t)i*Tf,hsc.begin()+(size_t)i*Tf+Tf);
        for(int k=0;k<topkc;++k){ float best=-1e30f;int bi=-1; for(int t=0;t<thr&&t<Tf;++t) if(s[t]>best){best=s[t];bi=t;} if(bi>=0){s[bi]=-1e30f; htop[(size_t)i*topkc+k]=nwin+bi;} } }
    CU(cudaMemcpyAsync(dtop,htop.data(),(size_t)K*topkc*4,cudaMemcpyHostToDevice,stream));
    // kv_all + combined idxs (window ⊕ indexer-selected compressed)
    int ntot=nwin+Tf; float* kv_all; kv_all=(float*)dmalloc((size_t)ntot*HEAD_DIM*4);
    CU(cudaMemcpyAsync(kv_all,win_kv,(size_t)nwin*HEAD_DIM*4,cudaMemcpyDeviceToDevice,stream));
    CU(cudaMemcpyAsync(kv_all+(size_t)nwin*HEAD_DIM,comp_kv,(size_t)Tf*HEAD_DIM*4,cudaMemcpyDeviceToDevice,stream));
    int wmax=0; for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;if(wid>wmax)wmax=wid;}
    int topk=wmax+topkc; std::vector<int> comb((size_t)K*topk,-1);
    for(int i=0;i<K;++i){int ig=pos+i,b=ig-WINDOW+1;if(b<0)b=0;int wid=ig+1-b;
        for(int k=0;k<wid;++k) comb[(size_t)i*topk+k]=b+k;
        for(int t=0;t<topkc;++t) comb[(size_t)i*topk+wmax+t]=htop[(size_t)i*topkc+t]; }
    int* dcomb; dcomb=(int*)dmalloc((size_t)K*topk*4); CU(cudaMemcpyAsync(dcomb,comb.data(),(size_t)K*topk*4,cudaMemcpyHostToDevice,stream));
    finish_attn(w, q, kv_all, dcomb, K, pos, ntot, topk, out, eps, stream);
    dsync(stream);
    dfree(q);dfree(xq2);dfree(xs2);dfree(qr2);dfree(iqrq);dfree(iqrs);dfree(qidx);dfree(qtmp);dfree(iw);dfree(iscore);dfree(dtop);dfree(kv_all);dfree(dcomb);
}

// ================= device-pos compressed decode (CUDA-graph capturable) =================
// combined cache kvc = [winmax(window) .. winmax+Tmax(compressed)][HEAD_DIM]; d_T = device compressed count.
#include "mla_forward.h"
__global__ void k_gather_win_dp(float* scr, const float* xin, const int* d_pos, int ntok, int tok_off, int dim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)ntok*dim) return; int r=i/dim, c=i%dim;
    int t=(*d_pos)+tok_off+r; scr[i]=xin[(size_t)t*dim+c]; }
__global__ void k_dg(int* d_g, const int* d_pos, int ratio){ if(!threadIdx.x&&!blockIdx.x) *d_g=(*d_pos)/ratio; }
__global__ void k_append_at2(float* dst, const float* scr, const int* d_idx, int hd){          // dst[(*d_idx)*hd + i]=scr[i]
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=hd) return; dst[(size_t)(*d_idx)*hd + i]=scr[i]; }
__global__ void k_commit_comp(float* comp_region, const float* cand, const int* d_T, const int* d_pos, int ratio, int hd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=hd) return;
    if(((*d_pos)+1)%ratio==0) comp_region[(size_t)(*d_T)*hd + i]=cand[i]; }          // append only when a group completes
__global__ void k_advance_T(int* d_T, const int* d_pos, int ratio){ if(!threadIdx.x&&!blockIdx.x && ((*d_pos)+1)%ratio==0) (*d_T)++; }
// strided combined idxs: window [base..pos] (region [0,winmax)) then compressed [winmax + t for t<*d_T]; fixed topk.
__global__ void k_comb_strided_dp(int* comb, const int* d_pos, const int* d_T, int winmax, int wtop, int Tmax){
    int k=blockIdx.x*blockDim.x+threadIdx.x; int topk=wtop+Tmax; if(k>=topk) return; int pos=*d_pos;
    if(k<wtop){ int base=pos-WINDOW+1; if(base<0)base=0; int v=base+k; comb[k]=(v<=pos)? v : -1; }
    else { int t=k-wtop; comb[k]=(t<*d_T)? winmax+t : -1; } }

// device-pos compressor emit: computes the candidate row from the last ntok tokens of xin (ending at *d_pos) and
// commits it to the compressed region + advances *d_T iff (*d_pos+1)%ratio==0. rope at compressed pos d_g.
static void emit_group_dp(float* comp_region, const float* xin, const int* d_pos, int* d_T, int* d_g, int ratio,
        const float* wkv, const float* wgate, const float* ape, const float* norm_w, const float* cc_cos, const float* cc_sin,
        int dim, int d, bool overlap, int rotate, float eps, cudaStream_t stream){
    int coff=overlap?2:1, od=coff*d; int ntok=overlap?2*ratio:ratio, tok_off=-(ntok-1), localg=overlap?1:0;
    float *scr,*kv,*score,*pooled,*cand;
    scr=(float*)dmalloc((size_t)ntok*dim*4); kv=(float*)dmalloc((size_t)ntok*od*4); score=(float*)dmalloc((size_t)ntok*od*4);
    pooled=(float*)dmalloc((size_t)(localg+1)*d*4); cand=(float*)dmalloc((size_t)d*4);
    k_gather_win_dp<<<((size_t)ntok*dim+255)/256,256,0,stream>>>(scr,xin,d_pos,ntok,tok_off,dim);
    gemm_fp32(kv,scr,wkv,ntok,od,dim,stream); gemm_fp32(score,scr,wgate,ntok,od,dim,stream);
    if(overlap) compressor_pool_overlap(pooled,kv,score,ape,localg+1,ratio,d,stream);
    else        compressor_pool(pooled,kv,score,ape,1,ratio,d,stream);
    rmsnorm(cand,pooled+(size_t)localg*d,norm_w,1,d,eps,true,stream);
    k_dg<<<1,1,0,stream>>>(d_g,d_pos,ratio);
    rope_interleaved_dp(cand+(d-ROPE_DIM),cc_cos,cc_sin,1,ROPE_DIM,false,d,1,d_g,stream);   // rope at compressed pos g
    if(rotate){ hadamard(cand,cand,1,d,stream); act_quant_fp4sim(cand,1,d,32,d,stream); }
    else       act_quant_fp8sim(cand,1,d-ROPE_DIM,64,d,stream);
    k_commit_comp<<<(d+255)/256,256,0,stream>>>(comp_region,cand,d_T,d_pos,ratio,d);   // commit at *d_T; caller advances
    dfree(scr);dfree(kv);dfree(score);dfree(pooled);dfree(cand);
}

// strided (ratio!=4) device-pos compressed decode step. kvc = combined cache; d_pos/d_T device.
void compressed_decode_step_strided_dp(float* out, const float* x, const float* xin, const CompressedAttnWeights& w,
        float* kvc, const int* d_pos, int* d_T, int* d_g, int winmax, int Tmax, int ratio, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA; const float scale=1.f/sqrtf((float)HEAD_DIM);
    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*o,*og,*kvs; int* comb;
    xq=(uint8_t*)dmalloc(DIM); xs=(float*)dmalloc((DIM/128)*4);
    qr=(float*)dmalloc(Q_LORA*4); qrq=(uint8_t*)dmalloc(Q_LORA); qrs=(float*)dmalloc((Q_LORA/128)*4);
    q=(float*)dmalloc(Kd*4); o=(float*)dmalloc(Kd*4); og=(float*)dmalloc(OB*4); ogq=(uint8_t*)dmalloc(OB); ogs=(float*)dmalloc((OB/128)*4);
    kvs=(float*)dmalloc(HEAD_DIM*4); int wtop=WINDOW; comb=(int*)dmalloc((size_t)(wtop+Tmax)*4);
    act_quant_fp8(xq,xs,x,1,DIM,128,stream);
    fp8_block_gemm(qr,xq,xs,a.wq_a,a.wq_a_s,1,Q_LORA,DIM,stream); rmsnorm(qr,qr,a.q_norm,1,Q_LORA,eps,true,stream);
    act_quant_fp8(qrq,qrs,qr,1,Q_LORA,128,stream); fp8_block_gemm(q,qrq,qrs,a.wq_b,a.wq_b_s,1,Kd,Q_LORA,stream);
    rmsnorm(q,q,nullptr,N_HEADS,HEAD_DIM,eps,false,stream);
    rope_interleaved_dp(q+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,d_pos,stream);
    // window kv -> kvc[*d_pos]
    fp8_block_gemm(kvs,xq,xs,a.wkv,a.wkv_s,1,HEAD_DIM,DIM,stream); rmsnorm(kvs,kvs,a.kv_norm,1,HEAD_DIM,eps,true,stream);
    rope_interleaved_dp(kvs+NOPE_DIM,a.cosT,a.sinT,1,ROPE_DIM,false,HEAD_DIM,1,d_pos,stream); act_quant_fp8sim(kvs,1,NOPE_DIM,64,HEAD_DIM,stream);
    k_append_at2<<<(HEAD_DIM+255)/256,256,0,stream>>>(kvc,kvs,d_pos,HEAD_DIM);
    // compressor emit (device-conditional) into the compressed region [winmax..]
    emit_group_dp(kvc+(size_t)winmax*HEAD_DIM, xin, d_pos, d_T, d_g, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,w.mc_norm,w.cc_cos,w.cc_sin,DIM,HEAD_DIM,false,0,eps,stream);
    k_advance_T<<<1,1,0,stream>>>(d_T,d_pos,ratio);
    // attention over combined cache
    k_comb_strided_dp<<<(wtop+Tmax+63)/64,64,0,stream>>>(comb,d_pos,d_T,winmax,wtop,Tmax);
    sparse_attn(o,q,kvc,a.attn_sink,comb,1,1,N_HEADS,HEAD_DIM,winmax+Tmax,wtop+Tmax,scale,stream);
    rope_interleaved_dp(o+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,d_pos,stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og,o,a.wo_a_fp8,a.wo_a_sc,1,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,a.wo_a,               1,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,1,OB,128,stream); fp8_block_gemm(out,ogq,ogs,a.wo_b,a.wo_b_s,1,DIM,OB,stream);
    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kvs);dfree(comb);
}

// device-pos indexer (ratio-4) compressed decode.
__global__ void k_mask_scores(float* score, const int* d_T, int Tmax){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<Tmax && t>=*d_T) score[t]=-1e30f; }
__global__ void k_topk_masked(int* out, const float* score, int Tmax, int topk, int winmax){   // top-k valid rows -> winmax+t, else -1
    if(threadIdx.x||blockIdx.x) return; extern __shared__ float sh[]; for(int t=0;t<Tmax;++t) sh[t]=score[t];
    for(int k=0;k<topk;++k){ float best=-1e29f; int bi=-1; for(int t=0;t<Tmax;++t) if(sh[t]>best){best=sh[t];bi=t;}
        if(bi>=0){ sh[bi]=-1e30f; out[k]=winmax+bi; } else out[k]=-1; } }
__global__ void k_comb_join(int* comb, const int* win, const int* sel, int wtop, int topk_c){   // [window ⊕ selected]
    int k=blockIdx.x*blockDim.x+threadIdx.x; int tot=wtop+topk_c; if(k>=tot) return; comb[k]=(k<wtop)?win[k]:sel[k-wtop]; }

void compressed_decode_step_indexer_dp(float* out, const float* x, const float* xin, const CompressedAttnWeights& w,
        float* kvc, float* idx_kvc, const int* d_pos, int* d_T, int* d_g, int winmax, int Tmax, int ratio, float eps, cudaStream_t stream){
    const auto& a=w.attn; const int Kd=N_HEADS*HEAD_DIM, GKd=Kd/O_GROUPS, OB=O_GROUPS*O_LORA;
    const int nH=w.index_n_heads, ihd=w.index_head_dim, QD=nH*ihd, rd=ROPE_DIM; const float scale=1.f/sqrtf((float)HEAD_DIM);
    const float wscale=rsqrtf((float)ihd)*rsqrtf((float)nH);
    uint8_t *xq,*qrq,*ogq,*iqrq; float *xs,*qrs,*ogs,*qr,*q,*o,*og,*kvs,*qidx,*qtmp,*iw,*isc; int *win,*sel,*comb;
    xq=(uint8_t*)dmalloc(DIM); xs=(float*)dmalloc((DIM/128)*4); qr=(float*)dmalloc(Q_LORA*4); qrq=(uint8_t*)dmalloc(Q_LORA); qrs=(float*)dmalloc((Q_LORA/128)*4);
    q=(float*)dmalloc(Kd*4); o=(float*)dmalloc(Kd*4); og=(float*)dmalloc(OB*4); ogq=(uint8_t*)dmalloc(OB); ogs=(float*)dmalloc((OB/128)*4); kvs=(float*)dmalloc(HEAD_DIM*4);
    iqrq=(uint8_t*)dmalloc(Q_LORA); float* iqrs=(float*)dmalloc((Q_LORA/128)*4);
    qidx=(float*)dmalloc((size_t)QD*4); qtmp=(float*)dmalloc((size_t)QD*4); iw=(float*)dmalloc((size_t)nH*4); isc=(float*)dmalloc((size_t)Tmax*4);
    int wtop=WINDOW, topk_c=(w.index_topk<Tmax)?w.index_topk:Tmax;
    win=(int*)dmalloc((size_t)wtop*4); sel=(int*)dmalloc((size_t)topk_c*4); comb=(int*)dmalloc((size_t)(wtop+topk_c)*4);
    act_quant_fp8(xq,xs,x,1,DIM,128,stream);
    fp8_block_gemm(qr,xq,xs,a.wq_a,a.wq_a_s,1,Q_LORA,DIM,stream); rmsnorm(qr,qr,a.q_norm,1,Q_LORA,eps,true,stream);
    act_quant_fp8(qrq,qrs,qr,1,Q_LORA,128,stream); fp8_block_gemm(q,qrq,qrs,a.wq_b,a.wq_b_s,1,Kd,Q_LORA,stream);
    rmsnorm(q,q,nullptr,N_HEADS,HEAD_DIM,eps,false,stream); rope_interleaved_dp(q+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,false,HEAD_DIM,N_HEADS,d_pos,stream);
    fp8_block_gemm(kvs,xq,xs,a.wkv,a.wkv_s,1,HEAD_DIM,DIM,stream); rmsnorm(kvs,kvs,a.kv_norm,1,HEAD_DIM,eps,true,stream);
    rope_interleaved_dp(kvs+NOPE_DIM,a.cosT,a.sinT,1,ROPE_DIM,false,HEAD_DIM,1,d_pos,stream); act_quant_fp8sim(kvs,1,NOPE_DIM,64,HEAD_DIM,stream);
    k_append_at2<<<(HEAD_DIM+255)/256,256,0,stream>>>(kvc,kvs,d_pos,HEAD_DIM);
    // main (overlap) + indexer (overlap+rotate) emits at the SAME *d_T, then advance once
    emit_group_dp(kvc+(size_t)winmax*HEAD_DIM, xin, d_pos, d_T, d_g, ratio, w.mc_wkv,w.mc_wgate,w.mc_ape,w.mc_norm,w.cc_cos,w.cc_sin,DIM,HEAD_DIM,true,0,eps,stream);
    emit_group_dp(idx_kvc, xin, d_pos, d_T, d_g, ratio, w.idx_c_wkv,w.idx_c_wgate,w.idx_c_ape,w.idx_c_norm,w.cc_cos,w.cc_sin,DIM,ihd,true,1,eps,stream);
    k_advance_T<<<1,1,0,stream>>>(d_T,d_pos,ratio);
    // indexer scoring for the single query -> select main-compressed rows
    act_quant_fp8(iqrq,iqrs,qr,1,Q_LORA,128,stream); fp8_block_gemm(qidx,iqrq,iqrs,w.idx_wq_b,w.idx_wq_b_s,1,QD,Q_LORA,stream);
    rope_interleaved_dp(qidx+(ihd-rd),a.cosT,a.sinT,nH,rd,false,ihd,nH,d_pos,stream); hadamard(qtmp,qidx,nH,ihd,stream); act_quant_fp4sim(qtmp,nH,ihd,32,ihd,stream);
    gemm_fp32(iw,x,w.idx_weights_proj,1,nH,DIM,stream); k_iw_scale<<<(nH+63)/64,64,0,stream>>>(iw,wscale,nH);
    index_score(isc,qtmp,idx_kvc,iw,1,Tmax,nH,ihd,stream); k_mask_scores<<<(Tmax+63)/64,64,0,stream>>>(isc,d_T,Tmax);
    k_topk_masked<<<1,32,(size_t)Tmax*4,stream>>>(sel,isc,Tmax,topk_c,winmax);
    k_comb_strided_dp<<<(wtop+63)/64,64,0,stream>>>(win,d_pos,d_T,winmax,wtop,0);   // window part only (Tmax=0)
    k_comb_join<<<(wtop+topk_c+63)/64,64,0,stream>>>(comb,win,sel,wtop,topk_c);
    sparse_attn(o,q,kvc,a.attn_sink,comb,1,1,N_HEADS,HEAD_DIM,winmax+Tmax,wtop+topk_c,scale,stream);
    rope_interleaved_dp(o+NOPE_DIM,a.cosT,a.sinT,N_HEADS,ROPE_DIM,true,HEAD_DIM,N_HEADS,d_pos,stream);
    if(a.wo_a_native) ogroup_gemm_fp8(og,o,a.wo_a_fp8,a.wo_a_sc,1,O_GROUPS,O_LORA,GKd,stream);
    else              ogroup_gemm    (og,o,a.wo_a,               1,O_GROUPS,O_LORA,GKd,stream);
    act_quant_fp8(ogq,ogs,og,1,OB,128,stream); fp8_block_gemm(out,ogq,ogs,a.wo_b,a.wo_b_s,1,DIM,OB,stream);
    dsync(stream);
    dfree(xq);dfree(xs);dfree(qr);dfree(qrq);dfree(qrs);dfree(q);dfree(o);dfree(og);dfree(ogq);dfree(ogs);dfree(kvs);
    dfree(iqrq);dfree(iqrs);dfree(qidx);dfree(qtmp);dfree(iw);dfree(isc);dfree(win);dfree(sel);dfree(comb);
}
