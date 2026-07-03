// compressed_decode.cu — M=1 KV-cache decode for a compressed (strided, ratio!=4) MLA layer. See header.
#include "compressed_decode.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "compressor.h"
#include "indexer.h"
#include "deepseek_v4.h"
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
    CU(cudaMalloc(&xq,(size_t)s0*DIM)); CU(cudaMalloc(&xs,(size_t)s0*(DIM/128)*4));
    act_quant_fp8(xq, xs, x, s0, DIM, 128, stream);
    fp8_block_gemm(win_kv, xq, xs, a.wkv, a.wkv_s, s0, HEAD_DIM, DIM, stream);
    rmsnorm(win_kv, win_kv, a.kv_norm, s0, HEAD_DIM, eps, true, stream);
    rope_interleaved(win_kv + NOPE_DIM, a.cosT, a.sinT, s0, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(win_kv, s0, NOPE_DIM, 64, HEAD_DIM, stream);
    CU(cudaStreamSynchronize(stream)); cudaFree(xq); cudaFree(xs);
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
    CU(cudaMalloc(&xq,DIM)); CU(cudaMalloc(&xs,(DIM/128)*4));
    CU(cudaMalloc(&qr,Q_LORA*4)); CU(cudaMalloc(&qrq,Q_LORA)); CU(cudaMalloc(&qrs,(Q_LORA/128)*4));
    CU(cudaMalloc(&q,Kd*4)); CU(cudaMalloc(&o,Kd*4)); CU(cudaMalloc(&og,OB*4));
    CU(cudaMalloc(&ogq,OB)); CU(cudaMalloc(&ogs,(OB/128)*4));

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
    float* kv_all; CU(cudaMalloc(&kv_all,(size_t)ntot*HEAD_DIM*4));
    CU(cudaMemcpyAsync(kv_all, win_kv, (size_t)nwin*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)nwin*HEAD_DIM, comp_kv, (size_t)Tn*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    // combined idxs: window [base..pos] ⊕ compressed [nwin + t] for t<Tn (strided: all t<(pos+1)/ratio == Tn)
    int base = pos - WINDOW + 1; if(base<0) base=0; int wwidth = pos+1-base;
    int tot = wwidth + Tn;
    std::vector<int> comb(tot);
    for(int k=0;k<wwidth;++k) comb[k]=base+k;
    for(int t=0;t<Tn;++t) comb[wwidth+t]=nwin+t;
    int* dcomb; CU(cudaMalloc(&dcomb,(size_t)tot*4));
    CU(cudaMemcpyAsync(dcomb, comb.data(), (size_t)tot*4, cudaMemcpyHostToDevice, stream));
    sparse_attn(o, q, kv_all, a.attn_sink, dcomb, 1, 1, N_HEADS, HEAD_DIM, ntot, tot, scale, stream);
    // de-rotate, grouped o-LoRA, wo_b
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, a.wo_a, 1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, 1, DIM, OB, stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);
    cudaFree(o);cudaFree(og);cudaFree(ogq);cudaFree(ogs);cudaFree(kv_all);cudaFree(dcomb);
}

// ================= ratio-4 (DSA indexer) decode =================
// Prefill cache for a ratio-4 layer: window KV + MAIN compressed KV (overlap) + INDEXER compressor KV
// (overlap+rotate, for scoring). All append-only. Sets *T = complete-group count.
void compressed_attn_cache_r4(float* win_kv, float* comp_kv, float* idx_ckv, int* T, const float* x,
                              const CompressedAttnWeights& w, int s0, int ratio, float eps, cudaStream_t stream){
    const auto& a = w.attn; const int idx_hd = w.index_head_dim;
    uint8_t* xq; float* xs;
    CU(cudaMalloc(&xq,(size_t)s0*DIM)); CU(cudaMalloc(&xs,(size_t)s0*(DIM/128)*4));
    act_quant_fp8(xq, xs, x, s0, DIM, 128, stream);
    fp8_block_gemm(win_kv, xq, xs, a.wkv, a.wkv_s, s0, HEAD_DIM, DIM, stream);
    rmsnorm(win_kv, win_kv, a.kv_norm, s0, HEAD_DIM, eps, true, stream);
    rope_interleaved(win_kv + NOPE_DIM, a.cosT, a.sinT, s0, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(win_kv, s0, NOPE_DIM, 64, HEAD_DIM, stream);
    CU(cudaStreamSynchronize(stream)); cudaFree(xq); cudaFree(xs);
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
    CU(cudaMalloc(&xq,DIM)); CU(cudaMalloc(&xs,(DIM/128)*4));
    CU(cudaMalloc(&qr,Q_LORA*4)); CU(cudaMalloc(&qrq,Q_LORA)); CU(cudaMalloc(&qrs,(Q_LORA/128)*4));
    CU(cudaMalloc(&q,Kd*4)); CU(cudaMalloc(&o,Kd*4)); CU(cudaMalloc(&og,OB*4));
    CU(cudaMalloc(&ogq,OB)); CU(cudaMalloc(&ogs,(OB/128)*4));
    CU(cudaMalloc(&iqrq,Q_LORA)); CU(cudaMalloc(&iqrs,(Q_LORA/128)*4));
    CU(cudaMalloc(&qidx,QD*4)); CU(cudaMalloc(&qtmp,QD*4)); CU(cudaMalloc(&iw,nH*4));

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
    CU(cudaMalloc(&iscore,(size_t)Tn*4));
    index_score(iscore, qtmp, idx_ckv, iw, 1, Tn, nH, idx_hd, stream);
    int topk = w.index_topk < Tn ? w.index_topk : Tn;
    int* dtop; CU(cudaMalloc(&dtop,(size_t)topk*4));
    k_topk_decode<<<1,32,(size_t)Tn*4,stream>>>(dtop, iscore, Tn, topk, nwin);
    // --- kv_all = [win_kv[0..pos] ; comp_kv[0..Tn-1]] ---
    int ntot = nwin + Tn;
    float* kv_all; CU(cudaMalloc(&kv_all,(size_t)ntot*HEAD_DIM*4));
    CU(cudaMemcpyAsync(kv_all, win_kv, (size_t)nwin*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)nwin*HEAD_DIM, comp_kv, (size_t)Tn*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    // --- combined idxs: window [base..pos] ⊕ indexer topk ---
    int base = pos - WINDOW + 1; if(base<0) base=0; int wwidth = pos+1-base;
    int tot = wwidth + topk;
    std::vector<int> hwin(wwidth); for(int k=0;k<wwidth;++k) hwin[k]=base+k;
    int* comb; CU(cudaMalloc(&comb,(size_t)tot*4));
    CU(cudaMemcpyAsync(comb, hwin.data(), (size_t)wwidth*4, cudaMemcpyHostToDevice, stream));
    CU(cudaMemcpyAsync(comb + wwidth, dtop, (size_t)topk*4, cudaMemcpyDeviceToDevice, stream));
    sparse_attn(o, q, kv_all, a.attn_sink, comb, 1, 1, N_HEADS, HEAD_DIM, ntot, tot, scale, stream);
    rope_interleaved(o + NOPE_DIM, cosP, sinP, N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, a.wo_a, 1, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, 1, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, 1, DIM, OB, stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(o);cudaFree(og);
    cudaFree(ogq);cudaFree(ogs);cudaFree(iqrq);cudaFree(iqrs);cudaFree(qidx);cudaFree(qtmp);cudaFree(iw);
    cudaFree(iscore);cudaFree(dtop);cudaFree(kv_all);cudaFree(comb);
}
