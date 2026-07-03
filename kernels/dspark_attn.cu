// dspark_attn.cu — DSparkAttention. See dspark_attn.h / DSPARK_HEAD_BUILD.md piece 4.
#include "dspark_attn.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "deepseek_v4.h"
#include <vector>
#include <cmath>
#include <cstdio>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// main-KV from main_x: wkv -> kv_norm -> rope(per-position) -> act_quant fp8sim(nope). [s,dim]->[s,HEAD_DIM].
void dspark_main_kv(float* main_kv, const float* main_x, const MLAWeights& w, int s, float eps, cudaStream_t stream){
    uint8_t* xq; float* xs;
    CU(cudaMalloc(&xq,(size_t)s*DIM)); CU(cudaMalloc(&xs,(size_t)s*(DIM/128)*4));
    act_quant_fp8(xq, xs, main_x, s, DIM, 128, stream);
    fp8_block_gemm(main_kv, xq, xs, w.wkv, w.wkv_s, s, HEAD_DIM, DIM, stream);
    rmsnorm(main_kv, main_kv, w.kv_norm, s, HEAD_DIM, eps, true, stream);
    rope_interleaved(main_kv + NOPE_DIM, w.cosT, w.sinT, s, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(main_kv, s, NOPE_DIM, 64, HEAD_DIM, stream);
    CU(cudaStreamSynchronize(stream)); cudaFree(xq); cudaFree(xs);
}

void dspark_attn_forward(float* out, const float* xin, const float* main_kv, int t,
                         const MLAWeights& w, const float* cosB, const float* sinB,
                         int block, int win, float eps, cudaStream_t stream){
    const int Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    int nwin = (t+1 < win) ? t+1 : win; int wstart = t+1-nwin; int n = nwin + block;

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*bkv,*kv_all,*o,*og;
    CU(cudaMalloc(&xq,(size_t)block*DIM)); CU(cudaMalloc(&xs,(size_t)block*(DIM/128)*4));
    CU(cudaMalloc(&qr,(size_t)block*Q_LORA*4)); CU(cudaMalloc(&qrq,(size_t)block*Q_LORA)); CU(cudaMalloc(&qrs,(size_t)block*(Q_LORA/128)*4));
    CU(cudaMalloc(&q,(size_t)block*Kd*4)); CU(cudaMalloc(&bkv,(size_t)block*HEAD_DIM*4));
    CU(cudaMalloc(&kv_all,(size_t)n*HEAD_DIM*4)); CU(cudaMalloc(&o,(size_t)block*Kd*4)); CU(cudaMalloc(&og,(size_t)block*OB*4));
    CU(cudaMalloc(&ogq,(size_t)block*OB)); CU(cudaMalloc(&ogs,(size_t)block*(OB/128)*4));

    // q
    act_quant_fp8(xq, xs, xin, block, DIM, 128, stream);
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, block, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, block, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, block, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, block, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, block*N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosB, sinB, block*N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // block kv
    fp8_block_gemm(bkv, xq, xs, w.wkv, w.wkv_s, block, HEAD_DIM, DIM, stream);
    rmsnorm(bkv, bkv, w.kv_norm, block, HEAD_DIM, eps, true, stream);
    rope_interleaved(bkv + NOPE_DIM, cosB, sinB, block, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(bkv, block, NOPE_DIM, 64, HEAD_DIM, stream);
    // kv = [main-KV window ⊕ block-KV]
    CU(cudaMemcpyAsync(kv_all, main_kv + (size_t)wstart*HEAD_DIM, (size_t)nwin*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)nwin*HEAD_DIM, bkv, (size_t)block*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    // dense idxs [block, n]: every block query attends to all n (window ⊕ block), per get_dspark_topk_idxs
    std::vector<int> hidx((size_t)block*n); for(int m=0;m<block;++m) for(int k=0;k<n;++k) hidx[(size_t)m*n+k]=k;
    int* idx; CU(cudaMalloc(&idx,(size_t)block*n*4)); CU(cudaMemcpyAsync(idx,hidx.data(),(size_t)block*n*4,cudaMemcpyHostToDevice,stream));
    sparse_attn(o, q, kv_all, w.attn_sink, idx, 1, block, N_HEADS, HEAD_DIM, n, n, scale, stream);
    rope_interleaved(o + NOPE_DIM, cosB, sinB, block*N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, w.wo_a, block, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, block, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, block, DIM, OB, stream);
    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(bkv);
    cudaFree(kv_all);cudaFree(o);cudaFree(og);cudaFree(ogq);cudaFree(ogs);cudaFree(idx);
}

// ---- DSparkBlock forward (block_forward with dspark_attn) ----
#include "hc.h"
#include "moe.h"
void dspark_block_forward(float* out, const float* x, const int* input_ids, const float* main_kv, int t,
                          const BlockWeights& w, const float* cosB, const float* sinB, int block, int win,
                          int iters, float eps, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    CU(cudaMalloc(&x1,(size_t)block*d*4)); CU(cudaMalloc(&post,(size_t)block*hc*4)); CU(cudaMalloc(&comb,(size_t)block*hc*hc*4));
    CU(cudaMalloc(&sub,(size_t)block*d*4)); CU(cudaMalloc(&res2,(size_t)block*hc*d*4));
    hc_pre(x1, post, comb, x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, block, hc, d, iters, eps, stream);
    rmsnorm(x1, x1, w.attn_norm, block, d, eps, true, stream);
    dspark_attn_forward(sub, x1, main_kv, t, w.attn, cosB, sinB, block, win, eps, stream);
    hc_post(res2, sub, x, post, comb, block, hc, d, stream);
    hc_pre(x1, post, comb, res2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, block, hc, d, iters, eps, stream);
    rmsnorm(x1, x1, w.ffn_norm, block, d, eps, true, stream);
    moe_forward(sub, x1, input_ids, w.ffn, block, stream);
    hc_post(out, sub, res2, post, comb, block, hc, d, stream);
    CU(cudaStreamSynchronize(stream));
    cudaFree(x1);cudaFree(post);cudaFree(comb);cudaFree(sub);cudaFree(res2);
}
