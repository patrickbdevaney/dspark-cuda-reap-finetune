// compressed_attn.cu — full compressed-layer MLA forward (prefill). See compressed_attn.h.
#include "compressed_attn.h"
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

// combined_idxs[q, 0:wwidth] = window[q], combined[q, wwidth:] = compress[q]. one thread per (q,j).
__global__ void k_combine_idxs(int* comb, const int* window, const int* compress, int s, int wwidth, int itopk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int tot = wwidth + itopk; if (i >= s * tot) return;
    int q = i / tot, j = i % tot;
    comb[i] = (j < wwidth) ? window[(size_t)q * wwidth + j] : compress[(size_t)q * itopk + (j - wwidth)];
}

void compressed_attn_forward(float* out, const float* x, const CompressedAttnWeights& w,
                             int s, int win, int ratio, float eps, cudaStream_t stream) {
    const int bs = s, Kd = N_HEADS * HEAD_DIM, GKd = Kd / O_GROUPS, OB = O_GROUPS * O_LORA, T = s / ratio;
    const float scale = 1.f / sqrtf((float)HEAD_DIM);
    const auto& a = w.attn;

    uint8_t *xq, *qrq, *ogq; float *xs, *qrs, *ogs, *qr, *q, *kv_win, *kv_comp, *kv_all, *o, *og;
    CU(cudaMalloc(&xq,(size_t)bs*DIM)); CU(cudaMalloc(&xs,(size_t)bs*(DIM/128)*4));
    CU(cudaMalloc(&qr,(size_t)bs*Q_LORA*4)); CU(cudaMalloc(&qrq,(size_t)bs*Q_LORA)); CU(cudaMalloc(&qrs,(size_t)bs*(Q_LORA/128)*4));
    CU(cudaMalloc(&q,(size_t)bs*Kd*4)); CU(cudaMalloc(&kv_win,(size_t)bs*HEAD_DIM*4));
    CU(cudaMalloc(&kv_comp,(size_t)T*HEAD_DIM*4)); CU(cudaMalloc(&kv_all,(size_t)(bs+T)*HEAD_DIM*4));
    CU(cudaMalloc(&o,(size_t)bs*Kd*4)); CU(cudaMalloc(&og,(size_t)bs*OB*4));
    CU(cudaMalloc(&ogq,(size_t)bs*OB)); CU(cudaMalloc(&ogs,(size_t)bs*(OB/128)*4));

    // --- q ---
    act_quant_fp8(xq, xs, x, bs, DIM, 128, stream);
    fp8_block_gemm(qr, xq, xs, a.wq_a, a.wq_a_s, bs, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, a.q_norm, bs, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, bs, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, a.wq_b, a.wq_b_s, bs, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, bs * N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, a.cosT, a.sinT, bs * N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);

    // --- window kv ---
    fp8_block_gemm(kv_win, xq, xs, a.wkv, a.wkv_s, bs, HEAD_DIM, DIM, stream);
    rmsnorm(kv_win, kv_win, a.kv_norm, bs, HEAD_DIM, eps, true, stream);
    rope_interleaved(kv_win + NOPE_DIM, a.cosT, a.sinT, bs, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(kv_win, bs, NOPE_DIM, 64, HEAD_DIM, stream);

    // --- main compressor -> compressed kv, then combined kv = [window ⊕ compressed] ---
    // ratio==4: overlapping compressor + DSA indexer. ratio==128: non-overlap compressor + strided idxs.
    const bool overlap = (ratio == 4), has_indexer = (ratio == 4);
    compressor_forward(kv_comp, x, w.mc_wkv, w.mc_wgate, w.mc_ape, w.mc_norm, w.cc_cos, w.cc_sin,
                       s, DIM, HEAD_DIM, ratio, overlap, ROPE_DIM, eps, false, stream);
    CU(cudaMemcpyAsync(kv_all, kv_win, (size_t)bs*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)bs*HEAD_DIM, kv_comp, (size_t)T*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));

    // --- compressed idxs (offset = s, into the compressed region) ---
    int itopk; int* compress_topk;
    if (has_indexer) {
        itopk = w.index_topk < T ? w.index_topk : T;
        float* idx_score; CU(cudaMalloc(&idx_score,(size_t)s*T*4)); CU(cudaMalloc(&compress_topk,(size_t)s*itopk*4));
        indexer_forward(idx_score, compress_topk, x, qr, w.idx_wq_b, w.idx_wq_b_s, w.idx_weights_proj,
                        w.idx_c_wkv, w.idx_c_wgate, w.idx_c_ape, w.idx_c_norm, a.cosT, a.sinT, w.cc_cos, w.cc_sin,
                        s, DIM, Q_LORA, w.index_n_heads, w.index_head_dim, ROPE_DIM, ratio, w.index_topk, s, eps, stream);
        cudaFree(idx_score);
    } else {
        // strided (get_compress_topk_idxs, prefill): compress[i,t] = (t >= (i+1)/ratio) ? -1 : t + s
        itopk = T; std::vector<int> hc((size_t)s * T);
        for (int i = 0; i < s; ++i) { int thr = (i + 1) / ratio;
            for (int t = 0; t < T; ++t) hc[(size_t)i * T + t] = (t >= thr) ? -1 : t + s; }
        CU(cudaMalloc(&compress_topk,(size_t)s*T*4));
        CU(cudaMemcpyAsync(compress_topk, hc.data(), (size_t)s*T*4, cudaMemcpyHostToDevice, stream));
    }

    // --- window idxs (host) ⊕ compressed idxs -> combined ---
    int wwidth = s < win ? s : win;
    std::vector<int> hw((size_t)s * wwidth);
    for (int i = 0; i < s; ++i) { int base = i - win + 1; if (base < 0) base = 0;
        for (int k = 0; k < wwidth; ++k) { int v = base + k; hw[(size_t)i * wwidth + k] = (v > i) ? -1 : v; } }
    int* window_dev; CU(cudaMalloc(&window_dev,(size_t)s*wwidth*4));
    CU(cudaMemcpyAsync(window_dev, hw.data(), (size_t)s*wwidth*4, cudaMemcpyHostToDevice, stream));
    int tot = wwidth + itopk; int* combined; CU(cudaMalloc(&combined,(size_t)s*tot*4));
    k_combine_idxs<<<(s*tot+255)/256,256,0,stream>>>(combined, window_dev, compress_topk, s, wwidth, itopk);

    // --- sparse attention over combined KV, then de-rotate + grouped o-LoRA + wo_b ---
    sparse_attn(o, q, kv_all, a.attn_sink, combined, 1, s, N_HEADS, HEAD_DIM, s + T, tot, scale, stream);
    rope_interleaved(o + NOPE_DIM, a.cosT, a.sinT, bs * N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, a.wo_a, bs, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, bs, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, a.wo_b, a.wo_b_s, bs, DIM, OB, stream);

    CU(cudaStreamSynchronize(stream));
    cudaFree(xq);cudaFree(xs);cudaFree(qr);cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(kv_win);
    cudaFree(kv_comp);cudaFree(kv_all);cudaFree(o);cudaFree(og);cudaFree(ogq);cudaFree(ogs);
    cudaFree(compress_topk);cudaFree(window_dev);cudaFree(combined);
}
