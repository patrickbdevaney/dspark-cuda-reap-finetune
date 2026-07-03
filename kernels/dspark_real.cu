// dspark_real.cu — real DSpark head composable pieces. See dspark_real.h / DSPARK_HEAD_BUILD.md.
#include "dspark_real.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // rmsnorm, act_quant_fp8
#include "compressor.h"    // gemm_fp32
#include "hc.h"            // hc_head
#include <cstdio>
#include <vector>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// main_x = main_norm( main_proj(main_hidden) ) : fp8 gemm (3d->d) then RMSNorm.
void dspark_main_x(float* main_x, const float* main_hidden, const uint8_t* main_proj, const float* main_proj_s,
                   const float* main_norm, int s, int dim, float eps, cudaStream_t stream){
    const int K = 3 * dim;                                   // main_hidden is [s, 3d]
    uint8_t* xq; float* xs;
    CU(cudaMalloc(&xq,(size_t)s*K)); CU(cudaMalloc(&xs,(size_t)s*(K/128)*4));
    act_quant_fp8(xq, xs, main_hidden, s, K, 128, stream);
    fp8_block_gemm(main_x, xq, xs, main_proj, main_proj_s, s, dim, K, stream);   // [s, dim]
    rmsnorm(main_x, main_x, main_norm, s, dim, eps, true, stream);
    CU(cudaStreamSynchronize(stream)); cudaFree(xq); cudaFree(xs);
}

// gather markov_w1[token] -> markov_embed[n, rank]  (rows of the rank-256 embedding table)
__global__ void k_gather_rows(float* out, const float* table, const int* ids, int n, int rank){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)n*rank) return;
    int t=i/rank, r=i%rank; out[i]=table[(size_t)ids[t]*rank + r];
}

// Markov head: embed = markov_w1[token] (rank); logits_bias = embed @ markov_w2^T  ([n,rank]x[vocab,rank]->[n,vocab]).
void dspark_markov(float* logits_bias, float* markov_embed, const int* token_ids,
                   const float* markov_w1, const float* markov_w2, int n, int vocab, int rank,
                   cudaStream_t stream){
    k_gather_rows<<<((size_t)n*rank+255)/256,256,0,stream>>>(markov_embed, markov_w1, token_ids, n, rank);
    gemm_fp32(logits_bias, markov_embed, markov_w2, n, vocab, rank, stream);      // C[n,vocab] = E[n,rank] @ W2[vocab,rank]^T
    CU(cudaStreamSynchronize(stream));
}

// ---- Piece 2: tap mean-pool over hc -> main_hidden[:, slot*d:] ----
__global__ void k_tap_pool(float* mh, const float* h, int s, int hc, int d, int slot, int n_taps){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*d) return; int t=i/d, j=i%d;
    float acc=0.f; for(int c=0;c<hc;++c) acc+=h[((size_t)t*hc+c)*d+j];
    mh[(size_t)t*(n_taps*d) + slot*d + j] = acc/(float)hc;                 // h.mean(dim=hc)
}
void dspark_tap_pool(float* main_hidden, const float* h, int s, int hc, int d, int slot, int n_taps, cudaStream_t stream){
    k_tap_pool<<<((size_t)s*d+255)/256,256,0,stream>>>(main_hidden,h,s,hc,d,slot,n_taps);
}

// ---- Piece 3: forward_head — block hidden -> greedy proposed block (with Markov bias) ----
__global__ void k_add_bias(float* logits, const float* bias, int n, int vocab){    // logits[t,:]+=bias[t,:]
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<(size_t)n*vocab) logits[i]+=bias[i];
}
void dspark_forward_head(int* output_ids, const float* x_block, const int* first_ids,
                         const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                         const float* norm, const float* lm_head, const float* markov_w1, const float* markov_w2,
                         int s, int block, int hc, int d, int vocab, int rank, float eps, cudaStream_t stream){
    const int N=s*block;
    float *collapsed,*logits,*bias,*membed; int *cur;
    CU(cudaMalloc(&collapsed,(size_t)N*d*4)); CU(cudaMalloc(&logits,(size_t)N*vocab*4));
    CU(cudaMalloc(&bias,(size_t)s*vocab*4)); CU(cudaMalloc(&membed,(size_t)s*rank*4)); CU(cudaMalloc(&cur,(size_t)s*4));
    // hc_head (hc 4->1) -> norm -> lm_head  over all N=s*block block-positions
    hc_head(collapsed, x_block, hc_head_fn, hc_head_scale, hc_head_base, N, hc, d, 1e-6f, stream);
    rmsnorm(collapsed, collapsed, norm, N, d, eps, true, stream);
    gemm_fp32(logits, collapsed, lm_head, N, vocab, d, stream);            // [s,block,vocab]
    CU(cudaStreamSynchronize(stream));
    // host AR loop: out[:,0]=first_ids; for i: markov(out[:,i]) bias into logits[:,i]; out[:,i+1]=argmax
    std::vector<int> out((size_t)s*(block+1)); std::vector<int> fid(s);
    CU(cudaMemcpy(fid.data(),first_ids,(size_t)s*4,cudaMemcpyDeviceToHost));
    for(int t=0;t<s;++t) out[(size_t)t*(block+1)+0]=fid[t];
    std::vector<float> lg((size_t)s*vocab);
    for(int i=0;i<block;++i){
        for(int t=0;t<s;++t) { int id=out[(size_t)t*(block+1)+i]; CU(cudaMemcpy(cur+t,&id,4,cudaMemcpyHostToDevice)); }
        dspark_markov(bias, membed, cur, markov_w1, markov_w2, s, vocab, rank, stream);
        // logits[:, i, :] += bias   (logits row for position i of each anchor)
        for(int t=0;t<s;++t) k_add_bias<<<(vocab+255)/256,256,0,stream>>>(logits+((size_t)t*block+i)*vocab, bias+(size_t)t*vocab, 1, vocab);
        CU(cudaStreamSynchronize(stream));
        for(int t=0;t<s;++t){ CU(cudaMemcpy(lg.data()+(size_t)t*vocab, logits+((size_t)t*block+i)*vocab, (size_t)vocab*4, cudaMemcpyDeviceToHost)); }
        for(int t=0;t<s;++t){ const float* r=lg.data()+(size_t)t*vocab; int a=0; for(int v=1;v<vocab;++v) if(r[v]>r[a])a=v; out[(size_t)t*(block+1)+i+1]=a; }
    }
    CU(cudaMemcpy(output_ids,out.data(),(size_t)s*(block+1)*4,cudaMemcpyHostToDevice));
    cudaFree(collapsed);cudaFree(logits);cudaFree(bias);cudaFree(membed);cudaFree(cur);
}
