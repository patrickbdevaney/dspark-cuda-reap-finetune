// dspark_real.cu — real DSpark head composable pieces. See dspark_real.h / DSPARK_HEAD_BUILD.md.
#include "dspark_real.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // rmsnorm, act_quant_fp8
#include "compressor.h"    // gemm_fp32
#include <cstdio>
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
