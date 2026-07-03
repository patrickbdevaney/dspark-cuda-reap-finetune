// moe.cu — MoE primitives, correctness-first (Gate K oracle: ref/gen_units.py).
#include "moe.h"
#include "dscratch.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>

__constant__ float E2M1_MAG[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float dec_e4m3(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}
__device__ __forceinline__ float dec_fp4(uint8_t nib) {   // sign(bit3) | mag-index(bits0-2)
    float m = E2M1_MAG[nib & 7];
    return (nib & 8) ? -m : m;
}

// ---------------- fp4_gemm (fp8 act x fp4 weight) ----------------
// One warp per (m,n). Walk K; per K-block accumulate raw dot then apply act(per-128) & weight(per-32) scales.
__global__ void fp4_gemm_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                const uint8_t* __restrict__ B, const float* __restrict__ bs,
                                int M, int N, int K) {
    int n = blockIdx.x, m = blockIdx.y; if (m >= M || n >= N) return;
    int lane = threadIdx.x & 31;
    int KBa = K / 128, KBw = K / 32;
    const uint8_t* Arow = A + (size_t)m * K;
    const uint8_t* Bpack = B + (size_t)n * (K / 2);          // packed nibbles
    const float* asr = as + (size_t)m * KBa;
    const float* bsr = bs + (size_t)n * KBw;
    float acc = 0.f;
    for (int kb = 0; kb < KBw; ++kb) {                       // per 32-weight-block (constant weight scale)
        float sub = 0.f; int base = kb * 32;
        for (int j = lane; j < 32; j += 32) {                // 1 iter (32 lanes cover 32)
            int k = base + j;
            float av = dec_e4m3(Arow[k]) * asr[k / 128];
            uint8_t byte = Bpack[k >> 1];
            uint8_t nib = (k & 1) ? (byte >> 4) & 0xF : byte & 0xF;
            sub += av * dec_fp4(nib);
        }
        acc += sub * bsr[kb];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
void fp4_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp4, const float* b_s,
              int M, int N, int K, cudaStream_t stream) {
    dim3 grid(N, M); fp4_gemm_kernel<<<grid, 32, 0, stream>>>(C, A_fp8, a_s, B_fp4, b_s, M, N, K);
}

// ---------------- moe_router_score ----------------
// One block per token. Compute n_routed scores (sqrtsoftplus), pick top-k of (score+bias) by iterative
// max, gather the PRE-bias scores, renormalize, scale.
__global__ void router_kernel(float* __restrict__ weights, int* __restrict__ indices,
                              const float* __restrict__ x, const float* __restrict__ gate_w,
                              const float* __restrict__ bias, int n, int dim, int n_routed, int topk,
                              float route_scale) {
    int tok = blockIdx.x; if (tok >= n) return;
    extern __shared__ float sh[];                 // [n_routed] orig scores + [n_routed] sel scores
    float* orig = sh; float* sel = sh + n_routed;
    const float* xr = x + (size_t)tok * dim;
    for (int e = threadIdx.x; e < n_routed; e += blockDim.x) {
        const float* gw = gate_w + (size_t)e * dim;
        float d = 0.f; for (int j = 0; j < dim; ++j) d += xr[j] * gw[j];
        float sp = (d > 20.f) ? d : log1pf(expf(d));          // softplus (stable)
        float s = sqrtf(sp);
        orig[e] = s; sel[e] = s + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float wsum = 0.f;
        for (int t = 0; t < topk; ++t) {
            float best = -1e30f; int bi = -1;
            for (int e = 0; e < n_routed; ++e) if (sel[e] > best) { best = sel[e]; bi = e; }
            sel[bi] = -1e30f;                                  // remove
            indices[(size_t)tok * topk + t] = bi;
            weights[(size_t)tok * topk + t] = orig[bi];
            wsum += orig[bi];
        }
        for (int t = 0; t < topk; ++t)
            weights[(size_t)tok * topk + t] = weights[(size_t)tok * topk + t] / wsum * route_scale;
    }
}
void moe_router_score(float* weights, int* indices, const float* x, const float* gate_w,
                      const float* bias, int n, int dim, int n_routed, int topk,
                      float route_scale, cudaStream_t stream) {
    router_kernel<<<n, 64, 2 * n_routed * sizeof(float), stream>>>(weights, indices, x, gate_w, bias,
                                                                   n, dim, n_routed, topk, route_scale);
}

// ================= MoE forward composition =================
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include <vector>
#include <cstdio>
void tc_fp4_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int, int, int, cudaStream_t); // Marlin TC W4A8 (tc_moe_gemm.cu)
bool g_moe_grouped=false;   // STRUCTURAL_PLAN Step 1b: zero-sync grouped-GEMM MoE (removes off[] D2H, graph-capturable)
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void compute_scores_kernel(float* sc, const float* x, const float* gw, int bs, int dim, int nr){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=bs*nr) return;
    int t=i/nr, e=i%nr; const float* xr=x+(size_t)t*dim; const float* gr=gw+(size_t)e*dim;
    float d=0.f; for(int j=0;j<dim;++j) d+=xr[j]*gr[j];
    float sp = d>20.f ? d : log1pf(expf(d));
    sc[i]=sqrtf(sp);
}
__global__ void gather_hash_kernel(int* idx, const long* tid2eid, const int* ids, int bs, int na){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=bs*na) return;
    int t=i/na, s=i%na; idx[i]=(int)tid2eid[(size_t)ids[t]*na + s];
}
__global__ void gather_scale_kernel(float* w, const float* sc, const int* idx, int bs, int na, int nr, float rs){
    int t=blockIdx.x; if(t>=bs) return; if(threadIdx.x) return;
    float sum=0.f; for(int s=0;s<na;++s){ float v=sc[(size_t)t*nr+idx[(size_t)t*na+s]]; w[(size_t)t*na+s]=v; sum+=v; }
    for(int s=0;s<na;++s) w[(size_t)t*na+s]=w[(size_t)t*na+s]/sum*rs;
}
__global__ void router_topk_kernel(float* w, int* idx, const float* sc, const float* bias,
                                   int bs, int nr, int na, float rs){
    int t=blockIdx.x; if(t>=bs||threadIdx.x) return;
    extern __shared__ float sel[];
    for(int e=0;e<nr;++e) sel[e]=sc[(size_t)t*nr+e]+(bias?bias[e]:0.f);
    float sum=0.f;
    for(int s=0;s<na;++s){ float best=-1e30f; int bi=-1;
        for(int e=0;e<nr;++e) if(sel[e]>best){best=sel[e];bi=e;}
        sel[bi]=-1e30f; idx[(size_t)t*na+s]=bi; float o=sc[(size_t)t*nr+bi];
        w[(size_t)t*na+s]=o; sum+=o; }
    for(int s=0;s<na;++s) w[(size_t)t*na+s]=w[(size_t)t*na+s]/sum*rs;
}
__global__ void swiglu_kernel(float* h, const float* g, const float* u, int n, float lim, float weight){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gg=g[i], uu=u[i];
    if(lim>0.f){ gg=fminf(gg,lim); uu=fminf(fmaxf(uu,-lim),lim); }
    float s = gg/(1.f+expf(-gg));               // silu
    h[i]= weight * s * uu;
}
__global__ void accum_kernel(float* y, const float* v, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=v[i];
}
// --- batched/grouped-dispatch helpers ---
__global__ void k_gather_x(float* Xe, const float* x, const int* tok, int me, int dim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)me*dim) return; int r=i/dim, c=i%dim;
    Xe[i]=x[(long)tok[r]*dim+c];
}
__global__ void k_scatter_add(float* out, const float* OE, const int* tok, int me, int dim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)me*dim) return; int r=i/dim, c=i%dim;
    atomicAdd(&out[(long)tok[r]*dim+c], OE[i]);
}
// swiglu with per-row (per-token) routing weight; gate+up already share the quantized input tile.
__global__ void swiglu_wrow(float* h, const float* g, const float* u, const float* wrow, int me, int inter, float lim){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)me*inter) return; int r=i/inter;
    float gg=g[i], uu=u[i]; if(lim>0.f){ gg=fminf(gg,lim); uu=fminf(fmaxf(uu,-lim),lim); }
    h[i]= wrow[r] * (gg/(1.f+expf(-gg))) * uu;
}

// device-side MoE grouping (Step 1 -> CUDA graphs): counting-sort tokens by expert on the GPU (no host vector
// work / big copies). Within-expert order is nondeterministic (atomic) but each token's expert output is scattered
// back independently, so the final result is order-invariant -> gates cosine 1.0 vs the host grouping.
__global__ void k_moe_count(int* counts, const int* idx, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(&counts[idx[i]],1); }
__global__ void k_moe_prefix(int* off, const int* counts, int nr){   // nr<=~160: single-thread exclusive scan
    if(threadIdx.x||blockIdx.x) return; off[0]=0; for(int e=0;e<nr;++e) off[e+1]=off[e]+counts[e]; }
__global__ void k_moe_scatter(int* alltok, float* allwt, int* cursor, const int* idx, const float* wt,
                              const int* off, int bs, int na){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=bs*na) return; int t=i/na, e=idx[i];
    int pos=atomicAdd(&cursor[e],1); alltok[off[e]+pos]=t; allwt[off[e]+pos]=wt[i]; }

void moe_forward(float* out, const float* x, const int* input_ids, const MoEWeights& w, int bs, cudaStream_t stream){
    const int dim=w.dim, inter=w.inter, nr=w.n_routed, na=w.n_act;
    float *sc,*wt,*g,*u,*h,*hs,*xs,*oe; uint8_t *xq,*hq; int *idx;
    sc=(decltype(sc))dmalloc((size_t)bs*nr*4); wt=(decltype(wt))dmalloc((size_t)bs*na*4); idx=(decltype(idx))dmalloc((size_t)bs*na*4);
    xq=(decltype(xq))dmalloc(dim); xs=(decltype(xs))dmalloc((dim/128)*4);
    g=(decltype(g))dmalloc(inter*4); u=(decltype(u))dmalloc(inter*4); h=(decltype(h))dmalloc(inter*4);
    hq=(decltype(hq))dmalloc(inter); hs=(decltype(hs))dmalloc((inter/128)*4); oe=(decltype(oe))dmalloc(dim*4);
    CU(cudaMemsetAsync(out,0,(size_t)bs*dim*4,stream));

    compute_scores_kernel<<<(bs*nr+63)/64,64,0,stream>>>(sc,x,w.gate_w,bs,dim,nr);
    if(w.is_hash){
        gather_hash_kernel<<<(bs*na+63)/64,64,0,stream>>>(idx,w.tid2eid,input_ids,bs,na);
        gather_scale_kernel<<<bs,32,0,stream>>>(wt,sc,idx,bs,na,nr,w.route_scale);
    } else {
        router_topk_kernel<<<bs,32,nr*sizeof(float),stream>>>(wt,idx,sc,w.gate_bias,bs,nr,na,w.route_scale);
    }
    // ================= GROUPED zero-sync MoE (STRUCTURAL_PLAN Step 1b) =================
    // ONE grouped GEMM per stage (gate/up/down) over ALL experts, expert map built on-device from off[] — the
    // last per-layer host sync (the off[] D2H copy) is GONE, so this path is CUDA-graph-capturable. Gated
    // cosine-1.0 vs the per-expert path (gate_grouped_moe + the full-forward argmax gate).
    extern bool g_moe_grouped;
    if(w.batched && w.device_route && g_moe_grouped && w.w1p){
        void tc_ensure_repacked(uint8_t*, int, int, cudaStream_t);
        void tc_a_to_fp16(__half*, const uint8_t*, const float*, int, int, cudaStream_t);
        void tc_build_tiles(int*, int*, int*, const int*, int, cudaStream_t);
        void tc_fp4_grouped_gemm(float*, const __half*, const uint8_t* const*, const float* const*,
                                 const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
        void tc_fp4_grouped_gemm_e8m0(float*, const __half*, const uint8_t* const*, const uint8_t* const*,
                                 const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
        const int maxm = bs*na;
        // -- device counting-sort grouping (off_d KEPT on device; NO D2H) --
        int *counts,*off_d,*cursor,*alltok_d; float* allwt_d;
        counts=(decltype(counts))dmalloc(nr*4); off_d=(decltype(off_d))dmalloc((nr+1)*4); cursor=(decltype(cursor))dmalloc(nr*4);
        alltok_d=(decltype(alltok_d))dmalloc((size_t)maxm*4); allwt_d=(decltype(allwt_d))dmalloc((size_t)maxm*4);
        CU(cudaMemsetAsync(counts,0,nr*4,stream)); CU(cudaMemsetAsync(cursor,0,nr*4,stream));
        k_moe_count<<<(bs*na+63)/64,64,0,stream>>>(counts,idx,bs*na);
        k_moe_prefix<<<1,1,0,stream>>>(off_d,counts,nr);
        k_moe_scatter<<<(bs*na+63)/64,64,0,stream>>>(alltok_d,allwt_d,cursor,idx,wt,off_d,bs,na);
        // -- tile descriptors (device) --
        int *tile_e,*tile_row0,*ntiles_d; tile_e=(decltype(tile_e))dmalloc(maxm*4); tile_row0=(decltype(tile_row0))dmalloc(maxm*4); ntiles_d=(decltype(ntiles_d))dmalloc(4);
        tc_build_tiles(tile_e,tile_row0,ntiles_d,off_d,nr,stream);
        // -- repack every expert weight in place once (idempotent) + upload per-expert ptr tables to device --
        for(int e=0;e<nr;++e){ tc_ensure_repacked((uint8_t*)w.w1p[e],inter,dim,stream);
            tc_ensure_repacked((uint8_t*)w.w3p[e],inter,dim,stream); tc_ensure_repacked((uint8_t*)w.w2p[e],dim,inter,stream); }
        const uint8_t **w1d,**w3d,**w2d; void **s1d,**s3d,**s2d;   // s*d hold float* (dequant) OR uint8_t* (e8m0)
        w1d=(decltype(w1d))dmalloc(nr*sizeof(void*)); w3d=(decltype(w3d))dmalloc(nr*sizeof(void*)); w2d=(decltype(w2d))dmalloc(nr*sizeof(void*));
        s1d=(decltype(s1d))dmalloc(nr*sizeof(void*)); s3d=(decltype(s3d))dmalloc(nr*sizeof(void*)); s2d=(decltype(s2d))dmalloc(nr*sizeof(void*));
        CU(cudaMemcpyAsync(w1d,w.w1p,nr*sizeof(void*),cudaMemcpyHostToDevice,stream)); CU(cudaMemcpyAsync(w3d,w.w3p,nr*sizeof(void*),cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(w2d,w.w2p,nr*sizeof(void*),cudaMemcpyHostToDevice,stream));
        const void *hs1 = w.e8m0_scales?(const void*)w.w1sp8:(const void*)w.w1sp;
        const void *hs3 = w.e8m0_scales?(const void*)w.w3sp8:(const void*)w.w3sp;
        const void *hs2 = w.e8m0_scales?(const void*)w.w2sp8:(const void*)w.w2sp;
        CU(cudaMemcpyAsync(s1d,hs1,nr*sizeof(void*),cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(s3d,hs3,nr*sizeof(void*),cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(s2d,hs2,nr*sizeof(void*),cudaMemcpyHostToDevice,stream));
        // -- scratch (maxm rows) --
        float *Xe,*Xes,*Gb,*Ub,*Hb,*Hsb,*OEb; uint8_t *Xeq,*Hqb; __half *x16,*h16;
        Xe=(decltype(Xe))dmalloc((size_t)maxm*dim*4); Xeq=(decltype(Xeq))dmalloc((size_t)maxm*dim); Xes=(decltype(Xes))dmalloc((size_t)maxm*(dim/128)*4);
        x16=(decltype(x16))dmalloc((size_t)maxm*dim*2); h16=(decltype(h16))dmalloc((size_t)maxm*inter*2);
        Gb=(decltype(Gb))dmalloc((size_t)maxm*inter*4); Ub=(decltype(Ub))dmalloc((size_t)maxm*inter*4); Hb=(decltype(Hb))dmalloc((size_t)maxm*inter*4);
        Hqb=(decltype(Hqb))dmalloc((size_t)maxm*inter); Hsb=(decltype(Hsb))dmalloc((size_t)maxm*(inter/128)*4); OEb=(decltype(OEb))dmalloc((size_t)maxm*dim*4);
        // -- routed experts: gather -> quant -> fp16 -> grouped gate/up -> swiglu -> quant -> fp16 -> grouped down -> scatter --
        k_gather_x<<<((size_t)maxm*dim+255)/256,256,0,stream>>>(Xe,x,alltok_d,maxm,dim);
        act_quant_fp8(Xeq,Xes,Xe,maxm,dim,128,stream);
        tc_a_to_fp16(x16,Xeq,Xes,maxm,dim,stream);
        if(w.e8m0_scales){
            tc_fp4_grouped_gemm_e8m0(Gb,x16,w1d,(const uint8_t* const*)s1d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
            tc_fp4_grouped_gemm_e8m0(Ub,x16,w3d,(const uint8_t* const*)s3d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
        } else {
            tc_fp4_grouped_gemm(Gb,x16,w1d,(const float* const*)s1d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
            tc_fp4_grouped_gemm(Ub,x16,w3d,(const float* const*)s3d,off_d,tile_e,tile_row0,ntiles_d,maxm,inter,dim,stream);
        }
        swiglu_wrow<<<((size_t)maxm*inter+255)/256,256,0,stream>>>(Hb,Gb,Ub,allwt_d,maxm,inter,w.swiglu_limit);
        act_quant_fp8(Hqb,Hsb,Hb,maxm,inter,128,stream);
        tc_a_to_fp16(h16,Hqb,Hsb,maxm,inter,stream);
        if(w.e8m0_scales) tc_fp4_grouped_gemm_e8m0(OEb,h16,w2d,(const uint8_t* const*)s2d,off_d,tile_e,tile_row0,ntiles_d,maxm,dim,inter,stream);
        else              tc_fp4_grouped_gemm(OEb,h16,w2d,(const float* const*)s2d,off_d,tile_e,tile_row0,ntiles_d,maxm,dim,inter,stream);
        k_scatter_add<<<((size_t)maxm*dim+255)/256,256,0,stream>>>(out,OEb,alltok_d,maxm,dim);
        // -- shared expert (fp8, all bs tokens, weight 1) --
        if(!getenv("NOSHARED")){
            act_quant_fp8(Xeq,Xes,x,bs,dim,128,stream);
            fp8_block_gemm(Gb,Xeq,Xes,w.sw1,w.sw1s,bs,inter,dim,stream);
            fp8_block_gemm(Ub,Xeq,Xes,w.sw3,w.sw3s,bs,inter,dim,stream);
            swiglu_kernel<<<((size_t)bs*inter+63)/64,64,0,stream>>>(Hb,Gb,Ub,bs*inter,w.swiglu_limit,1.f);
            act_quant_fp8(Hqb,Hsb,Hb,bs,inter,128,stream);
            fp8_block_gemm(OEb,Hqb,Hsb,w.sw2,w.sw2s,bs,dim,inter,stream);
            accum_kernel<<<((size_t)bs*dim+63)/64,64,0,stream>>>(out,OEb,bs*dim);
        }
        dsync(stream);
        dfree(counts);dfree(off_d);dfree(cursor);dfree(alltok_d);dfree(allwt_d);
        dfree(tile_e);dfree(tile_row0);dfree(ntiles_d);
        dfree(w1d);dfree(w3d);dfree(w2d);dfree(s1d);dfree(s3d);dfree(s2d);
        dfree(Xe);dfree(Xeq);dfree(Xes);dfree(x16);dfree(h16);
        dfree(Gb);dfree(Ub);dfree(Hb);dfree(Hqb);dfree(Hsb);dfree(OEb);
        dfree(sc);dfree(wt);dfree(idx);dfree(xq);dfree(xs);dfree(g);dfree(u);dfree(h);dfree(hq);dfree(hs);dfree(oe);
        return;
    }

    std::vector<int> hidx; std::vector<float> hw;
    if(!(w.batched && w.device_route)){                       // device_route does the grouping on-GPU (no host copy)
        hidx.resize((size_t)bs*na); hw.resize((size_t)bs*na);
        dsync(stream);
        CU(cudaMemcpy(hidx.data(),idx,(size_t)bs*na*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hw.data(),wt,(size_t)bs*na*4,cudaMemcpyDeviceToHost));
    }

    size_t w13n=(size_t)inter*(dim/2), w13s=(size_t)inter*(dim/32);
    size_t w2n=(size_t)dim*(inter/2),  w2s=(size_t)dim*(inter/32);
    void tc_fp4_gemm_pp_auto(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
    auto GEMM = w.use_tc_pp ? tc_fp4_gemm_pp_auto : (w.use_tc ? tc_fp4_gemm : fp4_gemm);
    if(w.batched){
        // Scratch must hold the LARGEST per-expert group: a token can route to the same expert in multiple na
        // slots (esp. hash layers), so me can exceed bs — up to bs*na total assignments. Size for bs*na.
        const int maxm = bs*na;
        float *Xe,*Xes2,*Gb,*Ub,*Hb,*Hsb,*OEb; uint8_t *Xeq,*Hqb;
        Xe=(decltype(Xe))dmalloc((size_t)maxm*dim*4); Xeq=(decltype(Xeq))dmalloc((size_t)maxm*dim); Xes2=(decltype(Xes2))dmalloc((size_t)maxm*(dim/128)*4);
        Gb=(decltype(Gb))dmalloc((size_t)maxm*inter*4); Ub=(decltype(Ub))dmalloc((size_t)maxm*inter*4); Hb=(decltype(Hb))dmalloc((size_t)maxm*inter*4);
        Hqb=(decltype(Hqb))dmalloc((size_t)maxm*inter); Hsb=(decltype(Hsb))dmalloc((size_t)maxm*(inter/128)*4); OEb=(decltype(OEb))dmalloc((size_t)maxm*dim*4);
        std::vector<int> off(nr+1,0); int* alltok_d; float* allwt_d;
        if(w.device_route){
            // DEVICE grouping: counting-sort on GPU (no host vector build / big idx-wt-alltok copies).
            int *counts,*off_d,*cursor;
            counts=(decltype(counts))dmalloc(nr*4); off_d=(decltype(off_d))dmalloc((nr+1)*4); cursor=(decltype(cursor))dmalloc(nr*4);
            alltok_d=(decltype(alltok_d))dmalloc((size_t)maxm*4); allwt_d=(decltype(allwt_d))dmalloc((size_t)maxm*4);
            CU(cudaMemsetAsync(counts,0,nr*4,stream)); CU(cudaMemsetAsync(cursor,0,nr*4,stream));
            k_moe_count<<<(bs*na+63)/64,64,0,stream>>>(counts,idx,bs*na);
            k_moe_prefix<<<1,1,0,stream>>>(off_d,counts,nr);
            k_moe_scatter<<<(bs*na+63)/64,64,0,stream>>>(alltok_d,allwt_d,cursor,idx,wt,off_d,bs,na);
            CU(cudaMemcpy(off.data(),off_d,(nr+1)*4,cudaMemcpyDeviceToHost));   // small sync (grid sizes); zero-sync = Step 1b
            dfree(counts); dfree(off_d); dfree(cursor);
        } else {
            // HOST grouping (oracle): build per-expert lists on host, upload flat arrays + offsets.
            std::vector<std::vector<int>> etok(nr); std::vector<std::vector<float>> ewt(nr);
            for(int t=0;t<bs;++t) for(int s=0;s<na;++s){ int e=hidx[(size_t)t*na+s]; etok[e].push_back(t); ewt[e].push_back(hw[(size_t)t*na+s]); }
            std::vector<int> alltok; std::vector<float> allwt;
            for(int e=0;e<nr;++e){ for(int t:etok[e]) alltok.push_back(t); for(float wv:ewt[e]) allwt.push_back(wv); off[e+1]=alltok.size(); }
            alltok_d=(decltype(alltok_d))dmalloc((size_t)alltok.size()*4); allwt_d=(decltype(allwt_d))dmalloc((size_t)allwt.size()*4);
            CU(cudaMemcpy(alltok_d,alltok.data(),(size_t)alltok.size()*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(allwt_d,allwt.data(),(size_t)allwt.size()*4,cudaMemcpyHostToDevice));
        }
        for(int e=0;e<nr;++e){ int me=off[e+1]-off[e]; if(!me) continue;
            const int* tok_d = alltok_d+off[e]; const float* wrow = allwt_d+off[e];
            const uint8_t *W1=w.w1p?w.w1p[e]:w.w1+(size_t)e*w13n, *W3=w.w3p?w.w3p[e]:w.w3+(size_t)e*w13n, *W2=w.w2p?w.w2p[e]:w.w2+(size_t)e*w2n;
            const float *W1s=w.w1sp?w.w1sp[e]:w.w1s+(size_t)e*w13s, *W3s=w.w3sp?w.w3sp[e]:w.w3s+(size_t)e*w13s, *W2s=w.w2sp?w.w2sp[e]:w.w2s+(size_t)e*w2s;
            k_gather_x<<<((size_t)me*dim+255)/256,256,0,stream>>>(Xe,x,tok_d,me,dim);
            act_quant_fp8(Xeq,Xes2,Xe,me,dim,128,stream);
            GEMM(Gb,Xeq,Xes2, W1,W1s, me,inter,dim,stream);   // gate+up share the quantized input tile Xeq
            GEMM(Ub,Xeq,Xes2, W3,W3s, me,inter,dim,stream);
            swiglu_wrow<<<((size_t)me*inter+255)/256,256,0,stream>>>(Hb,Gb,Ub,wrow,me,inter,w.swiglu_limit);
            act_quant_fp8(Hqb,Hsb,Hb,me,inter,128,stream);
            GEMM(OEb,Hqb,Hsb, W2,W2s, me,dim,inter,stream);
            k_scatter_add<<<((size_t)me*dim+255)/256,256,0,stream>>>(out,OEb,tok_d,me,dim);
        }
        // shared expert, all bs tokens (fp8, weight 1)
        if(!getenv("NOSHARED")){
        act_quant_fp8(Xeq,Xes2,x,bs,dim,128,stream);
        fp8_block_gemm(Gb,Xeq,Xes2, w.sw1,w.sw1s, bs,inter,dim,stream);
        fp8_block_gemm(Ub,Xeq,Xes2, w.sw3,w.sw3s, bs,inter,dim,stream);
        swiglu_kernel<<<((size_t)bs*inter+63)/64,64,0,stream>>>(Hb,Gb,Ub,bs*inter,w.swiglu_limit,1.f);
        act_quant_fp8(Hqb,Hsb,Hb,bs,inter,128,stream);
        fp8_block_gemm(OEb,Hqb,Hsb, w.sw2,w.sw2s, bs,dim,inter,stream);
        accum_kernel<<<((size_t)bs*dim+63)/64,64,0,stream>>>(out,OEb,bs*dim);
        }
        dsync(stream);
        dfree(Xe);dfree(Xeq);dfree(Xes2);dfree(Gb);dfree(Ub);dfree(Hb);dfree(Hqb);dfree(Hsb);dfree(OEb);dfree(alltok_d);dfree(allwt_d);
        dfree(sc);dfree(wt);dfree(idx);dfree(xq);dfree(xs);dfree(g);dfree(u);dfree(h);dfree(hq);dfree(hs);dfree(oe);
        return;
    }
    for(int t=0;t<bs;++t){
        const float* xr=x+(size_t)t*dim;
        // routed experts (fp4)
        for(int s=0;s<na;++s){
            int e=hidx[(size_t)t*na+s]; float wgt=hw[(size_t)t*na+s];
            // stacked base+stride, OR per-expert pointer table (real checkpoint) when w1p != null.
            const uint8_t *W1 = w.w1p? w.w1p[e] : w.w1+(size_t)e*w13n, *W3 = w.w3p? w.w3p[e] : w.w3+(size_t)e*w13n,
                          *W2 = w.w2p? w.w2p[e] : w.w2+(size_t)e*w2n;
            const float *W1s = w.w1sp? w.w1sp[e] : w.w1s+(size_t)e*w13s, *W3s = w.w3sp? w.w3sp[e] : w.w3s+(size_t)e*w13s,
                        *W2s = w.w2sp? w.w2sp[e] : w.w2s+(size_t)e*w2s;
            act_quant_fp8(xq,xs,xr,1,dim,128,stream);
            GEMM(g,xq,xs, W1, W1s, 1,inter,dim,stream);
            GEMM(u,xq,xs, W3, W3s, 1,inter,dim,stream);
            swiglu_kernel<<<(inter+63)/64,64,0,stream>>>(h,g,u,inter,w.swiglu_limit,wgt);
            act_quant_fp8(hq,hs,h,1,inter,128,stream);
            GEMM(oe,hq,hs, W2, W2s, 1,dim,inter,stream);
            accum_kernel<<<(dim+63)/64,64,0,stream>>>(out+(size_t)t*dim,oe,dim);
        }
        // shared expert (fp8), no routing weight
        if(getenv("NOSHARED")) continue;
        act_quant_fp8(xq,xs,xr,1,dim,128,stream);
        fp8_block_gemm(g,xq,xs, w.sw1, w.sw1s, 1,inter,dim,stream);
        fp8_block_gemm(u,xq,xs, w.sw3, w.sw3s, 1,inter,dim,stream);
        swiglu_kernel<<<(inter+63)/64,64,0,stream>>>(h,g,u,inter,w.swiglu_limit,1.f);
        act_quant_fp8(hq,hs,h,1,inter,128,stream);
        fp8_block_gemm(oe,hq,hs, w.sw2, w.sw2s, 1,dim,inter,stream);
        accum_kernel<<<(dim+63)/64,64,0,stream>>>(out+(size_t)t*dim,oe,dim);
    }
    dsync(stream);
    dfree(sc);dfree(wt);dfree(idx);dfree(xq);dfree(xs);dfree(g);dfree(u);
    dfree(h);dfree(hq);dfree(hs);dfree(oe);
}
