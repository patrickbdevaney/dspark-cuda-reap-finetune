// decode.cu — full 43-layer M=1 KV-cache DECODE driver for DeepSeek-V4-Flash-180B-REAP (Step 4 milestone 3).
// Prefill-populates per-layer KV caches over [id0..id_{PS-1}], then autoregressively decodes M=1 tokens and
// measures decode tok/s. Gate: the first decoded token (input id_{s-1} at pos s-1) must argmax==270 (the same
// next-token the gated prefill produces at logits[s-1] for the canonical prompt). Memory-safe: weights load
// native (WeightStore), scales/norms/wo_a re-dequant PER LAYER with release() — same peak as the prefill forward
// (the per-token re-dequant is the first thing the native-dtype optimization removes).
//   build: bash scripts/build_decode.sh -> build/decode
#include <unordered_map>
#include "weight_store.h"
#include "deepseek_v4.h"
#include "block.h"
#include "compressed_block.h"
#include "block_decode.h"
#include "hc.h"
#include "mla_attn.h"
#include "compressor.h"
#include "yarn.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static std::string key_map(const std::string& in){ std::string s=in; if(s.rfind("model.",0)==0) s=s.substr(6); return s; }

__global__ void k_deq_e8m0(float* o, const uint8_t* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=exp2f((float)in[i]-127.f); }
__global__ void k_deq_bf16(float* o, const __nv_bfloat16* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=__bfloat162float(in[i]); }
__global__ void k_deq_fp8_blk(float* o, const uint8_t* w, const uint8_t* sc, int rows, int cols, int blk){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)rows*cols) return; int r=i/cols, c=i%cols;
    __half_raw hr=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)w[i], __NV_E4M3);
    float wv=__half2float(*reinterpret_cast<__half*>(&hr));
    int scw=cols/blk; float sv=exp2f((float)sc[(size_t)(r/blk)*scw + c/blk]-127.f);
    o[i]=wv*sv;
}
__global__ void k_embed(float* h, const __nv_bfloat16* emb, const int* ids, int s, int dim){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*dim) return; int t=i/dim, j=i%dim;
    h[i]=__bfloat162float(emb[(size_t)ids[t]*dim + j]);
}
__global__ void k_hc_expand(float* out, const float* h, int s, int hc, int dim){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*hc*dim) return; int t=i/(hc*dim), j=i%dim;
    out[i]=h[(size_t)t*dim+j];
}

struct Loader {
    st::WeightStore& W; std::vector<void*> allocs;
    Loader(st::WeightStore& w):W(w){}
    ~Loader(){ for(void*p:allocs) cudaFree(p); }
    size_t mark(){ return allocs.size(); }
    void release(size_t m){ for(size_t i=m;i<allocs.size();++i) cudaFree(allocs[i]); allocs.resize(m); }
    const uint8_t* raw(const std::string& n){ return W.dev<uint8_t>(n); }
    const float* f32(const std::string& n){ return W.dev<float>(n); }
    float* alloc(size_t nb){ void* p; CU(cudaMalloc(&p,nb)); allocs.push_back(p); return (float*)p; }
    const float* scale(const std::string& n){ auto& t=W.get(n); size_t ne=t.numel(); float* o=alloc(ne*4);
        k_deq_e8m0<<<(ne+255)/256,256>>>(o,(const uint8_t*)t.dev,ne); return o; }
    const float* bf16(const std::string& n){ auto& t=W.get(n); size_t ne=t.numel(); float* o=alloc(ne*4);
        k_deq_bf16<<<(ne+255)/256,256>>>(o,(const __nv_bfloat16*)t.dev,ne); return o; }
    const float* wo_a(const std::string& wn, const std::string& sn){ auto& t=W.get(wn);
        int rows=t.shape[0], cols=t.shape[1]; size_t ne=(size_t)rows*cols; float* o=alloc(ne*4);
        k_deq_fp8_blk<<<(ne+255)/256,256>>>(o,(const uint8_t*)t.dev,(const uint8_t*)W.get(sn).dev,rows,cols,128); return o; }
};
static const float* up_f(const std::vector<float>& v, std::vector<void*>& keep){
    void* d; CU(cudaMalloc(&d,v.size()*4)); CU(cudaMemcpy(d,v.data(),v.size()*4,cudaMemcpyHostToDevice)); keep.push_back(d); return (const float*)d; }
static std::vector<float> stride_rows(const std::vector<float>& in, int s, int half, int ratio){
    std::vector<float> o((size_t)(s/ratio)*half); for(int g=0; g<s/ratio; ++g) for(int j=0;j<half;++j) o[(size_t)g*half+j]=in[(size_t)(g*ratio)*half+j]; return o; }

int main(int argc, char** argv){
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dir = argc>1?argv[1]:"/home/patrickd/models/DeepSeek-V4-Flash-180B";
    std::vector<int> ids;
    if(argc>2 && strchr(argv[2],',')){ char* tok=strtok(argv[2],","); while(tok){ ids.push_back(atoi(tok)); tok=strtok(nullptr,","); } }
    else { for(int i=0;i<8;++i) ids.push_back((int[]){671,6102,294,8760,344,270,106523,294}[i]); }
    int s = ids.size();
    int NDEC = argc>3?atoi(argv[3]):6;                 // tokens to decode (autoregressive) after prefill
    int PS = s-1;                                      // prefill positions 0..PS-1; decode starts at pos PS (=s-1)
    int seqmax = s + NDEC + 4;
    printf("[decode] loading %s ... s=%d NDEC=%d seqmax=%d\n", dir, s, NDEC, seqmax);
    st::WeightStore W(dir, key_map); Loader L(W);
    printf("[decode] loaded %.2f GiB, %zu tensors\n", W.loadedGiB(), W.count());
    const int half=ROPE_DIM/2, hc=HC_MULT, d=DIM;
    extern bool g_tc_fp8; g_tc_fp8=true; extern bool g_tc_ogroup; g_tc_ogroup=true;
    extern bool g_moe_grouped; g_moe_grouped=true; extern void tc_moe_clear_cache();

    // freqs over seqmax
    std::vector<void*> keep;
    std::vector<float> ssc,sss; yarn::freqs(ssc,sss,seqmax,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *slide_cos=up_f(ssc,keep), *slide_sin=up_f(sss,keep);
    std::vector<float> cqc_h,cqs_h; yarn::freqs(cqc_h,cqs_h,seqmax,ROPE_DIM,YARN_ORIG_MAXPOS,COMPRESS_ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *cqc=up_f(cqc_h,keep), *cqs=up_f(cqs_h,keep);
    const float *cc4c=up_f(stride_rows(cqc_h,seqmax,half,4),keep), *cc4s=up_f(stride_rows(cqs_h,seqmax,half,4),keep);
    const float *cc128c=up_f(stride_rows(cqc_h,seqmax,half,128),keep), *cc128s=up_f(stride_rows(cqs_h,seqmax,half,128),keep);

    // per-layer KV caches
    std::vector<LayerKV> KV(N_LAYERS);
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ int ratio=compress_ratio(Lyr);
        CU(cudaMalloc(&KV[Lyr].win_kv,(size_t)seqmax*HEAD_DIM*4));
        if(ratio){ CU(cudaMalloc(&KV[Lyr].xin,(size_t)seqmax*DIM*4));
            CU(cudaMalloc(&KV[Lyr].comp_kv,(size_t)(seqmax/ratio+2)*HEAD_DIM*4));
            if(ratio==4) CU(cudaMalloc(&KV[Lyr].idx_ckv,(size_t)(seqmax/ratio+2)*INDEX_HEAD_DIM*4)); }
    }
    int* d_ids; CU(cudaMalloc(&d_ids,seqmax*4));
    float *h0,*h,*h2,*collapsed,*logits;
    CU(cudaMalloc(&h0,(size_t)seqmax*d*4)); CU(cudaMalloc(&h,(size_t)seqmax*hc*d*4)); CU(cudaMalloc(&h2,(size_t)seqmax*hc*d*4));
    CU(cudaMalloc(&collapsed,(size_t)d*4)); CU(cudaMalloc(&logits,(size_t)VOCAB*4));
    // head weights (persistent)
    const float *head_w=L.bf16("head.weight"), *norm_w=L.bf16("norm.weight");
    const float *hc_fn=L.f32("hc_head_fn"), *hc_sc=L.f32("hc_head_scale"), *hc_bs=L.f32("hc_head_base");
    size_t head_mark=L.mark();                                   // keep head + freqs; per-layer dequant is above this

    std::vector<std::vector<const uint8_t*>> P1(N_LAYERS),P2(N_LAYERS),P3(N_LAYERS);
    std::vector<std::vector<const float*>> S1(N_LAYERS),S2(N_LAYERS),S3(N_LAYERS);
    auto fill_moe=[&](const std::string& pfx, bool is_hash, MoEWeights& m, int Lyr){
        std::string p=pfx+"ffn."; auto& p1=P1[Lyr];auto&p2=P2[Lyr];auto&p3=P3[Lyr];auto&s1=S1[Lyr];auto&s2=S2[Lyr];auto&s3=S3[Lyr];
        p1.clear();p2.clear();p3.clear();s1.clear();s2.clear();s3.clear();
        m.gate_w=L.bf16(p+"gate.weight"); m.is_hash=is_hash;
        m.gate_bias=is_hash?nullptr:(W.has(p+"gate.bias")?L.f32(p+"gate.bias"):nullptr);
        m.tid2eid=is_hash?(const long*)W.get(p+"gate.tid2eid").dev:nullptr;
        for(int e=0;e<N_ROUTED;++e){ std::string ep=p+"experts."+std::to_string(e)+".";
            p1.push_back(L.raw(ep+"w1.weight")); p2.push_back(L.raw(ep+"w2.weight")); p3.push_back(L.raw(ep+"w3.weight"));
            s1.push_back(L.scale(ep+"w1.scale")); s2.push_back(L.scale(ep+"w2.scale")); s3.push_back(L.scale(ep+"w3.scale")); }
        m.w1p=p1.data();m.w2p=p2.data();m.w3p=p3.data();m.w1sp=s1.data();m.w2sp=s2.data();m.w3sp=s3.data();
        std::string sp=p+"shared_experts.";
        m.sw1=L.raw(sp+"w1.weight");m.sw2=L.raw(sp+"w2.weight");m.sw3=L.raw(sp+"w3.weight");
        m.sw1s=L.scale(sp+"w1.scale");m.sw2s=L.scale(sp+"w2.scale");m.sw3s=L.scale(sp+"w3.scale");
        m.n_routed=N_ROUTED;m.n_act=N_ACT;m.dim=DIM;m.inter=MOE_INTER;m.vocab=VOCAB;m.route_scale=ROUTE_SCALE;m.swiglu_limit=SWIGLU_LIMIT;
        m.use_tc_pp=true;m.batched=true;m.device_route=true; };
    auto fill_attn=[&](const std::string& pfx, MLAWeights& a, bool compressed){
        std::string p=pfx+"attn.";
        a.wq_a=L.raw(p+"wq_a.weight");a.wq_a_s=L.scale(p+"wq_a.scale");a.wq_b=L.raw(p+"wq_b.weight");a.wq_b_s=L.scale(p+"wq_b.scale");
        a.wkv=L.raw(p+"wkv.weight");a.wkv_s=L.scale(p+"wkv.scale");a.wo_b=L.raw(p+"wo_b.weight");a.wo_b_s=L.scale(p+"wo_b.scale");
        a.q_norm=L.bf16(p+"q_norm.weight");a.kv_norm=L.bf16(p+"kv_norm.weight");
        a.wo_a=L.wo_a(p+"wo_a.weight",p+"wo_a.scale");a.attn_sink=L.f32(p+"attn_sink");
        a.cosT=compressed?cqc:slide_cos;a.sinT=compressed?cqs:slide_sin; };

    // build one layer's weights (dequant), run either prefill_cache (bs=PS) or a decode step (pos), then it's the
    // caller's job to L.release(mk). Returns via x_out.
    auto run_layer=[&](int Lyr, bool prefill, int pos, const float* x_in, float* x_out, const int* ids_dev){
        int ratio=compress_ratio(Lyr); std::string lp="layers."+std::to_string(Lyr)+".";
        if(ratio==0){
            BlockWeights b{}; fill_attn(lp,b.attn,false); fill_moe(lp,is_hash_layer(Lyr),b.ffn,Lyr);
            b.attn_norm=L.bf16(lp+"attn_norm.weight");b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn");b.hc_attn_scale=L.f32(lp+"hc_attn_scale");b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn");b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale");b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM;b.hc=HC_MULT;
            if(prefill) block_prefill_cache(x_out,x_in,ids_dev,b,PS,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            else        block_decode_step (x_out,x_in,ids_dev,b,pos,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
        } else {
            CompressedBlockWeights b{}; fill_attn(lp,b.attn.attn,true);
            std::string p=lp+"attn.";
            b.attn.mc_wkv=L.bf16(p+"compressor.wkv.weight");b.attn.mc_wgate=L.bf16(p+"compressor.wgate.weight");
            b.attn.mc_ape=L.f32(p+"compressor.ape");b.attn.mc_norm=L.bf16(p+"compressor.norm.weight");
            b.attn.cc_cos=(ratio==4)?cc4c:cc128c;b.attn.cc_sin=(ratio==4)?cc4s:cc128s;
            if(ratio==4){
                b.attn.idx_wq_b=L.raw(p+"indexer.wq_b.weight");b.attn.idx_wq_b_s=L.scale(p+"indexer.wq_b.scale");
                b.attn.idx_weights_proj=L.bf16(p+"indexer.weights_proj.weight");
                b.attn.idx_c_wkv=L.bf16(p+"indexer.compressor.wkv.weight");b.attn.idx_c_wgate=L.bf16(p+"indexer.compressor.wgate.weight");
                b.attn.idx_c_ape=L.f32(p+"indexer.compressor.ape");b.attn.idx_c_norm=L.bf16(p+"indexer.compressor.norm.weight");
            }
            b.attn.index_n_heads=INDEX_N_HEADS;b.attn.index_head_dim=INDEX_HEAD_DIM;b.attn.index_topk=INDEX_TOPK;
            fill_moe(lp,is_hash_layer(Lyr),b.ffn,Lyr);
            b.attn_norm=L.bf16(lp+"attn_norm.weight");b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn");b.hc_attn_scale=L.f32(lp+"hc_attn_scale");b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn");b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale");b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM;b.hc=HC_MULT;b.win=WINDOW;b.ratio=ratio;
            if(prefill) cblock_prefill_cache(x_out,x_in,ids_dev,b,PS,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            else        cblock_decode_step  (x_out,x_in,ids_dev,b,pos,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
        }
    };
    auto head_fwd=[&](const float* hstate, int* out_am){       // hc_head->norm->lm_head->argmax (1 token)
        hc_head(collapsed,hstate,hc_fn,hc_sc,hc_bs,1,hc,d,HC_EPS);
        rmsnorm(collapsed,collapsed,norm_w,1,d,EPS,true,0);
        gemm_fp32(logits,collapsed,head_w,1,VOCAB,d,0); CU(cudaDeviceSynchronize());
        std::vector<float> lg(VOCAB); CU(cudaMemcpy(lg.data(),logits,VOCAB*4,cudaMemcpyDeviceToHost));
        int am=0; for(int v=1;v<VOCAB;++v) if(lg[v]>lg[am]) am=v; *out_am=am; };

    // ---------------- PREFILL: populate caches over [id0..id_{PS-1}] ----------------
    CU(cudaMemcpy(d_ids,ids.data(),s*4,cudaMemcpyHostToDevice));
    k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
    k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
    printf("[decode] prefill %d positions...\n", PS);
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ size_t mk=L.mark();
        run_layer(Lyr,true,0,h,h2,d_ids); std::swap(h,h2);
        L.release(mk); tc_moe_clear_cache();
    }
    printf("[decode] prefill done. caches populated. starting decode.\n");

    // ---------------- DECODE: autoregressive M=1 ----------------
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    int cur = ids[s-1]; int first_am=-1; std::vector<int> gen;
    float total_ms=0;
    float *hd, *hd2; CU(cudaMalloc(&hd,(size_t)hc*d*4)); CU(cudaMalloc(&hd2,(size_t)hc*d*4));
    for(int step=0; step<NDEC; ++step){
        int pos = (s-1) + step;                                 // decode token `cur` at absolute position pos
        int* cur_dev; cur_dev=d_ids+pos; CU(cudaMemcpy(cur_dev,&cur,4,cudaMemcpyHostToDevice));
        cudaEventRecord(t0);
        k_embed<<<((size_t)d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,cur_dev,1,d);
        k_hc_expand<<<((size_t)hc*d+255)/256,256>>>(hd,h0,1,hc,d);
        float* xin=hd; float* xout=hd2;
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ size_t mk=L.mark();
            run_layer(Lyr,false,pos,xin,xout,cur_dev); std::swap(xin,xout);
            L.release(mk); tc_moe_clear_cache();
        }
        int am; head_fwd(xin,&am);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        if(step>0) total_ms+=ms;                                // exclude step 0 (warmup: repack + first dequant)
        if(step==0) first_am=am;
        gen.push_back(am); cur=am;
        printf("  step %d pos %d -> token %d  (%.1f ms%s)\n", step, pos, am, ms, step==0?" warmup":"");
    }
    double warm_ms = NDEC>1 ? total_ms/(NDEC-1) : total_ms;
    printf("\n[decode] first decoded token argmax = %d  (expect 270)  -> %s\n", first_am, first_am==270?"GATE PASS":"GATE FAIL");
    printf("[decode] generated:"); for(int g:gen) printf(" %d",g); printf("\n");
    printf("[decode] WARM decode: %.1f ms/tok = %.2f tok/s  (M=1 steady state, %d-step avg)\n", warm_ms, 1000.0/warm_ms, NDEC-1);
    size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[decode] mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0);
    return first_am==270?0:1;
}
