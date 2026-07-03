// forward.cu — full DeepSeek-V4-Flash-180B-REAP forward on Thor. embed -> HC-expand -> 43 blocks
// (block_forward L0-1 / compressed_block_forward L2-42) -> hc_head -> norm -> lm_head -> logits.
// Weights load zero-copy via WeightStore; dtypes the kernels need as fp32 (e8m0 scales, wo_a fp8, bf16
// norms/compressor/head) are dequantized at load. See ROADMAP Phase A. Gate 1 = this runs on real weights.
//   build: nvcc -O2 -std=c++17 -arch=sm_110a -I include src/forward.cu kernels/*.cu -o build/forward
#include "weight_store.h"
#include "deepseek_v4.h"
#include "block.h"
#include "compressed_block.h"
#include "dspark.h"        // DSpark MTP draft head
#include "hc.h"            // hc_head
#include "mla_attn.h"      // rmsnorm
#include "compressor.h"    // gemm_fp32
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

// ---------------- dequant kernels ----------------
__global__ void k_deq_e8m0(float* o, const uint8_t* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=exp2f((float)in[i]-127.f); }
__global__ void k_deq_bf16(float* o, const __nv_bfloat16* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=__bfloat162float(in[i]); }
// fp8(e4m3) weight * e8m0 block scale -> fp32. rows x cols, scale [rows/blk, cols/blk].
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
__global__ void k_hc_expand(float* out, const float* h, int s, int hc, int dim){   // [s,dim]->[s,hc,dim] repeat
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)s*hc*dim) return; int t=i/(hc*dim), j=i%dim;
    out[i]=h[(size_t)t*dim+j];
}

// ---------------- dequant-caching loader ----------------
struct Loader {
    st::WeightStore& W; std::vector<void*> allocs;
    Loader(st::WeightStore& w):W(w){}
    ~Loader(){ for(void*p:allocs) cudaFree(p); }
    size_t mark(){ return allocs.size(); }                           // per-layer dequant scoping: free
    void release(size_t m){ for(size_t i=m;i<allocs.size();++i) cudaFree(allocs[i]); allocs.resize(m); }  // buffers after the block runs (synced)
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
    setvbuf(stdout, nullptr, _IONBF, 0);          // unbuffered: see progress live (esp. if killed)
    const char* dir = argc>1?argv[1]:"/home/patrickd/models/DeepSeek-V4-Flash-180B";
    std::vector<int> ids; int s;
    if(argc>2 && strchr(argv[2],',')){                       // explicit comma-separated token ids
        char* tok=strtok(argv[2],","); while(tok){ ids.push_back(atoi(tok)); tok=strtok(nullptr,","); } s=ids.size();
    } else { s = argc>2?atoi(argv[2]):8; ids.resize(s); for(int i=0;i<s;++i) ids[i]=(i*131+7)%VOCAB; }
    printf("[forward] loading %s (single-tenant, ~96 GiB)...  s=%d ids:", dir, s); for(int v:ids) printf(" %d",v); printf("\n");
    st::WeightStore W(dir, key_map);
    printf("[forward] loaded %.2f GiB, %zu tensors. prefill s=%d\n", W.loadedGiB(), W.count(), s);
    Loader L(W);
    const int half=ROPE_DIM/2, hc=HC_MULT, d=DIM;

    std::vector<void*> keep;
    std::vector<float> ssc,sss; yarn::freqs(ssc,sss,s,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *slide_cos=up_f(ssc,keep), *slide_sin=up_f(sss,keep);
    std::vector<float> cqc_h,cqs_h; yarn::freqs(cqc_h,cqs_h,s,ROPE_DIM,YARN_ORIG_MAXPOS,COMPRESS_ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
    const float *cqc=up_f(cqc_h,keep), *cqs=up_f(cqs_h,keep);
    const float *cc4c=up_f(stride_rows(cqc_h,s,half,4),keep), *cc4s=up_f(stride_rows(cqs_h,s,half,4),keep);
    const float *cc128c=(s>=128)?up_f(stride_rows(cqc_h,s,half,128),keep):cqc, *cc128s=(s>=128)?up_f(stride_rows(cqs_h,s,half,128),keep):cqs;

    int* d_ids; CU(cudaMalloc(&d_ids,s*4)); CU(cudaMemcpy(d_ids,ids.data(),s*4,cudaMemcpyHostToDevice));
    float *h0, *h, *h2;
    CU(cudaMalloc(&h0,(size_t)s*d*4)); CU(cudaMalloc(&h,(size_t)s*hc*d*4)); CU(cudaMalloc(&h2,(size_t)s*hc*d*4));
    k_embed<<<((size_t)s*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,s,d);
    k_hc_expand<<<((size_t)s*hc*d+255)/256,256>>>(h,h0,s,hc,d);
    CU(cudaDeviceSynchronize());

    auto fill_moe=[&](const std::string& pfx, bool is_hash, MoEWeights& m, std::vector<const uint8_t*>& p1,std::vector<const uint8_t*>& p2,std::vector<const uint8_t*>& p3,
                      std::vector<const float*>& s1,std::vector<const float*>& s2,std::vector<const float*>& s3){
        std::string p=pfx+"ffn.";
        m.gate_w=L.bf16(p+"gate.weight"); m.is_hash=is_hash;
        m.gate_bias=m.is_hash?nullptr:(W.has(p+"gate.bias")?L.f32(p+"gate.bias"):nullptr);
        m.tid2eid=m.is_hash?(const long*)W.get(p+"gate.tid2eid").dev:nullptr;
        for(int e=0;e<N_ROUTED;++e){ std::string ep=p+"experts."+std::to_string(e)+".";
            p1.push_back(L.raw(ep+"w1.weight")); p2.push_back(L.raw(ep+"w2.weight")); p3.push_back(L.raw(ep+"w3.weight"));
            s1.push_back(L.scale(ep+"w1.scale")); s2.push_back(L.scale(ep+"w2.scale")); s3.push_back(L.scale(ep+"w3.scale")); }
        m.w1p=p1.data(); m.w2p=p2.data(); m.w3p=p3.data(); m.w1sp=s1.data(); m.w2sp=s2.data(); m.w3sp=s3.data();
        std::string sp=p+"shared_experts.";
        m.sw1=L.raw(sp+"w1.weight"); m.sw2=L.raw(sp+"w2.weight"); m.sw3=L.raw(sp+"w3.weight");
        m.sw1s=L.scale(sp+"w1.scale"); m.sw2s=L.scale(sp+"w2.scale"); m.sw3s=L.scale(sp+"w3.scale");
        m.n_routed=N_ROUTED; m.n_act=N_ACT; m.dim=DIM; m.inter=MOE_INTER; m.vocab=VOCAB; m.route_scale=ROUTE_SCALE; m.swiglu_limit=SWIGLU_LIMIT; };
    auto fill_attn=[&](const std::string& pfx, MLAWeights& a, bool compressed){
        std::string p=pfx+"attn.";
        a.wq_a=L.raw(p+"wq_a.weight"); a.wq_a_s=L.scale(p+"wq_a.scale"); a.wq_b=L.raw(p+"wq_b.weight"); a.wq_b_s=L.scale(p+"wq_b.scale");
        a.wkv=L.raw(p+"wkv.weight"); a.wkv_s=L.scale(p+"wkv.scale"); a.wo_b=L.raw(p+"wo_b.weight"); a.wo_b_s=L.scale(p+"wo_b.scale");
        a.q_norm=L.bf16(p+"q_norm.weight"); a.kv_norm=L.bf16(p+"kv_norm.weight");
        a.wo_a=L.wo_a(p+"wo_a.weight",p+"wo_a.scale"); a.attn_sink=L.f32(p+"attn_sink");
        a.cosT=compressed?cqc:slide_cos; a.sinT=compressed?cqs:slide_sin; };

    std::vector<std::vector<const uint8_t*>> P1(N_LAYERS),P2(N_LAYERS),P3(N_LAYERS);
    std::vector<std::vector<const float*>> S1(N_LAYERS),S2(N_LAYERS),S3(N_LAYERS);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){
        int ratio=compress_ratio(Lyr);
        std::string lp="layers."+std::to_string(Lyr)+".";
        size_t mk=L.mark();                                   // free this layer's dequant buffers after the block
        if(ratio==0){
            BlockWeights b{}; fill_attn(lp,b.attn,false);
            fill_moe(lp,is_hash_layer(Lyr),b.ffn,P1[Lyr],P2[Lyr],P3[Lyr],S1[Lyr],S2[Lyr],S3[Lyr]);
            b.attn_norm=L.bf16(lp+"attn_norm.weight"); b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn"); b.hc_attn_scale=L.f32(lp+"hc_attn_scale"); b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn"); b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale"); b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM; b.hc=HC_MULT;
            block_forward(h2,h,d_ids,b,s,HC_SINKHORN_ITERS,EPS);
        } else {
            CompressedBlockWeights b{}; fill_attn(lp,b.attn.attn,true);
            std::string p=lp+"attn.";
            b.attn.mc_wkv=L.bf16(p+"compressor.wkv.weight"); b.attn.mc_wgate=L.bf16(p+"compressor.wgate.weight");
            b.attn.mc_ape=L.f32(p+"compressor.ape"); b.attn.mc_norm=L.bf16(p+"compressor.norm.weight");
            b.attn.cc_cos=(ratio==4)?cc4c:cc128c; b.attn.cc_sin=(ratio==4)?cc4s:cc128s;
            if(ratio==4){
                b.attn.idx_wq_b=L.raw(p+"indexer.wq_b.weight"); b.attn.idx_wq_b_s=L.scale(p+"indexer.wq_b.scale");
                b.attn.idx_weights_proj=L.bf16(p+"indexer.weights_proj.weight");
                b.attn.idx_c_wkv=L.bf16(p+"indexer.compressor.wkv.weight"); b.attn.idx_c_wgate=L.bf16(p+"indexer.compressor.wgate.weight");
                b.attn.idx_c_ape=L.f32(p+"indexer.compressor.ape"); b.attn.idx_c_norm=L.bf16(p+"indexer.compressor.norm.weight");
            }
            b.attn.index_n_heads=INDEX_N_HEADS; b.attn.index_head_dim=INDEX_HEAD_DIM; b.attn.index_topk=INDEX_TOPK;
            fill_moe(lp,is_hash_layer(Lyr),b.ffn,P1[Lyr],P2[Lyr],P3[Lyr],S1[Lyr],S2[Lyr],S3[Lyr]);
            b.attn_norm=L.bf16(lp+"attn_norm.weight"); b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn"); b.hc_attn_scale=L.f32(lp+"hc_attn_scale"); b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn"); b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale"); b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM; b.hc=HC_MULT; b.win=WINDOW; b.ratio=ratio;
            compressed_block_forward(h2,h,d_ids,b,s,HC_SINKHORN_ITERS,EPS);
        }
        std::swap(h,h2);
        L.release(mk);                                        // block synced internally -> safe to free layer dequant
        if(Lyr%4==0){ size_t fb,tb; cudaMemGetInfo(&fb,&tb);
            printf("  layer %d/%d done (ratio %d)  mem %.1f/%.1f GiB\n",Lyr,N_LAYERS,ratio,(tb-fb)/1073741824.0,tb/1073741824.0); }
    }

    // --- head: hc_head (4->1) -> final norm -> lm_head -> logits[s,vocab] ---
    float* collapsed; CU(cudaMalloc(&collapsed,(size_t)s*d*4));
    hc_head(collapsed, h, L.f32("hc_head_fn"), L.f32("hc_head_scale"), L.f32("hc_head_base"), s, hc, d, HC_EPS);
    rmsnorm(collapsed, collapsed, L.bf16("norm.weight"), s, d, EPS, true, 0);
    float* logits; CU(cudaMalloc(&logits,(size_t)s*VOCAB*4));
    gemm_fp32(logits, collapsed, L.bf16("head.weight"), s, VOCAB, d, 0);   // [s,vocab]
    CU(cudaDeviceSynchronize()); cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);

    std::vector<float> lg((size_t)VOCAB); CU(cudaMemcpy(lg.data(),logits+(size_t)(s-1)*VOCAB,VOCAB*4,cudaMemcpyDeviceToHost));
    int am=0; for(int v=1;v<VOCAB;++v) if(lg[v]>lg[am]) am=v;
    size_t freeb,totb; cudaMemGetInfo(&freeb,&totb);
    printf("\n[Gate 1] prefill s=%d: %.1f ms (%.1f ms/tok). last-token argmax=%d logit=%.3f\n", s, ms, ms/s, am, lg[am]);
    printf("[Gate 1] GPU mem: %.1f/%.1f GiB used\n", (totb-freeb)/1073741824.0, totb/1073741824.0);
    printf("[Gate 1] FULL 180B FORWARD RAN ON THOR.\n");

    // ================= GATE 2: DSpark draft head + single-token acceptance tau =================
    std::vector<float> ml((size_t)s*VOCAB); CU(cudaMemcpy(ml.data(),logits,(size_t)s*VOCAB*4,cudaMemcpyDeviceToHost));
    std::vector<int> main_am(s);
    for(int t=0;t<s;++t){ const float* r=&ml[(size_t)t*VOCAB]; int a=0; for(int v=1;v<VOCAB;++v) if(r[v]>r[a])a=v; main_am[t]=a; }

    DSparkWeights dw{}; BlockWeights& mb=dw.block;
    std::vector<const uint8_t*> mp1,mp2,mp3; std::vector<const float*> ms1,ms2,ms3;
    fill_attn("mtp.0.", mb.attn, false);                                     // pure-sliding block
    fill_moe("mtp.0.", false, mb.ffn, mp1,mp2,mp3,ms1,ms2,ms3);
    mb.attn_norm=L.bf16("mtp.0.attn_norm.weight"); mb.ffn_norm=L.bf16("mtp.0.ffn_norm.weight");
    mb.hc_attn_fn=L.f32("mtp.0.hc_attn_fn"); mb.hc_attn_scale=L.f32("mtp.0.hc_attn_scale"); mb.hc_attn_base=L.f32("mtp.0.hc_attn_base");
    mb.hc_ffn_fn=L.f32("mtp.0.hc_ffn_fn"); mb.hc_ffn_scale=L.f32("mtp.0.hc_ffn_scale"); mb.hc_ffn_base=L.f32("mtp.0.hc_ffn_base");
    mb.dim=DIM; mb.hc=HC_MULT;
    dw.e_proj=L.raw("mtp.0.e_proj.weight"); dw.e_proj_s=L.scale("mtp.0.e_proj.scale");
    dw.h_proj=L.raw("mtp.0.h_proj.weight"); dw.h_proj_s=L.scale("mtp.0.h_proj.scale");
    dw.enorm=L.bf16("mtp.0.enorm.weight"); dw.hnorm=L.bf16("mtp.0.hnorm.weight"); dw.norm=L.bf16("mtp.0.norm.weight");
    dw.hc_head_fn=L.f32("mtp.0.hc_head_fn"); dw.hc_head_scale=L.f32("mtp.0.hc_head_scale"); dw.hc_head_base=L.f32("mtp.0.hc_head_base");
    dw.lm_head=L.bf16("head.weight"); dw.embed=(const __nv_bfloat16*)W.get("embed.weight").dev;
    dw.dim=DIM; dw.hc=HC_MULT; dw.vocab=VOCAB;

    float* draft; CU(cudaMalloc(&draft,(size_t)s*VOCAB*4));
    dspark_head_forward(draft, h, d_ids, dw, s, EPS);                        // tapped [s,hc,d] state
    std::vector<float> dl((size_t)s*VOCAB); CU(cudaMemcpy(dl.data(),draft,(size_t)s*VOCAB*4,cudaMemcpyDeviceToHost));
    std::vector<int> draft_am(s);
    for(int t=0;t<s;++t){ const float* r=&dl[(size_t)t*VOCAB]; int a=0; for(int v=1;v<VOCAB;++v) if(r[v]>r[a])a=v; draft_am[t]=a; }
    int m0=0,m1=0; for(int t=0;t<s;++t) if(draft_am[t]==main_am[t]) m0++;
    for(int t=0;t<s-1;++t) if(draft_am[t]==main_am[t+1]) m1++;
    printf("\n[Gate 2] DSpark draft head single-token acceptance (unfine-tuned on REAP):\n");
    printf("   tau@0  (draft[t]==main[t])   = %.3f  (%d/%d)\n", (double)m0/s, m0, s);
    printf("   tau@+1 (draft[t]==main[t+1]) = %.3f  (%d/%d)\n", (double)m1/(s-1), m1, s-1);
    printf("[Gate 2] main  argmax:"); for(int t=0;t<s;++t) printf(" %d",main_am[t]); printf("\n");
    printf("[Gate 2] draft argmax:"); for(int t=0;t<s;++t) printf(" %d",draft_am[t]); printf("\n");
    return 0;
}
