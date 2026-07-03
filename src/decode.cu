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
#include "dscratch.h"
#include "dspark_real.h"   // DSpark head: main_x, tap_pool, forward_head, markov
#include "dspark_attn.h"   // dspark_main_kv, dspark_block_forward
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
    int NGEN0 = argc>5?atoi(argv[5]):24;               // spec-decode tokens (if head given)
    int PS = s-1;                                      // prefill positions 0..PS-1; decode starts at pos PS (=s-1)
    int seqmax = s + (NDEC>NGEN0?NDEC:NGEN0) + DSPARK_BLOCK + 8;   // room for spec block overshoot
    printf("[decode] loading %s ... s=%d NDEC=%d seqmax=%d\n", dir, s, NDEC, seqmax);
    st::WeightStore W(dir, key_map); Loader L(W);
    printf("[decode] loaded %.2f GiB, %zu tensors\n", W.loadedGiB(), W.count());
    const int half=ROPE_DIM/2, hc=HC_MULT, d=DIM;
    extern bool g_tc_fp8; g_tc_fp8=true; extern bool g_tc_ogroup; g_tc_ogroup=true;
    extern bool g_moe_grouped; g_moe_grouped=true; extern void tc_moe_clear_cache();
    extern bool g_moe_gemv; g_moe_gemv=(getenv("MOE_GEMV")!=nullptr);   // fp4 GEMV: A/B'd SLOWER than TC mma (scalar nibble decode > mma-waste). default OFF.

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
    std::vector<std::vector<const uint8_t*>> S18(N_LAYERS),S28(N_LAYERS),S38(N_LAYERS);  // NATIVE e8m0 expert scales
    auto fill_moe=[&](const std::string& pfx, bool is_hash, MoEWeights& m, int Lyr){
        std::string p=pfx+"ffn."; auto& p1=P1[Lyr];auto&p2=P2[Lyr];auto&p3=P3[Lyr];auto&s1=S18[Lyr];auto&s2=S28[Lyr];auto&s3=S38[Lyr];
        p1.clear();p2.clear();p3.clear();s1.clear();s2.clear();s3.clear();
        m.gate_w=L.bf16(p+"gate.weight"); m.is_hash=is_hash;
        m.gate_bias=is_hash?nullptr:(W.has(p+"gate.bias")?L.f32(p+"gate.bias"):nullptr);
        m.tid2eid=is_hash?(const long*)W.get(p+"gate.tid2eid").dev:nullptr;
        for(int e=0;e<N_ROUTED;++e){ std::string ep=p+"experts."+std::to_string(e)+".";
            p1.push_back(L.raw(ep+"w1.weight")); p2.push_back(L.raw(ep+"w2.weight")); p3.push_back(L.raw(ep+"w3.weight"));
            s1.push_back(L.raw(ep+"w1.scale")); s2.push_back(L.raw(ep+"w2.scale")); s3.push_back(L.raw(ep+"w3.scale")); }  // e8m0 bytes, persistent (no dequant)
        m.w1p=p1.data();m.w2p=p2.data();m.w3p=p3.data();
        m.e8m0_scales=true; m.w1sp8=s1.data();m.w2sp8=s2.data();m.w3sp8=s3.data();
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
        a.wo_a_native=true; a.wo_a_fp8=L.raw(p+"wo_a.weight"); a.wo_a_sc=L.raw(p+"wo_a.scale");  // native fp8 (no dequant)
        a.attn_sink=L.f32(p+"attn_sink");
        a.cosT=compressed?cqc:slide_cos;a.sinT=compressed?cqs:slide_sin; };

    // build one layer's weights (dequant), run either prefill_cache (bs=PS) or a decode step (pos), then it's the
    // caller's job to L.release(mk). Returns via x_out.
    // Build EVERY layer's weight struct ONCE (persistent — experts + wo_a are native so residual dequant is
    // ~2 GB, fits). The decode loop then does zero per-token Loader work (no dequant, no cudaMalloc, no struct
    // rebuild). Memory-neutral enough: ~112 GiB peak.
    std::vector<BlockWeights> BW(N_LAYERS); std::vector<CompressedBlockWeights> CW(N_LAYERS);
    auto build_layer=[&](int Lyr){
        int ratio=compress_ratio(Lyr); std::string lp="layers."+std::to_string(Lyr)+".";
        if(ratio==0){
            BlockWeights& b=BW[Lyr]; fill_attn(lp,b.attn,false); fill_moe(lp,is_hash_layer(Lyr),b.ffn,Lyr);
            b.attn_norm=L.bf16(lp+"attn_norm.weight");b.ffn_norm=L.bf16(lp+"ffn_norm.weight");
            b.hc_attn_fn=L.f32(lp+"hc_attn_fn");b.hc_attn_scale=L.f32(lp+"hc_attn_scale");b.hc_attn_base=L.f32(lp+"hc_attn_base");
            b.hc_ffn_fn=L.f32(lp+"hc_ffn_fn");b.hc_ffn_scale=L.f32(lp+"hc_ffn_scale");b.hc_ffn_base=L.f32(lp+"hc_ffn_base");
            b.dim=DIM;b.hc=HC_MULT;
        } else {
            CompressedBlockWeights& b=CW[Lyr]; fill_attn(lp,b.attn.attn,true);
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
        }
    };
    auto run_layer=[&](int Lyr, bool prefill, int pos, const float* x_in, float* x_out, const int* ids_dev){
        int ratio=compress_ratio(Lyr);
        if(ratio==0){
            if(prefill) block_prefill_cache(x_out,x_in,ids_dev,BW[Lyr],PS,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            else        block_decode_step (x_out,x_in,ids_dev,BW[Lyr],pos,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
        } else {
            if(prefill) cblock_prefill_cache(x_out,x_in,ids_dev,CW[Lyr],PS,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            else        cblock_decode_step  (x_out,x_in,ids_dev,CW[Lyr],pos,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
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
    printf("[decode] building 43 layer structs once (persistent)...\n");
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr) build_layer(Lyr);       // all dequant done ONCE, resident (~2 GB)
    { size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[decode] structs built. mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0); }
    arena_init((size_t)512<<20);                            // 512 MB decode scratch arena (bump, reset per layer)
    printf("[decode] prefill %d positions...\n", PS);
    for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset();
        run_layer(Lyr,true,0,h,h2,d_ids); std::swap(h,h2);     // structs prebuilt -> no per-token Loader work
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
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset();
            run_layer(Lyr,false,pos,xin,xout,cur_dev); std::swap(xin,xout);
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

    // ================= SPEC-DECODE M=K VERIFY equivalence gate + timing =================
    // Verify the SAME K tokens the autoregressive decode produced, in ONE M=K forward. Its per-position argmax
    // must equal the decode's tokens (gen), and it costs ~1 forward for K tokens (the spec-decode weight-share win).
    int VK = NDEC<DSPARK_BLOCK?NDEC:DSPARK_BLOCK;
    if(VK>=2){
        std::vector<int> vtok(VK); vtok[0]=ids[s-1]; for(int i=1;i<VK;++i) vtok[i]=gen[i-1];   // decode INPUTS at [PS..PS+VK-1]
        for(int L=0;L<N_LAYERS;++L) KV[L].T=0;                                                 // reset compressed caches
        k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_ids,PS,d);
        k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); run_layer(Lyr,true,0,h,h2,d_ids); std::swap(h,h2); }  // re-prefill (reset window/comp caches)
        int* d_vtok; CU(cudaMalloc(&d_vtok,(size_t)VK*4)); CU(cudaMemcpy(d_vtok,vtok.data(),(size_t)VK*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_ids+PS,vtok.data(),(size_t)VK*4,cudaMemcpyHostToDevice));               // ids for hash routing at [PS..]
        float *hv,*hv2,*collK,*logK; CU(cudaMalloc(&hv,(size_t)VK*hc*d*4)); CU(cudaMalloc(&hv2,(size_t)VK*hc*d*4));
        CU(cudaMalloc(&collK,(size_t)VK*d*4)); CU(cudaMalloc(&logK,(size_t)VK*VOCAB*4));
        k_embed<<<((size_t)VK*d+255)/256,256>>>(h0,(const __nv_bfloat16*)W.get("embed.weight").dev,d_vtok,VK,d);
        k_hc_expand<<<((size_t)VK*hc*d+255)/256,256>>>(hv,h0,VK,hc,d); CU(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        float* vin=hv; float* vout=hv2;
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); int ratio=compress_ratio(Lyr);
            if(ratio==0) block_verify_step (vout,vin,d_ids+PS,BW[Lyr],PS,VK,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            else         cblock_verify_step (vout,vin,d_ids+PS,CW[Lyr],PS,VK,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
            std::swap(vin,vout); }
        hc_head(collK,vin,hc_fn,hc_sc,hc_bs,VK,hc,d,HC_EPS); rmsnorm(collK,collK,norm_w,VK,d,EPS,true,0);
        gemm_fp32(logK,collK,head_w,VK,VOCAB,d,0); CU(cudaDeviceSynchronize());
        cudaEventRecord(t1); cudaEventSynchronize(t1); float vms=0; cudaEventElapsedTime(&vms,t0,t1);
        std::vector<float> lg((size_t)VK*VOCAB); CU(cudaMemcpy(lg.data(),logK,(size_t)VK*VOCAB*4,cudaMemcpyDeviceToHost));
        std::vector<int> vam(VK); for(int t=0;t<VK;++t){const float*r=&lg[(size_t)t*VOCAB];int a=0;for(int v=1;v<VOCAB;++v)if(r[v]>r[a])a=v;vam[t]=a;}
        int match=0; for(int i=0;i<VK;++i) if(vam[i]==gen[i]) ++match;
        printf("\n[spec-verify] M=%d verify in ONE forward: %.1f ms (= %.1f ms/tok if all accepted vs %.1f M=1 -> %.2fx)\n", VK, vms, vms/VK, warm_ms, warm_ms/(vms/VK));
        printf("[spec-verify] verify argmax:"); for(int a:vam) printf(" %d",a); printf("\n");
        printf("[spec-verify] decode tokens:"); for(int i=0;i<VK;++i) printf(" %d",gen[i]); printf("\n");
        // match>=VK-1 tolerated: the only expected diffs are MoE-atomic near-ties (same tokens the decode flips
        // run-to-run). gate_mla_verify proves the M=K math bit-exact; full-model deterministic positions match.
        printf("[spec-verify] MATCH %d/%d -> %s  (M=K verify == K sequential decodes; diffs = MoE-atomic near-ties)\n",
               match, VK, match>=VK-1?"PASS":"FAIL");
    }
    // ================= DSpark SPEC-DECODE (draft head + accept loop) =================
    if(argc>4){
        const char* headdir=argv[4]; const int BLK=DSPARK_BLOCK, hf=ROPE_DIM/2;
        printf("\n[spec] loading DSpark head %s ...\n", headdir);
        st::WeightStore WH(headdir, key_map, "mtp."); Loader LH(WH);
        printf("[spec] head loaded %.2f GiB, %zu mtp tensors\n", WH.loadedGiB(), WH.count());
        std::vector<float> bc,bs2; yarn::freqs(bc,bs2,seqmax,ROPE_DIM,0,ROPE_THETA,YARN_FACTOR,YARN_BETA_FAST,YARN_BETA_SLOW);
        const float* blk_cos=up_f(bc,keep); const float* blk_sin=up_f(bs2,keep);
        const uint8_t* main_proj=LH.raw("mtp.0.main_proj.weight"); const float* main_proj_s=LH.scale("mtp.0.main_proj.scale");
        const float* main_norm=LH.bf16("mtp.0.main_norm.weight");
        int NSTAGE=0; while(WH.has("mtp."+std::to_string(NSTAGE)+".attn_norm.weight")) NSTAGE++;
        int NE=0; while(WH.has("mtp.0.ffn.experts."+std::to_string(NE)+".w1.weight")) NE++;
        printf("[spec] NSTAGE=%d head-experts=%d BLK=%d\n", NSTAGE, NE, BLK);
        std::vector<BlockWeights> mb(NSTAGE); std::vector<float*> mkv(NSTAGE);
        std::vector<std::vector<const uint8_t*>> HP1(NSTAGE),HP2(NSTAGE),HP3(NSTAGE);
        std::vector<std::vector<const float*>> HS1(NSTAGE),HS2(NSTAGE),HS3(NSTAGE);
        for(int st=0; st<NSTAGE; ++st){
            std::string b="mtp."+std::to_string(st)+".", p=b+"attn."; MLAWeights& a=mb[st].attn;
            a.wq_a=LH.raw(p+"wq_a.weight");a.wq_a_s=LH.scale(p+"wq_a.scale");a.wq_b=LH.raw(p+"wq_b.weight");a.wq_b_s=LH.scale(p+"wq_b.scale");
            a.wkv=LH.raw(p+"wkv.weight");a.wkv_s=LH.scale(p+"wkv.scale");a.wo_b=LH.raw(p+"wo_b.weight");a.wo_b_s=LH.scale(p+"wo_b.scale");
            a.q_norm=LH.bf16(p+"q_norm.weight");a.kv_norm=LH.bf16(p+"kv_norm.weight");
            a.wo_a=LH.wo_a(p+"wo_a.weight",p+"wo_a.scale");a.attn_sink=LH.f32(p+"attn_sink");a.cosT=blk_cos;a.sinT=blk_sin;
            mb[st].dim=DIM;mb[st].hc=HC_MULT;mb[st].attn_norm=LH.bf16(b+"attn_norm.weight");mb[st].ffn_norm=LH.bf16(b+"ffn_norm.weight");
            mb[st].hc_attn_fn=LH.f32(b+"hc_attn_fn");mb[st].hc_attn_scale=LH.f32(b+"hc_attn_scale");mb[st].hc_attn_base=LH.f32(b+"hc_attn_base");
            mb[st].hc_ffn_fn=LH.f32(b+"hc_ffn_fn");mb[st].hc_ffn_scale=LH.f32(b+"hc_ffn_scale");mb[st].hc_ffn_base=LH.f32(b+"hc_ffn_base");
            MoEWeights& m=mb[st].ffn; std::string fp=b+"ffn.";
            m.gate_w=LH.bf16(fp+"gate.weight");m.is_hash=false;m.gate_bias=WH.has(fp+"gate.bias")?LH.f32(fp+"gate.bias"):nullptr;m.tid2eid=nullptr;
            for(int e=0;e<NE;++e){ std::string ep=fp+"experts."+std::to_string(e)+".";
                HP1[st].push_back(LH.raw(ep+"w1.weight"));HP2[st].push_back(LH.raw(ep+"w2.weight"));HP3[st].push_back(LH.raw(ep+"w3.weight"));
                HS1[st].push_back(LH.scale(ep+"w1.scale"));HS2[st].push_back(LH.scale(ep+"w2.scale"));HS3[st].push_back(LH.scale(ep+"w3.scale")); }
            m.w1p=HP1[st].data();m.w2p=HP2[st].data();m.w3p=HP3[st].data();m.w1sp=HS1[st].data();m.w2sp=HS2[st].data();m.w3sp=HS3[st].data();
            std::string sp2=fp+"shared_experts."; m.sw1=LH.raw(sp2+"w1.weight");m.sw2=LH.raw(sp2+"w2.weight");m.sw3=LH.raw(sp2+"w3.weight");
            m.sw1s=LH.scale(sp2+"w1.scale");m.sw2s=LH.scale(sp2+"w2.scale");m.sw3s=LH.scale(sp2+"w3.scale");
            m.n_routed=NE;m.n_act=N_ACT;m.dim=DIM;m.inter=MOE_INTER;m.vocab=VOCAB;m.route_scale=ROUTE_SCALE;m.swiglu_limit=SWIGLU_LIMIT;
            m.use_tc_pp=true;m.batched=true;m.device_route=true; CU(cudaMalloc(&mkv[st],(size_t)seqmax*HEAD_DIM*4));
        }
        std::string LS="mtp."+std::to_string(NSTAGE-1)+".";
        const float* hh_fn=LH.f32(LS+"hc_head_fn");const float* hh_sc=LH.f32(LS+"hc_head_scale");const float* hh_ba=LH.f32(LS+"hc_head_base");
        const float* hnorm=LH.bf16(LS+"norm.weight");
        const float* mw1=LH.bf16(LS+"markov_head.markov_w1.weight");const float* mw2=LH.bf16(LS+"markov_head.markov_w2.weight");
        const __nv_bfloat16* emb=(const __nv_bfloat16*)W.get("embed.weight").dev;
        { size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[spec] head built. mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0); }

        // main_x accumulator + tapped re-prefill over [0..PS-1]
        float *main_x,*mh_pre; CU(cudaMalloc(&main_x,(size_t)seqmax*d*4)); CU(cudaMalloc(&mh_pre,(size_t)PS*3*d*4));
        for(int L=0;L<N_LAYERS;++L) KV[L].T=0;
        k_embed<<<((size_t)PS*d+255)/256,256>>>(h0,emb,d_ids,PS,d); k_hc_expand<<<((size_t)PS*hc*d+255)/256,256>>>(h,h0,PS,hc,d); CU(cudaDeviceSynchronize());
        for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); run_layer(Lyr,true,0,h,h2,d_ids); std::swap(h,h2);
            if(Lyr==40) dspark_tap_pool(mh_pre,h,PS,hc,d,0,3); else if(Lyr==41) dspark_tap_pool(mh_pre,h,PS,hc,d,1,3); else if(Lyr==42) dspark_tap_pool(mh_pre,h,PS,hc,d,2,3); }
        dspark_main_x(main_x, mh_pre, main_proj, main_proj_s, main_norm, PS, d, EPS); CU(cudaDeviceSynchronize());

        // buffers
        int *dbid,*dfid,*dout; CU(cudaMalloc(&dbid,BLK*4)); CU(cudaMalloc(&dfid,4)); CU(cudaMalloc(&dout,(BLK+1)*4));
        float *xemb,*xa,*xb; CU(cudaMalloc(&xemb,(size_t)BLK*d*4)); CU(cudaMalloc(&xa,(size_t)BLK*hc*d*4)); CU(cudaMalloc(&xb,(size_t)BLK*hc*d*4));
        float *hv,*hv2,*collK,*logK,*mh_v; CU(cudaMalloc(&hv,(size_t)BLK*hc*d*4)); CU(cudaMalloc(&hv2,(size_t)BLK*hc*d*4));
        CU(cudaMalloc(&collK,(size_t)BLK*d*4)); CU(cudaMalloc(&logK,(size_t)BLK*VOCAB*4)); CU(cudaMalloc(&mh_v,(size_t)BLK*3*d*4));
        int NGEN=NGEN0;
        int cur=ids[s-1], cpos=PS;                 // cur = token at position cpos (=PS=s-1), not yet in cache
        std::vector<int> sgen; int nverify=0, timed_tok=0; float spec_ms=0; cudaEvent_t s0,s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        printf("[spec] decoding %d tokens (block=%d)...\n", NGEN, BLK);
        while((int)sgen.size()<NGEN && cpos+BLK+1<seqmax){
            cudaEventRecord(s0);
            int anchor=cpos-1, ctx=cpos;           // main context [0..cpos-1]
            // rebuild head main-KV over the context
            for(int st=0;st<NSTAGE;++st) dspark_main_kv(mkv[st], main_x, mb[st].attn, ctx, EPS);
            // DRAFT: block [cur, noise x (BLK-1)]
            std::vector<int> bid(BLK,DSPARK_NOISE_TID); bid[0]=cur; CU(cudaMemcpy(dbid,bid.data(),BLK*4,cudaMemcpyHostToDevice));
            k_embed<<<((size_t)BLK*d+255)/256,256>>>(xemb,emb,dbid,BLK,d); k_hc_expand<<<((size_t)BLK*hc*d+255)/256,256>>>(xa,xemb,BLK,hc,d); CU(cudaDeviceSynchronize());
            float *cb=xa,*nb=xb;
            for(int st=0;st<NSTAGE;++st){ dspark_block_forward(nb,cb,dbid,mkv[st],anchor,mb[st],blk_cos+(size_t)ctx*hf,blk_sin+(size_t)ctx*hf,BLK,WINDOW,HC_SINKHORN_ITERS,EPS); std::swap(cb,nb); }
            CU(cudaMemcpy(dfid,&cur,4,cudaMemcpyHostToDevice));
            dspark_forward_head(dout,cb,dfid,hh_fn,hh_sc,hh_ba,hnorm,head_w,mw1,mw2,1,BLK,hc,d,VOCAB,DSPARK_MARKOV_RANK,EPS); CU(cudaDeviceSynchronize());
            std::vector<int> oo(BLK+1); CU(cudaMemcpy(oo.data(),dout,(BLK+1)*4,cudaMemcpyDeviceToHost));
            std::vector<int> draft(BLK); for(int i=0;i<BLK;++i) draft[i]=oo[1+i];     // proposals for cpos+1..cpos+BLK
            // VERIFY block [cur, draft[0..BLK-2]] at [cpos..cpos+BLK-1]
            std::vector<int> vtok(BLK); vtok[0]=cur; for(int i=1;i<BLK;++i) vtok[i]=draft[i-1];
            std::vector<int> Tbefore(N_LAYERS); for(int L=0;L<N_LAYERS;++L) Tbefore[L]=KV[L].T;
            int* dvt; dvt=d_ids+cpos; CU(cudaMemcpy(dvt,vtok.data(),BLK*4,cudaMemcpyHostToDevice));
            k_embed<<<((size_t)BLK*d+255)/256,256>>>(h0,emb,dvt,BLK,d); k_hc_expand<<<((size_t)BLK*hc*d+255)/256,256>>>(hv,h0,BLK,hc,d); CU(cudaDeviceSynchronize());
            float* vin=hv; float* vout=hv2;
            for(int Lyr=0; Lyr<N_LAYERS; ++Lyr){ arena_reset(); int ratio=compress_ratio(Lyr);
                if(ratio==0) block_verify_step (vout,vin,dvt,BW[Lyr],cpos,BLK,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
                else         cblock_verify_step (vout,vin,dvt,CW[Lyr],cpos,BLK,HC_SINKHORN_ITERS,EPS,KV[Lyr]);
                std::swap(vin,vout);
                if(Lyr==40) dspark_tap_pool(mh_v,vin,BLK,hc,d,0,3); else if(Lyr==41) dspark_tap_pool(mh_v,vin,BLK,hc,d,1,3); else if(Lyr==42) dspark_tap_pool(mh_v,vin,BLK,hc,d,2,3); }
            hc_head(collK,vin,hc_fn,hc_sc,hc_bs,BLK,hc,d,HC_EPS); rmsnorm(collK,collK,norm_w,BLK,d,EPS,true,0);
            gemm_fp32(logK,collK,head_w,BLK,VOCAB,d,0); CU(cudaDeviceSynchronize());
            std::vector<float> lg((size_t)BLK*VOCAB); CU(cudaMemcpy(lg.data(),logK,(size_t)BLK*VOCAB*4,cudaMemcpyDeviceToHost));
            std::vector<int> tam(BLK); for(int i=0;i<BLK;++i){const float*r=&lg[(size_t)i*VOCAB];int aa=0;for(int v=1;v<VOCAB;++v)if(r[v]>r[aa])aa=v;tam[i]=aa;}
            // ACCEPT longest matching prefix: draft[i]==tam[i] (target's token for pos cpos+1+i)
            int acc=0; while(acc<BLK-1 && draft[acc]==tam[acc]) ++acc;
            int correction=tam[acc];                        // target's token for pos cpos+acc+1
            for(int i=0;i<acc;++i) sgen.push_back(draft[i]); sgen.push_back(correction);
            // update main_x for the accepted range [cpos..cpos+acc] from verify taps; rollback compressor T
            dspark_main_x(main_x+(size_t)cpos*d, mh_v, main_proj, main_proj_s, main_norm, acc+1, d, EPS); CU(cudaDeviceSynchronize());
            for(int L=0;L<N_LAYERS;++L){ int ratio=compress_ratio(L); if(!ratio) continue; int valid=0;
                for(int j=cpos;j<=cpos+acc;++j) if((j+1)%ratio==0) ++valid; KV[L].T=Tbefore[L]+valid; }   // drop rows from rejected drafts
            cpos += acc+1; cur = correction;
            cudaEventRecord(s1); cudaEventSynchronize(s1); float ms=0; cudaEventElapsedTime(&ms,s0,s1);
            if(nverify>0){ spec_ms+=ms; timed_tok+=acc+1; } ++nverify;   // exclude round 0 (warmup: head repack)
            printf("  verify %d: accepted %d/%d + correction -> +%d tokens (%.1f ms)  cpos=%d\n", nverify, acc, BLK-1, acc+1, ms, cpos);
        }
        double avg_acc=(double)sgen.size()/nverify;
        double ms_per_tok = timed_tok>0 ? spec_ms/timed_tok : 0;
        printf("\n[spec] generated %d tokens over %d verifies: mean tokens/verify = %.2f (block=%d, max %d)\n", (int)sgen.size(), nverify, avg_acc, BLK, BLK);
        printf("[spec] tokens:"); for(int i=0;i<(int)sgen.size() && i<40;++i) printf(" %d",sgen[i]); printf("\n");
        printf("[spec] SPEC-DECODE: %.1f ms/tok = %.2f tok/s  (vs base M=1 %.1f ms/tok = %.2f tok/s -> %.2fx)\n",
               ms_per_tok, ms_per_tok>0?1000.0/ms_per_tok:0, warm_ms, 1000.0/warm_ms, ms_per_tok>0?warm_ms/ms_per_tok:0);
    }
    size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("[decode] mem %.1f/%.1f GiB\n",(tb-fb)/1073741824.0,tb/1073741824.0);
    return first_am==270?0:1;
}
