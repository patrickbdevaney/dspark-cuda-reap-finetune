// gate_cmla.cu — Gate K for the full compressed-layer MLA forward on REAL layer-2 REAP weights.
//   build: scripts/build_cmla.sh ; run: ./build/gate_cmla ref/goldens/cmla_layer2_seq16.safetensors
#include "safetensors.h"
#include "compressed_attn.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static int i32(const st::Tensor& t,int i){ return ((const int*)t.data)[i]; }
static const float* F(const st::Tensor& t){ return (const float*)t.data; }
template<class T> static const T* up(const st::Tensor& t){ void* d; CU(cudaMalloc(&d,t.nbytes)); CU(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); return (const T*)d; }

int main(int argc, char** argv){
    std::string path = argc>1?argv[1]:"ref/goldens/cmla_layer2_seq16.safetensors";
    st::SafeTensors S(path);
    const auto& dm=S.get("dims");
    int s=i32(dm,0),dim=i32(dm,1),q_lora=i32(dm,2),win=i32(dm,3),ratio=i32(dm,4),inh=i32(dm,5),ihd=i32(dm,6),itk=i32(dm,7);

    CompressedAttnWeights w{}; auto& a=w.attn;
    a.wq_a=up<uint8_t>(S.get("wq_a")); a.wq_a_s=up<float>(S.get("wq_a_s")); a.wq_b=up<uint8_t>(S.get("wq_b")); a.wq_b_s=up<float>(S.get("wq_b_s"));
    a.wkv=up<uint8_t>(S.get("wkv")); a.wkv_s=up<float>(S.get("wkv_s")); a.wo_b=up<uint8_t>(S.get("wo_b")); a.wo_b_s=up<float>(S.get("wo_b_s"));
    a.q_norm=up<float>(S.get("q_norm")); a.kv_norm=up<float>(S.get("kv_norm")); a.wo_a=up<float>(S.get("wo_a")); a.attn_sink=up<float>(S.get("attn_sink"));
    a.cosT=up<float>(S.get("cos")); a.sinT=up<float>(S.get("sin"));
    w.mc_wkv=up<float>(S.get("mc_wkv")); w.mc_wgate=up<float>(S.get("mc_wgate")); w.mc_ape=up<float>(S.get("mc_ape")); w.mc_norm=up<float>(S.get("mc_norm"));
    w.cc_cos=up<float>(S.get("cc_cos")); w.cc_sin=up<float>(S.get("cc_sin"));
    if (ratio == 4) {   // indexer only on ratio-4 layers
        w.idx_wq_b=up<unsigned char>(S.get("idx_wq_b")); w.idx_wq_b_s=up<float>(S.get("idx_wq_b_s")); w.idx_weights_proj=up<float>(S.get("idx_weights_proj"));
        w.idx_c_wkv=up<float>(S.get("idx_c_wkv")); w.idx_c_wgate=up<float>(S.get("idx_c_wgate")); w.idx_c_ape=up<float>(S.get("idx_c_ape")); w.idx_c_norm=up<float>(S.get("idx_c_norm"));
    }
    w.index_n_heads=inh; w.index_head_dim=ihd; w.index_topk=itk;

    const float* x=up<float>(S.get("x"));
    float* out; CU(cudaMalloc(&out,(size_t)s*dim*4));
    compressed_attn_forward(out, x, w, s, win, ratio, 1e-6f);
    CU(cudaDeviceSynchronize());
    std::vector<float> o((size_t)s*dim); CU(cudaMemcpy(o.data(),out,o.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=F(S.get("o_ref"));
    double mx=0,mabs=0,mrel=0; for(size_t i=0;i<o.size();++i) mx=fmax(mx,fabs((double)oref[i]));
    for(size_t i=0;i<o.size();++i){ double d=fabs((double)o[i]-oref[i]); mabs=fmax(mabs,d); mrel=fmax(mrel,d/(fabs((double)oref[i])+0.01*mx)); }
    bool ok = mrel < 3e-2;
    printf("[compressed_attn] s=%d dim=%d win=%d ratio=%d idx_topk=%d  |o|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",
           s,dim,win,ratio,itk,mx,mabs,mrel, ok?"PASS":"FAIL");
    printf("\nGate K (compressed MLA): %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
