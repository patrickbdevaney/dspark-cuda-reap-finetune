// gate_units.cu — Gate K: run the CUDA kernels on the reference unit goldens and compare within tolerance.
//   build: nvcc -O2 -arch=sm_110a -I include tests/gate_units.cu kernels/fp8_block_gemm.cu \
//          kernels/hc_sinkhorn.cu -o build/gate_units
//   run:   ./build/gate_units ref/goldens
#include "safetensors.h"
#include "fp8_block_gemm.h"
#include "hc_sinkhorn.h"
#include "mla_attn.h"
#include "moe.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static void* up(const st::Tensor& t) {                     // upload raw bytes to device
    void* d; CU(cudaMalloc(&d, t.nbytes)); CU(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice)); return d;
}
static const float* f32(const st::Tensor& t){ return (const float*)t.data; }
static int i32(const st::Tensor& t, int i){ return ((const int*)t.data)[i]; }

struct Err { double max_abs=0, max_rel=0; };
static Err compare(const std::vector<float>& got, const float* ref, int n, double denom) {
    Err e;
    for (int i=0;i<n;++i){ double d=fabs((double)got[i]-ref[i]); e.max_abs=fmax(e.max_abs,d);
        e.max_rel=fmax(e.max_rel, d/(fabs((double)ref[i])+0.01*denom)); }
    return e;
}

static bool gate_fp8_gemm(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_fp8_gemm.safetensors");
    const auto& dims = S.get("dims"); int M=i32(dims,0), N=i32(dims,1), K=i32(dims,2);
    void *dA=up(S.get("A_fp8")), *dB=up(S.get("B_fp8"));
    void *das=up(S.get("a_s")),  *dbs=up(S.get("b_s"));
    float* dC; CU(cudaMalloc(&dC, (size_t)M*N*4));
    fp8_block_gemm(dC, (const uint8_t*)dA, (const float*)das, (const uint8_t*)dB, (const float*)dbs, M,N,K);
    CU(cudaDeviceSynchronize());
    std::vector<float> C(M*N); CU(cudaMemcpy(C.data(), dC, (size_t)M*N*4, cudaMemcpyDeviceToHost));
    const float* Cref = f32(S.get("C_ref"));
    double mx=0; for(int i=0;i<M*N;++i) mx=fmax(mx,fabs((double)Cref[i]));
    Err e = compare(C, Cref, M*N, mx);
    bool ok = e.max_rel < 2e-2;
    printf("[fp8_block_gemm] M=%d N=%d K=%d  |C|max=%.4f  max_abs=%.5f  max_rel=%.5f  -> %s\n",
           M,N,K,mx,e.max_abs,e.max_rel, ok?"PASS":"FAIL");
    cudaFree(dA);cudaFree(dB);cudaFree(das);cudaFree(dbs);cudaFree(dC);
    return ok;
}

static bool gate_hc(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_hc_sinkhorn.safetensors");
    const auto& pp = S.get("params"); int n=i32(pp,0), hc=i32(pp,1), iters=i32(pp,2);
    void *dm=up(S.get("mixes")), *dsc=up(S.get("hc_scale")), *dba=up(S.get("hc_base"));
    float *dpre,*dpost,*dcomb;
    CU(cudaMalloc(&dpre,(size_t)n*hc*4)); CU(cudaMalloc(&dpost,(size_t)n*hc*4)); CU(cudaMalloc(&dcomb,(size_t)n*hc*hc*4));
    hc_sinkhorn(dpre,dpost,dcomb,(const float*)dm,(const float*)dsc,(const float*)dba,n,hc,iters,1e-6f);
    CU(cudaDeviceSynchronize());
    std::vector<float> pre(n*hc),post(n*hc),comb(n*hc*hc);
    CU(cudaMemcpy(pre.data(),dpre,(size_t)n*hc*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(post.data(),dpost,(size_t)n*hc*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(comb.data(),dcomb,(size_t)n*hc*hc*4,cudaMemcpyDeviceToHost));
    Err ep=compare(pre,f32(S.get("pre")),n*hc,1.0), eq=compare(post,f32(S.get("post")),n*hc,1.0),
        ec=compare(comb,f32(S.get("comb")),n*hc*hc,1.0);
    double mabs=fmax(ep.max_abs,fmax(eq.max_abs,ec.max_abs));
    bool ok = mabs < 1e-3;
    printf("[hc_sinkhorn] n=%d hc=%d iters=%d  max_abs pre=%.2e post=%.2e comb=%.2e  -> %s\n",
           n,hc,iters,ep.max_abs,eq.max_abs,ec.max_abs, ok?"PASS":"FAIL");
    cudaFree(dm);cudaFree(dsc);cudaFree(dba);cudaFree(dpre);cudaFree(dpost);cudaFree(dcomb);
    return ok;
}

static bool gate_sparse_attn(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_sparse_attn.safetensors");
    const auto& dm = S.get("dims");
    int b=i32(dm,0), m=i32(dm,1), h=i32(dm,2), d=i32(dm,3), n=i32(dm,4), topk=i32(dm,5);
    float scale = f32(S.get("scale"))[0];
    void *dq=up(S.get("q")), *dkv=up(S.get("kv")), *dsink=up(S.get("attn_sink")), *didx=up(S.get("topk_idxs"));
    float* d_o; CU(cudaMalloc(&d_o,(size_t)b*m*h*d*4));
    sparse_attn(d_o,(const float*)dq,(const float*)dkv,(const float*)dsink,(const int*)didx,b,m,h,d,n,topk,scale);
    CU(cudaDeviceSynchronize());
    std::vector<float> o((size_t)b*m*h*d); CU(cudaMemcpy(o.data(),d_o,o.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=f32(S.get("o_ref"));
    double mx=0; for(size_t i=0;i<o.size();++i) mx=fmax(mx,fabs((double)oref[i]));
    Err e=compare(o,oref,o.size(),mx);
    bool ok=e.max_rel<1e-2;
    printf("[sparse_attn] b=%d m=%d h=%d d=%d n=%d topk=%d  |o|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",
           b,m,h,d,n,topk,mx,e.max_abs,e.max_rel, ok?"PASS":"FAIL");
    cudaFree(dq);cudaFree(dkv);cudaFree(dsink);cudaFree(didx);cudaFree(d_o); return ok;
}

static bool gate_rope(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_rope.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), rd=i32(dm,1);
    void *dc=up(S.get("cos")), *ds=up(S.get("sin"));
    auto run=[&](const char* refname, bool inv)->Err{
        void* dx=up(S.get("x"));
        rope_interleaved((float*)dx,(const float*)dc,(const float*)ds,rows,rd,inv);
        CU(cudaDeviceSynchronize());
        std::vector<float> y((size_t)rows*rd); CU(cudaMemcpy(y.data(),dx,y.size()*4,cudaMemcpyDeviceToHost));
        Err e=compare(y,f32(S.get(refname)),y.size(),1.0); cudaFree(dx); return e;
    };
    Err ef=run("y_fwd",false), ei=run("y_inv",true);
    bool ok=fmax(ef.max_abs,ei.max_abs)<1e-4;
    printf("[rope] rows=%d rope_dim=%d  fwd_abs=%.2e inv_abs=%.2e -> %s\n",rows,rd,ef.max_abs,ei.max_abs,ok?"PASS":"FAIL");
    cudaFree(dc);cudaFree(ds); return ok;
}

static bool gate_rmsnorm(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_rmsnorm.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), dim=i32(dm,1); float eps=f32(S.get("eps"))[0];
    void *dx=up(S.get("x")), *dw=up(S.get("weight"));
    float* dy; CU(cudaMalloc(&dy,(size_t)rows*dim*4));
    rmsnorm(dy,(const float*)dx,(const float*)dw,rows,dim,eps,true); CU(cudaDeviceSynchronize());
    std::vector<float> yw((size_t)rows*dim); CU(cudaMemcpy(yw.data(),dy,yw.size()*4,cudaMemcpyDeviceToHost));
    rmsnorm(dy,(const float*)dx,nullptr,rows,dim,eps,false); CU(cudaDeviceSynchronize());
    std::vector<float> yn((size_t)rows*dim); CU(cudaMemcpy(yn.data(),dy,yn.size()*4,cudaMemcpyDeviceToHost));
    Err ew=compare(yw,f32(S.get("y_w")),yw.size(),1.0), en=compare(yn,f32(S.get("y_now")),yn.size(),1.0);
    bool ok=fmax(ew.max_abs,en.max_abs)<1e-4;
    printf("[rmsnorm] rows=%d dim=%d  w_abs=%.2e now_abs=%.2e -> %s\n",rows,dim,ew.max_abs,en.max_abs,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dw);cudaFree(dy); return ok;
}

static bool gate_act_quant(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_act_quant.safetensors");
    const auto& dm=S.get("dims"); int rows=i32(dm,0), dim=i32(dm,1), block=i32(dm,2);
    void* dx=up(S.get("x"));
    act_quant_fp8sim((float*)dx, rows, dim, block); CU(cudaDeviceSynchronize());
    std::vector<float> y((size_t)rows*dim); CU(cudaMemcpy(y.data(),dx,y.size()*4,cudaMemcpyDeviceToHost));
    const float* yr=f32(S.get("y_ref"));
    double mx=0; for(size_t i=0;i<y.size();++i) mx=fmax(mx,fabs((double)yr[i]));
    Err e=compare(y,yr,y.size(),mx);
    bool ok=e.max_rel<1e-3;
    printf("[act_quant] rows=%d dim=%d block=%d  max_abs=%.5f max_rel=%.5f -> %s\n",rows,dim,block,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dx); return ok;
}

static bool gate_ogroup(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_ogroup_gemm.safetensors");
    const auto& dm=S.get("dims"); int bs=i32(dm,0), G=i32(dm,1), R=i32(dm,2), Kd=i32(dm,3);
    void *doo=up(S.get("o")), *dw=up(S.get("wo_a"));
    float* dout; CU(cudaMalloc(&dout,(size_t)bs*G*R*4));
    ogroup_gemm(dout,(const float*)doo,(const float*)dw,bs,G,R,Kd); CU(cudaDeviceSynchronize());
    std::vector<float> out((size_t)bs*G*R); CU(cudaMemcpy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost));
    const float* oref=f32(S.get("out_ref"));
    double mx=0; for(size_t i=0;i<out.size();++i) mx=fmax(mx,fabs((double)oref[i]));
    Err e=compare(out,oref,out.size(),mx);
    bool ok=e.max_rel<1e-2;
    printf("[ogroup_gemm] bs=%d G=%d R=%d Kd=%d  |out|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",bs,G,R,Kd,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(doo);cudaFree(dw);cudaFree(dout); return ok;
}

static bool gate_fp4_gemm(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_fp4_gemm.safetensors");
    const auto& dm=S.get("dims"); int M=i32(dm,0), N=i32(dm,1), K=i32(dm,2);
    void *dA=up(S.get("A_fp8")), *das=up(S.get("a_s")), *dB=up(S.get("B_fp4")), *dbs=up(S.get("b_s"));
    float* dC; CU(cudaMalloc(&dC,(size_t)M*N*4));
    fp4_gemm(dC,(const uint8_t*)dA,(const float*)das,(const uint8_t*)dB,(const float*)dbs,M,N,K);
    CU(cudaDeviceSynchronize());
    std::vector<float> C(M*N); CU(cudaMemcpy(C.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    const float* Cref=f32(S.get("C_ref")); double mx=0; for(int i=0;i<M*N;++i) mx=fmax(mx,fabs((double)Cref[i]));
    Err e=compare(C,Cref,M*N,mx); bool ok=e.max_rel<2e-2;
    printf("[fp4_gemm] M=%d N=%d K=%d  |C|max=%.4f max_abs=%.5f max_rel=%.5f -> %s\n",M,N,K,mx,e.max_abs,e.max_rel,ok?"PASS":"FAIL");
    cudaFree(dA);cudaFree(das);cudaFree(dB);cudaFree(dbs);cudaFree(dC); return ok;
}

static bool gate_router(const std::string& dir) {
    st::SafeTensors S(dir + "/unit_router.safetensors");
    const auto& dm=S.get("dims"); int n=i32(dm,0), dim=i32(dm,1), nr=i32(dm,2), topk=i32(dm,3);
    float rs=f32(S.get("route_scale"))[0];
    void *dx=up(S.get("x")), *dgw=up(S.get("gate_w")), *dbias=up(S.get("bias"));
    float* dw; int* di; CU(cudaMalloc(&dw,(size_t)n*topk*4)); CU(cudaMalloc(&di,(size_t)n*topk*4));
    moe_router_score(dw,di,(const float*)dx,(const float*)dgw,(const float*)dbias,n,dim,nr,topk,rs);
    CU(cudaDeviceSynchronize());
    std::vector<float> wv(n*topk); std::vector<int> iv(n*topk);
    CU(cudaMemcpy(wv.data(),dw,(size_t)n*topk*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(iv.data(),di,(size_t)n*topk*4,cudaMemcpyDeviceToHost));
    const float* wref=f32(S.get("weights_ref")); const int* iref=(const int*)S.get("indices_ref").data;
    Err e=compare(wv,wref,n*topk,1.0); int idx_mism=0; for(int i=0;i<n*topk;++i) if(iv[i]!=iref[i]) idx_mism++;
    bool ok = e.max_abs<1e-3 && idx_mism==0;
    printf("[router] n=%d dim=%d n_routed=%d topk=%d  w_abs=%.2e idx_mismatch=%d -> %s\n",n,dim,nr,topk,e.max_abs,idx_mism,ok?"PASS":"FAIL");
    cudaFree(dx);cudaFree(dgw);cudaFree(dbias);cudaFree(dw);cudaFree(di); return ok;
}

int main(int argc, char** argv) {
    std::string dir = argc>1 ? argv[1] : "ref/goldens";
    bool ok = true;
    ok &= gate_fp8_gemm(dir);
    ok &= gate_hc(dir);
    ok &= gate_sparse_attn(dir);
    ok &= gate_rope(dir);
    ok &= gate_rmsnorm(dir);
    ok &= gate_act_quant(dir);
    ok &= gate_ogroup(dir);
    ok &= gate_fp4_gemm(dir);
    ok &= gate_router(dir);
    printf("\nGate K (units): %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
