// gate_units.cu — Gate K: run the CUDA kernels on the reference unit goldens and compare within tolerance.
//   build: nvcc -O2 -arch=sm_110a -I include tests/gate_units.cu kernels/fp8_block_gemm.cu \
//          kernels/hc_sinkhorn.cu -o build/gate_units
//   run:   ./build/gate_units ref/goldens
#include "safetensors.h"
#include "fp8_block_gemm.h"
#include "hc_sinkhorn.h"
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

int main(int argc, char** argv) {
    std::string dir = argc>1 ? argv[1] : "ref/goldens";
    bool a = gate_fp8_gemm(dir);
    bool b = gate_hc(dir);
    printf("\nGate K (units): %s\n", (a&&b)?"PASS":"FAIL");
    return (a&&b)?0:1;
}
