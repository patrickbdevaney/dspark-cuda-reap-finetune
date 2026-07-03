// gate_fp4_gemv.cu — M-row fp4 GEMV vs fp4_gemm oracle. GEMV reads original fp4 + e8m0 scale bytes;
// oracle reads original fp4 + f32 scale (= exp2(byte-127)). Expect cosine ~1.0 (same arithmetic).
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
void fp4_gemm(float*, const uint8_t*, const float*, const uint8_t*, const float*, int,int,int, cudaStream_t);
void tc_build_tiles(int*, int*, int*, const int*, int, cudaStream_t);
void tc_fp4_grouped_gemv_e8m0(float*, const uint8_t*, const float*, const uint8_t* const*, const uint8_t* const*,
        const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
int main(){
    const int N=64, K=256, M=3;  srand(9);            // 1 expert, M rows in one tile
    std::vector<uint8_t> W((size_t)N*(K/2)), Ssc((size_t)N*(K/32)), A((size_t)M*K); std::vector<float> As((size_t)M*(K/128));
    for(auto&v:W)v=rand()&0xff; for(auto&v:Ssc)v=120+rand()%12; for(auto&v:A)v=rand()%0x40; for(auto&v:As)v=0.5f+0.01f*(rand()%50);
    std::vector<float> Sf(Ssc.size()); for(size_t i=0;i<Ssc.size();++i) Sf[i]=exp2f((float)Ssc[i]-127.f);   // f32 for oracle
    uint8_t *dW,*dSsc,*dA; float *dAs,*dSf,*Cr,*Cg;
    CU(cudaMalloc(&dW,W.size()));CU(cudaMalloc(&dSsc,Ssc.size()));CU(cudaMalloc(&dA,A.size()));
    CU(cudaMalloc(&dAs,As.size()*4));CU(cudaMalloc(&dSf,Sf.size()*4));CU(cudaMalloc(&Cr,(size_t)M*N*4));CU(cudaMalloc(&Cg,(size_t)M*N*4));
    CU(cudaMemcpy(dW,W.data(),W.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(dSsc,Ssc.data(),Ssc.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(dAs,As.data(),As.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dSf,Sf.data(),Sf.size()*4,cudaMemcpyHostToDevice));
    fp4_gemm(Cr,dA,dAs,dW,dSf,M,N,K,0); CU(cudaDeviceSynchronize());       // oracle
    // GEMV: 1 expert, tiles from off=[0,M]
    const uint8_t **wd,**sd; CU(cudaMalloc(&wd,sizeof(void*)));CU(cudaMalloc(&sd,sizeof(void*)));
    CU(cudaMemcpy(wd,&dW,sizeof(void*),cudaMemcpyHostToDevice));CU(cudaMemcpy(sd,&dSsc,sizeof(void*),cudaMemcpyHostToDevice));
    int off[2]={0,M}; int*off_d,*te,*tr,*nt; CU(cudaMalloc(&off_d,8));CU(cudaMalloc(&te,M*4));CU(cudaMalloc(&tr,M*4));CU(cudaMalloc(&nt,4));
    CU(cudaMemcpy(off_d,off,8,cudaMemcpyHostToDevice)); tc_build_tiles(te,tr,nt,off_d,1,0);
    tc_fp4_grouped_gemv_e8m0(Cg,dA,dAs,(const uint8_t* const*)wd,(const uint8_t* const*)sd,off_d,te,tr,nt,M,N,K,0); CU(cudaDeviceSynchronize());
    std::vector<float> cr((size_t)M*N),cg((size_t)M*N); CU(cudaMemcpy(cr.data(),Cr,cr.size()*4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(cg.data(),Cg,cg.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,nr=0,ng=0,md=0; for(size_t i=0;i<cr.size();++i){dot+=cr[i]*cg[i];nr+=cr[i]*cr[i];ng+=cg[i]*cg[i];md=fmax(md,fabs(cr[i]-cg[i]));}
    double cos=dot/(sqrt(nr)*sqrt(ng)+1e-30); bool ok=cos>0.999999 && md<1e-3;
    printf("[fp4_gemv] N=%d K=%d M=%d cosine=%.7f maxabs=%.2e -> %s\n",N,K,M,cos,md,ok?"PASS":"FAIL"); return ok?0:1;
}
