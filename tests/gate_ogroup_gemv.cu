#include "mla_attn.h"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s\n",cudaGetErrorString(e));exit(1);} }while(0)
extern bool g_tc_ogroup;
__global__ void kdeq(float* o,const uint8_t* w,const uint8_t* sc,int GR,int Kd){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)GR*Kd)return; int r=i/Kd,c=i%Kd;
    __half_raw h=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)w[i],__NV_E4M3); float wv=__half2float(*reinterpret_cast<__half*>(&h)); int scw=Kd/128; o[i]=wv*exp2f((float)sc[(size_t)(r/128)*scw+c/128]-127.f); }
int main(){ int G=8,R=128,Kd=512; srand(5);
    std::vector<float> oh((size_t)G*Kd); for(auto&x:oh)x=0.02f*((rand()%200)-100)/100.f;
    std::vector<uint8_t> wo((size_t)G*R*Kd),sc((size_t)(G*R/128)*(Kd/128)); for(auto&x:wo)x=(rand()%0x40)|((rand()&1)<<7); for(auto&x:sc)x=120+rand()%12;
    float*dO,*Cr,*Cg,*wf; uint8_t*dW,*dS; CU(cudaMalloc(&dO,oh.size()*4));CU(cudaMalloc(&dW,wo.size()));CU(cudaMalloc(&dS,sc.size()));CU(cudaMalloc(&wf,wo.size()*4));CU(cudaMalloc(&Cr,(size_t)G*R*4));CU(cudaMalloc(&Cg,(size_t)G*R*4));
    CU(cudaMemcpy(dO,oh.data(),oh.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(dW,wo.data(),wo.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(dS,sc.data(),sc.size(),cudaMemcpyHostToDevice));
    kdeq<<<((size_t)G*R*Kd+255)/256,256>>>(wf,dW,dS,G*R,Kd); CU(cudaDeviceSynchronize());
    g_tc_ogroup=false; ogroup_gemm(Cr,dO,wf,1,G,R,Kd,0); CU(cudaDeviceSynchronize());   // oracle (f32 wo_a)
    ogroup_gemm_fp8(Cg,dO,dW,dS,1,G,R,Kd,0); CU(cudaDeviceSynchronize());               // GEMV (bs=1)
    std::vector<float> cr((size_t)G*R),cg((size_t)G*R); CU(cudaMemcpy(cr.data(),Cr,cr.size()*4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(cg.data(),Cg,cg.size()*4,cudaMemcpyDeviceToHost));
    double dot=0,nr=0,ng=0,md=0,mx=0; for(size_t i=0;i<cr.size();++i){dot+=cr[i]*cg[i];nr+=cr[i]*cr[i];ng+=cg[i]*cg[i];md=fmax(md,fabs(cr[i]-cg[i]));mx=fmax(mx,fabs(cr[i]));}
    double cos=dot/(sqrt(nr)*sqrt(ng)+1e-30); bool ok=cos>0.999999&&md/(mx+1e-30)<1e-3;
    printf("[ogroup_gemv] G=%d R=%d Kd=%d cosine=%.7f maxabs/|c|=%.2e -> %s\n",G,R,Kd,cos,md/(mx+1e-30),ok?"PASS":"FAIL"); return ok?0:1; }
