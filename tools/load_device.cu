// load_device.cu — register the 96 GiB checkpoint for ZERO-COPY GPU access (unified memory) and verify
// the GPU reads a real weight correctly. Foundation for forward.cu (no separate 96 GiB device alloc).
//   build: nvcc -O2 -arch=sm_110a -I include tools/load_device.cu -o build/load_device
//   run:   ./build/load_device /home/patrickd/models/DeepSeek-V4-Flash-180B
#include "safetensors.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

static std::string key_map(const std::string& in) {   // this REAP ckpt already uses attn/ffn names
    std::string s = in;
    if (s.rfind("model.", 0) == 0) s = s.substr(6);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ckpt_dir>\n", argv[0]); return 2; }
    // --- probe device capabilities (Jetson is integrated: host mem IS device mem) ---
    int integrated=0, hostReg=0, canUseHostPtr=0, mapHost=0;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0);
    cudaDeviceGetAttribute(&hostReg, cudaDevAttrHostRegisterSupported, 0);
    cudaDeviceGetAttribute(&canUseHostPtr, cudaDevAttrCanUseHostPointerForRegisteredMem, 0);
    cudaDeviceGetAttribute(&mapHost, cudaDevAttrCanMapHostMemory, 0);
    printf("integrated=%d hostRegisterSupported=%d canUseHostPtrForRegMem=%d canMapHostMem=%d\n",
           integrated, hostReg, canUseHostPtr, mapHost);

    st::ShardedSafeTensors S(argv[1], key_map);
    printf("shards=%zu tensors=%zu\n", S.shardCount(), S.count());

    // --- weight->device path for integrated Tegra (cudaHostRegister of file mmap is unsupported):
    //     single-copy shard-0 into cudaHostAlloc(Mapped) GPU-accessible memory, then GPU reads it.
    //     (forward.cu scales this to all 46 shards; peak ~96 GiB single copy, no mmap+device doubling.) ---
    auto regs = S.shardRegions();
    const void* h0 = regs[0].first; size_t bytes0 = regs[0].second;
    void* pinned = nullptr;
    cudaError_t e = cudaHostAlloc(&pinned, bytes0, cudaHostAllocMapped);
    if (e != cudaSuccess) { fprintf(stderr, "cudaHostAlloc %.2f GB failed: %s\n", bytes0/1e9, cudaGetErrorString(e)); return 1; }
    memcpy(pinned, h0, bytes0);                                    // faults mmap + single copy into GPU-accessible buf
    void* dptr; if (cudaHostGetDevicePointer(&dptr, pinned, 0) != cudaSuccess) { fprintf(stderr,"getDevicePtr failed\n"); return 1; }
    printf("shard-0: cudaHostAlloc(Mapped) %.3f GiB + copied; device pointer resolved\n", bytes0/1073741824.0);

    // verify GPU reads the FIRST 32 bytes of the pinned buffer correctly
    unsigned char hbuf[32], dbuf[32];
    memcpy(hbuf, pinned, 32);
    if (cudaMemcpy(dbuf, dptr, 32, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr,"D2H via dev-ptr failed\n"); return 1; }
    bool match = memcmp(hbuf, dbuf, 32) == 0;
    printf("GPU read of shard-0[:32] via device pointer: %s\n", match ? "MATCH host copy" : "MISMATCH");
    cudaFreeHost(pinned);
    printf("\nWeight->device path (cudaHostAlloc-mapped single-copy): %s\n", match ? "PASS" : "FAIL");
    return match ? 0 : 1;
}
