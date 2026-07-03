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
    st::ShardedSafeTensors S(argv[1], key_map);
    printf("shards=%zu tensors=%zu\n", S.shardCount(), S.count());

    // --- register every shard's data blob for mapped (zero-copy) device access ---
    size_t total = 0; int n = 0;
    for (auto& reg : S.shardRegions()) {
        cudaError_t e = cudaHostRegister((void*)reg.first, reg.second, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (e != cudaSuccess && e != cudaErrorHostMemoryAlreadyRegistered) {
            fprintf(stderr, "cudaHostRegister shard %d (%.2f GB) failed: %s\n", n, reg.second/1e9, cudaGetErrorString(e));
            return 1;
        }
        total += reg.second; n++;
    }
    printf("registered %d shards, %.3f GiB mapped for zero-copy GPU access\n", n, total/1073741824.0);

    // --- verify the GPU reads a real weight via its device pointer ---
    const st::Tensor& t = S.get("layers.1.attn.wq_a.weight");
    void* dptr; if (cudaHostGetDevicePointer(&dptr, (void*)t.data, 0) != cudaSuccess) { fprintf(stderr,"getDevicePtr failed\n"); return 1; }
    unsigned char hbuf[32], dbuf[32];
    memcpy(hbuf, t.data, 32);
    if (cudaMemcpy(dbuf, dptr, 32, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr,"D2H via dev-ptr failed\n"); return 1; }
    bool match = memcmp(hbuf, dbuf, 32) == 0;
    printf("device-pointer read of layers.1.attn.wq_a.weight[:32]: %s\n", match ? "MATCH host mmap" : "MISMATCH");

    // --- resolve device pointers for every tensor (the forward pass will use these) ---
    int resolved = 0; for (auto& kv : S.all()) { void* d; if (cudaHostGetDevicePointer(&d, (void*)kv.second.data, 0) == cudaSuccess) resolved++; }
    printf("resolved %d/%zu device pointers\n", resolved, S.count());
    printf("\nGate-1 (on-device weight access, zero-copy): %s\n", (match && resolved == (int)S.count()) ? "PASS" : "FAIL");
    return (match && resolved == (int)S.count()) ? 0 : 1;
}
