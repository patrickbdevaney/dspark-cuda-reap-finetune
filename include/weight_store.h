// weight_store.h — load the full sharded checkpoint into GPU-accessible memory on integrated Jetson.
// Single-copy: pread each shard's data blob straight into a cudaHostAlloc(Mapped) buffer (never faults the
// mmap data pages), then every tensor's device pointer = shard_base_dev + offset_in_blob. No mmap+device
// doubling -> no OOM. mmaps stay lazy (headers only) for metadata. See ROADMAP Phase A.
#pragma once
#include "safetensors.h"
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>

namespace st {

struct DevTensor { const void* dev = nullptr; std::string dtype; std::vector<int64_t> shape; size_t nbytes = 0;
                   int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; } };

class WeightStore {
public:
    WeightStore(const std::string& dir, std::string (*key_map)(const std::string&) = nullptr,
                const char* only_prefix = nullptr) {
        ShardedSafeTensors S(dir, key_map, only_prefix);
        // 1. load each shard's data blob into a mapped pinned buffer via pread (single copy, no mmap fault)
        std::unordered_map<const SafeTensors*, void*> base;   // shard -> pinned device-accessible base
        for (auto& kv : S.shards()) {
            SafeTensors* sh = kv.second.get(); size_t nb = sh->dataBytes();
            void* buf; cudaError_t e = cudaHostAlloc(&buf, nb, cudaHostAllocMapped);
            if (e != cudaSuccess) throw std::runtime_error(std::string("cudaHostAlloc shard failed: ") + cudaGetErrorString(e));
            size_t got = 0; off_t off = (off_t)sh->dataFileOffset();
            while (got < nb) { ssize_t r = pread(sh->fd(), (char*)buf + got, nb - got, off + got);
                if (r <= 0) throw std::runtime_error("pread shard failed: " + sh->path()); got += (size_t)r; }
            // drop the file pages from the page cache — we've copied them to `buf`. Reclaims ~96 GiB of
            // otherwise-"used" reclaimable cache that Tegra cudaMalloc won't auto-evict (memory headroom).
            posix_fadvise(sh->fd(), off, nb, POSIX_FADV_DONTNEED);
            void* dev; if (cudaHostGetDevicePointer(&dev, buf, 0) != cudaSuccess) throw std::runtime_error("getDevicePointer failed");
            pinned_.push_back(buf); base[sh] = dev; host_base_[sh] = sh->dataStart(); dev_base_[sh] = dev;
            loaded_ += nb;
        }
        // 2. resolve every tensor's device pointer = shard_dev_base + (t.data - shard_host_start)
        for (auto& kv : S.all()) {
            const Tensor& t = kv.second;
            const SafeTensors* owner = nullptr;
            for (auto& b : host_base_) { const uint8_t* hb = b.second;
                if (t.data >= hb && t.data < hb + b.first->dataBytes()) { owner = b.first; break; } }
            if (!owner) throw std::runtime_error("tensor not in any shard: " + kv.first);
            size_t offset = (size_t)(t.data - host_base_[owner]);
            DevTensor d; d.dev = (const uint8_t*)dev_base_[owner] + offset; d.dtype = t.dtype; d.shape = t.shape; d.nbytes = t.nbytes;
            t_.emplace(kv.first, std::move(d));
        }
    }
    ~WeightStore() { for (void* p : pinned_) cudaFreeHost(p); }

    bool has(const std::string& n) const { return t_.count(n) > 0; }
    const DevTensor& get(const std::string& n) const {
        auto it = t_.find(n); if (it == t_.end()) throw std::runtime_error("weight not found: " + n); return it->second; }
    template<class T> const T* dev(const std::string& n) const { return (const T*)get(n).dev; }
    size_t count() const { return t_.size(); }
    double loadedGiB() const { return loaded_ / 1073741824.0; }

private:
    std::unordered_map<std::string, DevTensor> t_;
    std::unordered_map<const SafeTensors*, const uint8_t*> host_base_;
    std::unordered_map<const SafeTensors*, void*> dev_base_;
    std::vector<void*> pinned_;
    size_t loaded_ = 0;
};

} // namespace st
