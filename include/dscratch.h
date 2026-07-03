// dscratch.h — decode scratch arena. At M=1 the per-call cudaMalloc/cudaFree/cudaStreamSynchronize in every
// sub-function dominate. When the arena is ON (decode), dmalloc bumps from a pre-allocated slab (reset per
// layer), dfree/dsync are no-ops, and the whole token runs as one stream with a single final sync. When OFF
// (gates / prefill forward), it falls back to real cudaMalloc/cudaFree/cudaStreamSynchronize — zero behavior
// change for existing callers. Single-stream, single-threaded decode only.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
extern bool   g_arena_on;
extern char*  g_arena;
extern size_t g_arena_off, g_arena_cap;

static inline void* dmalloc(size_t n){
    if(g_arena_on){ n=(n+255)&~((size_t)255); void* p=g_arena+g_arena_off; g_arena_off+=n;
        if(g_arena_off>g_arena_cap){ fprintf(stderr,"[dscratch] arena overflow %zu>%zu\n",g_arena_off,g_arena_cap); abort(); }
        return p; }
    void* p; cudaMalloc(&p,n); return p;
}
static inline void dfree(void* p){ if(!g_arena_on && p) cudaFree(p); }
static inline void dsync(cudaStream_t s){ if(!g_arena_on) cudaStreamSynchronize(s); }

void arena_init(size_t cap);   // allocate the slab once, set g_arena_on
void arena_reset();            // g_arena_off = 0 (call at the top of each layer's work)
