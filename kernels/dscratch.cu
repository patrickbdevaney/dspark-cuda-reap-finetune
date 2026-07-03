// dscratch.cu — decode scratch arena globals. See dscratch.h.
#include "dscratch.h"
bool   g_arena_on  = false;
char*  g_arena     = nullptr;
size_t g_arena_off = 0;
size_t g_arena_cap = 0;
void arena_init(size_t cap){
    if(g_arena){ cudaFree(g_arena); }
    cudaMalloc((void**)&g_arena, cap); g_arena_cap = cap; g_arena_off = 0; g_arena_on = true;
}
void arena_reset(){ g_arena_off = 0; }
