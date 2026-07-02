// hc_sinkhorn.cu — HC split + Sinkhorn, correctness-first (Gate K oracle: ref/gen_units.py).
// One block per token; a single thread does the tiny hc x hc (=4x4) Sinkhorn. hc is small (4), iters=20;
// this is latency-trivial per token. Faithful to kernel.py:371-438 op order (row-softmax+eps, col-norm,
// then iters-1 of {row-norm, col-norm}). Vectorize/warp-per-token later, AFTER the gate passes.
#include "hc_sinkhorn.h"

#define HCMAX 8   // max hc (config uses 4)

__global__ void hc_sinkhorn_kernel(float* __restrict__ pre, float* __restrict__ post,
                                   float* __restrict__ comb,
                                   const float* __restrict__ mixes, const float* __restrict__ hc_scale,
                                   const float* __restrict__ hc_base, int n, int hc, int iters, float eps) {
    int i = blockIdx.x;
    if (i >= n || threadIdx.x != 0) return;
    int mh = (2 + hc) * hc;
    const float* mx = mixes + (size_t)i * mh;
    float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

    // pre / post
    for (int j = 0; j < hc; ++j) {
        pre[(size_t)i * hc + j]  = 1.f / (1.f + expf(-(mx[j]      * s0 + hc_base[j])))       + eps;
        post[(size_t)i * hc + j] = 2.f / (1.f + expf(-(mx[hc + j] * s1 + hc_base[hc + j])));
    }
    // comb[hc,hc] = mixes[2hc:] * s2 + base
    float c[HCMAX * HCMAX];
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            c[j * hc + k] = mx[2 * hc + j * hc + k] * s2 + hc_base[2 * hc + j * hc + k];

    // comb = softmax(-1) + eps   (row-wise)
    for (int j = 0; j < hc; ++j) {
        float mmax = -1e30f; for (int k = 0; k < hc; ++k) mmax = fmaxf(mmax, c[j*hc+k]);
        float sum = 0.f; for (int k = 0; k < hc; ++k) { c[j*hc+k] = expf(c[j*hc+k]-mmax); sum += c[j*hc+k]; }
        for (int k = 0; k < hc; ++k) c[j*hc+k] = c[j*hc+k] / sum + eps;
    }
    // comb = comb / (comb.sum(-2)+eps)   (col-normalize)
    for (int k = 0; k < hc; ++k) {
        float cs = 0.f; for (int j = 0; j < hc; ++j) cs += c[j*hc+k];
        cs += eps; for (int j = 0; j < hc; ++j) c[j*hc+k] /= cs;
    }
    // iters-1 Sinkhorn passes: row-normalize then col-normalize
    for (int it = 0; it < iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) { float rs=0.f; for (int k=0;k<hc;++k) rs+=c[j*hc+k]; rs+=eps; for (int k=0;k<hc;++k) c[j*hc+k]/=rs; }
        for (int k = 0; k < hc; ++k) { float cs=0.f; for (int j=0;j<hc;++j) cs+=c[j*hc+k]; cs+=eps; for (int j=0;j<hc;++j) c[j*hc+k]/=cs; }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            comb[((size_t)i * hc + j) * hc + k] = c[j * hc + k];
}

void hc_sinkhorn(float* pre, float* post, float* comb,
                 const float* mixes, const float* hc_scale, const float* hc_base,
                 int n, int hc, int iters, float eps, cudaStream_t stream) {
    hc_sinkhorn_kernel<<<n, 32, 0, stream>>>(pre, post, comb, mixes, hc_scale, hc_base, n, hc, iters, eps);
}
