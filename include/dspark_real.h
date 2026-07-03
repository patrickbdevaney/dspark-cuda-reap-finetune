// dspark_real.h — real DSpark block-diffusion head pieces (DSpark-head repo model.py:744-874).
// Composable core: main_x (tap-proj) + Markov head. The block attention (main-KV window ⊕ block) and the
// host AR/verify loop are built on top (see DSPARK_HEAD_BUILD.md). Reuses gated primitives throughout.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// main_x = main_norm( main_proj(main_hidden) ). main_hidden:[s, 3*dim] (cat of mean-pooled taps L40/41/42).
// main_proj: fp8 [dim, 3*dim] + e8m0->f32 scale.  main_norm: rmsnorm weight [dim] (f32). -> main_x:[s, dim].
void dspark_main_x(float* main_x, const float* main_hidden, const uint8_t* main_proj, const float* main_proj_s,
                   const float* main_norm, int s, int dim, float eps, cudaStream_t stream = 0);

// DSparkMarkovHead: token_ids:[n] -> logits_bias:[n, vocab] (bigram correction) + markov_embed:[n, rank].
// markov_w1: [vocab, rank] f32 (embedding rows).  markov_w2: [vocab, rank] f32 (rank->vocab head).
void dspark_markov(float* logits_bias, float* markov_embed, const int* token_ids,
                   const float* markov_w1, const float* markov_w2, int n, int vocab, int rank,
                   cudaStream_t stream = 0);
