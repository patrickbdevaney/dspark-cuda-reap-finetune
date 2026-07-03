#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."
nvcc -O2 -std=c++17 -arch=sm_110a -I include \
  src/forward.cu kernels/fp8_block_gemm.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu \
  kernels/hc_sinkhorn.cu kernels/mla_forward.cu kernels/block.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/compressed_attn.cu kernels/compressed_block.cu \
  -o build/forward
echo "built build/forward"
