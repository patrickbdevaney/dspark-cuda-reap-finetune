#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."
nvcc -O2 -lineinfo -std=c++17 -arch=sm_110a -I include \
  src/decode.cu kernels/fp8_block_gemm.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu \
  kernels/hc_sinkhorn.cu kernels/mla_forward.cu kernels/block.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/compressed_attn.cu kernels/compressed_block.cu kernels/tc_moe_gemm.cu \
  kernels/tc_fp8_gemm.cu kernels/mla_decode.cu kernels/compressed_decode.cu kernels/block_decode.cu kernels/dscratch.cu kernels/dspark_real.cu kernels/dspark_attn.cu kernels/dspark.cu \
  -o build/decode
echo "built build/decode"
