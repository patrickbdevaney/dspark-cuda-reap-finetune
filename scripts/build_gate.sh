#!/usr/bin/env bash
# Build the Gate-K unit test (host CUDA). Run: ./build/gate_units ref/goldens
set -e
cd "$(dirname "$0")/.."
nvcc -O2 -std=c++17 -arch=sm_110a -I include \
  tests/gate_units.cu kernels/fp8_block_gemm.cu kernels/hc_sinkhorn.cu kernels/mla_attn.cu kernels/moe.cu \
  -o build/gate_units
echo "built build/gate_units"
