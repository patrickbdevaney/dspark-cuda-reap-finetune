# Decode Speed Gap — Step 1 Inventory (dedup key)

**Headline:** best measured decode **7.89 tok/s** (126.7 ms/tok). Generic vLLM on same checkpoint+HW class: **18.9–24.4 tok/s** (0xSero). We are **2.5–3× slower than generic vLLM.**

**Bandwidth diagnostic (computed):** active weights per M=1 token ≈ **8.77 GB** (attn fp8 ~0.115 GB/layer + 6/160 experts fp4 ~0.076 GB/layer + shared, ×43 layers). Floor = 8.77 GB ÷ 273 GB/s = **32 ms/tok**. We hit 126.7 ms → **~69 GB/s achieved = 25% of peak**. vLLM at 18.9–24.4 tok/s ⇒ **60–77% of peak**. **The gap is bandwidth efficiency, not algorithm.**

## Methods already implemented
- **M=1 KV-cache decode engine** — 43-layer AR, MLA + KV-compressor + DSA-indexer hybrid attention (sliding / 4:1 sparse / 128:1 compressed per layer), Hyper-Connections, YaRN RoPE — the enabling function.
- **Quant: W4A8 MoE (fp8-act × fp4-weight, e8m0 block scales) + W8A8 fp8 dense/attn** — matches checkpoint.
- **structs-once** — build layer weight structs once, not per token — removed per-token Loader host work.
- **native e8m0 / wo_a** — no per-layer re-dequant; scales consumed in-kernel.
- **bump-arena scratch** (dmalloc/dfree/dsync, g_arena_on) — no mid-step malloc/sync.
- **tc_fp8** — native FP8 tensor-core GEMM for M≥2 (prefill/verify) — ~18× over the warp oracle.
- **M=1 fp8 GEMV** (fp8_gemv_m1_kernel) — uint-vectorized weight read, warp-per-output — beats TC at M=1.
- **fp4 grouped MoE GEMM** (k_grouped_w4a8_e8m0) — tensor-core, e8m0 scales, funnel-shift for unaligned fp4.
- **fused ogroup fp8 TC** (tc_ogroup) — o(f32)→f16 + fused fp8 wo_a decode, no per-token wo16 convert.
- **M=1 ogroup GEMV** (ogroup_gemv_fp8_kernel) — **WIN −15% (148→126.7 ms/tok)**; killed the ogroup's scalar-byte m16-mma running at ~40 GB/s.
- **determinized MoE scatter** — fixed-order sum vs atomicAdd — reproducible + spec-accept 1.9→2.5.
- **full 43-layer CUDA-graph capture** (device-pos, all 3 attn flavors, device-conditional compressor emit) — bit-exact, **measured PARITY (0.99×) → engine is GPU-bound, NOT launch-bound**.
- **spec-decode**: M=K verify (weights read once for K tokens) + DSpark block-diffusion draft head + accept-longest-prefix.
- Cumulative: **0.50 → 7.89 tok/s (15.9×)**.

## Methods considered but rejected (do NOT re-research without new evidence)
- **fp4 GEMV at M=1** — scalar nibble decode slower than TC mma. REGRESSED.
- **fp8 M=K GEMV** (verify) — TC reads weight once AND does M×N via mma; GEMV does M scalar dots. 334→362 ms. REGRESSED.
- **ogroup M=K GEMV** — acc[bs] register array kills occupancy; regressed even bs=1 (126→196 ms). REGRESSED.
- **CUDA graphs for speedup** — PARITY (0.99×). Not launch-bound, so no win. (Kept as infra.)
- **verify indexer host-sync removal** — no timing change (sync wasn't the bottleneck at this scale).
- **warp-per-output fp8 oracle at M=1** — slower than the vectorized GEMV.
- **persistent GPU memory across power-cycle** — hard constraint: forbidden (power-cycle risk at ~120 GiB).

## Known open questions / unverified assumptions
- **[MEASURING NOW via ncu]** Are the decode GEMV/MoE kernels COMPUTE-bound (software fp8/fp4 decode: dec_e4m3 intrinsic + exp2f per byte) rather than memory-bound? At 25% of peak BW this is the leading hypothesis — the tensor cores decode fp8/fp4 in HARDWARE (free), our GEMVs decode in SOFTWARE per byte.
- We use **NO vendor library** — not DeepGEMM, FlashMLA, FlashInfer, CUTLASS, Marlin/Machete. All kernels hand-rolled. Unknown how much a purpose-built fp8×fp4 GEMM would close the 25%→70% gap.
- Kernels are launched **fully serialized** (dsync after each op, no PDL / no multi-stream overlap). CUDA-graph parity suggests this is NOT the current bottleneck (GPU is busy back-to-back), but confirm.
- **No cross-op kernel fusion** — each GEMM / norm / quant / RoPE is a separate kernel with an arena round-trip through DRAM. Activations are tiny at M=1, so this is likely minor vs the weight-read compute-bound issue, but unquantified.
- Unknown whether vLLM's 18.9–24.4 tok/s uses DeepGEMM / Marlin / FlashInfer, and what % bandwidth those achieve on THIS model — reverse-engineering that tells us the target.
- Whether sm_110 (Thor, Blackwell-class) tensor-core FP4 paths (tcgen05, TMA) are usable for M=1 decode without the row-waste that made our TC lose at M=1.
