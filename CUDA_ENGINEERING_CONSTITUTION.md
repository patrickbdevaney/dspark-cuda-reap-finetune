# CUDA Engineering Constitution — gemma-4-26B-A4B + DFlash on Jetson Thor
**Ground-truth state & cumulative champion knowledge. Read this first. Maintained across context resets.**
Last major state: DFlash **~118 tok/s**, base ~45, gate PASS, tau 13.33. **We BEAT vLLM (107) by +10%.** Arc 85.65->118 (+38%). Repos: `~/gemma-cuda-hybrid` (active), `~/gemma-cuda-server` (v1.0 banked).

---

## 0. MISSION & CURRENT STATE
Purpose-built pure-CUDA NVFP4 + DFlash spec-decode server for `google/gemma-4-26B-A4B-it` on Jetson Thor. NEVER sacrifice correctness (`scripts/gate_self.sh` must PASS; every change verified BIT-EXACT vs champion on `/tmp/share/bench_primes.txt` GEN=8 tokens `236778 236764 236743 236800 236764 236743 236810 236764`).
- **Champion metric:** `DFLASH=1 GEN=80 ./build/forward` median-of-5-6, on `/tmp/share/bench_primes.txt` (prompt = "List the first 40 prime numbers, comma separated.").
- **Arc:** 82 (start) → 85.65 → **108** (+26% this arc). vLLM gemma DFlash = **107** (85ms/step, tau 9.21). Our tau **13.33** = 1.45x moat → **ceiling ~157 tok/s** if we hit vLLM's step time.

## 1. HARDWARE — Jetson Thor (VERIFIED, some corrected the hard way)
- **sm_110a** (Blackwell). CUDA 13.0; renumbered from sm_101 (12.8/12.9). Build: `nvcc -arch=sm_110a`. CUTLASS: `-DCUTLASS_NVCC_ARCHS=110a` (no `Sm110` C++ tag in `arch/arch.h` — stock SM100 kernels need arch-guard patching).
- **20 SMs** (NOT 96 — the "96" is max-config *tensor cores*). 2560 CUDA cores. **228 KB smem/SM** (datacenter-class, CC 11.0 Table 31). Has **TMEM + tcgen05 + TMA** (like B200, unlike DGX Spark/RTX50).
- **~273 GB/s LPDDR5X unified** — but GPU only gets **~73% achievable** (~200 GB/s memcpy measured); shared arbiter. LPDDR5X latency ~500-800ns (unmeasured; do a pointer-chase before sizing cp.async STAGES).
- **THERMAL THROTTLING IS REAL** after long sessions: absolute tok/s drifts 94↔108 for the SAME code. **ALWAYS measure back-to-back A/B (stash baseline, rebuild, compare)** — never trust absolute numbers across time. Rebooting kills the Claude session — do NOT reboot.
- LD_PRELOAD for vLLM docker: `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`. Stop vLLM with `docker kill` NEVER `docker stop` (page-cache leak). Drop caches: `sudo sysctl -w vm.drop_caches=3`.

## 2. MODEL + DFLASH structure (constants in src/forward.cu / src/draft.cu)
- gemma-4: H=2816, NLAYER=30, **NHEAD=16**, sliding layers HD_S=256/NKV_S=8 (G=2), FULL layers (L=5,11,17,23,29) HD_F=512/NKV_F=2 (G=8), MoE 128 experts top-8, MOE_INT=704, VOCAB=262144. NVFP4 = E2M1 codes + E4M3 per-16 block scale + FP32 per-tensor global. gemma double-norm (~10 RMSNorm/layer).
- DFlash draft: 5-layer qwen3-style block-diffusion, BLK=16, k=14 spec tokens, shares frozen embed+lmhead (FP4). Draft weights MUST stay bf16 (see §4). Verify = M=15 forward.
- Profiled DFlash step @118: **MoE ~40% (biggest: gateup_grouped 19% + down_bw 14% + base 8%)**, tc_w4a16 (lm_heads+dense) 25%, sdpa 10%, tc_bf16 (draft linears) 4.5%. Bottleneck MOVED here twice — always re-profile. History: was draft-lm_head-dominated (22%) until routed to tc; draft-linear (12%) until routed to bf16 TC.

## 3. CHAMPION KERNEL STACK (what's in place, all bit-exact)
- **NVFP4 lm_head** (the original breakthrough, +35%): tied bf16 embed quantized to E2M1+E4m3 (`g_ewp/g_ews/g_egs`, separate cudaMalloc → 256B aligned). base M=1 = `fp4_gemv`; **M=15 draft+verify lm_heads now route through the Marlin `tc_w4a16_gemm`** (kernels/tc_verify_gemm.cu) — this was +9.7%, the single biggest win.
- **Marlin-class TC verify GEMM** `tc_w4a16_gemm` (handles ALL M≤16 dense verify GEMMs N≤8192 AND the lm_heads): raw `mma.sync.m16n8k16.f32.f16.f16.f32`, 1 warp = 8 N-cols, **A fragment read direct from L2-cached global x** (no shared/sync), **B in-register FP4→fp16 dequant**, **offline weight REPACK** → 16-byte `int4` coalesced loads via `__ldcs` (evict-first). Cached by src-ptr, lazy repack on warm-up (graph-safe). WARPS=1 = max grid fill.
- **MoE grouped**: `k_moe_invert` (expert→token map) + `k_moe_gateup_grouped` + `k_moe_down_bw` (weight-resident, no atomics, per-assignment partials + `k_moe_finalize`). Both have U=4 K-unroll prefetch. W4A16 (fp16 act, NO activation quant → dodges vLLM's ~75µs FP4-quant-at-bs=1 trap).
- **Attention head-pack** `sdpa_cache_kernel`: one block per (query, kv_head), all G sibling heads, **KV read once** (was 4x GQA-redundant); qs/acc in registers, only `red[G*hd]` in shared (16KB even for hd=512/G=8).
- CUDA graphs on base + verify decode. HW FP4 decode `__nv_cvt_fp4x2_to_halfraw2`, HW e4m3 `__nv_cvt_fp8_to_halfraw` (register-only, replaced divergent constant C_LUT).

## 4. MEASURED DECISION SURFACE (what WON / LOST / NEUTRAL — do not re-litigate)
**WON (banked):** NVFP4 lm_head (+35%); bandwidth-down MoE (+2.8%); MoE C_LUT→HW cvt (+0.5%); **route lm_heads through tc (+9.7%)**; **16-byte int4 + __ldcs evict-first (+3.5%)**; tc weight prefetch U=8 (+1.9%); tc grid-fill WARPS=1 + no-shared-A (+3%); tc offline repack (+1.6%); MoE gateup+down U=4 prefetch (+5%); dense TC raw-mma in-register dequant (+3.3%); head-pack attn (+0.6% short-ctx); **draft linears through bf16 TC GEMM (+9.3%!, tau 13.33 unchanged, output bit-exact — the same warp-per-column->mma win as the lm_head; draft numerics only touch tau, verify guarantees output).**
**LOST (reverted, don't retry):** draft→FP4 AND draft→FP8 (both collapse acceptance 13.33→11.14 — the bf16 draft IS the moat); cp.async on the *base M=1 gateup* (SMEM overhead > register gain at M=1); fp4_gemv/gateup 128-bit widen (per-EXPERT FP4 ptrs not 16B-aligned — but the lm_head EMBED buffer IS, that's why 16B works there); grid-cooperative megakernel (grid-barrier too expensive for M=1 tiny ops); TC grouped MoE (128-tile padding waste at ~2 tok/expert); full-step megakernel per-unit (barrier overhead); FP4 verify lm_head vs bf16 k_lmhead_batched (neutral — bf16 already efficient block-per-vocab).
**NEUTRAL:** lm_head via TC when replacing already-efficient bf16 (memory-bound); base gateup micro-tuning (U=8, launch_bounds — register wall).
**LOST — cp.async pipeline on tc GEMM (2026-07-01, STAGES=2/3/4/6 ALL ~107 vs 108.3 __ldcs):** SoL shows 90.9% L1TEX scoreboard stall (mma waits for weight) — LOOKS like cp.async territory, but our MAX-GRID-FILL structure (1 warp/block, 32768 blocks for lm_head) hides the per-warp stall via BLOCK-level parallelism (SM switches to another resident block). Async win < shared-hop + pipeline-fill overhead (kg8=22 groups). Marlin cp.async is for FEW LARGE register-heavy blocks (warp-level hiding); wrong tool for many-tiny-blocks. Register-prefetch of int4 (UG) also regressed (register pressure). **__ldcs sequential int4 + max-grid-fill is champion; tc GEMM at structural limit ~108, 57% mem.** NOTE: pushing tc 57→90% mem = only ~+2% (tc weight ~1.1GB/step is small vs 123ms block); the +9.7% lm_head win was KERNEL EFFICIENCY (CUDA-core→tc), NOT memory %.
**LOST — MoE gateup 2-way accumulator ILP split (2026-07-01, -1.3% 116.3 vs 117.8):** MoE grouped kernels are latency-bound (SoL 55% compute / 30% mem) but NOT on the accumulator chain — 8 extra accumulators' register pressure > ILP gain. At bs=1 verify most experts get cnt=1 token (only 2 accumulators) but splitting further regressed. MoE (~40% of step now, the biggest category) is resistant: 16B loads won't help (30% mem), ILP won't help (register wall), TC-mma won't help (~1 tok/expert padding waste). Left as-is. Next MoE idea (untried): reduce the invert/router glue, or 2 outputs/warp.
**KEY META-LESSON:** several "LOST/NEUTRAL" verdicts were SHALLOW — the tc kernel was neutral on lm_head UNTIL it had repack+prefetch+16B, then routing the (w4a16 CUDA-core) draft lm_head through it was +9.7%. **Re-test dismissed levers after the kernel they'd use improves.**

## 5. BEST KERNEL PATTERNS (reusable, proven on Thor)
1. **SoL profiling drives everything.** `sudo -E env DFLASH=1 GEN=20 ncu --launch-skip N --launch-count K --kernel-name regex:... --section SpeedOfLight --csv ./build/forward`. Read Memory% vs Compute%: both <60% = LATENCY-bound → **prefetch (raise MLP)**; "grid too small" → **more blocks (fewer warps/block, WARPS=1)**; Memory>60% Compute low = bandwidth → **16B int4 loads + evict-first + coalesce (repack)**.
2. **Marlin memory recipe (the 60→90% path):** (a) weights `__ldcs`/`cp.async.cg.L2::evict_first` (read once, don't thrash L2 that activations reuse); activations plain (evict-normal). (b) **16-byte int4 loads only** — sub-16B can't use cp.async.cg + wastes MSHR/LSU. (c) offline REPACK weight into `[n_block][k_group8][lane*16]` so a warp reads 512 contiguous B/group, feeding mma register-order directly (no ldmatrix for B). (d) cp.async 4-6 shared stages, `cp.async.wait_group stages-2`, prefetch stage `pipe+stages-1` two k-steps early (Thor has 228KB smem → go deep). (e) XOR swizzle `^(row%8)` for bank-conflict-free A.
3. **mma.sync m16n8k16 fragment layout** (verified): A row-major 8 halves/thread (a0=As[gid][2t4..], a1=[gid+8], a2=[gid][2t4+8..], a3=[gid+8]); B col-major 4 halves/thread (b0=W[n0+gid][2t4,2t4+1], b1=[2t4+8,2t4+9]); C 4 f32 (c0=D[gid][2t4], c1=[gid][2t4+1], c2=[gid+8][2t4], c3=[gid+8][2t4+1]). gid=lane/4, t4=lane%4.
4. **K-unroll register prefetch** (latency-bound M=1..16): issue U independent weight loads into `w[U]` before decode/FMA. U=4 for MoE, U=8 for tc GEMM.
5. **In-register FP4 dequant** (no shared round-trip): `__hmul2(tcv_fp4x2(byte), sc2)` straight into mma B fragment. Pre-fold `global·block_scale` into one `__hmul2`.
6. **Head-pack GQA**: one block per (q, kv_head), G sibling heads share the one KV read; qs/acc in registers, batched reduction (red[G*hd]) to keep syncs low.
7. **NVFP4 dequant alt (not yet used):** bit-splice E2M1→fp16 via `&0x70007000 >>3 | sign` + one `__hmul2` (cheaper than cvt; vLLM marlin/dequant.h:397).

## 6. vLLM / SGLang — what we learned & stole (code-level, cited in git log)
- vLLM gemma DFlash = **107 tok/s / 85ms step / tau 9.21 / 55% accept** (measured on THIS box, docker `vllm/vllm-openai:gemma-aarch64-cu130`, transformers 5.10.2, `--speculative-config method=dflash --attention-backend triton_attn`). We exceed it (108) with tau 13.33.
- **tcgen05 is NOT the answer** (research-confirmed): block-scaled NVFP4 locked to MMA_M∈{128,256}; at M=15 we're 60× under compute roof (crossover M≈900). Padding M→128 = zero extra HBM (weight streamed once) but adds tcgen05.alloc/cp/ld latency. `mma.sync` is correct.
- vLLM verify attention = FlashInfer `BatchPrefillWithPagedKVCache` with **GQA head-packing** (packed_qo_len = M×group, CTA_TILE_Q=64, grid=8 CTAs/kv-head, KV read once). Non-causal = just `causal=False` (drop diagonal test). We reproduced the head-pack.
- vLLM greedy verify = full M×VOCAB logits + dense argmax (`rejection_greedy_sample_kernel` = pure equality `draft!=target_argmax`). **Opportunity we have and they don't:** fused chunked-vocab argmax (never materialize [M,256k]) since we don't do temp>0 — but measured NEUTRAL for us (lm_head not bandwidth-bound; our bottleneck was the GEMM efficiency, now fixed by tc).
- SGLang 1.78x vLLM MoE at bs=1 via `M<=E → BLOCK_SIZE_M=16/GROUP_SIZE_M=1` + single-CTA smem `moe_align` (no global atomics) + trtllm tile_tokens_dim=8. MoE glue (router+topk+align+quant) ≈ expert matmul cost at bs=1.
- Files to model on: `IST-DASLab/marlin/marlin_cuda_kernel.cu`, vLLM `csrc/.../marlin/{dequant.h,gptq_marlin_repack.cu,marlin_template.h}`, flashinfer `attention/{prefill,scheduler}.cuh`, sglang `fused_moe_triton_*`, `moe_align_kernel.cu`.

## 7. METHODOLOGY (non-negotiable)
- **Back-to-back A/B for every change** (thermal). `run5(){ ...median NR==3...}; A=$(run5); git stash; rebuild; B=$(run5); stash pop; rebuild; A2=$(run5)`.
- **Bit-exact gate every change**: GEN=8 tokens must == the 8-token reference AND `gate_self.sh` PASS.
- Commit each champion atomically with before/after in the message. Revert regressions/neutrals immediately.
- Weight-repack caches keyed by src ptr; lazy on warm-up so CUDA-graph capture never sees a cudaMalloc.

## 8. ROADMAP — RESEARCH-SYNTHESIZED SURFACE (see RESEARCH_FINDINGS.md for full detail + citations; champion 118, 4-agent deep research done 2026-07-01)
**Corrected roofline (NVFP4):** ~2.25 GB wt + 0.3 GB KV = 2.55 GB/tok → base ceiling **~78 tok/s** (measured 45 = 58% → ~1.7× base headroom); full-peak 107. We are NOT bandwidth/KV-bound. Verify (MoE 40% + sdpa 10% + dense) dominates the step; draft is small (block-diffusion = ONE forward, not 15).
**GROUND TRUTH (nsys @118):** MoE verify 40% (biggest), tc_w4a16 lm_heads+dense 25%, sdpa 10%, tc_bf16 draft-linears 4.5%, k_attn 1.8%.
IMMEDIATE (cheap, bit-exact — do first, compound):
1. **FlashNorm** — fold RMSNorm into next GEMM weights (gemma: fold **(1+g)**, zero-centered). Removes kernel + activation round-trip. +1-3%, BIT-EXACT. arXiv:2407.09577.
2. **Fused add+RMSNorm + fused gate+up SwiGLU** — ~35% MLP traffic cut, +3-6% (match reduce dtype to keep tau). arXiv:2602.11808.
KERNEL #1 (biggest bottleneck):
3. **MoE verify kernel** (40%, latency-bound 55%compute/30%mem, ~1-2 tok/expert). ILP-split LOST (register wall). Untried: fuse router+gather+GEMM+scatter; 2-outputs/warp; cut invert/router glue. (engines+MoE agent pending — MOST IMPORTANT result.)
ALGO (testable, bit-exact):
4. **Bigger block BLK 16→24** — acceptance-saturated (95%) so more-tokens/cycle is the ONLY tau lever + improves MoE amortization. GATE: measure tau holds (drafter trained for BLK=16).
5. **Adaptive draft length** (AdaEDL entropy-stop) — fewer draft on hard tokens. +5-15%, lossless.
THE BLACK SWAN (big effort, biggest prize ~1.5-1.8×):
6. **Activation sparsity (TEAL/CATS)** — training-free magnitude threshold, skip weight COLUMNS at M=1 (fewer LPDDR bytes), stacks on NVFP4. Risk: gemma SiLU less sparse than ReLU; re-measure tau. Only lever that breaks the roofline. arXiv:2408.14690.
PERSISTENT MEGAKERNEL (revised verdict): our loss was `grid.sync()` (~35% of token time); winners use sentinel-poll counters (Hazy/Kog/MPK). CAN win + bit-exact BUT bounded ~10-25% at huge effort (we're already ~73% of floor + have graphs). Not highest-EV.
DEAD/DON'T (evidence in RESEARCH_FINDINGS): FP8/FP4 draft (tested-LOST tau); trees; drafter-swap (τ downgrade); KV-quant <32k; PowerInfer/hot-expert (unified mem = no tier); tcgen05/TMEM (MMA_M=128); DSMEM clusters (Thor 2-SM cap); split-K/stream-K/TMA (not BW levers); cp.async on tc (block-parallelism hides latency).
4. **Whole-step CUDA graph** coverage (~20% launch overhead in general; we're pure-CUDA so partial).
5. **FP8 KV cache** (long-context; head-pack already cut KV 4x).
6. Re-profile after each — the bottleneck moves (it was lm_head all along, not what earlier filtered profiles showed).

## 9. GOTCHAS
- Per-expert / per-layer FP4 weight ptrs are NOT 16B-aligned (safetensors packing) → uint4 loads CRASH. Only the standalone-cudaMalloc'd embed (g_ewp) and REPACKED buffers are 16B-safe.
- N must be divisible by 8 for the tc (true for qkv/o/VOCAB); K divisible by 128 for the 16B-int4 repack (2816, 4096 OK).
- Full layers hd=512 → naive shared attention exceeds 48KB → keep qs/acc in registers.
- `k_lmhead_batched` (bf16) is ALREADY efficient (block-per-vocab + shared reuse) — don't "optimize" to FP4 (neutral).

---

## 10. FULL CUMULATIVE IMPROVEMENT LEDGER (both repos — transferable to ANY MoE/NVFP4 inference on Thor)
Every measured improvement across `gemma-cuda-server` (v1.0, built decode 0→82) and `gemma-cuda-hybrid` (v2.0, 82→118). Each is a TRANSFERABLE pattern for MoE + NVFP4 + spec-decode on sm_110a. Grouped by technique class.

### A. CORRECTNESS FOUNDATION (gemma-cuda-server, must-have before any speed work)
- NVFP4 safetensors loader + cublasLt NVFP4 GEMM + scale-swizzle converter (validated maxrel 0.4% vs fp32 ref on real 16GB ckpt).
- Full 30-layer forward → **gate PASS = top-1 matches vLLM** on confident prompts. Precision unification: **W4A16 everywhere** (fp16 act, FP4 weight) → DFlash==base==vLLM bit-exact. (W4A4 tested: 5.5% quant err erodes acceptance — W4A16 is the accuracy/speed sweet spot.)
- DFlash draft/verify end-to-end, acceptance gate (tau 4.0 → tuned DK=14 → tau 13.33). Draft bug fixes: within-block attention must be CAUSAL for sliding layers 0-3, non-causal only full layer.

### B. BASE DECODE M=1 (gemma-cuda-server, 6.45 → ~45 tok/s) — transferable GEMV/MoE patterns
- **fp16 activations for GEMV** (ncu-guided): decode is memory-bound; fp16 acts halve the act traffic. +key.
- **HW FP4 decode** `cvt.f16x2.e2m1x2` (later `__nv_cvt_fp4x2_to_halfraw2`) + **half2** math: 15.81→18.52 (+17%). Replaced shared-mem e4m3 LUT with register HW cvt.
- **warp-per-output GEMV** (vs block-per-output): 6.45→11.02 (+71%). Vectorized uint32 weight loads.
- **MoE HW decode + fp16 + half2** in gateup/down: 18.52→21.40 (+15%).
- **MoE down warp-per-(t,d), 8 experts FUSED into one accumulator** → ONE shfl reduce (vs 8 block reductions, 34% util before): 21.3→25.9 (+22%). `__ldcs` streaming.
- **Parallel router top-8**: 128-thread reduction-argmax (was 1 thread/1024 ops): +5.5%.
- **CUDA-graph base decode**: 30.73→34.18 (+11.2%). Kills launch overhead for the M=1 tiny-op chain.
- LOST here: uint4 lmhead (embed 8B-align), constant-mem e4m3 LUT (neutral), FP8 lm_head (fp8 decode not free on Thor), warp-per-output MoE (2× tried, failed — block-per-output better for that shape).

### C. DFLASH SPEC-DECODE (gemma-cuda-server, the multiplier: base×tau) — transferable to ANY draft+verify
- **half2 draft linear**: DFlash 31.34→37.42 (+19.4%). **half2 verify lmhead** (weight decoded once, reused across M): 42.05→51.22 (+21.8%).
- **incremental draft context** (only new positions' K/V): 37.42→42.05 (+12.4%).
- **grouped verify MoE gateup** (group tokens by expert, weight-resident): 51.87→59.72 (+15.1%).
- **batched lm_head** (warp-per-n, reuse W across M positions) + **device argmax** (no D2H copy + host argmax): 26.2→31.46.
- **NVFP4 verify lm_head** (+6.9%) then **NVFP4 draft lm_head** (+26%, 65→82): the tied embed quantized to FP4 = 4× less lm_head traffic. THE v1.0 breakthrough.
- **T>0 typical-acceptance** feature (exact-parity greedy; helps low-accept workloads; no-op on primes).
- LOST: tree/multi-round verify (net-negative on primes — depth-dominated, linear DFlash better), draft→FP4/FP8 (collapses tau 13.33→11.14 — **draft MUST stay bf16, it IS the moat**).

### D. MARLIN-CLASS TC + DEEP-RESEARCH ARC (gemma-cuda-hybrid, 82 → 118) — the biggest transferable wins
- **Raw `mma.sync.m16n8k16` TC GEMM with in-register FP4→fp16 dequant** (B straight to fragment, no shared round-trip): the naive wmma was −5%; the Marlin-class hand-pipeline BEAT CUDA-core (+3.3% dense). Fragment layout derived in §5.3.
- **Offline weight REPACK** into mma-fragment/coalesced order (cached by ptr, lazy on warm-up = graph-safe): +1.6%. Then **16-byte int4 loads + `__ldcs` evict-first**: +3.5%. **Software-pipelined weight prefetch U=8**: +1.9%. **WARPS=1 grid-fill + no-shared-A (direct L2)**: +3.1%.
- **THE TWO BIG ONES (deep-research-driven):** route **draft+verify LM_HEADS through the tc kernel** (they were still warp-per-column CUDA-core — the biggest kernel at 22%): **+9.7%** → beat vLLM. Then **draft LINEARS through a bf16 TC GEMM** (mma.f16.f16.f32, bf16→f16 weights, no dequant): **+9.3%, tau unchanged, output bit-exact**.
- **MoE gateup+down U=4 K-unroll prefetch** (SoL: latency-bound): +5%. **MoE weight-resident down** (no atomics + finalize): +2.8%.
- **Attention head-pack** (one block/(query,kv_head), KV read once vs 4× GQA-redundant): +0.6% short-ctx (bigger long-ctx).
- **DEAD-ENDS measured (don't repeat):** CUTLASS/tcgen05 TC for M≤16 (padding waste, M=15 is 60× under compute roof, MMA_M locked 128); megakernel M1/M2/full-step (grid-barrier overhead dominates M=1 tiny ops — wrong regime vs Hazy big-op); cp.async pipeline (max-grid-fill hides latency at BLOCK level, not warp); MoE ILP split (register wall); TC grouped MoE (~1 tok/expert padding).

### E. THE TRANSFERABLE META-PLAYBOOK (apply to any new MoE/NVFP4 model on Thor)
1. Correctness first: W4A16, gate vs reference, bit-exact. 2. Profile with ncu SoL EVERY step — latency-bound (both<60%)→prefetch/more-ILP-or-blocks; bandwidth→16B int4+evict-first+repack; grid-too-small→WARPS=1. 3. The bottleneck MOVES after each win — re-profile (ours went lm_head→draft-linear→MoE). 4. At M=8-16 (spec verify), warp-per-column CUDA-core GEMMs are LATENCY-BOUND — route through raw `mma.sync` TC (NOT CUTLASS/tcgen05, which need M≥128). This single pattern gave +9.7% AND +9.3%. 5. Spec-decode draft must stay high-precision (bf16) — quantizing it destroys the tau moat. 6. Back-to-back A/B always (thermal). 7. Repack weights offline into fragment/coalesced order, cache by ptr, lazy on warm-up (graph-safe).
