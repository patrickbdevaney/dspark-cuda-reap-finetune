# Deep Research Findings + Synthesis — Decode Zenith (gemma-4-26B-A4B + DFlash, Thor)
Champion at research time: DFlash 118 tok/s, base 45, tau 13.33. 4 agents launched; 2 returned (below), 2 pending (engines+MoE, black-swan-kernels — APPEND when they land).

## ⚠️ META-LESSON (validates the inside-view critique): both returned agents reasoned from LITERATURE, not our PROFILE — and both top picks are flawed
- Ceiling agent's **#1 "FP8-quantize the draft"** = **ALREADY TESTED, LOST** (constitution: FP8 AND FP4 draft both collapse tau 13.33→11.14; draft MUST stay bf16). The agent didn't know our tested dead-ends.
- Ceiling agent's **"draft is 76%, we're draft-DEPTH-bound"** = **WRONG**. It modeled the draft as 15 SEQUENTIAL autoregressive forwards (3.88 T_base). Ours is **block-diffusion = ONE forward for the whole block**. Our nsys PROFILE is ground truth: **MoE verify 40%, tc_w4a16 (lm_heads+dense) 25%, sdpa 10%, tc_bf16 (draft linears) only 4.5%, k_attn (draft attn) 1.8%.** The VERIFY dominates, not the draft.
- Spec agent's **#1 "MoE verify pruning" (EVICT/FASER)** = LOW-EV for us: it's a TREE-regime lever (prune many low-value branches). Our LINEAR 95%-accept draft already commits ~all tokens; the accepted tokens need ~the whole expert union anyway, so pruning the ~1.67 rejected saves little.
→ **GROUND TRUTH = the profile. The MoE verify (40%, latency-bound at ~1-2 tok/expert) is the real #1 bottleneck. It's a KERNEL problem, not an algorithm problem.**

## VALID, HIGH-VALUE findings (corroborated / correct)
1. **We are ACCEPTANCE-SATURATED: 13.33/14 ≈ 95%.** So the entire EAGLE/Medusa/distillation "raise acceptance α" frontier has ~no runway. Nothing beats block-diffusion here (EAGLE-3 τ~6, MTP τ~1.9, GRIFFIN τ~5 are all DOWNGRADES). **DO NOT swap drafters.** (spec agent, high-confidence, cross-checked)
2. **Trees contraindicated** for batch-1 MoE (expert-union grows per branch; depth-dominated). Matches our own primes measurement + Sequoia/MoESD/EVICT. **DO NOT revisit tree verify.**
3. **Corrected NVFP4 roofline** (ceiling agent, this part is right): NVFP4 = 0.5625 B/wt; ~4B active → **2.25 GB weights + ~0.3 GB KV = ~2.55 GB/token**. Base ceiling = 200 GB/s ÷ 2.55 = **~78 tok/s**; measured base 45 = **58% of roofline → ~1.7× headroom on the base path** (small-batch GEMV inefficiency). Full-peak (273) = 107.
4. **KV compression = DEAD at our context** (2-8k): FP8-KV net-negative below ~7k (fixed overhead), crossover >~50k. Only revisit past ~16-32k context. Breaks bit-exact.
5. **PowerInfer / hot-cold expert / neuron streaming = DEAD on Thor** (definitively refuted): every such win needs a TWO-TIER bandwidth hierarchy (VRAM/RAM, HBM/DDR, DRAM/flash). Thor is ONE 273 GB/s unified pool — a cold expert costs the same as a hot one, no tier to promote into. Also gemma/SwiGLU has no ReLU activation sparsity. **The seductive black swan that isn't.**
6. **Expert-union math**: M=15 tokens × 8/128 experts → verify activates the UNION (~60-78 of 128 experts, roughly uniform routing). Per-token that's ~5.85 experts/token vs base's 8 → DFlash MoE is ALREADY amortized BETTER than base. Bigger block improves amortization further (BLK=32 → ~110 experts / ~28 tokens = ~3.9/token).

## The TESTABLE levers that survive scrutiny (ranked by EV for OUR measured machine)
1. **[KERNEL, #1 bottleneck] MoE verify efficiency** — it's 40%, latency-bound (SoL 55% compute / 30% mem, ~1-2 tok/expert). ILP-split LOST (register wall). Untried: fuse router+gather+GEMM+scatter into fewer kernels; 2-outputs/warp; reduce the invert/router glue; a fundamentally different grouped-GEMM structure for M~1/expert. (pending engines+MoE agent targets exactly this.)
2. **[ALGO, testable now] Bigger block BLK 16→24/32** — we're acceptance-saturated so more tokens/cycle is the ONLY tau lever, AND it improves MoE amortization (fewer experts/token). Bit-exact (verify still fully checks). GATE: does the drafter hold accuracy at longer blocks (it was trained for BLK=16)? MEASURE tau at BLK=20/24. If tau/BLK stays high → direct win. Low effort.
3. **[ALGO, testable] Adaptive draft length (AdaEDL, training-free, entropy stop)** — draft fewer on hard tokens (avoid full M=15 MoE verify when accept will be short), extend on easy. Bit-exact, lossless, low effort. +5-15%.
4. **[KERNEL] Base decode efficiency** 45→toward 78 ceiling (1.7× headroom) — helps the base path; GEMV/dequant at M=1.
5. **[ALGO, workload-specific] Zero-cost n-gram/prompt-lookup draft branch** stacked with bf16 draft — +10-40% on code/repetitive text only, ~0 draft FLOPs, bit-exact. Ref REST 2311.08252.

## DO-NOT (confirmed dead / low-EV — save the effort)
FP8/FP4 draft (tau collapse, TESTED); tree/multi-candidate verify (MoE+depth); swap to EAGLE-3/Medusa/MTP (τ downgrade); MoE verify pruning (low-EV for linear 95%-accept); KV quant (<32k ctx); PowerInfer/hot-expert (unified memory, no tier); target sub-4-bit/2:4 sparsity (breaks bit-exact, verify only 24% anyway... NOTE: our profile says verify is MORE than 24% — re-check when MoE agent lands).

## PENDING (append on arrival — protect via commit)
- Agent af6e2fc50aff19bea: engines + MoE-at-bs=1 frontier (the #1 bottleneck — MOST IMPORTANT pending result).
- Agent a181982ab5290ad04: CUDA/Blackwell black-swan kernels (persistent-kernel, TMA/DSMEM, fusion, small-N split-K).

## ACTION PLAN (post-synthesis)
Priority = the MoE verify kernel (ground-truth #1) + bigger-block (testable algo). When the MoE agent lands, synthesize its actionable kernel ideas with our profile and implement top-down, bit-exact + gate + back-to-back A/B, per the constitution's method.

## AGENT 3 RETURNED: CUDA/Blackwell black-swan kernels (concrete, ranked — GOLD)
Governing fact confirmed: M=1 decode pinned to ~200 GB/s (73% of 273). BW-tricks give single-digit-to-15%; STEP-CHANGES must READ FEWER BYTES.

### TIER 0 — do first, cheap, (mostly) bit-exact:
- **FlashNorm** (arXiv:2407.09577): fold RMSNorm into the FOLLOWING GEMM weights. Gemma: fold **(1+g_i)** (zero-centered norm). Removes a kernel + activation round-trip. **+1-3% e2e, BIT-EXACT.** Low effort. github.com/OpenMachine-ai/transformer-tricks
- **Fused add+RMSNorm + fused gate+up SwiGLU** (arXiv:2602.11808): ~35% MLP memory-traffic cut, **+3-6% e2e**. NOT guaranteed bit-exact (accum dtype — match reference reduction dtype to keep tau).
- **Allocator fix**: on Thor, `cudaMallocManaged` is NOT GPU-cached; `cudaMalloc`/`malloc+cudaHostRegister` IS L2-cached under HW coherence. Use cudaMalloc for weights so router/norm/embed hot re-reads hit L2. Eliminate cudaMemcpy (shared DRAM). Free. (CUDA-13-for-Thor blog)
- **jetson_clocks + nvpmodel -m 0 (MAXN)**: MANDATORY baseline — without EMC pinned max you never reach 73%. Verify EMC CurrentFreq==MaxFreq.
- **Kill concurrent CPU/IO DRAM traffic** during decode (shared EMC arbiter is the ~73% cause): busy-poll not memcpy, move tokenize/logits-copy off critical path. Low-single to low-double-digit % on contended runs.
- **128-bit v4 aligned weight loads**, pad non-pow2 dims.

### TIER 1 — BLACK SWANS (read fewer bytes = break roofline):
- **⭐ #7 HIGHEST-EV BLACK SWAN: Activation sparsity (TEAL arXiv:2408.14690 / CATS arXiv:2404.08763)** — training-free magnitude thresholding, 40-50% sparsity → **skip whole weight COLUMNS at M=1 → never read from LPDDR.** Measured **1.53× @40% / 1.8× @50%** single-batch decode. STACKS multiplicatively on NVFP4. github.com/FasterDecoding/TEAL. RISK: gemma GeLU/SiLU less sparse than ReLU (PowerInfer's ReLU assumption is DEAD here); must build thresholded-gather into mma.sync + **re-measure DFlash tau** (sparsity shifts draft/target distributions). Effort HIGH. **+30-80% if gemma tolerates 40-50%.** *The only lever with a credible path to another ~1.5-1.8× beyond 118.*
- **#8 Sparse-Marlin 2:4** (github.com/IST-DASLab/Sparse-Marlin, `mma.sp.sync`): ~1.8× byte reduction stacked on NVFP4. Needs pruning+calibration (SparseGPT), NOT bit-exact, tau risk, quality risk on 26B MoE. Verify sm_110a assembles mma.sp.
- **#9 MoE expert caching** (lossless if caching not pruning): keep hot experts near-resident. BUT — likely low given uniform routing (MoE-Infinity: <5% hot per-prompt but uniform across workload).

### TIER 2 — Persistent megakernel, HONEST verdict:
- Our earlier LOSS was the MISTAKE of using `grid.sync()` (measured ~35% of token time, ~10× worse than the fix). Winners (Hazy/Kog/MPK) use **sentinel-poll on global-memory counters** (producer/consumer, no barrier) + shmem paging. Kog: 0.80µs vs 7.59µs sync. **CAN be made to win + bit-exact BUT bounded: we're already ~70-73% of floor + have CUDA graphs, so only the bubble term ≈ 10-25% at HUGE effort. NOT highest-EV.**

### CONFIRMED DEAD (de-prioritized with evidence): tcgen05/TMEM (MMA_M=128, +bandwidth-bound anyway — our mma.sync.m16n8k16 / m16n8k64-for-NVFP4 is correct); DSMEM clusters (Thor capped 2-SM cta_group); split-K/stream-K (occupancy not BW — FlashInfer: split-KV doesn't help low-BW GPUs = Thor); TMA (can't raise LPDDR ceiling); warp-spec ping-pong (large-batch only).
- ON-DEVICE TODO: verify Thor L2 size + persistingL2CacheMaxSize via cudaGetDeviceProperties (bounds L2-pin of hot set).

## REVISED ACTION PLAN (3 of 4 agents in; engines+MoE pending)
IMMEDIATE cheap/bit-exact (do first, compound): (1) jetson_clocks+MAXN baseline check, (2) allocator audit (no Managed for weights), (3) FlashNorm fold (bit-exact +1-3%), (4) fused add+RMSNorm/SwiGLU (+3-6%, match dtype).
THE BLACK SWAN (big effort, biggest prize): activation sparsity TEAL/CATS -> thresholded-gather mma.sync, re-validate tau. Path to ~1.5-1.8×.
KERNEL #1 bottleneck (pending MoE agent): the MoE verify (40%). 
SPEC-DECODE (testable): bigger block BLK 16->24, adaptive draft length.
DO-NOT: FP8/FP4 draft (tested-LOST), trees, drafter-swap, KV-quant<32k, PowerInfer, tcgen05, DSMEM.

## AGENT 4 RETURNED (relaunched, grounded): engines + MoE-at-bs=1 — CODE-PROVEN, reorders the MoE plan
DIAGNOSIS (high-conf): MoE 55%compute/30%mem = ISSUE/LATENCY-bound on a dependency chain (NOT compute, NOT bandwidth). Warps eligible-but-stalled on dequant→FMA→shuffle chain. NOT grid-starved. So: cut the chain latency (ILP + tail), not bytes.
### ★ #1 HIGHEST-EV (implement first): MULTIPLE OUTPUTS PER WARP w/ SHARED ACTIVATION operand
- Each warp owns R=2-4 independent OUTPUT rows, R-wide reg accumulator, reuses the loaded activation across all R. llama.cpp mmvq.cu:566/579-582/632 (`float tmp[ncols_dst][rows_per_cuda_block]`).
- DIFFERENT from our failed 2-way ILP split: that split ONE output's reduction (no reuse→register wall); THIS is R DIFFERENT outputs sharing the activation → registers buy ILP AND reuse. **down_bw ALREADY does this (RB=2); gateup is 1-output/warp = the gap.**
- +20-40% on GEMM (×0.4 step). BIT-EXACT (per-dot summation order unchanged). Start R=2 measure then R=4. PAIR with #2.
### #2 cap maxrregcount 40-56 (enabler for #1; frees regs for occupancy, unblocks without spill). +5-15%, bit-exact.
### #3 persistent expert-major kernel + whole-step ONE CUDA graph (removes 60-78 per-expert launches + tail-straggler). +15-30% BUT check current graph coverage first (if verify already graphed, launch part already banked). High cost. Hand-roll scheduler (MPK can't express MoE routing).
### #4 fuse gate+up into one w1 GEMM + in-register SwiGLU + fold router weight (DeepGEMM/vLLM-marlin/llama.cpp). +10-20% GEMM1. Bit-exact if fp32 accum + matched SwiGLU order.
### #5 fuse finalize/scatter into GEMM2 epilogue (atomic scatter-add). +5-15%. Bit-exact ONLY if fp32 accum before bf16 store.
### #6 gather-in-prologue via ids-indirection (no permute pass; Thor unified = pointer arith). Low cost, enables #3. Bit-exact.
### #7 interleave FP4 unpack/scale ALU with prev K-chunk FMAs (Marlin ALU pipelining, NOT cp.async). +5-15%. Subsumed if #1 fills issue slots.
### #8 shorten/overlap warp-reduce tail. +5-10%.
### DO-NOT (confirmed): block-sparse/MegaBlocks/DeepGEMM tile-padded masked GEMM (16-128× padding at 1-2 tok/expert — matches our tcgen05 dead-end); smaller FlashInfer tile (bottoms at 8); memory-side as PRIMARY lever (30% mem = not the wall).
### PLAN: #1+#2 together first (gateup RB=2, reg cap), measure, R=4. Then (after checking graph coverage) #3. Fusions #4-#6 = bit-exact secondary.

## IMPLEMENTED + RESULT: MoE gateup RB=2 (agent #1) = LOST (map-vs-territory)
Tried multiple-outputs-per-warp (RB=2, shared activation) on gateup: U=2 → -0.8% (116 vs 117); U=4 + __launch_bounds__(256,6) → -9% (107, register-cap forced spilling). BIT-EXACT but regresses. WHY: llama.cpp mmvq RB is code-proven for a DENSE single-weight GEMV; our gateup has 4 weight streams (gate+up × 2 rows) = 2× down_bw's register load, so RB=2 blows the register budget and the activation-reuse gain doesn't outweigh it. down_bw already has RB=2 and works BECAUSE it's single-stream. → MoE kernel is resistant to ILP levers (both 2-way-split AND RB=2 lost to the register wall). Next MoE lever = STRUCTURAL: fuse gate+up (#4) or persistent expert-major (#3) — big restructures. OR pivot to the activation-sparsity black swan (highest-EV overall, reads fewer bytes).

## MEASURED + REFUTED: activation sparsity (TEAL/CATS, the #1 black swan) — profile-first killed it before the build
Measured gemma-4 MoE activations at verify (M=15), SPARSITY=1 probe (kept in forward.cu, guarded, zero prod overhead):
- x2 (gate/up input, H=2816): hard-zeros 0.1-0.2%; retained-L2 @ 50% sparsity = 0.986, @40% = 0.994.
- hbuf (down input=silu*up, 704): hard-zeros 0.5-0.9%; retained-L2 @ 50% = 0.994, @40% = 0.997.
VERDICT: activations are magnitude-DENSE (≈0% hard zeros) → TEAL/CATS is inherently SOFT/LOSSY here (thresholding small-nonzero, ~1.4% x2-norm loss @50%). That's TEAL's normal ~1-2%-accuracy-loss regime but NOT bit-exact. The 40%-bottleneck is the VERIFY MoE which MUST be bit-exact (perturbed verify → wrong argmax → non-greedy output, nothing catches it). The draft (lossy-safe) isn't the bottleneck. → **Activation sparsity is REFUTED for our bit-exact constraint — the seductive black swan that doesn't apply, now MEASURED (not assumed), like PowerInfer.** Saved a multi-hour gather-kernel build.
REMAINING bit-exact surface (largely exhausted at 117-118): bigger-block BLK16→24 (τ lever, acceptance-saturated, testable, bit-exact); persistent expert-major kernel (#3, big, ~10-25% bounded, check graph coverage first). Base decode 45→78 (not the served path).
