# OPTIMIZATION_LEDGER.md — decode-speed campaign (per OPTIMIZATION_METHODOLOGY.md)

Durable, auditable record of the decode-optimization campaign: baseline → champions → A/B deltas → stop-rule
decisions → black-swan leaps. Correctness gates never loosen (each optimized kernel must pass its bit-exact/
cosine gate vs the correctness-first version before becoming champion).

## BASELINE (measured, `src/forward.cu` correctness-first kernels)
| run | s | total ms | ms/tok |
|---|---|---|---|
| Gate 1 | 27 | 12621 | **467** |
| Gate 1 | 8 | 5494 | 687 |
| Gate 1 | 5 | 4726 | 945 |
- **Base decode ≈ 0.5–1.5 tok/s** (single-token decode ≈ one full forward; prefill amortizes fixed cost).
- τ baseline (proxy MTPBlock, single-token) = **0.815**. Block-diffusion block-τ: pending piece-5 run.
- **TARGET: 38–50 tok/s** (base + DSpark spec-decode, in concert). Gap: **~30–100×** → structural, not marginal.

## BOTTLENECK (root cause, from code — not guessed)
1. **Every GEMM is `<<<grid(N,M), 32>>>` = ONE WARP PER OUTPUT ELEMENT walking K serially** (fp8_block_gemm,
   fp4_gemm). No tiling, no shared mem, no tensor cores, no vectorization. The single dominant cost.
2. **MoE dispatch loops per-token on the HOST** (`for t<bs`) × per-expert kernel launches (6 routed×3 + shared×3)
   with `cudaStreamSynchronize` → thousands of tiny serialized synced launches per forward (×43 layers).
These are the correctness-first shapes — chosen for provable correctness, always meant to be replaced.

## CHAMPION TO PORT (black-swan lever #1 — already proven in the lineage)
`~/gemma-cuda-hybrid/kernels/tc_verify_gemm.cu` — **Marlin-class raw `mma.sync.m16n8k16` TC GEMM** (W4A16,
+3.3%, bit-exact in the Gemma campaign). Also `tc_w4a16_gemm`, `tc_bf16_gemm` (draft). Adapt to OUR dtypes:
- **fp8-act × fp4-weight (W4A8)** for routed experts — `mma.sync` fp8 path (m16n8k32) + e8m0 block-scale apply.
- **fp8 × fp8 (W8A8)** for dense/attn linears — same, e4m3.
- Keep e8m0 scales read in-kernel (also the MEMORY.md dequant-free win).

## GRIND PLAN (methodology: harness → champion A/B → stop-rule → black-swan; literature scan FIRST)
1. **[BLOCKED — session limit, retry after 11:10pm ET via resumeFromRunId wf_b54e3f4d-169] Literature/landscape scan** — deep-research on fastest fp8/fp4 MoE GEMM + MLA/sparse-attn +
   spec-decode verify kernels on Blackwell `sm_110a` (mma.sync shapes, cp.async, ldmatrix, grouped/batched MoE,
   megakernel fusion, CUDA graphs). Mine for structural leaps beyond Marlin before micro-tuning.
2. **Decode benchmark harness** — `build/forward` timing per-component (attn vs MoE vs head), fixed prompt,
   report tok/s. No change is "an improvement" without an A/B delta here. Log every variant below.
3. **Target #1 — MoE expert GEMM + dispatch (biggest cost):** port tc_verify_gemm → W4A8 `mma.sync` fp4-expert
   GEMM; batch the dispatch (grouped GEMM, device-side, kill the host per-token loop + syncs). Gate bit-exact
   vs fp4_gemm (gate_units MoE). Expect the largest single jump.
4. **Target #2 — dense/attn fp8 GEMM (W8A8 mma.sync)** — same treatment; gate vs fp8_block_gemm.
5. **Target #3 — fusion/graphs** — megakernel fuse RMSNorm+GEMM+act-quant; CUDA-graph the decode step (kill
   launch overhead). 6. **DSpark head kernels** overlap with target verify. Co-optimize SYSTEM tok/s.
7. **Stop-rule** per component at <~2–3%/effort → escalate to the next black-swan (from the scan).

## LEDGER (variant → measured tok/s → keep/kill → note)
| # | component | variant | tok/s | vs champ | decision |
|---|---|---|---|---|---|
| 0 | MoE fp4 GEMM | fp4_gemm (warp-per-output, oracle) | 1.412 ms/call | — | baseline/gate-oracle |
| 1 | MoE fp4 GEMM | **tc_fp4_gemm** (Marlin TC W4A8) — per-call repack | 0.462 ms | 3.06× | superseded |
| 2 | MoE fp4 GEMM | **tc_fp4_gemm + CACHED repack** (ptr-keyed) | **M=8: 0.080 ms / M=1: 0.065 ms** | **M=8: 19.7× / M=1: 2.71×** | **CHAMPION** (cosine 1.0). Swapped into moe.cu behind MoEWeights.use_tc. TC win GROWS with M -> BATCH tokens. |
| — | note | full-model enable: caching every expert repack DOUBLES expert mem (~82GB) -> OOM. Repack at LOAD (store repacked in place of original) or per-layer scope. | | | TODO (blocks full-model use_tc) |
| — | next | **FP4 COMPUTE (2070 TFLOPS = Thor's strongest, 4x fp16)** — HW present but NOT in ptxas for sm_110 (CUDA 13); reach via cuBLASLt/cuDNN FP4 GEMM (library) NOW, or hand-PTX when exposed. A/B vs our TC path. | | up to ~4× | INVESTIGATE (top lever; see FP4_COMPUTE_NOTE.md — cuBLASLt lightest) |
| E2E | forward.cu | tc_fp8 dense + batched | 559.8 ms/tok | 1.23× | banked, correct (argmax stable). |
| E2E | forward.cu | + tc_fp4 repack-at-load pp (single fwd) | 555.9 ms/tok | — | double-counts one-time repack. |
| E2E | forward.cu | 2x WARM (repack amortized) | 451.2 ms/tok (2.22 tok/s) | 1.52× | pass0 539.2/pass1 451.2. |
| E2E | forward.cu | + TC attn-output (fp16 mma/group) | 388.5 ms/tok | 1.77× | −14%. |
| E2E | forward.cu | + funnel-shift MoE + device-route (Step 1) | **378.8 ms/tok (2.64 tok/s)** | **1.81× base** | funnel −1.3% (MoE GEMM ~14%), device-route −1.2%. Kernel phase tapped; correct, argmax 270. |
| 3 | dense/attn fp8 GEMM | **tc_fp8_gemm** (native FP8 mma m16n8k32.e4m3, W8A8) | **0.023 ms** (vs 0.413) | **17.88×** | **CHAMPION** (cosine 1.0, max_rel 2e-5 vs fp8_block_gemm; fp8 in, no fp16 upconvert). Dense/attn/shared-expert path. |
| — | (was) | native FP8 mma m16n8k32 (dequant fp4-wt→fp8, acts already fp8, FP8 tensor core = 2× fp16; FP4 mma NOT on sm_110) | | ~2× over current fp16 TC | TODO (HW-verified path) |
| — | next | batched/grouped MoE dispatch (kill host per-token loop) | | | TODO |

## Target #1 PORT DESIGN (turnkey — adapt gemma tc_w4a16 → our W4A8 MoE-expert GEMM)
Champion `tc_verify_gemm.cu` is fp4-weight × **fp16**-act, mma.sync.m16n8k16.f32.f16.f16.f32, 1 warp=8 N-cols,
weight-repack + `__ldcs` 16B loads + in-register FP4→fp16 dequant × per-k-tile fp8 scale. Our fp4_gemm inputs
are fp8-e4m3 act (A_fp8[M,K]) + fp4-e2m1 weight (B_fp4[N,K/2]) + per-32 e8m0 weight scale (b_s[N,K/32]).
**PORT (minimal, reuse the proven kernel):**
1. Convert A fp8→fp16 once (cheap kernel) → feed the fp16 A path unchanged. (Later: native fp8 mma m16n8k32 = 2× — a follow-up A/B.)
2. Weight path identical (fp4 → the `tcv_fp4x2` register dequant already matches e2m1).
3. Replace gemma's per-k-tile fp8 scale (`tcv_e4m3(sb[..])*wg_inv`) with our **per-32 e8m0**: scale = `exp2(b_s_byte-127)`;
   the mma k-tile is 16, our scale block is 32 → one e8m0 covers 2 k-tiles; index `b_s[n, k_tile/2]`, apply as the fp16 B multiplier.
4. Repack weights per-expert (reuse `k_tc_repack_w`) at load (once, cache by ptr) — for all 160 experts × 43 layers.
**GATE:** bit-exact (or cosine>0.9999 for the fp16-act rounding) vs `fp4_gemm` on the gate_units MoE golden. Then
A/B in the decode harness → log tok/s here. Then batch the MoE dispatch (device grouped GEMM) as the paired win.
**Files:** new `kernels/tc_moe_gemm.cu` + gate; swap into `moe.cu` behind a flag (keep fp4_gemm as the gate oracle).

## STATUS (campaign): baseline+bottleneck+champion+plan committed; scan retry @11:10pm ET (resumeFromRunId
## wf_b54e3f4d-169). NEXT EXECUTION = write kernels/tc_moe_gemm.cu per the design above, gate bit-exact vs
## fp4_gemm, A/B in the decode harness. This is the single largest expected jump (naive→TC GEMM, ~10-30×).

## LITERATURE SCAN RESULTS (deep-research wf_b54e3f4d-169, 23 sources, 24 confirmed / 1 refuted)
**REFRAMING INSIGHT (highest value):** *low-batch decode is MEMORY-BANDWIDTH-bound — cost is dominated by
weight loading, not compute or routing* (confirmed, fused-MoE + FlashMLA sources). This EXPLAINS our own A/B:
M=1 → 2.71× (bandwidth-capped, weight load dominates) vs M=8 → 19.7× (load amortized, compute-where-TC-wins).
So the biggest decode levers are **(a) reduce weight TRAFFIC and (b) BATCH to amortize loads** — not just faster MMA.

### Ranked black-swans for next rounds (with source-backed numbers)
1. **Batch/amortize weight loads** — the TC win scales with M (our data + theory). Batch the decode microstep and
   the spec-decode block(5) through the MoE so weight loads amortize → toward the 19.7× regime. (Highest ROI, free.)
2. **Fuse gate+up (w1,w3) projections** — share input tile, keep intermediate in registers → ~35% less global
   memory traffic, **1.16–1.40×** (source: fused-MoE-dispatch). Directly attacks the bandwidth bound. Clean win.
3. **No-fp32-dequant / native-dtype** (e8m0 scales in-kernel, fp8 wo_a, bf16 lm_head) — pure traffic reduction,
   also the MEMORY.md win. Bandwidth-bound decode makes this a SPEED win too, not just memory.
4. **Grouped GEMM (single-launch, device permute/unpermute + row_id_map)** for MoE dispatch — kills the host
   per-token loop + per-expert launches; block-scheduled Triton variant (BLOCK_M fixed) cut Mixtral 24→5 launches,
   beat Megablocks +131% @32 tok (LOW-batch win, falls off @512 — our regime). NOTE: CUTLASS grouped GEMM has
   **no FP4/W4A8** — we build our own W4A8 grouped GEMM (extend tc_moe_gemm to multi-expert single launch).
5. **CUDA graphs** to kill 43-layer launch overhead. **Megakernel REFUTED** (0-3 vote: does NOT reliably beat
   CUDA-graphed cuBLAS at int8 batch-1) → prefer CUDA graphs, do NOT over-invest in a full megakernel.
6. **MLA sparse decode is DEQUANT-bound** (dequant ~50 cyc vs MMA ~34 cyc/token), not MMA-bound. FlashMLA is
   SM90/SM100 only (no sm_110a) → adapt, don't port. The CTA-cluster "crossover" exploits MQA (1 KV head, all
   query heads share it) to cut dequant 50% — applicable to our MLA (1 KV head). FP8 KV = 656 B/token layout.
Sources: DeepGEMM, FlashMLA hopper-fp8-sparse-deep-dive, fused-moe-dispatch-triton, momoe, NV_grouped_gemm,
hazyresearch no-bubbles, AutoMegaKernel(refuted), NVIDIA DFlash + Jetson-Thor-7x blogs. journal.jsonl has all 111.

## REVISED GRIND ORDER (bandwidth-first): batch loads (1) → fuse gate+up (2) → native-dtype no-dequant (3) →
## W4A8 grouped GEMM single-launch (4) → CUDA graphs (5) → MLA dequant-cut crossover (6). Each A/B'd + logged.

## Grind #2 — DONE + COMPOUNDED PATH VALIDATED
- batched (grouped dispatch) cosine 1.0 vs oracle; **batched + use_tc (TC W4A8 at M=count) cosine 1.0** — the
  compounded fast MoE path is correct. Carries: TC 19.7x (M=8) + dispatch amortization. Enable both in the
  multi-token forward (prefill/spec-block/capture). Per-token+fp4_gemm stays the bit-exact oracle.

## Grind round #2 (batch MoE + fuse gate+up) — IN PROGRESS, gate caught a bug
- **gate+up input-share: DONE** — the routed w1/w3 GEMMs share the single quantized input tile `Xeq` (act_quant
  once, feed both). (Full register-fusion of g,u+swiglu into one kernel = further follow-up.)
- **Batched/grouped dispatch: DONE — gate PASSES cosine 1.0** (moe.cu, `MoEWeights.batched`). Group tokens by
  expert -> one GEMM per expert at M=count -> swiglu_wrow -> scatter_add; shared M=bs. Per-token oracle kept
  as `else` (default, bit-exact). ROOT CAUSE of the earlier failures (found via compute-sanitizer + per-expert
  dumps): (1) pageable async cudaMemcpyAsync of etok/ewt raced the gather -> OOB garbage; (2) then a reused
  per-expert blocking cudaMemcpy into tok_d silently failed to update (every expert saw stale [1,0,..]) ->
  only tok1's output landed. FIX: upload ALL tokens/weights ONCE as flat device arrays + per-expert OFFSET
  pointers (no reused buffer, no per-expert copy). cosine 1.0 vs oracle; oracle stays bit-exact.

## Grind — STRUCTURAL_PLAN Step 1b: zero-sync GROUPED-GEMM MoE — DONE, WIN
- **A/B (same session, s=8 prefill, warm ms/tok, argmax=270 both):**
  per-expert device_route (`NOGROUPED=1`) **365.7** → grouped (`g_moe_grouped`, default) **319.8** = **−12.5% (1.14×)**.
  Memory 109.6/122.8 both (memory-neutral). Unit gate cosine 0.9999999 vs fp4_gemm oracle.
- **Mechanism:** one grouped W4A8 GEMM per stage (gate/up/down) over ALL experts, replacing ~48 tiny per-expert
  launches ×3 ×43 layers; tile→expert map built on-device from `off[]` (`k_build_tiles`), so the per-layer
  `off[]` D2H copy — the last mid-forward host sync — is GONE. Grid.y = bs*na (host upper bound, extra tiles
  early-exit). Weights = same in-place-repacked fp4 bytes as the pp path (funnel-shift alignment per tile).
- **Structural payoff (beyond the 12.5%):** MoE is now CUDA-graph-capturable → enables Step 3 (graphs), where
  the M=1 launch-bound decode regime is where this compounds hardest.
- kernel: `kernels/tc_moe_gemm.cu` (`k_grouped_w4a8_kernel`, `k_build_tiles`, `tc_fp4_grouped_gemm`);
  integration `kernels/moe.cu` (`g_moe_grouped` branch); gate `tests/gate_grouped_moe.cu`.

## Decode opt #1 — native-e8m0 expert scales (no per-token dequant) — WIN 3x
- **A/B (full 43-layer M=1 decode, argmax=270 both, generated seq identical):** per-layer f32 dequant
  **2019 ms/tok (0.50 tok/s)** -> native e8m0 **678 ms/tok (1.47 tok/s)** = **-66% (3.0x)**. mem 110.7/122.8.
- **Mechanism:** the grouped MoE GEMM reads the ORIGINAL e8m0 scale BYTES (F8_E8M0) from the WeightStore and
  computes `exp2f(byte-127)` in-register (bit-identical to the pre-dequanted f32 pow2 -> same argmax/tokens).
  Removes ~160x3 `cudaMalloc`+dequant-kernel launches PER LAYER PER TOKEN (~20,640/token) AND keeps the scale
  pointers persistent (no f32 buffer). kernel `k_grouped_w4a8_e8m0_kernel` (tc_moe_gemm.cu); `MoEWeights.e8m0_scales`.
- Remaining per-token dequant (next levers): wo_a fp8->f32 (134 MB/layer), attn fp8 scales, norms bf16->f32,
  shared-expert scales; + thousands of scratch mallocs/syncs per token (pre-alloc = Step 2) + launch overhead
  (CUDA graphs = Step 3). Physics floor ~273 GB/s -> active-weight reads ~20-25 ms/tok base; still overhead-bound.

## Decode opt #4 — native wo_a (fp8+e8m0 -> f16 in one pass, no f32 dequant) — WIN -22%
- **A/B (warm M=1 decode, argmax=270):** 581 ms/tok (1.72) -> **453 ms/tok (2.21 tok/s)** = -22%. mem 110.6.
- **Mechanism (nsys-guided):** profile showed `k_deq_fp8_blk` (wo_a fp8->f32, 2.02 ms x43 = 87 ms/tok) +
  `k_f2h` (f32->f16, ~53 ms/tok) dominating. `ogroup_gemm_fp8`/`k_wo_fp8_to_f16` convert the fp8 wo_a straight
  to f16 with the e8m0 block-scale in-register (bit-identical: same fp8 decode x exp2(byte-127) x float2half),
  killing the f32 dequant buffer AND the double conversion. wo_a stays native/persistent (MLAWeights.wo_a_native).
- **NOTE (non-determinism):** multi-token decode sequences vary run-to-run (MoE scatter_add atomics -> near-tie
  argmax flips downstream); token-0 argmax is stable (==270) so the gate is deterministic. Benign; a sorted
  scatter would make it bit-reproducible if ever needed.

## Decode opt #5 — build per-layer weight structs ONCE (persistent) — WIN -27%
- **A/B (warm M=1, argmax=270):** 453 -> **331 ms/tok (3.02 tok/s)** = -27%. mem 110.6/122.8 (structs resident,
  no OOM: experts + wo_a native, so residual dequant ~2 GB).
- **Mechanism:** previously run_layer rebuilt BlockWeights/CompressedBlockWeights EVERY token (fill_attn/fill_moe
  -> Loader dequant of scales/norms/gate/compressor + cudaMalloc + host struct-build). Now build all 43 structs
  ONCE up front (build_layer), keep dequant resident, and the decode loop does zero per-token Loader work.
  Removed ~120 ms/token of per-token dequant+malloc. Cumulative: 0.50 -> 3.02 tok/s (6.0x) across opts #1-5.
- Remaining (nsys, per token): tc_fp8 attn GEMMs at M=1 ~91 ms (need M=1 GEMV), MoE grouped GEMM ~78 ms,
  tc_ogroup ~44 ms, MoE router compute_scores ~19 ms (serial per-thread dot), + launch overhead (CUDA graphs).

## Decode opt #6 — fused fp8 wo_a in TC ogroup (no per-token wo16 conversion) — WIN -29%
- **A/B (warm M=1, argmax=270):** 331 -> **234 ms/tok (4.27 tok/s)** = -29%, memory-neutral (110.7).
- **Mechanism (nsys):** after build-once, `k_wo_fp8_to_f16` was converting the ENTIRE wo_a (33.5M elems x43) to
  f16 EVERY token = ~80 ms/token (wo_a is constant!). `tc_ogroup_fp8_kernel` decodes fp8 wo_a * e8m0 scale ->
  f16 INSIDE the mma inner loop (no wo16 buffer, reads fp8 = half the bytes). Bit-identical. argmax=270.
- **Cumulative decode: 0.50 -> 4.27 tok/s = 8.5x** (opts #1-6). Remaining (nsys): tc_fp8 attn GEMMs ~91 ms,
  MoE grouped ~78 ms, tc_ogroup mma ~30 ms, router compute_scores ~19 ms (serial dot), + launch overhead.
  Path to ~50 tok/s = reduce base toward bandwidth floor (~40 ms) + DSpark spec-decode (block verify, ~3-4x).

## Decode opt #8 — vectorized M=1 fp8 GEMV (attn dense GEMMs) — WIN -30%
- **A/B (warm M=1, argmax=270):** 214 -> **148.9 ms/tok (6.71 tok/s)** = -30%. Unit gate cosine 1.0 vs oracle.
- **Mechanism:** at M=1 the m16-tile TC (tc_fp8) is mma-LATENCY bound (15/16 rows wasted, fixed mma cost). The
  GEMV (`fp8_gemv_m1_kernel`) is one warp per output n, reads the B[n] row uint-vectorized (4 fp8/load, 32
  lanes = 128 contiguous bytes, fully coalesced), per-128 scales per element, single warp-reduce -> pure
  bandwidth. Routed for M==1 in fp8_block_gemm (NO_GEMV=1 falls back to TC for A/B). **Cumulative 0.50 -> 6.71
  tok/s = 13.4x.** (Earlier the byte-load warp-per-output oracle was SLOWER than TC — the uint vectorization
  is what makes the GEMV win.) Next biggest: MoE grouped fp4 GEMM at M=1 (analogous fp4 GEMV).

## Decode try — M=1 fp4 MoE GEMV — REVERTED (survivor, not champion)
- fp4 GEMV on ORIGINAL fp4 (funnel-aligned uint4 weight load, e8m0 in-register), unit-gated cosine 1.0 vs
  fp4_gemm (tests/gate_fp4_gemv). But full-decode A/B: **148.9 -> 230 ms/tok (SLOWER)**. Root cause: fp4's
  packed-nibble scalar decode (32 fp8+fp4 decodes/lane/32-block) costs more than the m16-mma waste it removes;
  the TC mma decodes fp4x2->half2 in hardware far cheaper. UNLIKE fp8 (opt #8) where simple uint decode won.
  LESSON: M=1 GEMV wins only when the per-element decode is cheap (fp8), not for packed fp4. Kept default-OFF
  (`g_moe_gemv`, env MOE_GEMV=1) as a gated reference; MoE stays on the TC grouped path. Champion 148.5 ms/tok.
