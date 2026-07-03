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
