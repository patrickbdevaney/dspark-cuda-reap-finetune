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
| 1 | MoE fp4 GEMM | **tc_fp4_gemm** (Marlin TC mma.sync.m16n8k16, W4A8) | **0.462 ms/call** | **3.06× FASTER** | **CHAMPION** (cosine 1.0, rms 0.03% vs fp4_gemm; incl per-call repack — cache it for more) |
| — | next | cache weight repack at load (kill per-call repack+malloc) | | (expect >3×) | TODO |
| — | next | native fp8 mma m16n8k32 (skip fp8→fp16) | | (~2× more) | TODO |
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
