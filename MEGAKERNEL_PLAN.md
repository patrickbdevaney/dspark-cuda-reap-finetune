# Megakernel build plan — the only remaining lever to the ~137 roofline (base 44.6 → ~60-89 target)

## Why this is the last lever (established by 7 measured loop iterations)
Every per-kernel micro-opt is closed: prefetch/widen/C_LUT neutral, cp.async LOST (SMEM round-trip >
register gain at M=1), draft-quant LOST (bf16 is the acceptance moat), TC/grouped/graph closed. The base
runs at ~32% of peak BW (88 of 273 GB/s); individual M=1 kernels are latency-bound and structurally can't
go higher alone. The megakernel is the ONLY thing that raises whole-step BW efficiency (Hazy: 50%→78%).

## Realistic payoff (research-anchored, honest)
- MPK on Qwen3-30B-A3B MoE (structurally ~ours): ~1.16-1.7x over ALREADY-fused vLLM. Hazy 2.5x is Llama-1B dense.
- For our M=1 MoE base: expect ~1.3-1.7x -> base 44 -> ~57-75; DFlash 85 -> possibly >100 IF the verify/step
  pipelining lands. NOT a guaranteed 137. The win is cross-op weight prefetch + killing the ~10 norm CTAs/layer.

## Build (multi-session; SAFE = behind a MEGAKERNEL=1 flag, current kernels stay the gate-passing champion)
Structure (Hazy "No Bubbles" + MPK, adapted to Thor 20 SMs / sm_110a):
1. **Persistent grid** (cudaLaunchCooperativeKernel or grid-stride persistent), ~40-80 CTAs to fill 20 SMs.
   On-GPU instruction interpreter: each CTA consumes a pre-scheduled instruction list (built once on host,
   reused every step — like the CUDA graph but with cross-op overlap).
2. **Instruction types** (start minimal, grow): (a) fused input_rmsnorm+QKV+RoPE, (b) attention, (c) O-proj+
   residual, (d) fused post_attn_rmsnorm + router + expert-select, (e) fused gate+up+SiLU+down per active
   expert (register/SMEM intermediate), (f) final norm + FP4 lm_head. Fold each RMSNorm into the next GEMM
   prologue (FlashNorm: W' = diag(g)W, defer 1/RMS) so norms stop being 1-CTA launches.
3. **Cross-op sync = global-memory counter array** (NOT grid.sync — too slow). Each instruction bumps its
   counter on completion; consumers spin on sentinel. Kog: sentinel-poll cut sync 7.6us->0.9us.
4. **The BW win**: start loading the next instruction's weights (cp.async/direct) before the current finishes.
5. **SMEM paging**: partition Thor SMEM into pages; instructions request/release. Keep the token's activations
   resident across the whole step (no HBM round-trips between ops).

## Incremental, gate-safe milestones (each a loop iteration, champion never broken)
- M1: persistent skeleton + counter-sync + ONE fused instruction (input_rmsnorm+QKV), MEGAKERNEL=1, verify
  bit-exact vs current path on one layer. No speed goal yet.
- M2: add attention + O-proj+residual; verify one full attention sublayer.
- M3: add the fused MoE instruction (gate+up+SiLU+down, on-chip intermediate); verify one full layer.
- M4: chain all 30 layers + lm_head; verify end-to-end == current output; THEN measure. Keep only if faster+gate.
- M5: add cross-op weight prefetch (the 50%->78% BW lever); tune.
- Draft variant: same skeleton for the 5-layer draft (smaller, the research-endorsed hand-writable first target),
  keeps bf16 weights (moat intact).

## Risk / honest note
This is days-to-weeks of careful work with real correctness risk at each milestone. The flag keeps the champion
(base 44.6 / DFlash 85) safe throughout. If M4 doesn't beat the graphed baseline, the honest outcome is that
the current architecture was already near the achievable Thor ceiling for a single author, and DFlash 85 with
the bf16-draft moat (tau 13.3 vs vLLM 7.84) stands as the result.
