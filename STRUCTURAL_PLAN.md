# STRUCTURAL_PLAN.md — the multiplier phase (device routing → CUDA graphs → KV-decode → spec-decode)

**Why this is a separate phase.** Kernel-compute is largely tapped: TC-everything took the warm forward 687 →
383.5 ms/tok (1.8×), and the last lever (funnel-shift MoE coalescing) gave only −1.3% — the profile confirms
compute is now a small slice. The remaining ~15× to the 38–50 tok/s target is **structural overhead + the
decode regime itself**, not more GEMM tuning. Each step below is gated cosine-1.0 vs the current path and run
**detached-to-file** (memory constraint: forward uses ~90% of shared RAM; see DECODE_HORIZON hard-constraint).

## Crucial reframe: we've been measuring PREFILL, not DECODE
`forward.cu` processes 8 tokens together (M=8, no KV cache, no autoregressive loop). The 38–50 tok/s target is
**autoregressive DECODE** — 1 token/step, M=1, KV-cached. At M=1 the GEMMs are tiny and **launch + host
overhead dominate** — which is exactly what device-routing + graphs kill. So the multipliers pay off in the
decode regime, and building a single-token decode step is part of this phase (not optional).

## Step 1 — Device-side MoE routing (removes the per-layer host round-trip; unblocks graphs)
Current (`moe.cu:167-170`): `cudaStreamSynchronize` + copy `idx/wt` to host + `std::vector` grouping + copy back
— a HARD sync every layer (×43). Replace with on-device grouping:
1. `k_count`: histogram tokens per expert (atomicAdd over the (token,slot) assignments).
2. prefix-sum → `off[nr+1]` (device).
3. `k_scatter`: place each (token,slot) into `alltok[atomicAdd(&cursor[e],1)]` at its expert bucket.
For the expert GEMM loop WITHOUT a per-`me` host sync: launch a **fixed grid** sized for `bs*na` (the max any
expert can hold) per expert and have the kernel read `off[]` on device and early-exit extra threads; OR a single
**grouped-GEMM** kernel over all buckets. Gate: cosine 1.0 vs the current host-grouped batched path. This is the
prerequisite for graphs (a mid-forward host sync makes a graph uncapturable).

## Step 2 — Pre-allocate + static launch sequence (graph-capturable)
Remove mid-forward `cudaMalloc/Free` (Loader per-layer dequant, pp `x16`, funnel temps): allocate a fixed set of
reused scratch buffers up front (memory-neutral — same peak, just not alloc'd/freed each layer). Then the layer
loop is a fixed sequence of kernels on fixed pointers → capturable.

## Step 3 — CUDA-graph capture of the decode step
With Steps 1–2, capture the single-token forward (43 layers + head) as ONE `cudaGraph` via
`cudaStreamBeginCapture`/`EndCapture`, instantiate once, `cudaGraphLaunch` per token. Kills the hundreds of
per-kernel launch overheads that dominate M=1 decode. Gate: identical logits vs the un-captured path.

## Step 4 — KV-cache single-token decode path
Add the autoregressive decode step: MLA/attention over the **cached** latent KV (append new token's KV, attend
over history) instead of recomputing. This is the real decode regime + where tok/s is measured. Reuse the gated
attention kernels; add KV append + the single-token (M=1) path. (KV is tiny by MLA+SWA+DSA design — memory-safe.)

## Step 5 — DSpark speculative decode (the throughput multiplier)
Draft head proposes a block of `block_size=5` tokens; the target verifies the block in ONE graph-launched
forward; accept the longest matching prefix (τ≈0.75–0.8 measured). ~2.5–4× decode throughput on top of the base.
Overlap draft ∥ target on separate streams. This is the payoff the whole draft-head effort is for.

## Expected trajectory (order-of-magnitude, to verify by measurement)
base decode M=1 (graphs + device routing kills overhead) → then spec-decode ×2.5–4 → target 38–50 tok/s. The
kernel TC wins (banked) are the per-step compute floor; graphs remove the per-step overhead; spec-decode
multiplies the token rate. Measure each on a **decode** step (M=1), not the s=8 prefill.

## Discipline (unchanged)
Gate every step cosine-1.0 vs the prior path; run detached-to-file; never a persistent memory addition; log the
measured delta (`OPTIMIZATION_LEDGER.md`) + rationale (`GATE_LOG.md`). Stop-rule + black-swan search still apply.
