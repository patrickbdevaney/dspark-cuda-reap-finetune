# Capture + Train wall-time — optimal path (research-backed)

Goal: cut the ~1–4 day capture+train run for the DSpark head → REAP fine-tune to as short as possible on
one Jetson Thor. Backed by deep-research #2 (`research/deepresearch2_final.json`, 102 agents, cited).
Supersedes the naive pre-flight estimate.

## The honest Thor correction (read first)
The eye-popping "30–90× from batching" numbers are from **offloading** regimes (model does NOT fit GPU →
baseline is disk-crippled: MoE-Gen 236B 0.8→31 tok/s, MoE-Lightning 10.3×). **Thor's 96 GiB fits in the
122.8 GiB unified pool**, so our single-stream baseline is not offload-crippled. Realistic batching gain is
the **dense roofline ~3–5×** (batch 16→128: vLLM 318→964 TPS ≈3×), and it **saturates early for MoE**
because activated experts grow with batch (arithmetic intensity I ≈ n·Nk/Ne; a 256-expert top-6 head needs
large *per-expert* token counts to reach compute-bound). So batching helps, but it is not the 30× miracle.

## Levers, ranked by real wall-time impact on Thor
1. **Gate 2 FIRST — let measured τ decide how much fine-tune we even need.** The unpruned DSpark head may
   already give usable τ on REAP (modest active-path shift; REAP kept top-6 routing). If τ is decent, a
   *light* regenerated fine-tune — or none — suffices. This is the single biggest wall-time lever: it can
   collapse the whole capture+train to near-zero. **Do not generate a byte of training data before Gate 2.**
2. **Piggyback capture on runs we already do.** The head's INPUT is the target's layer-40/41/42 hidden
   states (`main_proj`); EAGLE-3 dropped the feature-regression *loss* but still *consumes* fused low/mid/high
   hidden states — so **taps must be captured, cannot be skipped**. BUT SpecForge shows reusing
   *inference-time* hidden states eliminates the dedicated capture prefill (their 6.16 hr prefill → 0, 1.67×).
   We run the full target for Gate 2 and for serving anyway → **harvest {taps, tokens, top-k logits} during
   those runs** instead of a separate capture pass.
3. **Small dataset via warm-start + "regenerated ≈ on-policy".** Warm-started heads converge on far less
   data. And the adversarial pass **REFUTED** that strict on-policy beats regenerated (4.63×→4.29×, within
   noise) — so we can **regenerate responses from an existing instruction corpus** (ShareGPT/UltraChat-style
   + our agentic/code/math) with the target at temp≈0.8 rather than pure from-scratch self-generation.
   Target ~5–20M tokens, not 50M+.
4. **Throughput-batched generation** for whatever dedicated regeneration we do: ~3–5× over single-stream on
   Thor. Batch many prompts; watch per-expert token count, not global batch.
5. **Lossless spec-decode bootstrap of the capture generation** — use the *unfine-tuned* DSpark head to
   spec-decode the target's own regeneration: measured 1.5–1.8× (Nemotron RL rollouts), up to ~50% (DAS),
   and provably on-policy. **CRITICAL: only with EXACT rejection sampling** — relaxed/typical acceptance
   (Medusa-style) is NOT distribution-preserving and would corrupt the captured distribution. Compose with (4).
6. **Cache-once, train-many.** SpecForge offline: precompute taps once, train the head many epochs on 1 GPU
   with the 96 GiB target NOT loaded. Disk cost is the tradeoff (~12 TB for full UltraChat+ShareGPT; scales
   with tokens). At 3 taps × 4096 × bf16 = 24 KB/token: 20M tokens ≈ **480 GB** (fp8 taps ≈ 240 GB). **User
   can free disk → cache-once viable; else stream from NVMe or shrink the set.** Respects Axiom 8 (target and
   trainer never co-resident — capture first, free the model, then train from cache).
7. **Training is NOT the bottleneck.** FastMTP: 210 M head, 3 epochs, 389 K samples, global batch 64,
   **<1 day on one H20**. Full-head fine-tune (not LoRA) is the disclosed recipe; warm-start + token-CE+KD.
   Thor is slower absolute but training stays hours-scale and is amortized by cache-once.

## The path
```
Gate 2 (measure unfine-tuned τ on REAP)  ── harvest taps/tokens/logits during this run
   │  τ decent? ─────────────► ship it, or a LIGHT regenerated fine-tune (small set)
   │  τ poor?  ─────────────► fuller self-distillation:
   ▼
Capture: batched (3–5×) + exact-rejection spec-decode (1.5–1.8×) regeneration @temp0.8 of a curated
         prompt set → cache {layer40-42 taps, target tokens, top-k(16-32) logits} to NVMe (cache-once)
   ▼  (free the 96 GiB target — Axiom 8)
Train:  head-only, warm-start, token-CE (decay β0.6) + logit-KD, LR 5e-5 cosine, 1–3 epochs from cache
   ▼
Eval (ABBA) → τ + tok/s vs the ~41–57 tok/s goal
```

## Revised wall-time
- If Gate 2 τ is already good → **hours** (light fine-tune) or **zero** (ship unfine-tuned).
- If full path, with piggyback capture + small set + batching + spec-decode + cache-once → capture drops
  from ~16–70 hr toward **~half a day to ~1.5 days**, train **hours**. Total plausibly **~1 day**, and the
  first real number comes from measuring Gate-1 batched throughput once the GPU is free.

## Sources (deep-research #2, all 3-0 unless noted)
MoE batching: MoE-Gen 2503.09716, Bench360 2511.16682, MoE-Lightning ASPLOS'25, MoE-Lens 2504.09345,
2411.08982. Lossless spec-decode capture: Nemotron RL-spec-decode, DAS 2511.13841 (exact-rejection caveat:
Draft-OPD 2605.29343). What-to-cache: EAGLE-3 2503.01840 (features consumed as input), SpecForge offline
(lmsys 2025-07-25, ~12TB), vLLM Speculators v0.3.0. Training cost: FastMTP 2509.18362 (<1 day/H20).
