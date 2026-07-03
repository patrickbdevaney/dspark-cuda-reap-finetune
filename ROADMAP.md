# ROADMAP — full build state & path (capture → fine-tune → inference server)

**Durable progress record for `dspark-cuda-reap-finetune`.** If context is lost, start here. Read alongside
`CONSTITUTION.md` (rules), `ADAPTATION_PLAN.md` (arch delta), `CAPTURE_TRAIN_PLAN.md` (training research),
`reference/DEEPSEEK_V4_MODELING_NOTES.md` (numeric spec), `GATE_LOG.md` (finding+rationale per gate), `FP4_COMPUTE_NOTE.md` (Thor FP4 strategy), `DECODE_HORIZON.md` (grind to 38-50 tok/s), and the memory file
`~/.claude/projects/-home-patrickd/memory/dspark-v4flash-180b-thor.md`.

## HIGH-LEVEL OVERVIEW (user, turn ~55) — AGILE RE-SEQUENCE: ship the product first, then improve the head
**Rationale:** a fast, stable, memory-lean production server running the 180B + the EXISTING (unpruned) DSpark
head is a real, banked, shippable win — beats vLLM/SGLang on memory + startup + decode on Thor — *before* the
fine-tune. Improve the draft head afterward as a follow-on optimization.
1. **Optimize base decode + DSpark decode** — fastest kernels for the REAP target model AND the unpruned DSpark
   block-diffusion head. TC mma.sync GEMMs (champion #1 done, 19.7×) + batched MoE + fusion/graphs. Goal: decode
   "decent" (toward 38–50 tok/s) for base + unpruned-head spec-decode.
2. **OpenAI-compatible server — SEPARATE FOLDER in the repo** (e.g. `serve/`). Full **feature parity with
   gemma-cuda-hybrid's server, ADAPTED for DeepSeek-V4-Flash**: OpenAI API + **streaming**, **tool-calling schema**,
   **think-block / reasoning delineation** (DSpark chat format), **terminal + WebUI clients (UI/UX optimized)**,
   **configurable KV**, prefix cache, model+hardware-tuned kernels. Inherit the gemma server abstractions
   (`reference/GEMMA_ENGINE_README.md`, inherited `src/`+`include/{webui,tokenizer,httplib}.h`+`server/`) and
   re-target to this model's tokenizer/chat-template/tool-schema/reasoning-format. **This is the PRODUCTION
   DELIVERABLE + banked win**: as good as vLLM/SGLang but memory-lean, quick-to-start, faster decode on Thor.
3. **THEN capture + train the draft head** — minimized-wall-time optimal capture + training for a NEW, better draft
   head: higher τ acceptance + ultimate decode for REAP on this hardware. (Fastest kernels from (1) make capture
   feasible; representative on-policy coverage; block-acceptance objective.) Drops into the server as an upgrade.
Note: base-inference kernels, the server (`serve/`), and the capture/train kernels are SEPARATE artifacts, all
preserved in-repo. Step 2 gives a satisfying, stable, efficient product; step 3 makes it even faster.

## Mission
Fine-tune the DeepSeek "DSpark" spec-decode **draft head** onto `0xSero/DeepSeek-V4-Flash-180B-REAP`
(K160, NVFP4/FP8), and serve the pair at max decode — **all pure CUDA on one Jetson Thor** (`sm_110a`,
CUDA 13, 122.8 GiB unified). Every kernel hand-rolled and gated bit-exact vs a PyTorch oracle in `ref/`.
Three products: (1) end-to-end pure-CUDA draft-head fine-tune, (2) optimal `sm_110a` inference server,
(3) transferable reference. See `README.md` for the north star. **All CUDA is a preserved first-class repo
artifact — nothing disposable.**

---

## WHERE WE ARE NOW (turn ~40) — GATES 1, 1.5, 2 PASSED ✅ (project thesis proven end-to-end)
**Gate 2 GO: DSpark head unfine-tuned tau@0 = 0.815 (22/27) on REAP** — head transfers, light fine-tune should suffice.
**Gate 1.5: 'The capital of France is' -> ' Paris' (correct).  Gate 1: full 180B runs, mem 107.6/122.8.**
NEXT PHASE: **C capture** (user: review wall-clock optimality first) -> D training (pure-CUDA aspiration, Muon-vs-AdamW review) -> E server. See CAPTURE_TRAIN_PLAN.md.

### (historical) GATE 1 PASSED
**The full DeepSeek-V4-180B-REAP forward RUNS on Thor** (`src/forward.cu`, `build/forward`, `scripts/build_forward.sh`).
s=8 prefill: all 43 layers, memory FLAT 120.5/122.8 GiB, finite sane logits (argmax=1822 logit=16.4),
5494 ms (687 ms/tok, unoptimized). OOM fixed by per-layer dequant scoping (`Loader::mark/release`).
MEMORY: after `fadvise(DONTNEED)` fix (weight_store.h) = **107.6/122.8 GiB, ~15 GiB headroom** (page cache was the inflator, NOT the GUI; KV is <0.1GiB by MLA+SWA+DSA design). See MEMORY.md. Correctness: prompt 'The capital of France is' -> ' Paris' (Gate 1.5 PASS).
RUN: `./build/forward /home/patrickd/models/DeepSeek-V4-Flash-180B <s>`.

**NEXT (recursion):**
- **Gate 1.5 — correctness:** per-layer math already gated on real weights (block 0.23%, compressed cosine
  1.0, same dequant path as goldens). Validate END-TO-END: feed a real tokenized prompt (inherited
  `include/tokenizer.h`, `server/tok_test.cpp`; check model tokenizer files) → greedy-decode → sensible text?
  OR compare logits to a reference (full 96 GiB ref forward is infeasible on CPU — prefer tokenizer sanity +
  spot-checking dequant/freqs/head pieces). Risks are forward.cu-new: freqs indexing, embed/HC-init, head.
- **Speed:** 687 ms/tok is correctness-first (per-token host loops, warp-per-output GEMMs). Phase E optimizes.
- **Gate 2 — DSpark MTP head** (pivotal): implement `mtp.0.*` head, measure unfine-tuned τ on REAP.

### (historical) Weight-to-device loader path: SOLVED & PROVEN (`tools/load_device.cu`)
- Probed: `integrated=1, hostRegisterSupported=1, canUseHostPtrForRegMem=1, canMapHostMem=1`.
- `cudaHostRegister` of the file-backed `MAP_PRIVATE` mmap → "operation not supported" (Tegra limitation on
  that mapping type — NOT registration in general).
- **WORKING PATH (verified on shard-0, GPU read-back MATCH):** per shard `cudaHostAlloc(bytes, cudaHostAllocMapped)`
  → copy the shard data blob in → `cudaHostGetDevicePointer`. Integrated GPU ⇒ that buffer IS device memory,
  **single copy, no mmap+device doubling → no OOM.**
- **forward.cu TODO:** scale to all 46 shards; to stay single-copy at 96 GiB peak, either `pread` each shard's
  data region straight into the pinned buffer (no mmap), or `munmap` each shard right after copying it in.
  Then build the name→device-ptr map from tensor offsets within each shard's pinned buffer.

---

## WHAT'S BUILT & GATED (all bit-exact vs `ref/` oracle unless noted)

**Gate 0 (compat):** loader confirmed — `ShardedSafeTensors` mmaps 46 shards, 96.02 GiB, 1383 shape-checks.

**Gate K — 20 kernel primitives** (`./build/gate_units ref/goldens`, all PASS):
fp8_block_gemm · fp4_gemm · gemm_fp32 · hc_sinkhorn · moe_router_score · sparse_attn · rope_interleaved ·
rmsnorm · act_quant_{fp8,fp8sim,fp4sim,fp4} · ogroup_gemm · hadamard · index_score ·
compressor_pool(+overlap) · compressor_forward(+rotate) · yarn freqs (host).

**Composed forwards, validated on REAL 180B-REAP weights:**
- `mla_forward` — pure-sliding attention (layer 1): cosine 1.0, rms ~1e-4 (0.36% legacy metric).
- `compressed_attn_forward` — MLA + KV-compressor + DSA-indexer:
  - ratio-4 (indexer, layer 2): cosine 1.0, rms 1.6e-4.
  - ratio-128 (strided, layer 3): cosine 0.9999997, rms 7.5e-4.  *(metric lesson below)*
- `moe_forward` — hash + noaux_tc + SwiGLU FP4 experts + FP8 shared: bit-exact.
- `hc_pre/hc_post/hc_head` — Hyper-Connections: ~1e-6.
- **`block_forward`** — full transformer Block (layer 1) on **2.76 GB real weights: 0.23%**.

**Milestone commits:** M0 fork → M1 loader → M2.1 MLA → M2.2 MoE/HC/Block → M2.3 hadamard/index_score →
M2.4 compressor → M2.5 compressor-rotate → M2.6 indexer_forward → M2.7 compressed_attn (ratio-4) →
M2.8 yarn → M2.9 ratio-128 (+ metric resolution) → (now) forward.cu / load_device.

**METRIC LESSON (important):** per-element `max_rel = |diff|/(|ref|+0.01·mx)` is PATHOLOGICAL for deep
fp8/fp4 compositions — it grows with seq length on near-zero outputs regardless of correctness (proven:
accepted ratio-4 path rose 1.6%→4.1% max_rel from seq16→seq256 while cosine stayed 1.0000000). For deep
compositions use **rms_rel(relative-L2)<1e-2 AND cosine>0.9999 AND max_abs/|o|max<5e-3** (see gate_cmla.cu).

---

## REPO MAP
- `include/deepseek_v4.h` — all config constants + `compress_ratio(L)`, `is_hash_layer`, `has_indexer/compressor`.
- `include/safetensors.h` — mmap reader + `ShardedSafeTensors` (index.json, lazy per-shard, `shardRegions()`).
- `kernels/` — `fp8_block_gemm · fp4_gemm(moe) · mla_attn(rope/rmsnorm/act_quant/sparse_attn) · moe · hc ·
  hc_sinkhorn · mla_forward · block · compressor · indexer · compressed_attn`.
- `include/{...}.h` — matching headers; `mla_forward.h`(MLAWeights) `block.h`(BlockWeights)
  `moe.h`(MoEWeights) `compressed_attn.h`(CompressedAttnWeights) `yarn.h`(host freqs).
- `ref/` — pure-torch oracle: `kernel.py` `fast_hadamard_transform.py` `deepseek_v4_ref.py`(verbatim model.py,
  rotate_activation assert relaxed for fp32) `raw_loader.py` `gen_units.py`(all unit goldens) `gen_golden.py`
  (mla/block/cmla modes, `build_args` HF→ModelArgs mapper).
- `tests/` — `gate_units.cu`(20 prims) `gate_mla.cu` `gate_block.cu` `gate_cmla.cu`(ratio-4 & 128).
- `tools/` — `load_device.cu`(WIP weight→device) `inspect_weights.cpp`.
- `scripts/` — `build_gate.sh` `build_block.sh` `build_cmla.sh`.
- `src/` — inherited Gemma `forward.cu/megakernel.cu/draft.cu` (NOT yet re-targeted; reference for server).

## BUILD & GATE
```bash
# goldens (in torch container): mount ref/ + model, run gen_units.py / gen_golden.py {mla,block,cmla}
docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" -v $PWD/ref:/ref [-v <model>:/model:ro] \
  vllm-dflash-thor:sglang bash -lc 'python3 /ref/gen_units.py --out /ref/goldens'
bash scripts/build_gate.sh  && ./build/gate_units ref/goldens
bash scripts/build_block.sh && ./build/gate_block ref/goldens/block_layer1_seq16.safetensors
bash scripts/build_cmla.sh  && ./build/gate_cmla  ref/goldens/cmla_layer2_seq16.safetensors   # + layer3_seq256
```
Harness note: CUDA kernels compile+run on HOST (`nvcc -arch=sm_110a`). Goldens generated in CPU-torch
container `vllm-dflash-thor:sglang`. Never `--runtime nvidia` (wedges containers). Single-tenant weights
(Axiom 8): the 96 GiB model can't be co-loaded with another — load sequentially.

---

## REMAINING PATH

### Phase A — `forward.cu` → GATE 1  (current)
1. **Weight loader** — DONE: `include/weight_store.h` `WeightStore` loads all shards (pread→cudaHostAlloc-
   mapped, single-copy) and exposes `dev<T>(name)` device pointers for all 43843 tensors. `tools/load_device.cu`
   is the load+verify test.
   - **MoE INTEGRATION — RESOLVED (finding):** checked shard-4 header. Experts are **NOT byte-contiguous and
     NOT expert-ordered** (e0's w1/w2/w3 grouped consecutively, but e2 jumps to a different 1.2 GB region).
     So `MoEWeights`'s stacked-`[E,...]`+stride assumption (used by the pre-stacked block golden) **fails on
     the real checkpoint.** Per-expert layout: `layers.L.ffn.experts.{e}.w1.weight` = `[inter=2048, dim/2=2048]`
     dtype **I8** (FP4 packed, 4.19 MB each); `.w2.weight`=[dim, inter/2]; `.w3` like w1; scales `.scale`
     (F32). shared_experts `.{w1,w2,w3}.{weight,scale}`. gate `.weight` + `.tid2eid` (layers 0,1,2 are hash).
     **FIX — DONE:** `moe_forward` now takes optional per-expert device-pointer tables
     (`const uint8_t* w1[E], w2[E], w3[E]` + `const float* w1s[E]...`) instead of base+stride — moe.cu already
     loops per selected expert, so swap `w1 + e*stride` → `w1_ptrs[e]` (localized; keep the stacked version so
     gate_block/gate_units still pass). Model loader builds the 160-ptr tables per layer via WeightStore.
   - embed = `embed_tokens.weight` (bf16 lookup). `lm_head.weight` (bf16/fp8) + final `norm.weight`.
2. **Model assembly** — `compressed_block_forward` DONE (`kernels/compressed_block.cu`, compiles). Build
   per-layer `BlockWeights`(L0-1)/`CompressedBlockWeights`(L2-42) by name lookup from `WeightStore`.
   **EXACT CHECKPOINT NAMES (verified from index):**
   - attn: `layers.{L}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}` · `.q_norm.weight` · `.kv_norm.weight`
     · `.attn_sink`. compressor: `.compressor.{wkv,wgate}.weight` · `.compressor.ape` · `.compressor.norm.weight`.
     indexer: `.indexer.wq_b.{weight,scale}` · `.indexer.weights_proj.weight` · `.indexer.compressor.{wkv,wgate}.weight`
     · `.indexer.compressor.ape` · `.indexer.compressor.norm.weight`.
   - block: `layers.{L}.{attn_norm,ffn_norm}.weight` · `layers.{L}.hc_{attn,ffn}_{fn,scale,base}`.
   - ffn: `layers.{L}.ffn.gate.{weight,tid2eid}` · `.experts.{e}.{w1,w2,w3}.{weight,scale}` ·
     `.shared_experts.{w1,w2,w3}.{weight,scale}`.
   - top level: `embed.weight` (bf16), `norm.weight` (final), `lm_head`? (**CHECK**: not in grep above — the
     model may TIE lm_head to `embed.weight`; verify in model.py `ModelArgs`/forward).
   - **`wo_a` IS fp8 (`.scale` present)** but `mla_forward`/`ogroup_gemm` expect **fp32** (goldens did
     `.float()`). LOADER MUST DEQUANT `wo_a` fp8→fp32 per layer (~134 MB/layer fp32, ~5.8 GB total) OR add an
     fp8 path to `ogroup_gemm`. Same-class check for any other tensor a kernel assumes fp32.
   - YaRN freqs via `yarn.h`: per layer type (compressed base=compress_rope_theta orig=65536; sliding
     base=rope_theta orig=0). query freqs = `freqs[:s]`, compressed = `freqs[:s:ratio]`.
3. **Output head (model.py:771-826, `ParallelHead`):** after 43 blocks `h` is `[s, hc, dim]`; final =
   **hc_head collapse (hc 4→1)** with top-level `hc_head_{fn,scale,base}` → final `norm.weight` (RMSNorm) →
   `lm_head` linear → logits `[s, vocab]`. lm_head weight is **bf16 in ckpt, dequant to fp32** (model.py:730
   comment). embed = `embed.weight` (bf16 lookup kernel). Grep top-level names: `embed.weight`, `norm.weight`,
   `hc_head_fn/scale/base`, and the lm_head/`head.weight` tensor (confirm exact name). Need a small `hc_head`
   kernel (already in `hc.h`? `hc_head` exists) + bf16 embed-lookup + a bf16/fp32 lm_head GEMM.
4. **DSpark head names for Phase B:** `mtp.0.{enorm,hnorm,attn_norm,ffn_norm,norm}.weight`,
   `mtp.0.attn.{q_norm,kv_norm}.weight`, + mtp attn/ffn weights; `mtp[-1].head` shares the main output head.

**forward.cu is now FULLY SPECIFIED** — loader (done) + per-layer struct population by the exact names above
(with wo_a fp8→fp32 dequant + per-expert ptr tables) + YaRN freqs + 43-layer loop (block_forward L0-1 /
compressed_block_forward L2-42) + hc_head/norm/lm_head → **Gate 1** (first full 180B run: memory, KV, tok/s).
2. **Model assembly:** `embed(input_ids)` → HC-expand → 43 layers → final `rmsnorm` → `lm_head` → logits.
   - Layers 0–1: pure-sliding block (`block_forward` uses `mla_forward`).
   - Layers 2–42: compressed block — need a `compressed_block_forward` = block wrapping
     `compressed_attn_forward` (even L → ratio-4 indexer, odd L → ratio-128 strided). Wire HC + norms + MoE
     around it (mirror `block_forward`).
   - YaRN freqs via `yarn.h`: query freqs (per layer type) + compressed freqs (stride-ratio).
   - Hash layers (first 3): MoE uses `tid2eid` hash routing (already in `moe_forward`).
3. **GATE 1:** run a real prefill+decode → measure resident memory (target ≤122 GiB), real KV capacity,
   **decode tok/s** (baseline single-stream ~18.9). First time the whole 180B runs on Thor.
   - Optional full-model golden: dump ref logits for a short prompt, compare (or spot-check top-k).

### REORDERED PLAN (user directive, turn ~42) + strategy corrections
**Order:** (a) build the REAL DSpark block-diffusion head + true block-τ  →  (b) **optimize decode kernels
until no more levers** (per **OPTIMIZATION_METHODOLOGY.md**: champion-survivor A/B evolution → stop at
diminishing returns → literature-driven black-swan saltatory leaps; co-optimize target+head SYSTEM tok/s) (Marlin-class mma.sync GEMMs + gemma-cuda-hybrid techniques + our opts, in concert
with the draft head; target **38–50 tok/s**) — PREREQUISITE that makes capture feasible (current correctness-
first kernels ~1 tok/s → capture would take years)  →  (c) SOTA **representative** capture set + max block-
acceptance training.
**STRATEGY CORRECTION (user):** the head predicts *what the base model would say* — CANNOT domain-specialize
beyond faithful mimicry. Acceptance varies by domain only because the target's own output ENTROPY varies (code
low-entropy → long accepted blocks; open reasoning high-entropy → short). Capture = **representative on-policy
coverage** of the served workload, NOT per-domain skill tuning. Lever = faithful mimicry (token-CE + top-k
logit-KD) + coverage + block/Markov structure. Optimize **E[accepted block length]**, measured per-domain.
**Wall-time (optimized):** capture ~½–1 day dedicated (≪ if harvested during serving — SpecForge taps-during-
inference drops prefill 6.16hr→0); training hours (head-only ~2.5GB, 3 epochs). Total ~1–2 days, capture-
dominated. Warm-start → ~5–20M tokens (τ already ~0.8-class).

### THE REAL DSpark head (`~/models/DeepSeek-V4-Flash-DSpark-head`, 48 shards) — SPEC for build (a)
NOT the REAP built-in `mtp.0` MTPBlock (plain next-token; my `dspark.cu` = that stepping stone, proxy τ@0=0.815).
REAL head (modeling-notes §10; READ the DSpark-head repo's own `inference/model.py` — differs from REAP one):
block-diffusion MTP, shares main embed+head, `compress_ratio=0` (pure sliding).
- **Tap:** main model appends `h.mean(dim=2)` (HC 4→mean) at layers **40,41,42** → `main_hidden=[b,s,3d]`.
- **forward_embed:** `main_x=main_norm(main_proj(main_hidden))` (fp8 3d→d); draft input `[ids, noise×4]`
  (block_size=5, rest=`noise_token_id 128799`) → embed → 4 HC copies.
- **DSparkAttention:** start_pos==0 builds sliding-128 KV from main_x; decode: q/kv over block, sparse_attn
  over `[sliding main-KV ⊕ block]`.
- **forward_head:** hc_head→norm→head → per-block **AR** sample of 5; each step adds **Markov bias**
  `markov_head(prev_id)` (bigram embed rank-256→head); a **confidence head** scores each draft token from
  `[hidden ⊕ markov_embed]`. Returns (output_ids[6], logits, confidence).
- **Weights:** `mtp.0.{main_proj(+scale),main_norm}`+full block; `markov_head.markov_w1/w2` (on mtp.2 — CONFIRM
  3-block wiring). **Verify/accept loop NOT in reference** — build in harness (draft output_ids vs target,
  accept longest matching prefix, confidence early-exit). Loader: WeightStore reads raw shards; mp=1.
- **LOGISTICS (verified):** DSpark-head repo = ~11 GB (shards 46-48 = the mtp.* head: 3 MTP blocks,
  256-expert MoE, markov_head, main_proj, confidence). Extract ONLY mtp.* and run on REAP taps 40/41/42 →
  fits (96 REAP + ~3 GB head). Head keeps 256 experts (its own capacity); fine-tune adapts to REAP distribution.

### Phase B(proxy, DONE) — REAP built-in MTPBlock → measured τ@0=0.815
**Architecture (model.py MTPBlock:756-783), fully mapped:** the head is a `Block` subclass, layer_id=43 →
`compress_ratio(43)=0` → **PURE-SLIDING block → reuses `block_forward`** (+ `mtp.0.*` weights), wrapped:
```
e  = enorm( embed(input_ids) )                 # enorm: RMSNorm bf16;  embed shared (bf16 lookup)
xh = hnorm( x )                                # x = main model's final [s,hc,d] HC state; hnorm bf16
x' = e_proj(e).unsqueeze(hc) + h_proj(xh)      # e_proj,h_proj: FP8 [4096,4096]+scale (act_quant+fp8_block_gemm)
x' = block_forward(x', mtp_block_weights, s)   # full pure-sliding Block (attn+MoE+HC), mtp.0.* weights
logits = hc_head(x', mtp.hc_head_{fn,scale,base}) -> mtp.norm(bf16) -> lm_head(head.weight, shared)
```
**mtp.0.* weights (999 tensors):** full block set (`attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}`,
`attn.{q_norm,kv_norm}`, `attn_sink`, `{attn_norm,ffn_norm}`, `hc_{attn,ffn}_{fn,scale,base}`,
`ffn.gate.{weight,bias}` [NOT hash — layer 43], `ffn.experts.{0..159}.{w1,w2,w3}.{weight,scale}`,
`ffn.shared_experts.*`) PLUS head-specific: `e_proj.{weight,scale}` `h_proj.{weight,scale}` (FP8 4096×4096)
`enorm.weight` `hnorm.weight` `norm.weight` (bf16) `hc_head_{fn,scale,base}` (f32). +~2.5 GiB resident.
**BUILD:** `include/dspark.h` (DSparkWeights = BlockWeights + e_proj/h_proj+scales + enorm/hnorm/norm +
hc_head_*) + `kernels/dspark.cu` `dspark_head_forward(logits, x[s,hc,d], input_ids, W, head_w, s)`. In
forward.cu, after the 43 layers, tap the `[s,hc,d]` state (before the main hc_head) → run dspark head.
**GATE 2 (τ):** run main model → hidden states h_t + greedy tokens x_{t+1}; feed (h_t, x_{t+1}) to the head →
predict x_{t+2}; **single-token acceptance = fraction matching the target's greedy next token** (first proxy;
full block-diffusion DSPARK_BLOCK=5 acceptance later). τ decides light-finetune sufficiency. Validate the
head forward numerically first (a real-weights golden like block/cmla, or the Paris-style sanity: does the
head's greedy prediction track the target's?).

### (original Phase B note)
- Implement the DSpark block-diffusion MTP head forward (it's an MLA + 256-expert top-6 MoE MTP layer;
  REAP built-in mtp has 160). Reference: model.py MTP module + `src/draft.cu` (inherited DFlash draft, adapt).
- **GATE 2:** measure **unfine-tuned acceptance τ** of the existing head on the REAP target. This decides
  whether a light fine-tune suffices (head exists for unpruned → small step to REAP) or more is needed.
  Estimated served target ~41–57 tok/s if τ good.

### Phase C — CAPTURE  (after Gate 2; user: review wall-clock optimality first)
- On-policy self-distillation: the REAP+NVFP4 target generates responses; cache what the head needs.
- Research done (`CAPTURE_TRAIN_PLAN.md`): tokens + top-k target logits (k≈16–32), likely SKIP hidden
  states (EAGLE-3), hybrid regenerated-vs-static data, lossless spec-decode bootstrap to speed capture,
  continuous batching for aggregate tok/s. **User directive: re-review that capture is TRULY wall-clock-
  optimal + the correct CUDA solution before running.**
- Benchmark full wall-time before the real run.

### Phase D — TRAINING  (user: pure CUDA if viable; first-principles optimal setup)
- Cache-once-train-many: precompute target outputs once, train only the ~3% head, 1–3 warm-started epochs.
- **User directive:** aspire to do training in **pure CUDA** (as wall-clock optimization + a CUDA reference);
  at that time review the OPTIMAL setup from first principles — optimizer (**Muon vs AdamW**, July 2026),
  precision, data pipeline for the small MoE head (only active experts get gradients). Token-CE (decay β0.6,
  K=3) + logit-KD, LR ~5e-5 cosine, warm-start.

### Phase E — INFERENCE SERVER  (when head is trained)
- Take the inherited `gemma-cuda-hybrid` abstractions (`src/`, NVFP4 decode, mma.sync GEMM, KV mgmt,
  OpenAI API, prefix cache — see `reference/GEMMA_ENGINE_README.md`) and tune to `sm_110a` × this model's
  attention/arch × the DSpark head to maximize spec-decode decode. Production-grade.

---

## KEY DECISIONS / GOTCHAS (hard-won)
- Pivoted OFF SGLang (absent on Thor) and vLLM (unified-mem double-copy OOM risk) → pure CUDA.
- Portability: only the **weights + trained head** are portable (run in vLLM/DGX-Spark/B200); our CUDA is
  Thor-tuned but PRESERVED as the canonical reference. Unpruned↔REAP = one constant `N_ROUTED` (160↔256) +
  different checkpoint; per-token compute path is identical (top-6 + shared expert) so kernels & decode
  speed are invariant — only memory + a few shapes change (see README).
- **hc_post transposition bug** (model.py:692 sums comb's FIRST index) — caught only by real-weights Block
  gate, not synthetic (self-consistent goldens hide shared bugs). Lesson: real-weights integration gates.
- MoE args silently wrong from HF field-name filter → added explicit `build_args` `_HF2MA` map. MoE act
  scale must be `ue8m0` (pow2), not None.
- Reference `rotate_activation` asserts bf16 → relaxed to allow fp32 in harness copy for fp32 goldens.
- `cudaHostRegister` unsupported on Jetson for file mmap (current blocker; see resolution above).
