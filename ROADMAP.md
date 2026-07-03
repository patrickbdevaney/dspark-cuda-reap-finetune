# ROADMAP — full build state & path (capture → fine-tune → inference server)

**Durable progress record for `dspark-cuda-reap-finetune`.** If context is lost, start here. Read alongside
`CONSTITUTION.md` (rules), `ADAPTATION_PLAN.md` (arch delta), `CAPTURE_TRAIN_PLAN.md` (training research),
`reference/DEEPSEEK_V4_MODELING_NOTES.md` (numeric spec), and the memory file
`~/.claude/projects/-home-patrickd/memory/dspark-v4flash-180b-thor.md`.

## Mission
Fine-tune the DeepSeek "DSpark" spec-decode **draft head** onto `0xSero/DeepSeek-V4-Flash-180B-REAP`
(K160, NVFP4/FP8), and serve the pair at max decode — **all pure CUDA on one Jetson Thor** (`sm_110a`,
CUDA 13, 122.8 GiB unified). Every kernel hand-rolled and gated bit-exact vs a PyTorch oracle in `ref/`.
Three products: (1) end-to-end pure-CUDA draft-head fine-tune, (2) optimal `sm_110a` inference server,
(3) transferable reference. See `README.md` for the north star. **All CUDA is a preserved first-class repo
artifact — nothing disposable.**

---

## WHERE WE ARE NOW (turn ~28)
Building **`forward.cu`** (the full-model loop → Gate 1). All kernels + both attention variants + full
Block are validated on real weights. **Weight-to-device loader path: SOLVED & PROVEN** (`tools/load_device.cu`).
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
1. **Weight loader** (blocker above): single-copy into GPU-accessible memory, build name→device-ptr map,
   populate per-layer weight structs.
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

### Phase B — DSpark MTP draft head → GATE 2  (pivotal go/no-go)
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
