# gemma-cuda-hybrid

**A from-scratch, pure-CUDA inference server for `google/gemma-4-26B-A4B-it` (NVFP4) with DFlash speculative decoding, hand-optimized for NVIDIA Jetson Thor (sm_110a) — ~118 tok/s, beating vLLM's 107 by +10% on the same model and hardware, in a single lean binary with a complete OpenAI-compatible API, reasoning + tool parsing, prefix caching, FP8 64K KV, and built-in web + terminal chat.**

No PyTorch. No Python runtime on the hot path. No framework. Every kernel is readable CUDA, tuned to this model's exact shapes, this GPU's exact architecture, and the NVFP4 numeric format.

---

## What it is

`gemma-4-26B-A4B-it` is Google's April-2026 Gemma-4 MoE: **25.2B total / 3.8B active** params (128 experts, top-8 + 1 shared), 256K context, 262,144 vocab, `<|turn>`/`<|channel>thought`/`<|tool_call>` chat grammar. This project serves it — quantized to **NVFP4** (4.25-bit weights) — as a local OpenAI endpoint on a **Jetson Thor** (Blackwell `sm_110a`, 20 SMs, 273 GB/s LPDDR5x, CUDA 13), using **DFlash** block-diffusion speculative decoding for a >2.5× decode speedup over autoregressive.

It began as a correctness-first pure-CUDA reimplementation (`gemma-cuda-server`, banked at DFlash 82 tok/s) and became a Marlin-class, research-driven, hand-tuned kernel ensemble that **exceeds vLLM's production stack on this exact configuration**.

## Headline numbers (measured, back-to-back on the same Thor)

| | This server | vLLM (gemma-4 DFlash) |
|---|---|---|
| Decode | **~118 tok/s** | 107 tok/s |
| Acceptance length τ | **13.33** | 9.21 |
| Runtime | single ~1.5 MB binary, no Python | Python + torch + CUDA stack |
| Bit-exact greedy | ✅ (gate-verified) | — |

Arc: base decode 6→45 tok/s; DFlash 5→**118**. Every step gated bit-exact against the reference. Theoretical ceiling (τ × vLLM step time) ≈ 157 tok/s.

## Everything it has

**Inference engine**
- NVFP4 **W4A16** decode (E2M1 weights + E4M3 group-16 scales + FP32 global), HW FP4/FP8 decode intrinsics.
- **Marlin-class raw `mma.sync.m16n8k16` tensor-core GEMM** for all M≤16 GEMMs (verify dense, both lm_heads, draft linears): in-register FP4→fp16 dequant, offline weight repack, 16-byte `int4` coalesced loads + `__ldcs` evict-first, max-grid-fill.
- **Grouped weight-resident MoE** (no atomics, U-unroll prefetch, HW-decode SwiGLU).
- **Head-packed GQA attention** (KV read once per kv-head, not per query-head).
- **DFlash** speculative decoding (5-layer qwen3-style block-diffusion draft, k=14 draft tokens/block, shared frozen embed+lm_head; draft stays bf16 — the acceptance moat).
- CUDA-graph capture of the verify step; the tied embed quantized to NVFP4 for a 4× lighter lm_head.

**Server (`SERVE=1 ./build/forward`)**
- **OpenAI-compatible** `POST /v1/chat/completions` (streaming SSE + non-streaming) + `GET /v1/models` + a web UI at `GET /`.
- **Lossless temperature sampling** (Gumbel-max target sampling + sample-match acceptance; temp=0 reduces to exact greedy).
- **Prefix caching** — LCP KV reuse across turns (the agentic win: a long system prompt is prefilled once, reused every turn).
- **FP8 (e4m3) KV cache**, configurable context (default **64K**, `CTX`/`FP8KV` env).
- **Reasoning delineation** — `ChanRouter` parses gemma-4's `<|channel>thought…<channel|>` into `reasoning_content` vs `content` (streaming deltas + non-stream field), triggered by `<|think|>` (`enable_thinking`). Hand-written equivalent of vLLM's `--reasoning-parser gemma4`.
- **Tool calling** — request `tools` → gemma `<|tool>` declarations; parse `<|tool_call>call:name{args}<tool_call|>` → OpenAI `tool_calls` with valid-JSON `arguments` + `finish_reason:"tool_calls"`. Equivalent of vLLM's `--tool-call-parser gemma4`.

**Clients**
- Self-contained single-file **WebUI** (no CDN/build/npm, works offline): streaming, live markdown (code blocks w/ copy, headers/lists/quotes), collapsible **Thinking** blocks, settings, tok/s, dark theme.
- Pure-C++ **terminal client** (`build/chat`): streaming multi-turn REPL, dimmed thinking, tok/s.

## Build & run
```bash
bash scripts/build.sh                 # -> build/forward (single binary)
DFLASH=1 GEN=80 ./build/forward       # benchmark the decode engine
SERVE=1 PORT=8080 ./build/forward     # OpenAI server + WebUI at http://localhost:8080
g++ -O2 -std=c++17 -I include server/chat.cpp -o build/chat -lpthread && ./build/chat  # terminal chat
```
Server env: `CTX=65536` (context, default 64K), `FP8KV=1` (FP8 KV, default on), `DK=14` (draft tokens). Per-request: `temperature`, `max_tokens`, `stream`, `enable_thinking`, `tools`.

---

## Advantages vs vLLM / SGLang (and the honest trade)

**Why it's faster here (+10%):** every kernel is tuned to *this* model's exact tensor shapes, *this* GPU's exact arch (`sm_110a`, its L2/smem sizes, its LPDDR5x roofline), and the NVFP4 format — with no framework dispatch, no Python per-step overhead, no graph-break surprises. Plus a measurably better draft (τ 13.33 vs 9.21).

**Why it's leaner:** one ~1.5 MB binary vs a multi-GB Python+torch+CUDA install; lower RAM, instant startup, deterministic, fully auditable (every kernel readable), edge-deployable.

**The honest trade — this is a *specialist*, not a *generalist*:** vLLM/SGLang serve *any* model on *any* CUDA GPU with batching, paged attention, multi-LoRA, tensor parallelism, and a huge feature surface. This serves *one* model on *one* chip, faster and leaner. It is single-instance (mutex-serialized), single-GPU, no continuous batching. **For a fleet of models/GPUs and multi-tenant scale, use vLLM/SGLang. For maximum single-stream decode speed and minimum footprint of a fixed model on a fixed edge device, a hand-tuned kernel ensemble like this wins.**

## Why it's NVIDIA-only, and specialized to this model + sm_110a + NVFP4

The speed comes from binding tightly to the hardware and format — which is also what makes it non-portable:

- **CUDA-specific:** `nvcc`/PTX `mma.sync` tensor-core instructions, `__nv_cvt_fp4x2/fp8` hardware-decode intrinsics, CUDA graphs, `cp.async`, `__ldcs` cache hints, warp `__shfl`. None exist outside NVIDIA's toolchain.
- **sm_110a-specific:** compiled `-arch=sm_110a`; the mma fragment layouts and the 228 KB-smem / 20-SM / ~200 GB/s-achievable tuning (WARPS=1 grid-fill, U-unroll prefetch depth) are calibrated to Thor's Blackwell die. (We *researched and rejected* `tcgen05` here: at M=15 the verify is 60× under the compute roof and tcgen05's MMA_M is locked to 128 — raw `mma.sync` is correct for this regime.)
- **NVFP4-specific:** the E2M1 + E4M3-block-scale + FP32-global dequant, the offline repack into mma-fragment order, the group-16 scale handling — all assume the NVFP4 layout.
- **Model-specific:** hard constants for gemma-4 (H=2816, 30 layers, 16 heads, sliding hd=256/nkv=8 + full hd=512/nkv=2 pattern, 128 experts top-8, VOCAB=262144, gemma double-norm, rope θ), the gemma-4 BPE tokenizer + `<|turn>`/`<|channel>`/`<|tool_call>` chat grammar, and the DFlash draft.

## Significance: a feature-complete CUDA OpenAI server for agentic use

Agent harnesses (the OpenAI SDK, LangChain, LlamaIndex, Claude-Code-style loops) speak the OpenAI API. A **drop-in local endpoint** with streaming + **reasoning separation** + **tool calling** means an agent runs **fully on-device, at max speed, no cloud, no Python-server tax** — critical for robotics/edge (Thor is a robotics chip). The reasoning/tool plumbing (separated `reasoning_content`, JSON `tool_calls`) is exactly what an agent loop needs: it can read the model's thinking, dispatch tools, feed results back (`<|tool_response>`), and iterate — all locally, in a single lean process.

---

## Adapting this to other setups

### Other NVIDIA GPUs (A100/H100/RTX 50xx/other Jetson)
- Change `-arch=sm_110a` (→ `sm_90a` Hopper, `sm_100`/`sm_120` other Blackwell, `sm_87` Orin). `mma.sync` shapes are portable across Ampere→Blackwell; **NVFP4 HW intrinsics need Blackwell** (Ampere/Hopper must emulate or use INT4/FP8). Retune `WARPS`, prefetch depth, and the FP8-KV/grid-fill choices to the target's SM count, L2/smem, and achievable-BW fraction. `tcgen05` is Blackwell-only.

### Other MoEs (Qwen3-MoE, Nemotron, DeepSeek, Mixtral)
- Update the model constants (H, layers, heads, hd, nkv, experts/top-k, sliding pattern, vocab, rope θ, norm structure) in `src/forward.cu`. Swap the **tokenizer** (`include/tokenizer.h` is gemma-4 BPE — different vocab/merges/normalizer) and the **chat template + parsers** (`chat_prompt`/`ChanRouter` encode gemma-4's `<|turn>`/`<|channel>`/`<|tool_call>` grammar; Qwen/Llama use `<|im_start|>`/`<think>`/different tool formats). The MoE router (top-k, shared-expert) and the grouped-GEMM tiling carry over. Provide a matching speculative draft (DFlash is gemma-specific).

### Dense models (Llama, Qwen dense, gemma dense)
- Skip the MoE path entirely (the dense `linear()`/tc GEMM already handles all projections). Everything else — NVFP4 dequant, tc GEMM, attention, tokenizer/server — is the same. Simpler than the MoE case.

### The adaptation discipline
Correctness first (a bit-exact gate vs a reference like vLLM), then **profile-driven** kernel tuning — see `AGENTIC_OPTIMIZATION_METHODOLOGY.md` and the `CUDA_ENGINEERING_CONSTITUTION.md` for the exact loop and the full won/lost/neutral ledger of what works on this hardware.

---

## The broader pattern: hardware/vendor speciation vs. portability

To extract peak performance from a device you **speciate to its vendor stack** — and every major stack is a near-clone of CUDA's SIMT model, because that model won:

- **NVIDIA → CUDA** (warps, shared memory, `mma`/tensor cores, `cp.async`).
- **AMD → HIP/ROCm** — deliberately CUDA-shaped (`hipify` is largely mechanical; `__shfl`→`__shfl`, `mma`→MFMA/WMMA, LDS≈smem). This project's kernels would port to HIP with intrinsic swaps + arch retuning.
- **Moore Threads → MUSA**, **others** — again SIMT clones with a CUDA-like API surface.

The convergence is real: threads/warps/blocks, a scratchpad (smem/LDS), matrix-core instructions, and async copy appear in every vendor's model. Porting a kernel *ensemble* between them is intrinsic-substitution + per-arch retuning — not a rewrite.

**The alternative is to generalize — at a measured cost in speed:**
- **Vulkan compute / OpenCL** run on *any* GPU (NVIDIA, AMD, Intel, mobile) but expose no vendor tensor-core intrinsics or NVFP4 paths → typically **1.5–3× slower** than hand-tuned native for LLM decode.
- **Triton / compiler IR** (what vLLM/SGLang lean on) gives one kernel that JITs per-arch — excellent productivity and good performance, but it still can't beat a hand-tuned per-shape/per-arch assembly-level kernel at the margin (and it carries the Python/JIT runtime).

**The spectrum:** hand-tuned vendor-native (this repo) → Triton/compiler → Vulkan/portable. Each step buys portability and generality; each step spends peak performance and lean footprint.

## Why a lean, hardware-optimized kernel ensemble is worth it

For a **fixed model on a fixed device** (the edge/robotics/appliance case), the specialist wins on the axes that matter there:
- **Performance maximization** — every kernel sits near the memory roofline for its exact shapes; no dispatch, no graph breaks, no per-step interpreter.
- **Efficient resource use** — a single small binary, minimal and predictable memory, instant startup, no multi-GB framework — leaving the device's scarce RAM/compute for the model and the KV cache.
- **Auditability & determinism** — every line is readable CUDA; greedy output is bit-exact and gated; no hidden JIT.
- **Full capability** — and, as this project shows, "lean and fast" need not mean "bare": it still ships streaming, sampling, prefix caching, 64K FP8 KV, reasoning separation, tool calling, and a rich UI.

The cost is engineering effort and non-generality. When you know the model and the chip and you want the most tokens per second per watt out of them, that trade is the right one — and this repo is a worked example of paying it end-to-end, from raw `mma.sync` PTX to an agent-ready OpenAI endpoint.

---

## Repository layout & docs
- `src/forward.cu` — model, engine (`forward_block`/`engine_prefill`/`engine_generate_dflash`), kernels, and the server (`run_server`).
- `src/draft.cu` — the DFlash draft model + propose.
- `kernels/` — `tc_verify_gemm.cu` (Marlin tc GEMM + bf16 tc), `fp4_gemm.cu`, `nvfp4_quant.cu`, `attention.cu`, `elementwise.cu`.
- `include/` — `tokenizer.h` (gemma-4 BPE + chat/thinking/tool grammar), `webui.h`, `safetensors.h`, `third_party/{httplib,json}.hpp`.
- `server/` — `chat.cpp` (terminal client), `tok_test.cpp` (tokenizer validation, 5/5 vs HF), `README.md`.
- **`CUDA_ENGINEERING_CONSTITUTION.md`** — ground-truth state, champion kernel stack, full won/lost/neutral ledger, transferable patterns, roadmap.
- **`AGENTIC_OPTIMIZATION_METHODOLOGY.md`** — the research-grounded optimization loop (map vs territory, profile-first, the black-swan budget).
- **`RESEARCH_FINDINGS.md`** / **`DEEP_RESEARCH_PROMPT.md`** — the deep-research surface (Marlin/FlashInfer/tcgen05/roofline; what was tried, won, and refuted with evidence).

Sibling repo `gemma-cuda-server` = the banked v1.0 baseline (DFlash 82, the stable NVFP4-lm_head reference).
