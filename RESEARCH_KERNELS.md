# RESEARCH_KERNELS.md — decode-kernel literature scan (deep-research wf_b54e3f4d-169)

**Preserved verbatim — this scan cost ~1.6M subagent tokens (105 agents, 23 sources, 111 claims, 24 confirmed
/ 1 refuted).** Question: fastest CUDA kernels for low-batch decode of a large quantized MoE (fp8/fp4) on
Blackwell sm_110a / Jetson Thor toward 38–50 tok/s. Actionable synthesis + grind order live in
`OPTIMIZATION_LEDGER.md`; this is the full evidence base. Claims are adversarially-verified (2/3 vote) unless noted.

**TOP REFRAMING:** low-batch decode is **memory-BANDWIDTH-bound** (weight loading dominates, not compute/routing)
— explains our own A/B (M=1 → 2.71×, M=8 → 19.7×). Levers: batch to amortize loads + cut weight/activation traffic.

## 6. Other

- DeepSeek Sparse Attention (DSA) reduces core attention complexity from O(L^2) to O(Lk) via a Lightning Indexer (FP8 scorer) plus top-k token selection, operating up to 128K context.
- SGLang's DSA implementation uses FlashMLA (DeepSeek's optimized MLA/multi-query attention kernel) and a FlashAttention-3 sparse variant adapted for kernel reuse.
- The sparse attention backend uses different page sizes within one backend: page size 64 for the indexer and page size 1 for the sparse forward, with a dedicated key&key_scale cache for fast token scoring.
- TileLang kernels are cited as a useful path for flexible attention-kernel development in this stack.
- FP8 KV cache is a planned optimization claimed to roughly double the number of tokens that fit in KV cache.
- FlashMLA's dense MLA decode kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in compute-bound configuration on H800 SXM5 (BF16), showing MLA decode has both a bandwidth-bound and compute-bound regime.
- FlashMLA supports SM90/SM100 architectures (Hopper and Blackwell B200), requiring CUDA 12.8+ (CUDA 12.9+ for SM100 kernels) — it does not list Jetson Thor sm_110a support.
- FlashMLA provides token-level sparse MLA attention with configurable top-k selection, achieving up to 350 TFLOPS on B200 for FP8 sparse decode.
- FlashMLA's sparse decode uses an FP8-with-scale KV cache format with bfloat16 computation, and supports paged KV cache with block tables.
- FlashMLA sparse MLA prefill reaches up to 1450 TFLOPS on B200 with CUDA 12.9, indicating strong Blackwell-generation MLA performance headroom.
- A grouped GEMM executes all per-expert GEMMs in a single kernel launch on-device, taking a list of tokens and experts as input, eliminating the host-side per-expert launch loop that dominates MoE small-GEMM cost.
- Naively looping and launching one GEMM per expert is dominated by kernel-launch overhead because MoE expert GEMMs are small (few tokens, small expert matrices).
- Grouped GEMM differs from batched GEMM because experts receive differing numbers of tokens (ragged/variable-size inputs), so it cannot be expressed as a uniform batched GEMM.
- Reference grouped-GEMM implementations for MoE dispatch exist in both TritonLang and NVIDIA CUTLASS.
- Router assignment is a lightweight linear layer plus softmax after attention that selects experts per token, and tokens are then gathered per expert via an all-to-all across devices.
- cuDNN's DSA indexer top-k selection uses a radix top-K kernel to select candidate KV indices from indexer scores, supporting top_k up to 2048 on SM100+ (Blackwell) GPUs.
- The DSA lightning indexer forward computes dense indexer scores as S[b,q,k] = sum_h ReLU(Q_h . K_h^T) . W_h, and requires SM100+ with head_dim==128 and qhead_per_kv_head in {32,64}.
- The production sparse-attention forward kernel is FlashMLA (C++) and is NOT integrated into cuDNN; cuDNN only packages the indexer, top-k, score-recompute, and backward operations.
- cuDNN DSA/FlashMLA supports MLA shapes with head_dim in {512, 576}, and this shape is supported on SM90 as well as SM100.
- The Indexer Forward and Indexer Top-K kernels are restricted to SM100+ only, while backward/score-recompute kernels run on SM90 or SM100.
- For MLA decode, the kernel uses 128 query heads and 1 KV head (MQA-like), with head_dim_k=576 and head_dim_v=512, so a single key head serves all query heads during decode.
- The FP8 sparse decode kernel is dequantization-bound, not MMA-bound: dequantizing one token's KVCache costs ~50 cycles versus only ~34 cycles for the MMA operations per K/V token.
- FP8 KV quantization uses tile-level quantization (tile size 1x128) on the first 512 elements, storing 512 float8_e4m3 values plus 4 float32 scale factors and 64 unquantized bf16 RoPE elements, for 656 bytes per token.
- A CTA-cluster 'crossover' technique using Hopper Distributed Shared Memory exploits the MQA property (all query heads attend the same key heads) to cooperatively load and cut dequantization work by 50%.
- The FP8 sparse kernel reaches 410 TFLOPS (batch=128, heads=128, s_q=2, topk=2048) versus 250 TFLOPS for the previous FP8 sparse kernel, with sparse execution time comparable to the dense bf16 kernel around sequence length ~3000.
- DeepSeek Sparse Attention (DSA) selects the top-2048 tokens per query for the sparse attention computation, materializing a logits tensor then running a row-wise top-k to produce a (2048,) integer index tensor.
- The sparse-attention indexer logits are computed with an FP8 MQA-logits GEMM via DeepGEMM (deep_gemm.fp8_mqa_logits with q_fp8, kv_fp8, weights, ks, ke).
- The MLA FP8 KV cache lays out each entry as 512 float8_e4m3 values for the quantized NoPE part plus a 128-byte RoPE part of 64 bfloat16 values left unquantized for accuracy.
- The sparse MLA path uses the FlashMLA sparse attention kernel which requires a block size of 64, and the indexer K cache is allocated in a separate buffer from the MLA K cache.
- A fused top-k kernel is called out as an optimization priority because the naive approach materializes the full logits tensor before the row-wise top-k.
- The library provides device-side permute/unpermute operations that reorganize token-to-expert assignments into contiguous per-expert groups, returning a permuted activation tensor and a row_id_map, enabling grouped GEMM without a host per-token loop.
- The grouped GEMM supports FP8 precision (in addition to FP32/FP16/BF16) across NVIDIA architectures from SM70 through SM90, but does not document FP4/W4A8 support.
- The repository provides no benchmark or performance numbers quantifying grouped-GEMM speedup, expert count per kernel, or low-batch decode behavior, so its concrete speedup for single-stream decode is unverifiable from this source.
- A block-scheduled grouped-GEMM Triton MoE kernel with a precomputed mapping from program blocks to (expert, token_offset) pairs replaces per-expert kernel launches — reducing Mixtral (8 experts, top-2) from 24 separate cuBLAS calls to 5 Triton launches, giving up to 9.1x speedup over PyTorch reference at batch=1.
- At small/low-batch decode (1–128 tokens), the fused MoE kernel is memory-bandwidth-bound: cost is dominated by weight loading rather than expert routing.
- Fusing the gate and up projections so both share a single input tile and keep the intermediate in registers saves ~470 MB of memory traffic per forward pass, ~35% reduction in global memory traffic, yielding 1.16–1.40x speedup over the unfused approach on DeepSeek-V3 (256 experts, top-8).
- For the block-scheduled grouped GEMM to be correct, BLOCK_M must be fixed rather than autotuned, otherwise the precomputed block-to-expert schedule disagrees with the kernel tiling.
- The fused Triton MoE dispatch kernel beats Megablocks by 131% at 32 tokens (2.13ms vs 2.78ms) and 124% at 128 tokens, but falls to 89% of Megablocks performance at 512 tokens — i.e., the Triton win is specific to low batch.
- For NVFP4 grouped GEMM on Blackwell (SM100), warp specialization (dedicated producer/consumer warps in a 4-stage pipeline) delivered a ~60% reduction in kernel time (v2→v3), and cluster multicast a further ~63% (v3→v4), yielding a final ~10x speedup over reference (238 us → 23.8 us).
- NVFP4 tensor-core MMA on Blackwell uses the tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 instruction with E2M1 weights and per-16-element block scale factors, applied as x_fp = x_nvfp4 * scale.
- The accumulator plus block-scale factors fill the 512-column TMEM partition so only one CTA fits per SM, capping theoretical occupancy at 12.5% (12.4% achieved), making the kernel latency/barrier-bound rather than DRAM-bound at these shapes (DRAM throughput only 27% of peak, tensor-core utilization 50%, dominant stall barrier wait).
- Cluster multicast (__cluster_dims__(2,1,1)) sharing the A tile across CTAs cuts matrix-A read traffic by up to 4x; the optimal choice of which operand to keep hot in L2 depends on problem shape (keep the smaller reusable operand hot, stream the larger).
- For MoE grouped GEMM, a specialized low-M epilogue (M <= 96), host-side expert sorting by descending M, and persistent scheduling via a global atomic counter (replacing static blockIdx.x) were key wins; the fastest known submission reached 16.029 us using cooperative cta_group::2 MMA across CTAs.
- GVR, an exact bit-exact Top-K algorithm for DeepSeek Sparse Attention decoding on NVIDIA Blackwell, achieves an average 1.88x single-operator speedup over the production radix-select Top-K kernel, with up to 2.42x per layer per step and no loss in output accuracy.
- In DSA decode, the Top-K selector's memory traffic (R·N·4B) grows linearly with sequence length N while sparse MLA stays constant at K·d·2B (K=2048 fixed), so the Top-K fraction of total DSA decode latency increases monotonically and becomes the major bottleneck at long context; all three decode components are memory-bound.
- GVR integrated into TensorRT-LLM improves end-to-end time-per-output-token (TPOT) by up to 7.52% at 100K context in TEP8 min-latency deployment, with larger gains at longer contexts and smaller but still positive gains under speculative decoding.
- The production TensorRT-LLM radix-select Top-K baseline (ported to Blackwell sm_100) already achieves 7.4x speedup over torch.topk, using a half->11->11->10-bit radix decomposition (4 passes) with a split-CTA path across 10 CTAs for sequences over 200K.
- Decode-step Top-K index sets exhibit strong temporal correlation across consecutive steps in DeepSeek-V3.2: layers 20-60 show 35-50% average raw overlap (max ~60%), while layers 0-1 show near-zero (~1-2%) overlap, enabling the previous step's Top-K to serve as a prediction signal.
- MPK compiles multi-GPU LLM inference into a single megakernel that performs all computation and communication within one kernel launch, eliminating per-operator kernel launch overhead.
- MPK reduces LLM inference latency by 1.2x to 6.7x versus baseline approaches.
- MPK provides fused layer primitives such as rmsnorm_linear_layer that combine RMSNorm and Linear into the megakernel, matching the fuse-RMSNorm+GEMM decode-step fusion goal.
- MPK was released June 2025 with an accompanying paper at OSDI 2026, but the public repo documentation gives no tokens/sec, per-token latency, or single-GPU low-batch decode numbers.
- A fused MoE dispatch pipeline can reduce kernel launches from 3E+4 (naive per-expert: 24 for Mixtral-8 experts, 768 for DeepSeek-V3 256 experts) down to a fixed 5 Triton launches regardless of expert count, via a block-scheduled grouped GEMM that maps program blocks to (expert_id, token_offset) pairs in a single launch without padding waste.
- At low batch (256 experts, top-8, ~512 tokens so ~2 tokens per expert), the expert FFN becomes memory-bound rather than compute-bound because per-expert GEMM tiles (2,2048)x(2048,7168) are too small to fill tensor cores and weight loading dominates; dispatch-level kernel fusion is insufficient and fundamentally different strategies (expert parallelism, weight caching) are required.
- Fusing the SwiGLU gate and up projections into one kernel that shares L2-cached input tiles with in-register SiLU eliminates the gate_out and up_out global-memory buffers, cutting expert-FFN memory traffic by ~35% (6TF+2Td bytes saved) and yielding a 1.15x end-to-end speedup on Mixtral-8x7B at 512 tokens.
- At inference-relevant small batch sizes (<=128 tokens), a portable Triton fused MoE dispatch is 1.18-1.31x faster than the CUDA-optimized Megablocks (attributed to lower launch overhead of 5 launches vs multi-stage dispatch), but Megablocks' hand-tuned block-sparse CUDA kernels win at 2048+ tokens by better saturating tensor cores.
- A fixed BLOCK_M grouped-GEMM schedule (chosen at compile time) underperforms Megablocks' dynamic block-sparse layout under extreme routing skew at 64+ experts: on Qwen2-MoE-57B at 128 tokens, going from uniform to Zipfian alpha=2.0 drops the speedup from 1.03x to 0.70x because Megablocks consolidates the dominant expert's tokens into one large sparse block.
- DFlash speculative decoding increases decode throughput by more than 15x versus autoregressive decoding, and 1.5x over EAGLE-3, for gpt-oss-120b (a MoE model) on Blackwell.
- At batch size 1 (single-stream/low-batch decode), DFlash more than doubles interactivity on Blackwell.
- The DFlash drafter predicts a block of masked future tokens in a single forward pass rather than generating them sequentially, and uses target hidden-state conditioning and KV injection to raise acceptance.
- On a single Blackwell Ultra GPU, DFlash delivers up to 5.8x higher throughput for Gemma 4 31B (Math500) via vLLM.
- DFlash integrates into SGLang, vLLM (via the open-source Speculators library), and TensorRT-LLM with no code changes required.
- MoMoE eliminates Python per-token/per-expert for-loops by packing token indices per expert into fused Triton kernels using cumulative-sum offset tables, avoiding padding waste.
- MoMoE fuses scatter/gather directly into its Triton kernels instead of using grouped GEMMs, and at low sparsity/high density where compute is GEMM-dominated a CUTLASS grouped-GEMM approach (TEGrouped/Megatron) is expected to outperform MoMoE.
- MoMoE's advantage comes from effective sparsity and is strongest at small context lengths / high sparsity, converging to PyTorch performance asymptotically at large sequence lengths.
- MoMoE is the fastest MoE implementation tested for K (experts-per-token) less than or equal to 16.
- MoMoE implements no quantization scheme, operating in BF16/FP32 rather than W4A8/W8A8 low-precision formats.
- Fusing an entire model forward pass into a single megakernel (via a pipelined on-SM instruction interpreter) eliminates kernel-launch boundaries and their memory-pipeline bubbles, which is directly the megakernel/persistent-kernel fusion technique for killing launch overhead in a multi-layer decode.
- The megakernel achieves per-user (low-batch, Llama-1B context) throughput around 50% higher than SGLang and vLLM, evidencing that megakernel fusion helps latency-bound single-stream/low-batch decode, not just high-batch throughput.
- Inter-instruction (persistent-kernel) pipelining inside the megakernel yields a measurable 6.1% throughput gain by overlapping memory pipeline bubbles across fused instructions.
- The megakernel fuses model operations into 9 combined instructions (e.g. RMSNorm+all-gather, QKV matmul+RoPE, attention+distributed transpose, O-projection+residual, MLP), demonstrating concrete op-fusion boundaries for a decode step.
- A global work queue and instruction interleaving each provide additional gains (14.2% and 6.4% respectively at batch 8,192), showing scheduling/load-balancing wins within a persistent megakernel.
- Fusing the entire single-batch decode forward pass into one persistent megakernel (eliminating the ~100 separate kernel launches) reaches 78% of H100 memory bandwidth and beats existing engines (vLLM/SGLang) by 1.5x-3.5x for Llama-1B single-stream decode.
- Standard per-kernel launch/teardown overhead is the bottleneck for low-batch decode; conventional engines hit at most 50% of GPU bandwidth because a forward pass is decomposed into ~100 kernels whose setup stalls weight loading.
- CUDA graphs reduce but do not eliminate launch overhead — dummy-kernel launch cost drops from 2.1us (standard stream) to 1.3us (CUDA graphs), still leaving 'unnecessary stalls' that make graphs suboptimal for low-latency decode versus a megakernel.
- The megakernel is structured as an on-SM instruction interpreter running pre-scheduled fused instruction types (e.g. RMSnorm+QKV+RoPE, attention, O-proj+residual, RMSnorm+up-gate+SiLU, down-proj+residual), with a global-memory counter array for on-GPU synchronization and shared-memory paging to overlap loads across instruction boundaries.
- Even inside the optimal megakernel, single-batch decode is memory-bandwidth-bound: on B200 a forward pass is ~680us (implicit ~1,470 passes/s) against a theoretical ~3,000 passes/s ceiling, with ~250us of the 600us runtime spent on activation store/load plus synchronization and 95% of compute time in matrix-vector ops.
- On Jetson AGX Thor, EAGLE-3 speculative decoding gave a 2.5x throughput uplift, boosting Llama 3.3 70B decode from 6.27 to 16.19 tokens/sec at concurrency 1 (batch-1 single stream) using vLLM with W4A16 quantization.
- Software optimization alone (updated monthly vLLM containers) delivered 3.3-3.5x decode speedup on the same model and same quantization vs launch day: Llama 3.3 70B went from 12.64 to 41.5 output tok/s and DeepSeek R1 70B from 11.5 to 40.29 output tok/s.
- Adding speculative decoding pushed Llama 3.3 70B on Jetson Thor to 88.62 output tokens/sec, a 7x gain over launch-day throughput.
- A 70B model fits in Thor's memory at reduced precision: ~140GB in FP16, ~70GB in FP8, ~35GB in 4-bit, making quantization the enabler for large-model decode on the 128GB unified device.
- Jetson Thor's Blackwell architecture supports NVFP4 and FP8 quantization, with the demonstrated decode wins measured on W4A16 (4-bit weight, 16-bit activation) Llama 3.3 70B.
- Compiling a whole model forward pass into a single persistent cooperative megakernel (one threadblock per SM, counter-synchronized) eliminates the kernel-launch bubble between ops and beats CUDA-graphed cuBLAS at int8 batch-1 decode, e.g. 1.18-1.33x on L4, 1.25-1.27x on L40S, 1.19-1.23x on RTX 5090.
- Single-stream batch-1 decode is bandwidth-bound, and the megakernel exploits this by keeping activations in on-chip pages and prefetching the next layer's weights while the current layer computes.
- The megakernel does NOT beat cuBLAS/vLLM at bf16 batch-1: its bf16 kernel runs ~1.38 ms/token, about 1.24x slower than CUDA-graphed cuBLAS; the wins are specifically for quantized (int8/W8A16) weights.
- The megakernel fuses RMSNorm, RoPE, SwiGLU, and dequant micro-kernels together with gemv/gemm and attention tiles inside the single kernel, matching eager PyTorch output to ~1e-7 (fp32)/bf16 tolerance.
- An unattended 10-minute autotuning/autoresearch run self-improves the megakernel schedule by 1.47x over its own starting schedule, with every proposed schedule statically DAG-validated before launch.
- On edge devices the LLM decoding phase is memory-bandwidth-bound, which is why heterogeneous processing units cannot be fully exploited by naive sequential decoding — directly supporting the research premise that low-batch decode is bandwidth-limited.
- Ghidorah achieves up to 7.6x speedup in the LLM decoding phase on NVIDIA Jetson NX by combining speculative decoding with heterogeneous-core parallelism versus sequential decoding.
- The core method distributes speculative-decoding workloads across multiple heterogeneous processing units (CPU/GPU/NPU) to convert bandwidth-bound decode into more parallel compute, rather than optimizing a single GEMM kernel.
- Their ARCA profiling explicitly trades off draft acceptance rate against parallel capability to maximize net speedup — a scheduling insight relevant to spec-decode verify-step design.
- Ghidorah introduces hetero-core model parallelism (HCMP) adapted specifically for unified-memory edge platforms and speculative decoding, matching the Jetson/unified-memory constraint of the research question.
- LiquidGEMM's LiquidQuant performs W4A8 dequantization using only two arithmetic instructions per four weight elements, avoiding the CUDA-core dequantization bottleneck that throttles other W4A8 kernels.
- LiquidGEMM uses an implicit fine-grained pipeline that fully overlaps weight loading, dequantization, and MMA (tensor-core) across warp groups without software synchronization or redundant memory traffic.
- The root cause of slow existing W4A8 kernels is that CUDA-core dequantization cannot keep pace with tensor-core throughput — a design lesson directly applicable to W4A8 decode GEMMs.
- LiquidGEMM achieves up to 2.90x speedup over existing W4A8 GEMM kernels and 1.12-1.63x over TensorRT-LLM's quantized GEMM, with up to 4.94x end-to-end system speedup.
- In DeepGEMM, small-batch (decode) MoE GEMM is bandwidth-bound and the win comes from latency, whereas large-batch is compute-bound and the win comes from TFLOPS — directly answering whether small-M decode is memory- or compute-bound.
- DeepGEMM only implements SM90 (Hopper) and SM100 (Blackwell B200/GB200) code paths; SM120 (RTX 5090 / RTX Pro 6000) is unsupported, implying no path for Jetson Thor sm_110a.
- For MoE decode under CUDA graphs where the host cannot know per-expert token counts, DeepGEMM uses an M-grouped masked grouped-GEMM that takes a mask tensor instead of a host-side per-token loop.
- DeepGEMM FP8 GEMM uses E4M3 inputs with FP32 accumulation, casting back to BF16/FP32 on store, and applies fine-grained scaling of one scale per 128 columns of K.
- On Blackwell SM100, block scale factors are packed as UE8M0 — four UE8M0 floats packed into a single int32 — versus FP32 scale factors on SM90.
- In MoE models, speculative decoding increases data movement and verification time by 2-3x because draft tokens each independently select their own expert subset (e.g., Mixtral activates 2 of 8 experts per token, so 3 draft tokens can activate up to 6 experts, tripling weight fetch), unlike dense LLMs where verification traffic is unchanged.
- Because the extra expert-loading cost can exceed the throughput gain, always-on speculation causes up to 1.5x slowdown in MoEs, and even a conservative K=1 can lose over 25% performance on some tasks; the optimal draft length K is task- and model-dependent and varies per request/iteration.
- The Cascade framework uses a 'speculation utility' metric (ratio of ETR gain to verification cost), disabling speculation when utility < 1 and hill-climbing K otherwise; implemented in vLLM across five MoEs, it limits slowdown to 5% (vs 1.5x) and improves throughput 7-14% over static-K speculation.
- In single-batch (batch-1) serving, compute units are underutilized and each decode step's latency is governed by memory bandwidth to fetch model parameters, confirming low-batch decode is memory-bandwidth-bound rather than compute-bound.
- For dense models, speculative decoding drafters run 50x-100x faster than the target with minimal overhead and yield consistent 1.4-1.8x speedup on LLaMA-3-8B, because verification loads all weights regardless and adds no memory pressure.
- At batch size 1 (single-stream decode), SGLang's FP4 MoE kernel runs a MoE layer in 206.9μs vs vLLM's 369.5μs — a 1.78x speedup from Blackwell-specific kernel engineering, not quantization format alone.
- SGLang FP4 MoE is 1.32x faster than vLLM FP4 at batch size 1, showing that at low batch the kernel implementation (TMA alignment, padding) dominates over the numeric format.
- SGLang aligns FP4 block-scale MoE GEMMs to 128-token boundaries to satisfy the Tensor Memory Accelerator (TMA) requirement, whereas vLLM uses generic CUTLASS configs without Blackwell-specific padding — the source of the low-batch performance gap.
- Kernel fusion cuts activation memory traffic by 21.9% (26.5 MB vs 20.7 MB at batch 128), indicating low/moderate-batch FP4 MoE decode is sensitive to activation memory bandwidth.
- FP4 MoE reaches only ~1262 TFLOPS peak (SGLang, batch 4096) and scales toward 3.5-4.7x over BF16 only at large batch, implying single-stream decode is far from compute-bound and dominated by memory/launch overhead.

## REFUTED (killed by adversarial verify — do NOT pursue)


## Source list (23)
DeepGEMM guide, Blackwell NVFP4 comparison (HF), grouped-GEMMs+MoE (ianbarber), nvfp4-group-gemm (mufeez),
fused-moe-dispatch-triton (subhadipmitra), NV_grouped_gemm, momoe (tilderesearch), hazyresearch no-bubbles,
mirage-project, AutoMegaKernel (refuted), tp-llama, FlashMLA + hopper-fp8-sparse-deep-dive, vllm deepseek-v3-2,
cudnn DSA, lmsys deepseek-V32, NVIDIA DFlash spec-decode blog, Jetson-AGX-Thor-7x blog, arxiv 2509.01229/2505.23219/2506.20675.
