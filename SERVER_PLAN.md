# Production Serving Build Plan — pure-CUDA gemma-4 + DFlash OpenAI server on Thor
Banking the decode win (DFlash 118 tok/s, +10% vs vLLM, lean pure-CUDA MoE runtime) as a usable serving system.
Architecture decision (user): **PURE C++/CUDA SINGLE BINARY** (leanest, zero extra RAM, matches ethos). No Python at runtime.
Deps vendored (header-only): `include/third_party/httplib.h`, `include/third_party/json.hpp`.

## Ceiling: CONFIRMED at 118 (current draft). BLK=16 baked-in (static const), DK=14 already-optimal. Bigger block needs draft retraining (separate ML project). Bank 118.

## PHASES (resumable)
### Phase 1 — Persistent Engine + OpenAI HTTP server  [FOUNDATION, in progress]
1a. **BPE tokenizer in C++** [CRITICAL PATH — build+validate FIRST]. Spec below. `include/tokenizer.h`.
1b. **Engine refactor**: split forward.cu main() into a load-once `Engine{ load(); prefill(ids); step()->tok; reset(); }` reusable across requests. Keep DFlash decode loop intact.
1c. **HTTP server** (`server/server.cpp`, cpp-httplib): `POST /v1/chat/completions` (+ `/v1/completions`, `/v1/models`). Parse JSON (json.hpp), apply chat template, tokenize, run engine, return OpenAI-schema JSON.
1d. **SSE token streaming** (`stream:true` → `data: {chunk}\n\n` deltas).
1e. **Sampling**: greedy + temperature + top-p (TEMP/TYP_EPS already in forward.cu; add top-p). Respect OpenAI params (temperature, max_tokens, stop).

### Phase 2 — Prefix caching (agentic turns)
Persist KV across requests; hash prefix blocks (system prompt + history); on new request, match longest cached prefix, skip its re-prefill. vLLM automatic-prefix-caching model. Big win when a long system prompt is reused every turn.

### Phase 3 — FP8 KV cache @ 64k
Extend KV alloc to 64k ctx + FP8 store/load in sdpa_cache_kernel + draft attention (+ sliding-window layers). LOSSY but standard for serving (NOT the bit-exact decode path). Both a memory enabler (fit 64k) AND a speed win >32k (research). Validate quality delta.

### Phase 4 — Clients
Terminal chat REPL + minimal single-file web UI (both hit the OpenAI endpoint; streaming).

## TOKENIZER SPEC (gemma-4, from tokenizer.json)
- type BPE, vocab 262144, merges 514906 (ordered by rank = priority), byte_fallback=true, ignore_merges=false.
- **Normalizer**: Replace " " (U+0020) → "▁" (U+2581).
- **Pre-tokenizer**: Split on " " MergedWithPrevious — effectively NO-OP after normalization (no spaces remain); BPE runs over the whole normalized string.
- **BPE**: initial symbols = unicode chars of normalized string; any char-token not in vocab → byte_fallback to its UTF-8 bytes as `<0xXX>` tokens (ids ~3-258). Then repeatedly merge the adjacent pair with the LOWEST merge-rank until none applies. Map final symbols → ids.
- **Decoder**: Sequence[ Replace "▁"→" ", ByteFallback (fuse <0xXX> runs → bytes), Fuse ]. i.e. concat token strings, ▁→space, reassemble byte-fallback runs into UTF-8.
- Specials: <pad>=0 <eos>=1 <bos>=2 ; added_tokens are `special`, matched BEFORE bpe (never split). `<end_of_turn>` is the turn stop.
- Default encode adds NO bos; the chat template supplies <bos> + turn markers.
- **Chat template (gemma)**: `<bos>` then per turn `<start_of_turn>{role}\n{content}<end_of_turn>\n`, roles user/model (map system→prepend to first user). Generation prompt ends `<start_of_turn>model\n`. Stop on `<end_of_turn>` / <eos>.

## REFERENCE TEST VECTORS (validate the C++ BPE against these — from HF tokenizers)
- "Hello, world!" -> [9259, 236764, 1902, 236888]
- "List the first 40 prime numbers." -> [1613, 506, 1171, 236743, 236812, 236771, 8355, 4945, 236761]
- "def fib(n):" -> [2063, 10779, 236769, 236749, 1473]
- "  spaces" -> [138, 35220]
- "\n" -> [107]
- decode([9259,236764]) == "Hello,"

## ENGINE-REFACTOR NOTES (forward.cu)
- main() currently: load ckpt (arg1) + draft (~/models/...DFlash) + DScratch, read ids from file, run GEN loop, print. Has incremental decode_step (mtok=1) + DFlash verify path + CUDA graphs.
- Refactor: `struct Engine` holds Model + DraftModel + DScratch + Session (KV). Methods: prefill(vector<int> ids), generate(params, callback(tok)) streaming, reset()/rewind(n) for prefix reuse. Keep the DFlash verify graph.
- KV cache = Session.Kc/Vc per layer; currently bf16, CAP context. For prefix cache, keep populated across requests + track prefix length.
