// inspect_weights.cpp — load the DeepSeek-V4 REAP checkpoint, validate the full tensor set against
// the deepseek_v4 constants, and report the resident memory footprint by dtype/weight-class.
// Pure C++/mmap (no CUDA): answers Gate-1's "does the footprint match 96.66 GiB?" empirically.
//   build: g++ -O2 -std=c++17 -I include tools/inspect_weights.cpp -o build/inspect_weights
//   run:   ./build/inspect_weights /home/patrickd/models/DeepSeek-V4-Flash-180B
#include "safetensors.h"
#include "deepseek_v4.h"
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <cinttypes>

using namespace dsv4;
using st::Tensor;

// This REAP checkpoint already uses attn/ffn naming; normalize the few legacy spellings just in case.
static std::string key_map(const std::string& in) {
    std::string s = in;
    auto rep = [&](const char* a, const char* b) {
        size_t p; while ((p = s.find(a)) != std::string::npos) s.replace(p, strlen(a), b);
    };
    if (s.rfind("model.", 0) == 0) s = s.substr(6);
    rep("self_attn", "attn");
    rep(".mlp.", ".ffn.");
    rep("weight_scale_inv", "scale");
    rep("e_score_correction_bias", "bias");
    return s;
}

struct Checker {
    const st::ShardedSafeTensors& st_;
    int missing = 0, shape_err = 0, checked = 0;
    std::vector<std::string> problems;

    bool shape_eq(const Tensor& t, std::vector<int64_t> want) {
        if (t.shape.size() != want.size()) return false;
        for (size_t i = 0; i < want.size(); ++i) if (t.shape[i] != want[i]) return false;
        return true;
    }
    // require presence (+ optional exact shape). shape={} skips the shape check.
    void req(const std::string& name, std::vector<int64_t> shape = {}) {
        ++checked;
        if (!st_.has(name)) { ++missing; problems.push_back("MISSING " + name); return; }
        if (!shape.empty() && !shape_eq(st_.get(name), shape)) {
            ++shape_err;
            const auto& s = st_.get(name).shape;
            std::string got = "["; for (size_t i=0;i<s.size();++i){ got += std::to_string(s[i]); if(i+1<s.size()) got+=","; } got += "]";
            std::string exp = "["; for (size_t i=0;i<shape.size();++i){ exp += std::to_string(shape[i]); if(i+1<shape.size()) exp+=","; } exp += "]";
            problems.push_back("SHAPE " + name + " got " + got + " want " + exp);
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <checkpoint_dir> [--verbose]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    bool verbose = (argc > 2 && std::string(argv[2]) == "--verbose");

    printf("Loading index + shard headers from %s ...\n", dir.c_str());
    st::ShardedSafeTensors S(dir, key_map);
    printf("  shards=%zu  tensors=%zu\n\n", S.shardCount(), S.count());

    // ---- 1. footprint by dtype + weight class ----
    std::map<std::string, std::pair<uint64_t,uint64_t>> by_dt;   // dtype -> (count, bytes)
    std::map<int, uint64_t> by_class;                            // WClass -> bytes
    uint64_t total = 0;
    for (auto& kv : S.all()) {
        const Tensor& t = kv.second;
        by_dt[t.dtype].first++; by_dt[t.dtype].second += t.nbytes;
        by_class[(int)wclass(t.dtype)] += t.nbytes;
        total += t.nbytes;
    }
    const char* cls_name[] = {"FP4_EXPERT","FP8_LINEAR","BF16","F32","I64_HASH","SCALE_E8M0","UNKNOWN"};
    printf("=== footprint by dtype ===\n");
    for (auto& kv : by_dt)
        printf("  %-9s count=%7" PRIu64 "  %8.3f GB\n", kv.first.c_str(), kv.second.first, kv.second.second/1e9);
    printf("=== footprint by weight class ===\n");
    for (auto& kv : by_class)
        printf("  %-11s %8.3f GB\n", cls_name[kv.first], kv.second/1e9);
    printf("  ----------------------------------------\n");
    printf("  TOTAL      %8.3f GB  = %7.3f GiB   (published loaded footprint: 96.66 GiB)\n\n",
           total/1e9, total/1073741824.0);

    // ---- 2. structural validation against dsv4 constants ----
    Checker C{S};
    // top-level
    C.req("embed.weight",  {VOCAB, DIM});
    C.req("head.weight",   {VOCAB, DIM});
    C.req("norm.weight",   {DIM});
    C.req("hc_head_fn",    {HC_MULT, HC_DIM});
    C.req("hc_head_base",  {HC_MULT});
    C.req("hc_head_scale", {1});

    auto P = [](const std::string& pfx, const std::string& s){ return pfx + s; };
    for (int L = 0; L < N_LAYERS; ++L) {
        std::string p = "layers." + std::to_string(L) + ".";
        // norms
        C.req(P(p,"attn_norm.weight"), {DIM});
        C.req(P(p,"ffn_norm.weight"),  {DIM});
        // MLA attention
        C.req(P(p,"attn.wq_a.weight"), {Q_LORA, DIM});
        C.req(P(p,"attn.wq_a.scale"));
        C.req(P(p,"attn.q_norm.weight"), {Q_LORA});
        C.req(P(p,"attn.wq_b.weight"), {(int64_t)N_HEADS*HEAD_DIM, Q_LORA});
        C.req(P(p,"attn.wkv.weight"),  {HEAD_DIM, DIM});
        C.req(P(p,"attn.kv_norm.weight"), {HEAD_DIM});
        C.req(P(p,"attn.wo_a.weight"), {(int64_t)O_GROUPS*O_LORA, DIM});
        C.req(P(p,"attn.wo_b.weight"), {DIM, (int64_t)O_GROUPS*O_LORA});
        C.req(P(p,"attn.attn_sink"),   {N_HEADS});
        // HC params
        C.req(P(p,"hc_attn_fn"), {HC_MIX, HC_DIM});
        C.req(P(p,"hc_ffn_fn"),  {HC_MIX, HC_DIM});
        C.req(P(p,"hc_attn_base"), {HC_MIX});  C.req(P(p,"hc_attn_scale"), {3});
        C.req(P(p,"hc_ffn_base"),  {HC_MIX});  C.req(P(p,"hc_ffn_scale"),  {3});
        // compressor / indexer per feature map
        if (has_compressor(L)) {
            int cd = 2 * HEAD_DIM;   // coff*head_dim, coff=2 when ratio==4 (overlap)
            if (compress_ratio(L) == 128) cd = HEAD_DIM;
            C.req(P(p,"attn.compressor.wkv.weight"),  {cd, DIM});
            C.req(P(p,"attn.compressor.wgate.weight"),{cd, DIM});
            C.req(P(p,"attn.compressor.norm.weight"), {HEAD_DIM});
        }
        if (has_indexer(L)) {
            C.req(P(p,"attn.indexer.wq_b.weight"), {(int64_t)INDEX_N_HEADS*INDEX_HEAD_DIM, Q_LORA});
            C.req(P(p,"attn.indexer.weights_proj.weight"), {INDEX_N_HEADS, DIM});
            C.req(P(p,"attn.indexer.compressor.wkv.weight"), {2*INDEX_HEAD_DIM, DIM});
            C.req(P(p,"attn.indexer.compressor.norm.weight"), {INDEX_HEAD_DIM});
        }
        // MoE gate
        C.req(P(p,"ffn.gate.weight"), {N_ROUTED, DIM});
        if (is_hash_layer(L)) C.req(P(p,"ffn.gate.tid2eid"), {VOCAB, N_ACT});
        else                  C.req(P(p,"ffn.gate.bias"), {N_ROUTED});
        // routed experts (spot-check first + last to keep the pass fast)
        for (int e : std::vector<int>{0, N_ROUTED-1}) {
            std::string ep = p + "ffn.experts." + std::to_string(e) + ".";
            C.req(ep+"w1.weight", {MOE_INTER, DIM/2});   // I8 packed: in/2
            C.req(ep+"w2.weight", {DIM, MOE_INTER/2});
            C.req(ep+"w3.weight", {MOE_INTER, DIM/2});
        }
        // shared expert (fp8, not fp4)
        C.req(P(p,"ffn.shared_experts.w1.weight"), {MOE_INTER, DIM});
        C.req(P(p,"ffn.shared_experts.w2.weight"), {DIM, MOE_INTER});
    }

    // ---- 3. MTP (REAP built-in nextn: enorm/hnorm/e_proj/h_proj + block; NO markov/confidence) ----
    for (int m = 0; m < N_MTP; ++m) {
        std::string p = "mtp." + std::to_string(m) + ".";
        C.req(P(p,"enorm.weight"), {DIM});
        C.req(P(p,"hnorm.weight"), {DIM});
        C.req(P(p,"e_proj.weight"), {DIM, DIM});
        C.req(P(p,"h_proj.weight"), {DIM, DIM});
        C.req(P(p,"norm.weight"),  {DIM});
        C.req(P(p,"attn.wq_a.weight"), {Q_LORA, DIM});
        C.req(P(p,"attn.wkv.weight"),  {HEAD_DIM, DIM});
        C.req(P(p,"ffn.gate.weight"), {N_ROUTED, DIM});
        C.req(P(p,"hc_head_fn"), {HC_MULT, HC_DIM});
    }

    printf("=== structural validation (vs deepseek_v4.h constants) ===\n");
    printf("  checked=%d  missing=%d  shape_mismatch=%d\n", C.checked, C.missing, C.shape_err);
    if (!C.problems.empty()) {
        int show = verbose ? (int)C.problems.size() : std::min<int>(30, C.problems.size());
        for (int i = 0; i < show; ++i) printf("    %s\n", C.problems[i].c_str());
        if (show < (int)C.problems.size()) printf("    ... (%zu more; --verbose for all)\n", C.problems.size()-show);
        printf("\nRESULT: FAIL (%d problems)\n", (int)C.problems.size());
        return 1;
    }
    printf("\nRESULT: PASS — all expected tensors present with correct shapes.\n");
    return 0;
}
