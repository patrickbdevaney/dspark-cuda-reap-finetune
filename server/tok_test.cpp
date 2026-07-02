// tok_test.cpp — validate the C++ BPE tokenizer against HF reference vectors.
#include "tokenizer.h"
#include <cstdio>
#include <string>
static bool eq(const std::vector<int>&a,const std::vector<int>&b){ if(a.size()!=b.size())return false; for(size_t i=0;i<a.size();++i)if(a[i]!=b[i])return false; return true; }
static void pr(const char* s,const std::vector<int>&v){ printf("%-40s -> [",s); for(size_t i=0;i<v.size();++i)printf("%s%d",i?", ":"",v[i]); printf("]\n"); }
int main(int argc,char**argv){
    Tokenizer tk; tk.load(argc>1?argv[1]:std::string(getenv("HOME")+std::string("/models/gemma-4-26B-A4B-it-NVFP4/tokenizer.json")));
    printf("loaded: vocab=%zu merges=%zu specials=%zu eos=%d bos=%d turn_end=%d\n",tk.vocab.size(),tk.merges.size(),tk.specials.size(),tk.eos_id,tk.bos_id,tk.turn_end);
    struct T{ const char* s; std::vector<int> ref; };
    std::vector<T> tests={
        {"Hello, world!", {9259,236764,1902,236888}},
        {"List the first 40 prime numbers.", {1613,506,1171,236743,236812,236771,8355,4945,236761}},
        {"def fib(n):", {2063,10779,236769,236749,1473}},
        {"  spaces", {138,35220}},
        {"\n", {107}},
    };
    int pass=0;
    for(auto& t : tests){ auto got=tk.encode(t.s); bool ok=eq(got,t.ref); pass+=ok;
        printf("[%s] ",ok?"PASS":"FAIL"); pr(t.s,got); if(!ok){ printf("   expected: "); pr("",t.ref); } }
    // roundtrip decode
    printf("decode([9259,236764]) = %s (expect 'Hello,')\n", tk.decode({9259,236764}).c_str());
    printf("\n%d/%zu passed\n", pass, tests.size());
    return pass==(int)tests.size()?0:1;
}
