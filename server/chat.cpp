// chat.cpp — lean pure-C++ terminal chat client for the DFlash OpenAI server (streaming, thinking, multi-turn).
// build: g++ -O2 -std=c++17 -I include server/chat.cpp -o build/chat -lpthread ; run: ./build/chat [host] [port]
#include "third_party/httplib.h"
#include "third_party/json.hpp"
#include <iostream>
#include <string>
using json=nlohmann::json;
int main(int argc,char**argv){
    std::string host=argc>1?argv[1]:"localhost"; int port=argc>2?atoi(argv[2]):8080;
    bool think = getenv("THINK")!=nullptr; float temp = getenv("TEMP")?atof(getenv("TEMP")):0.f;
    httplib::Client cli(host,port); cli.set_read_timeout(600,0);
    json msgs=json::array();
    std::cout<<"\033[1;35m▌ Gemma-4 · DFlash · Thor\033[0m  "<<host<<":"<<port<<"   (/new resets · Ctrl-D quits)\n";
    std::string line;
    while(true){
        std::cout<<"\n\033[1;36myou ›\033[0m "; std::cout.flush();
        if(!std::getline(std::cin,line)) break;
        if(line=="/new"){ msgs.clear(); std::cout<<"\033[2m[history cleared]\033[0m\n"; continue; }
        if(line.empty()) continue;
        msgs.push_back({{"role","user"},{"content",line}});
        json body={{"messages",msgs},{"stream",true},{"max_tokens",512},{"temperature",temp},{"enable_thinking",think}};
        std::cout<<"\033[1;32mgemma ›\033[0m "; std::cout.flush();
        std::string buf, answer; bool inR=false;
        auto t0=std::chrono::steady_clock::now(); int ntok=0;
        cli.Post("/v1/chat/completions", httplib::Headers{}, body.dump(), "application/json",
            [&](const char* data,size_t len)->bool{
                buf.append(data,len); size_t i;
                while((i=buf.find("\n\n"))!=std::string::npos){ std::string ln=buf.substr(0,i); buf.erase(0,i+2);
                    if(ln.rfind("data: ",0)!=0) continue; std::string p=ln.substr(6); if(p=="[DONE]") continue;
                    try{ auto j=json::parse(p); auto& d=j["choices"][0]["delta"];
                        if(d.contains("reasoning_content")){ if(!inR){std::cout<<"\033[2m🤔 ";inR=true;} std::cout<<d["reasoning_content"].get<std::string>(); }
                        if(d.contains("content")){ if(inR){std::cout<<"\033[0m\n";inR=false;} std::string c=d["content"].get<std::string>(); std::cout<<c; answer+=c; ntok++; }
                        std::cout.flush();
                    }catch(...){}
                }
                return true; });
        if(inR) std::cout<<"\033[0m";
        double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::cout<<"\033[2m   ["<<ntok<<" tok · "<<(dt>0?ntok/dt:0)<<" tok/s]\033[0m\n";
        msgs.push_back({{"role","assistant"},{"content",answer}});
    }
    std::cout<<"\nbye\n"; return 0;
}
