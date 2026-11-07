// LucinaInference oracle: dump llama.cpp's own tokens and logits so the Rust engine can be
// checked against the original (porting guidelines: the original is the source of truth).
//
// Built against the pinned llama.cpp checkout (see docs/upstreams.toml), CPU backend only:
// no device is offered to the model and op offload is off, so no Vulkan kernel runs.
//
//   llama-logits --model M.gguf --prompt-file p.txt --out work/ref/x [--ubatch 512]
//                [--threads 8] [--one-by-one] [--kv f16|f32] [--no-special]
//   llama-logits --model M.gguf --tokenize-lines corpus.txt --out work/ref/tok
//
// Outputs (logits mode):  <out>.tokens.txt  one token id per line
//                         <out>.logits.f32  n_tokens x n_vocab little-endian f32, row = position
//                         <out>.meta.json   settings used
// Outputs (tokenize mode): <out>.tokens.txt  one line per input line, ids space-separated
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void die(const char * msg) {
    fprintf(stderr, "llama-logits: %s\n", msg);
    exit(1);
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die(("cannot read " + path).c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, /*add_special*/ false, special);
    std::vector<llama_token> out(n < 0 ? 0 : n);
    if (n > 0) {
        int got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), out.data(), n, false, special);
        if (got != n) die("tokenize size mismatch");
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, prompt_file, lines_file, out;
    int ubatch = 512, threads = 8;
    bool one_by_one = false, special = true;
    std::string kv = "f16";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--tokenize-lines") lines_file = next();
        else if (a == "--out") out = next();
        else if (a == "--ubatch") ubatch = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--one-by-one") one_by_one = true;
        else if (a == "--kv") kv = next();
        else if (a == "--no-special") special = false;
        else die(("unknown argument " + a).c_str());
    }
    if (model_path.empty() || out.empty() || (prompt_file.empty() == lines_file.empty()))
        die("need --model, --out and exactly one of --prompt-file / --tokenize-lines");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices = no_devices;   // CPU only: no GPU device is offered to the model
    mp.n_gpu_layers = 0;
    mp.load_mtp = false;       // the MTP head is not part of the main forward
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (!lines_file.empty()) {
        std::ifstream f(lines_file, std::ios::binary);
        std::ofstream o(out + ".tokens.txt");
        std::string line;
        while (std::getline(f, line)) {
            // corpus lines carry literal "\n" escapes so multi-line cases fit on one line
            std::string text;
            for (size_t k = 0; k < line.size(); k++) {
                if (line[k] == '\\' && k + 1 < line.size() && line[k + 1] == 'n') { text += '\n'; k++; }
                else if (line[k] == '\\' && k + 1 < line.size() && line[k + 1] == 't') { text += '\t'; k++; }
                else if (line[k] == '\\' && k + 1 < line.size() && line[k + 1] == '\\') { text += '\\'; k++; }
                else text += line[k];
            }
            auto toks = tokenize(vocab, text, special);
            for (size_t k = 0; k < toks.size(); k++) o << (k ? " " : "") << toks[k];
            o << "\n";
        }
        llama_model_free(model);
        return 0;
    }

    const std::string text = read_file(prompt_file);
    std::vector<llama_token> toks = tokenize(vocab, text, special);
    const int n = (int) toks.size();
    if (n == 0) die("empty prompt");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::max(256, ((n + 255) / 256) * 256);
    cp.n_batch = std::max(n, 1);
    cp.n_ubatch = one_by_one ? 1 : std::min(ubatch, n);
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.op_offload = false;
    cp.type_k = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.type_v = cp.type_k;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");

    std::vector<float> logits((size_t) n * n_vocab);
    if (one_by_one) {
        for (int i = 0; i < n; i++) {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.n_tokens = 1;
            b.token[0] = toks[i]; b.pos[0] = i; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            if (llama_decode(ctx, b) != 0) die("decode failed");
            memcpy(&logits[(size_t) i * n_vocab], llama_get_logits_ith(ctx, 0), sizeof(float) * n_vocab);
            llama_batch_free(b);
        }
    } else {
        llama_batch b = llama_batch_init(n, 0, 1);
        b.n_tokens = n;
        for (int i = 0; i < n; i++) {
            b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        if (llama_decode(ctx, b) != 0) die("decode failed");
        for (int i = 0; i < n; i++)
            memcpy(&logits[(size_t) i * n_vocab], llama_get_logits_ith(ctx, i), sizeof(float) * n_vocab);
        llama_batch_free(b);
    }

    {
        std::ofstream o(out + ".tokens.txt");
        for (auto t : toks) o << t << "\n";
    }
    {
        FILE * f = fopen((out + ".logits.f32").c_str(), "wb");
        if (!f) die("cannot write logits");
        fwrite(logits.data(), sizeof(float), logits.size(), f);
        fclose(f);
    }
    {
        std::ofstream o(out + ".meta.json");
        o << "{\"model\":\"" << model_path << "\",\"n_tokens\":" << n << ",\"n_vocab\":" << n_vocab
          << ",\"ubatch\":" << cp.n_ubatch << ",\"threads\":" << threads << ",\"one_by_one\":" << (one_by_one ? "true" : "false")
          << ",\"kv\":\"" << kv << "\",\"special\":" << (special ? "true" : "false") << ",\"backend\":\"cpu\"}\n";
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
