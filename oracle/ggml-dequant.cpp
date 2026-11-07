// LucinaInference oracle: dequantize the first rows of one GGUF tensor with ggml's own `to_float`
// (llama.cpp pin in docs/upstreams.toml), so Rust dequant ports can be checked bit for bit.
//
//   ggml-dequant <model.gguf> <tensor name> <rows> <out.f32>
//   ggml-dequant <model.gguf> --list            (name, ggml type id, type name, ne0, ne1)
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ggml-dequant <model.gguf> <tensor> <rows> <out.f32> | <model.gguf> --list\n");
        return 1;
    }
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &meta };
    gguf_context * g = gguf_init_from_file(argv[1], ip);
    if (!g) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

    if (std::string(argv[2]) == "--list") {
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(meta, name);
            printf("%s %d %s %lld %lld\n", name, (int) t->type, ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1]);
        }
        return 0;
    }
    if (argc < 5) { fprintf(stderr, "need <rows> <out.f32>\n"); return 1; }
    const char * name = argv[2];
    const int64_t id = gguf_find_tensor(g, name);
    if (id < 0) { fprintf(stderr, "no tensor %s\n", name); return 1; }
    ggml_tensor * t = ggml_get_tensor(meta, name);
    const int64_t ne0 = t->ne[0];
    const int64_t rows = std::min<int64_t>(atoll(argv[3]), ggml_nrows(t));
    const size_t row_bytes = ggml_row_size(t->type, ne0);
    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    if (!tr->to_float) { fprintf(stderr, "type %s has no to_float\n", ggml_type_name(t->type)); return 1; }

    FILE * f = fopen(argv[1], "rb");
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    if (fseek(f, (long) off, SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); return 1; }
    std::vector<uint8_t> raw(row_bytes * rows);
    if (fread(raw.data(), 1, raw.size(), f) != raw.size()) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    std::vector<float> out(ne0 * rows);
    for (int64_t r = 0; r < rows; r++) tr->to_float(raw.data() + r * row_bytes, out.data() + r * ne0, ne0);
    FILE * o = fopen(argv[4], "wb");
    fwrite(out.data(), sizeof(float), out.size(), o);
    fclose(o);
    fprintf(stderr, "%s: %s, %lld rows x %lld\n", name, ggml_type_name(t->type), (long long) rows, (long long) ne0);
    return 0;
}
