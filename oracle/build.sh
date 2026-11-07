#!/usr/bin/env bash
# Build the llama.cpp oracle against the pinned checkout (docs/upstreams.toml).
set -euo pipefail
LLAMA="${LLAMA_CPP:-~/llama.cpp}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../work/bin"
mkdir -p "$out"
g++ -O2 -std=c++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$here/llama-logits.cpp" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$out/llama-logits"
g++ -O2 -std=c++17 -I"$LLAMA/ggml/include" "$here/ggml-dequant.cpp" \
  -L"$LLAMA/build/bin" -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$out/ggml-dequant"
echo "$out/llama-logits, $out/ggml-dequant (llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD))"
