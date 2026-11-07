#!/usr/bin/env bash
# Build the llama.cpp oracle against the pinned checkout (docs/upstreams.toml).
set -euo pipefail
LLAMA="${LLAMA_CPP:-~/llama.cpp}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../work/bin"
mkdir -p "$out"
g++ -O2 -std=c++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$here/llama-logits.cpp" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$out/llama-logits"
echo "$out/llama-logits (llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD))"
