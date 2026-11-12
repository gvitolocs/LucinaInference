#!/usr/bin/env bash
# Build the llama.cpp oracle against the pinned checkout (docs/upstreams.toml).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/scripts/local.env" ] && . "$ROOT/scripts/local.env"
[ -n "${LI_LLAMA_CPP_DIR:-}" ] || { echo "error: LI_LLAMA_CPP_DIR is unset; set it in scripts/local.env (see scripts/local.env.example)" >&2; exit 1; }
LLAMA="${LLAMA_CPP:-$LI_LLAMA_CPP_DIR}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../work/bin"
mkdir -p "$out"
g++ -O2 -std=c++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$here/llama-logits.cpp" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$out/llama-logits"
g++ -O2 -std=c++17 -I"$LLAMA/ggml/include" "$here/ggml-dequant.cpp" \
  -L"$LLAMA/build/bin" -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$out/ggml-dequant"
echo "$out/llama-logits, $out/ggml-dequant (llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD))"
