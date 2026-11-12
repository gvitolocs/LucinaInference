#!/usr/bin/env bash
# Window 1: VRAM at 122,880 ctx (q4_0, ub 256) with infr tg at depth 120k; correctness of infr Vulkan against
# llama.cpp Vulkan (4k, 32k, 27B short) and against the f64/CPU references (all rows, 27B short and 9B chat).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
[ -f scripts/local.env ] && . scripts/local.env
for v in LI_MODELS_DIR; do [ -n "${!v:-}" ] || { echo "error: $v is unset; set it in scripts/local.env (see scripts/local.env.example)" >&2; exit 1; }; done
source scripts/gpuwin/win-common.sh
M27=$LI_MODELS_DIR/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
M9=$LI_MODELS_DIR/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf
KV="--set kv.type_k=q4_0 --set kv.type_v=q4_0"
mkdir -p work/vk work/vkl
step vram_ctx122880_tg_d120k target/release/infr bench $M27 -p 0 -n 32 -d 120000 --ctx 122880 -u 256 -r 1 $KV
step infr_27b_short_rows target/release/lucina-parity vulkan-window --model $M27 --tokens work/ref/q38-27b-short-b512.tokens.txt \
  --all-rows --kv q4_0 --out work/vk/q38-27b-short-q4
step infr_27b_long target/release/lucina-parity vulkan-window --model $M27 --tokens work/long/p32k.tokens.txt \
  --prefixes 2048,4096,32768 --kv q4_0 --out work/vk/q38-27b-agent
step llama_27b_long_ub256 work/bin/llama-logits --gpu --model $M27 --tokens work/long/p32k.tokens.txt --positions 2047,4095,32767 \
  --ubatch 256 --kv q4_0 --out work/vkl/q38-27b-agent-ub256
step llama_27b_long_ub512 work/bin/llama-logits --gpu --model $M27 --tokens work/long/p32k.tokens.txt --positions 2047,4095,32767 \
  --ubatch 512 --kv q4_0 --out work/vkl/q38-27b-agent-ub512
step llama_27b_short work/bin/llama-logits --gpu --model $M27 --tokens work/ref/q38-27b-short-b512.tokens.txt --ubatch 256 --kv q4_0 \
  --out work/vkl/q38-27b-short-q4
step infr_9b_chat_rows target/release/lucina-parity vulkan-window --model $M9 --tokens work/ref/q35-9b-chat-b512.tokens.txt \
  --all-rows --kv f16 --out work/vk/q35-9b-chat-f16
step llama_9b_chat work/bin/llama-logits --gpu --model $M9 --tokens work/ref/q35-9b-chat-b512.tokens.txt --ubatch 512 --kv f16 \
  --out work/vkl/q35-9b-chat-f16
