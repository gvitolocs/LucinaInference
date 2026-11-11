#!/usr/bin/env bash
# Night 2, window B: llama.cpp 3612199ed, production flags. llama-bench matrix, then MTP replay on a private server.
cd ~/Projects/Lucina/LucinaInference
source scripts/gpuwin/win-common.sh
M=~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
LB=~/llama.cpp/build/bin/llama-bench
P="-m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -b 256 -o json"
KEYF=~/.config/llama-server-api.key
~/llama.cpp/build/bin/llama-server -m $M --host 127.0.0.1 --port <bench-port> --no-webui --api-key-file $KEYF \
  -ngl 99 -c 122880 -np 1 --flash-attn on --jinja --reasoning off --no-reasoning-preserve \
  --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp --spec-draft-n-max 2 -b 512 -ub 256 > $L/llama-server.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do [ "$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:<bench-port>/health)" = 200 ] && break; sleep 1; done
echo "$(date -u +%T) private llama-server health $(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:<bench-port>/health)"
step client_llama_mtp target/release/lucina-bench-client --url http://127.0.0.1:<bench-port> --api-key-file $KEYF --model qwen3.8 \
  --requests work/long/requests.jsonl --repeat 2 --out $L/client-llama-mtp.json
step client_llama_depth target/release/lucina-bench-client --url http://127.0.0.1:<bench-port> --api-key-file $KEYF --model qwen3.8 --requests work/long/depth.jsonl --repeat 1 --timeout-s 900 --out $L/client-llama-depth.json
kill $SRV; sleep 3; kill -9 $SRV 2>/dev/null
