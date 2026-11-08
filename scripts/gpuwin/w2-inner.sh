#!/usr/bin/env bash
# Window 2: llama.cpp 3612199ed with production flags. llama-bench matrix, then a private llama-server (production
# flags incl. MTP draft n=2, port <bench-port>) for the client replay.
cd ~/Projects/Lucina/LucinaInference
source scripts/gpuwin/win-common.sh
M27=~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
LB=~/llama.cpp/build/bin/llama-bench
P="-m $M27 -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -b 512 -o json"
step lb_pp_d0 $LB $P -p 512,4096 -n 0 -r 3
step lb_tg_d0_d4k $LB $P -p 0 -n 128 -d 0,4096 -r 3
step lb_pp512_d4k $LB $P -p 512 -n 0 -d 4096 -r 3
step lb_deep $LB $P -p 512 -n 128 -d 32768,65536,120000 -r 1
# MTP replay against a private server with the production flags
KEYF=~/.config/llama-server-api.key
~/llama.cpp/build/bin/llama-server -m $M27 --host 127.0.0.1 --port <bench-port> --no-webui --api-key-file $KEYF \
  -ngl 99 -c 122880 -np 1 --flash-attn on --jinja --reasoning off --no-reasoning-preserve \
  --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp --spec-draft-n-max 2 -b 512 -ub 256 \
  > $L/llama-server.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:<bench-port>/health)" = 200 ] && break; sleep 1; done
echo "$(date -u +%T) private llama-server health $(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:<bench-port>/health)"
step client_llama_mtp target/release/lucina-bench-client --url http://127.0.0.1:<bench-port> --api-key-file $KEYF --model qwen3.8 \
  --requests work/long/requests.jsonl --repeat 2 --out $L/client-llama-mtp.json
kill $SRV; sleep 3; kill -9 $SRV 2>/dev/null
