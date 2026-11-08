#!/usr/bin/env bash
# Window 3: infr with the same shapes and flags as window 2 (q4_0 KV, ub 256), then infr serve with MTP for the replay.
cd ~/Projects/Lucina/LucinaInference
source scripts/gpuwin/win-common.sh
M27=~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
I="target/release/infr bench $M27 -u 256 --set kv.type_k=q4_0 --set kv.type_v=q4_0"
step ib_pp512_d0 $I -p 512 -n 0 -r 3
step ib_pp4096_d0 $I -p 4096 -n 0 -r 3
step ib_tg128_d0 $I -p 0 -n 128 -r 3
step ib_tg128_d4k $I -p 0 -n 128 -d 4096 -r 3
step ib_pp512_d4k $I -p 512 -n 0 -d 4096 -r 3
step ib_pp512_d32k $I -p 512 -n 0 -d 32768 -r 1
step ib_tg128_d32k $I -p 0 -n 128 -d 32768 -r 1
step ib_pp512_d64k $I -p 512 -n 0 -d 65536 -r 1
step ib_tg128_d64k $I -p 0 -n 128 -d 65536 -r 1
step ib_pp512_d120k $I -p 512 -n 0 -d 120000 --ctx 122880 -r 1
# (tg128 at 120k depth: window 1, tg32 at d120000)
INFR_MTP=1 target/release/infr serve $M27 --addr 127.0.0.1:<bench-port> --ctx 122880 -u 256 --no-think \
  --set kv.type_k=q4_0 --set kv.type_v=q4_0 > $L/infr-serve.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:<bench-port>/health)" = 200 ] && break; sleep 1; done
echo "$(date -u +%T) infr serve health $(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:<bench-port>/health)"
step client_infr_mtp target/release/lucina-bench-client --url http://127.0.0.1:<bench-port> --model qwen3.8 \
  --requests work/long/requests.jsonl --repeat 2 --out $L/client-infr-mtp.json
kill $SRV; sleep 3; kill -9 $SRV 2>/dev/null
