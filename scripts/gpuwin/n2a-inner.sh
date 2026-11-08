#!/usr/bin/env bash
# Night 2, window A: infr q4_0 KV (now honoured), prepass vs inline decode, depths 0..120k, VRAM at 122,880.
cd ~/Projects/Lucina/LucinaInference
source scripts/gpuwin/win-common.sh
M=~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
I="target/release/infr bench $M -u 256 --set kv.type_k=q4_0 --set kv.type_v=q4_0"
step tg_d120k_prepass $I -p 0 -n 64 -d 120000 --ctx 122880 -r 1
step tg_d120k_inline env INFR_KV_INLINE=1 $I -p 0 -n 64 -d 120000 --ctx 122880 -r 1
for d in 0 4096 32768 65536; do
  step tg_d${d}_prepass $I -p 0 -n 128 -d $d -r 1
  step tg_d${d}_inline env INFR_KV_INLINE=1 $I -p 0 -n 128 -d $d -r 1
done
