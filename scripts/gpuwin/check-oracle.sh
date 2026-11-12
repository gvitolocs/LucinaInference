#!/usr/bin/env bash
# Regression check of the oracle's new --tokens / --positions path on CPU (waits for Tctl < 88 °C).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
[ -f scripts/local.env ] && . scripts/local.env
[ -n "${LI_MODELS_DIR:-}" ] || { echo "error: LI_MODELS_DIR is unset; set it in scripts/local.env (see scripts/local.env.example)" >&2; exit 1; }
M=$LI_MODELS_DIR/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf; V=248320
# gate (coordinator): MemAvailable >= 20 GB, load < 16, Tctl < 88 C; 3 threads, CPUQuota 300%
ok() { [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 20 ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 16 ] && [ "$(sensors | awk '/Tctl/{gsub(/[+°C]/,"",$2); print int($2)}')" -lt 88 ]; }
until ok; do sleep 60; done
mkdir -p work/chk
systemd-run --user --scope -q -p MemoryMax=24G -p CPUQuota=300% work/bin/llama-logits --model $M --tokens work/ref/q38-27b-short-b512.tokens.txt \
  --ubatch 512 --threads 3 --out work/chk/all > /dev/null 2>&1
cmp work/chk/all.logits.f32 work/ref/q38-27b-short-b512.logits.f32 && echo "ALL ROWS via --tokens: bit-identical to the earlier run"
systemd-run --user --scope -q -p MemoryMax=24G -p CPUQuota=300% work/bin/llama-logits --model $M --tokens work/ref/q38-27b-short-b512.tokens.txt \
  --positions 24,34 --ubatch 512 --threads 3 --out work/chk/pos > /dev/null 2>&1
cat work/chk/pos.positions.txt | tr '\n' ' '; echo
for i in 0 1; do r=$([ $i = 0 ] && echo 24 || echo 34)
  dd if=work/ref/q38-27b-short-b512.logits.f32 of=work/chk/ref-r$r.f32 bs=$((V*4)) skip=$r count=1 status=none
  dd if=work/chk/pos.logits.f32 of=work/chk/pos-r$r.f32 bs=$((V*4)) skip=$i count=1 status=none
  echo "row $r positions-mode vs all-rows: $(target/release/lucina-parity compare --n-vocab $V --ref work/chk/ref-r$r.f32 --test work/chk/pos-r$r.f32 | tr -d ' \n')"
done
