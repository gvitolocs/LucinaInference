#!/usr/bin/env bash
# One GPU window (owner-approved stage-2 windows, 2026-11-08 night). Usage: gpuwin.sh NAME INNER MAX_S HARD_END_EPOCH
# 1) wait for and HOLD the coding-agent queue lock  2) queue our lease  3) stop llama-server-qwen (clock starts)
# 4) run INNER under the lease with DEADLINE = min(T0 + MAX_S, HARD_END)  5) VRAM watchdog: card > 23.5 GB kills the
#    running measurement  6) restore: kill leftovers, restart llama-server-qwen, /health, one gateway request  7) unlock.
set -u
NAME=$1 INNER=$2 MAX_S=$3 HARD_END=$4
cd ~/Projects/Lucina/LucinaInference
L=work/logs/$NAME; mkdir -p $L
MEAS='target/release/infr|lucina-parity vulkan-window|work/bin/llama-logits|llama-bench|llama-server .*--port <bench-port>'
# Headroom gate (coordinator, 2026-11-07 22:15): never start while build-host is swapping.
# OK = MemAvailable >= 20 GB, 1-min load < 16, swap-in+out < 256 pages/s over 5 s.
res_ok() {
  local avail load sio
  avail=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)
  load=$(awk "{print int(\$1)}" /proc/loadavg)
  sio=$(vmstat 5 2 | tail -1 | awk "{print \$7+\$8}")
  # Lease budget (2026-11-08: windows 1 and 2 stopped production and then never got their lease, because another
  # job held 4000 MiB): every other ledger entry (held, linger, waiting) plus ours must fit once llama-server releases.
  local others budget
  others=$($LI_GPU_LEASE status --json | jq '[.entries[] | select(.name != "llama-server") | .mb] | add // 0')
  budget=$($LI_GPU_LEASE status --json | jq '.vramTotalMb - .reserveMb')
  echo "$(date -u +%T) $NAME: headroom avail ${avail} GB, load ${load}, swap io ${sio}/s, other leases ${others} MiB of ${budget}"
  if [ "$avail" -ge 20 ] && [ "$load" -lt 16 ] && [ "$sio" -lt 256 ] && [ $(( others + 21500 )) -le "$budget" ]; then
    echo "$(date -u +%T) $NAME: headroom ok"; return 0
  fi
  echo "$(date -u +%T) $NAME: headroom low"; return 1
}
exec 9>$LI_GPU_FLOCK
while true; do
  if [ $(( HARD_END - $(date +%s) )) -lt 600 ]; then echo "$(date -u +%T) $NAME: SKIPPED, less than 10 min before the hard end"; exit 0; fi
  until res_ok; do sleep 60; [ $(( HARD_END - $(date +%s) )) -lt 600 ] && { echo "$(date -u +%T) $NAME: SKIPPED, no headroom before the hard end"; exit 0; }; done
  echo "$(date -u +%T) $NAME: waiting for the coding-agent queue lock"
  flock 9
  echo "$(date -u +%T) $NAME: holding the coding-agent queue lock"
  res_ok && break
  echo "$(date -u +%T) $NAME: headroom gone while waiting; releasing the flock"; flock -u 9; sleep 60
done

restore() {
  echo "$(date -u +%T) RESTORE: killing leftovers"
  [ -n "${WATCH:-}" ] && kill $WATCH 2>/dev/null
  [ -n "${LEASE:-}" ] && kill $LEASE 2>/dev/null
  pkill -f "$MEAS"; sleep 3; pkill -9 -f "$MEAS"
  for i in $(seq 1 30); do $LI_GPU_LEASE status 2>&1 | grep -q lucina-inference || break; sleep 1; done
  systemctl --user start llama-server-qwen
  echo "$(date -u +%T) RESTORE: started llama-server-qwen"
  for i in $(seq 1 450); do
    [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:<origin-port>/health)" = 200 ] && break; sleep 2
  done
  echo "$(date -u +%T) RESTORE: origin health $(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:<origin-port>/health)"
  local ok=0 R=""
  for i in $(seq 1 20); do
    KEY=$(cat ~/.config/llama-server-api.key)
    R=$(curl -s -m 180 http://127.0.0.1:<gateway-port>/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.8","messages":[{"role":"user","content":"What is 17+25? Answer with the number only."}],"max_tokens":16}')
    unset KEY
    if echo "$R" | grep -q '"content":"'; then ok=1; break; fi
    echo "$(date -u +%T) RESTORE: gateway not answering yet (try $i)"; sleep 30
  done
  if [ $ok = 1 ]; then
    echo "$(date -u +%T) RESTORE: gateway confirmed: $(echo "$R" | grep -o '"content":"[^"]*"' | head -1)"
  else
    echo "$(date -u +%T) RESTORE: ALERT gateway did not answer in 10 min; production NOT confirmed"
  fi
  $LI_GPU_LEASE status 2>&1 | head -3
  echo "$(date -u +%T) $NAME: releasing the coding-agent queue lock"
}
trap restore EXIT

rm -f $L/.go
( $LI_GPU_LEASE run --name lucina-inference --mb 21500 --wait 180 -- bash -c "while [ ! -f $L/.go ]; do sleep 0.2; done; DEADLINE=\$(cat $L/.go) L=$L exec bash $INNER" ) > $L/inner.log 2>&1 &
LEASE=$!
sleep 2
echo "$(date -u +%T) $NAME: stopping llama-server-qwen (clock starts)"
T0=$(date +%s); systemctl --user stop llama-server-qwen
DEADLINE=$(( T0 + MAX_S )); [ $DEADLINE -gt $HARD_END ] && DEADLINE=$HARD_END
echo $DEADLINE > $L/.go.tmp; mv $L/.go.tmp $L/.go
echo "$(date -u +%T) $NAME: deadline $(date -u -d @$DEADLINE +%T)"
# VRAM sampler + watchdog (card-wide, sysfs): over 24064 MiB (23.5 GB) kills the running measurement
( d=/sys/class/drm/card0/device; while true; do
    v=$(( $(cat $d/mem_info_vram_used) >> 20 )); g=$(( $(cat $d/mem_info_gtt_used) >> 20 ))
    echo "$(date +%s) $v $g $(sensors 2>/dev/null | awk '/Tctl/{gsub(/[+°C]/,"",$2); print $2}')"
    if [ $v -gt 24064 ]; then echo "$(date -u +%T) VRAM LIMIT $v MiB: killing the running measurement" >> $L/vram-limit.txt; pkill -f "$MEAS"; fi
    sleep 0.5; done ) > $L/vram.txt &
WATCH=$!
while kill -0 $LEASE 2>/dev/null && [ $(date +%s) -lt $(( DEADLINE + 5 )) ]; do sleep 1; done
echo "$(date -u +%T) $NAME: window over after $(( $(date +%s) - T0 )) s"
cat $L/inner.log
rm -f $L/.go
