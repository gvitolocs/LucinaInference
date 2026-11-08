#!/usr/bin/env bash
# Stage-2 night 2 (2026-11-09): three GPU windows from 01:00 UTC, each <= 38 min of measurement, >= 10 min of production
# Qwen between them, all measurement over by 03:40 UTC. Each window log is checked against the proved protocol after.
cd ~/Projects/Lucina/LucinaInference
HARD=$(date -d "2026-11-09 03:40:00 UTC" +%s)
echo "$(date -u +%T) night 2 start"
for w in n2a n2b n2c; do
  bash scripts/gpuwin/gpuwin.sh $w scripts/gpuwin/$w-inner.sh 2280 $HARD > work/logs/$w.driver.log 2>&1
  cat work/logs/$w.driver.log
  target/release/lucina-gate-check work/logs/$w.driver.log
  [ $w != n2c ] && { echo "$(date -u +%T) production gap"; sleep 600; }
done
echo "$(date -u +%T) night 2 done"
