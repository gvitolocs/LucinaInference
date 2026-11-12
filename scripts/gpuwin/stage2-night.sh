#!/usr/bin/env bash
# Stage-2 night (owner-approved): three GPU windows from 01:00 UTC, each <= 38 min of measurement (+ ~1.5 min restore),
# >= 10 min of production Qwen between them, all measurement over by 03:40 UTC (everything restored by 03:45).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
[ -f scripts/local.env ] && . scripts/local.env
HARD=$(date -d "2026-11-08 03:40:00 UTC" +%s)
echo "$(date -u +%T) stage2 night start"
bash scripts/gpuwin/gpuwin.sh w1 scripts/gpuwin/w1-inner.sh 2280 $HARD
echo "$(date -u +%T) production gap"; sleep 600
bash scripts/gpuwin/gpuwin.sh w2 scripts/gpuwin/w2-inner.sh 2280 $HARD
echo "$(date -u +%T) production gap"; sleep 600
bash scripts/gpuwin/gpuwin.sh w3 scripts/gpuwin/w3-inner.sh 2280 $HARD
echo "$(date -u +%T) oracle regression check (self-gated: 20 GB free, load < 16, Tctl < 88, 3 threads)"
timeout 10800 bash scripts/gpuwin/check-oracle.sh > work/logs/check-oracle.log 2>&1
echo "$(date -u +%T) stage2 night done"
