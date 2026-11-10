# Sourced by the window inner scripts: production GPU env and a deadline-bounded step runner.
export MESA_VK_DEVICE_SELECT=1002:744c MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 RADV_PERFTEST=nogttspill
export GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 GGML_VK_ENABLE_MEMORY_PRIORITY=1
left() { echo $(( DEADLINE - $(date +%s) )); }
step() { # name, cmd...
  local n=$1; shift; local t; t=$(left)
  if [ "$t" -le 10 ]; then echo "$(date -u +%T) SKIP $n (window over)"; return; fi
  echo "$(date -u +%T) START $n (${t}s left)"
  local s0; s0=$(date +%s)
  timeout -k 5 "$t" "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date -u +%T) END $n exit $rc after $(( $(date +%s) - s0 )) s"
}
