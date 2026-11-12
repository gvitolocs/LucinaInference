#!/usr/bin/env bash
# Merge gate: fail if a tracked file — or a commit message — carries a private string
# (host names, the home path, service ports, local tool names). The regex lives only in the
# gitignored scripts/local.env as LI_PRIVATE_PATTERNS and must never be a tracked literal;
# this script is that gate. Procedure: docs/ci/PUBLIC-SYNC.md.
#
# usage:
#   scripts/check-private-strings.sh [--tree]     # the working tree's tracked files (default)
#   scripts/check-private-strings.sh <rev>        # tracked files at <rev> AND <rev>'s message
#
# exit: 0 clean · 1 a private string was found (file:line listed) · 2 LI_PRIVATE_PATTERNS unset
#
# Set LI_PRIVATE_REPO to scan a repository other than this one (public-sync.sh points it at
# the scrubbed clone to check the commits it re-makes there). The pattern is always read from
# this tree's local.env, never the scanned repo's.
set -u

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo=${LI_PRIVATE_REPO:-$root}

# --- pattern ------------------------------------------------------------------
# Honour an already-set environment value verbatim (even if empty): a caller passing
# LI_PRIVATE_PATTERNS= is asserting "unset", and must get exit 2, not a silent load from
# local.env. Only when the variable is genuinely absent do we source it. In a git worktree
# scripts/local.env lives in the primary checkout, so fall back to the common dir.
if [ -z "${LI_PRIVATE_PATTERNS+set}" ]; then
  candidates=("$root/scripts/local.env")
  if common_dir=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null); then
    candidates+=("$(cd "$common_dir/.." && pwd)/scripts/local.env")
  fi
  for c in "${candidates[@]}"; do
    # shellcheck source=/dev/null disable=SC1090
    [ -f "$c" ] && { . "$c"; break; }
  done
fi

if [ -z "${LI_PRIVATE_PATTERNS:-}" ]; then
  cat >&2 <<'EOF'
check-private-strings: LI_PRIVATE_PATTERNS is unset or empty.
  It must be set to a single extended regex (grep/git grep -E) matching every private
  string — host names, the home path, service ports, local tool names. Define it in
  scripts/local.env (copy the placeholder from scripts/local.env.example). This gate
  never passes silently, so a missing pattern is a hard stop, not a green check.
EOF
  exit 2
fi

# --- what to scan -------------------------------------------------------------
mode=tree; rev=""
case "${1:-}" in
  ""|--tree) ;;
  -*) echo "check-private-strings: unknown option $1 (usage: [--tree | <rev>])" >&2; exit 2 ;;
  *) mode=rev; rev=$1 ;;
esac
if [ "$mode" = rev ] && ! git -C "$repo" rev-parse --verify -q "${rev}^{commit}" >/dev/null; then
  echo "check-private-strings: '$rev' is not a commit in $repo" >&2; exit 2
fi

matches=""
note() { [ -n "$matches" ] && matches+="
$1" || matches=$1; }

grep_args=(-nIE "$LI_PRIVATE_PATTERNS")
[ "$mode" = rev ] && grep_args+=("$rev")
grep_args+=(--)

scan=$(git -C "$repo" grep "${grep_args[@]}" 2>/dev/null); rc=$?
case $rc in
  0) while IFS= read -r line; do [ -n "$line" ] && note "$line"; done <<<"$scan" ;;
  1) : ;;                     # git grep: no match — clean
  *) echo "check-private-strings: git grep failed (exit $rc) on ${rev:-working tree}" >&2; exit 1 ;;
esac

if [ "$mode" = rev ]; then
  msg=$(git -C "$repo" log -1 --format=%B "$rev" 2>/dev/null | grep -nE "$LI_PRIVATE_PATTERNS"); mrc=$?
  case $mrc in
    0) while IFS= read -r line; do [ -n "$line" ] && note "commit ${rev} message ${line}"; done <<<"$msg" ;;
    1) : ;;
    *) echo "check-private-strings: message scan failed on $rev" >&2; exit 1 ;;
  esac
fi

if [ -n "$matches" ]; then
  echo "check-private-strings: PRIVATE STRINGS found in ${rev:-the working tree}:" >&2
  printf '%s\n' "$matches" >&2
  exit 1
fi
exit 0
