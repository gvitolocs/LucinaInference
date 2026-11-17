#!/usr/bin/env bash
# Re-make the public copy of `main` as one scrubbed commit per source commit.
# docs/ci/PUBLIC-SYNC.md holds the procedure and recovery.
#
# It reads the last synced source commit from a mapping file in the clone's .git (creating it
# from the clone's current baseline the first time), and for each new first-parent commit of
#   git rev-list --reverse --first-parent <last>..main
# it re-makes a commit on the clone's `public-main` with the external sync tool — same author,
# dates and message, a scrubbed tree, merges collapsed to their single scrubbed mainline
# parent, exactly as the first sync did. It then runs the private-string gate in PUBLIC mode on
# every re-made commit — a public commit may carry neither a private string nor one of the ledger
# trailers the private main keeps — and gitleaks over the new range, with this repository's
# .gitleaksignore as the ignore file (the scrubbed clone has no copy of its own yet), and STOPS.
# It pushes only with --push, and then only `public-main` -> the remote's `main`.
#
# A gate that stops does so after the re-make: public-main has moved while the mapping file has
# not. The next run notices that and resets public-main back to the mapping's last_public before
# re-making, locally — it never touches the remote.
#
# usage: scripts/public-sync.sh [--push]
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prog=public-sync
die() { echo "$prog: $*" >&2; exit "${2:-2}"; }
info() { echo "$prog: $*"; }

do_push=0
for a in "$@"; do
  case "$a" in
    --push) do_push=1 ;;
    -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "usage: $prog [--push]" ;;
  esac
done

# --- config (scripts/local.env; in a worktree it lives in the primary checkout) -----------
# Caller-provided values win: only read the env file when none of the three is already set,
# so a test or a wrapper can point the sync at another clone without editing local.env.
if [ -z "${LI_PUBLIC_CLONE:-}" ] && [ -z "${LI_PUBLIC_SYNC_TOOL:-}" ] && [ -z "${LI_PUBLIC_REMOTE:-}" ]; then
  envf=$root/scripts/local.env
  if [ ! -f "$envf" ] && common=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null); then
    envf=$(cd "$common/.." && pwd)/scripts/local.env
  fi
  # shellcheck source=/dev/null disable=SC1090
  [ -f "$envf" ] && . "$envf"
fi
need() { [ -n "${!1:-}" ] || die "$1 is unset; set it in scripts/local.env (see scripts/local.env.example)"; }
need LI_PUBLIC_CLONE; need LI_PUBLIC_SYNC_TOOL; need LI_PUBLIC_REMOTE
clone=$LI_PUBLIC_CLONE; tool=$LI_PUBLIC_SYNC_TOOL; remote=$LI_PUBLIC_REMOTE
source_ref=refs/remotes/origin/main   # the clone's 'origin' is the source repository

[ -d "$clone/.git" ] || die "'$clone' is not a git clone (LI_PUBLIC_CLONE)"
[ -x "$tool" ] || die "the sync tool '$tool' is missing or not executable (LI_PUBLIC_SYNC_TOOL)"
[ -z "$(git -C "$clone" status --porcelain)" ] || die "'$clone' has uncommitted changes"
git -C "$clone" rev-parse --verify -q refs/heads/public-main >/dev/null \
  || die "'$clone' has no 'public-main' branch to append to"

info "fetching the latest source main into the clone"
git -C "$clone" fetch --quiet origin main
git -C "$clone" rev-parse --verify -q "$source_ref" >/dev/null \
  || die "'$clone' has no $source_ref — its 'origin' must be the source repository"
src=$(git -C "$clone" rev-parse "$source_ref")

# --- mapping file: the last synced source <-> public pair ---------------------------------
map=$clone/.git/li-public-sync
if [ ! -f "$map" ]; then
  pub=$(git -C "$clone" rev-parse refs/heads/public-main)
  base=$(git -C "$clone" merge-base "$source_ref" refs/heads/public-main) \
    || die "cannot find a fork base between $source_ref and public-main in the clone"
  # Adopt the current clone as the synced baseline only when its trees already agree, so a
  # first run never silently blesses an un-synced or out-of-tree public-main.
  [ "$(git -C "$clone" rev-parse "$source_ref^{tree}")" = "$(git -C "$clone" rev-parse 'refs/heads/public-main^{tree}')" ] \
    || die "no mapping file and public-main's tree does not match source main's; establish the baseline by hand (docs/ci/PUBLIC-SYNC.md)"
  printf '# Last synced source <-> public pair for scripts/public-sync.sh (docs/ci/PUBLIC-SYNC.md).\n# base is the fork commit the scrub rebuild re-makes from. Regenerated on each sync; never committed.\nbase=%s\nlast_source=%s\nlast_public=%s\n' "$base" "$src" "$pub" >"$map"
  info "created $map from the current synced baseline"
fi
base=$(sed -n 's/^base=//p' "$map")
last_source=$(sed -n 's/^last_source=//p' "$map")
last_public=$(sed -n 's/^last_public=//p' "$map")

# The fork base must not move: everything before it is shared, clean history.
mb=$(git -C "$clone" merge-base "$source_ref" refs/heads/public-main)
[ "$mb" = "$base" ] || die "the public/source fork base moved ($base -> $mb); refusing to sync — investigate first"

# --- is there anything new to re-make? ----------------------------------------------------
# The tool re-derives the whole base..main chain from the fork base: already-synced commits
# reproduce byte-for-byte (deterministic commit objects), new ones append, and a merge is
# re-made against its single scrubbed mainline parent — exactly as the first sync did.
new_src=$(git -C "$clone" rev-list --first-parent "$last_source..$source_ref")
tip_before=$(git -C "$clone" rev-parse refs/heads/public-main)
did_remake=0
if [ -n "$new_src" ]; then
  # A previous run can stop after the re-make and before the mapping is advanced (a gate refused,
  # the tool refused mid-chain), leaving public-main ahead of last_public. Start from the recorded
  # baseline instead of building on that half-synced chain: a local update-ref, nothing pushed.
  if [ "$tip_before" != "$last_public" ]; then
    git -C "$clone" rev-parse --verify -q "${last_public}^{commit}" >/dev/null \
      || die "the mapping's last_public ($last_public) is not a commit in the clone; fix $map by hand" 1
    if git -C "$clone" merge-base --is-ancestor "$last_public" "$tip_before"; then
      info "public-main is ahead of the mapping's last_public — a previous run stopped after its re-make; resetting it to $(git -C "$clone" rev-parse --short "$last_public")"
      git -C "$clone" update-ref refs/heads/public-main "$last_public"
      tip_before=$last_public
    else
      die "public-main ($tip_before) has diverged from the mapping's last_public ($last_public) — not resetting it; see docs/ci/PUBLIC-SYNC.md to resolve by hand" 1
    fi
  fi
  info "re-making the scrubbed chain from base $(git -C "$clone" rev-parse --short "$base") onto public-main"
  if ! "$tool" rebuild --repo "$clone" --ref refs/heads/public-main --base "$base" -- --first-parent "$base..$source_ref"; then
    die "the sync tool refused (a residual private string, or a blob it could not scrub). public-main may have moved to $(git -C "$clone" rev-parse --short refs/heads/public-main); see docs/ci/PUBLIC-SYNC.md to reset it" 1
  fi
  did_remake=1
elif [ "$do_push" != 1 ]; then
  info "nothing to sync — public-main already reflects $source_ref ($(git -C "$clone" rev-parse --short "$src"))"
  exit 0
fi

# --- the commits to verify before any push ------------------------------------------------
# A re-make verifies the commits it just made. When nothing was re-made but we are asked to
# push, verify whatever is on public-main but not yet on the remote (all of it, if the remote
# has no main we can see).
if [ "$did_remake" = 1 ]; then
  verify_range="$tip_before..refs/heads/public-main"
else
  git -C "$clone" fetch --quiet "$remote" 2>/dev/null \
    || info "note: could not fetch '$remote' — verifying every scrubbed commit since the fork base"
  remote_tip=$(git -C "$clone" rev-parse --verify -q "refs/remotes/$remote/main" || true)
  verify_range="${remote_tip:-$base}..refs/heads/public-main"
fi
nverify=$(git -C "$clone" rev-list --first-parent --count "$verify_range")
info "verifying $nverify commit(s) in $verify_range"

# gate 1: the private-string check, in public mode, on every commit in the range — so a ledger
# trailer the scrub tool failed to drop is refused here too, not only by the tool
gate=$root/scripts/check-private-strings.sh
[ -x "$gate" ] || die "the private-string gate '$gate' is missing or not executable"
while IFS= read -r c; do
  [ -n "$c" ] || continue
  if ! LI_PRIVATE_REPO="$clone" "$gate" --public "$c"; then
    die "public commit $(git -C "$clone" rev-parse --short "$c") still carries a private string or a ledger trailer — NOT pushing" 1
  fi
done < <(git -C "$clone" rev-list --first-parent "$verify_range")
info "private-string gate passed on all $nverify commit(s)"

# gate 2: gitleaks over the same range. The ignore file comes from THIS repository's root: the
# scrubbed clone's own tree only gets its copy on a later sync, so its root is not the place to
# look for it. `--gitleaks-ignore-path` needs the file to exist, so it is passed only when it does.
gitleaks_bin=$(command -v gitleaks 2>/dev/null || true)
[ -n "$gitleaks_bin" ] || gitleaks_bin=$HOME/.local/bin/gitleaks
[ -x "$gitleaks_bin" ] || die "gitleaks not found (on PATH or \$HOME/.local/bin) — refusing to finish without the secret scan"
gl_args=(--no-banner --redact --log-opts="$verify_range")
if [ -f "$root/.gitleaksignore" ]; then
  gl_args+=(--gitleaks-ignore-path="$root/.gitleaksignore")
  info "gitleaks: ignoring the fingerprints listed in this repository's .gitleaksignore"
fi
if ! "$gitleaks_bin" git "${gl_args[@]}" "$clone"; then
  die "gitleaks reported a finding in $verify_range — NOT pushing" 1
fi

# --- record the new source <-> public baseline after a re-make ----------------------------
new_pub=$(git -C "$clone" rev-parse refs/heads/public-main)
if [ "$did_remake" = 1 ]; then
  tmp=$map.tmp.$$
  printf '# Last synced source <-> public pair for scripts/public-sync.sh (docs/ci/PUBLIC-SYNC.md).\n# base is the fork commit the scrub rebuild re-makes from. Regenerated on each sync; never committed.\nbase=%s\nlast_source=%s\nlast_public=%s\n' "$base" "$src" "$new_pub" >"$tmp"
  mv -f "$tmp" "$map"
fi

# --- push only when asked, and only public-main -> the remote's main ----------------------
if [ "$do_push" != 1 ]; then
  info "re-made $(git -C "$clone" rev-parse --short "$new_pub") on public-main and passed both gates — not pushed."
  info "review it, then publish with:  $prog --push"
  exit 0
fi
info "pushing public-main -> $remote/main"
git -C "$clone" push "$remote" refs/heads/public-main:refs/heads/main
info "pushed $(git -C "$clone" rev-parse --short "$new_pub") to $remote/main"
