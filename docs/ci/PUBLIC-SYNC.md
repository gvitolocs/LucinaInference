# Public sync

How the private `main` of this repository becomes the public `main`, and the gate that keeps
machine-private strings out of the public history (and out of the private history too).

This is a **local** procedure, run by hand on the build host. It is deliberately not a CI job:
its inputs are the private-string pattern and the scrub tool, and neither is in the repository,
so a public runner could not perform it. See [../ci.md](../ci.md) for what CI does run.

## The invariant

Some strings must never appear in a tracked file or a commit message: the build host's name, the
Verus host's name, the developer's home path, the local service ports, and the names of the
private queue/lease/merge tools. They live in exactly two places, both outside the repository:

- the gitignored `scripts/local.env`, as `LI_PRIVATE_PATTERNS` (a single extended regex listing
  every one of them) and the machine paths the sync reads;
- the scrub tool's own rule table (see below).

Because the regex is the only description of what is private, it stays in `local.env` and is
never a tracked literal — the very file that would name the strings is the one git ignores.
`scripts/local.env.example` carries an empty `LI_PRIVATE_PATTERNS=` placeholder and a comment.

## The gate: `scripts/check-private-strings.sh`

```
scripts/check-private-strings.sh            # the working tree's tracked files (default)
scripts/check-private-strings.sh <rev>      # tracked files at <rev> AND <rev>'s commit message
scripts/check-private-strings.sh --tree     # same as no argument
```

It greps the tracked content with `LI_PRIVATE_PATTERNS` and prints `file:line` for each hit.

- **exit 0** — nothing private found.
- **exit 1** — at least one private string; the offending `file:line` (and, for a `<rev>`, the
  matching message line) are listed on stderr.
- **exit 2** — `LI_PRIVATE_PATTERNS` is unset or empty. **The gate never passes silently:** a
  missing pattern is a hard stop, so an unconfigured machine cannot merge or sync green.

The pattern is read from `scripts/local.env`; in a git worktree that file lives in the primary
checkout, which the script resolves through the git common dir. Set `LI_PRIVATE_REPO=<path>` to
scan a *different* repository (the sync tool points it at the scrubbed clone to check the commits
it re-makes there); the pattern is always taken from this tree, never the scanned one.

This gate is wired into the local merge pipeline as a declared gate in
[`.lucina/pipeline.toml`](../../.lucina/pipeline.toml), the same kind of file LucinaOS uses to
declare its merge gates. Its `run` is `scripts/check-private-strings.sh --tree` with `paths = []`
(always runs), so any staged merge that would put a private string in a tracked file is refused
before it lands.

## The scrub tool (not in the repository)

Re-making the public history needs a tool that knows every private string and its replacement.
That table is the sensitive part, so the scrub binary and its source are kept **outside** the
repository — under the gitignored `work/` tree (`work/public-sync/`). It is referenced only by
path, through `LI_PUBLIC_SYNC_TOOL`. The repository never contains it, so the repository can never
leak it.

## The clone, the branch, and the mapping file

The public copy lives in a **scrubbed clone** — a separate checkout whose `origin` is this
repository and whose remote named by `LI_PUBLIC_REMOTE` is the public repository. The clone carries
a `public-main` branch: the first-parent chain of scrubbed commits that mirrors `main`.

`scripts/public-sync.sh` records progress in a **mapping file**, `<clone>/.git/li-public-sync`,
which is not a tracked file and not part of any commit. It stores:

- `last_source` — the newest source `main` commit already reflected in `public-main`;
- `last_public` — the `public-main` commit that reflects it;
- `base` — the **fork commit**: the newest commit shared by `main` and `public-main`, i.e. the
  clean history before the two diverged. Everything from `base` back is already public.

On the first run it creates the file from the clone's current baseline, adopting the present
`public-main` as the sync of the present `main` — but only after checking their trees agree, so a
first run never silently blesses an un-synced clone.

## The procedure: `scripts/public-sync.sh [--push]`

Inputs come from `scripts/local.env`: `LI_PUBLIC_CLONE`, `LI_PUBLIC_SYNC_TOOL`,
`LI_PUBLIC_REMOTE`.

1. Confirm the clone is clean, has `public-main`, and its `origin` tracks the source `main`; fetch
   the latest source `main` into the clone.
2. Read the mapping file (create it as above if absent); assert the fork `base` has not moved.
3. Compute `git rev-list --reverse --first-parent <last_source>..main`. **If empty → "nothing to
   sync" and stop** (this is the everyday dry run, and it touches nothing on the remote).
4. For the new commits, re-make `public-main` with the scrub tool **exactly as the first sync did**:
   same author, dates and message per commit, each tree scrubbed, and each merge collapsed to its
   single scrubbed mainline parent. The tool re-derives the whole `base..main` chain from the fork
   base, which is deterministic — commits already on `public-main` reproduce byte-for-byte and only
   the new ones append — so `public-main` stays a clean fast-forward.
5. **Gate 1** — run `check-private-strings.sh <rev>` on **every** re-made commit, in the clone.
6. **Gate 2** — run `gitleaks` over the new range (`<tip-before>..public-main`).
7. Only after both gates pass is the mapping file advanced to the new `source ↔ public` pair.
8. **Stop.** Without `--push` nothing leaves the machine: `public-main` holds the new scrubbed
   commits, ready for review, and the script prints the `--push` command to publish them.

`--push` publishes **only** `public-main → the remote's `main`** (a normal fast-forward; it never
force-pushes and never pushes any other ref). If either gate fails, or the scrub tool refuses, the
run stops with a non-zero exit, the mapping is not advanced, and no push happens.

A note on scratch-directory clones: if the clone was set up under a temporary, OS-wiped scratch
directory it can disappear between sessions. Move it somewhere durable — `work/public-sync/clone`
is the suggested home — and update `LI_PUBLIC_CLONE` to the new path. The mapping file travels with
the clone (it lives in the clone's `.git`), and `base`/`last_source`/`last_public` are plain object
IDs, so a move needs no re-sync.

## What it checks, in one line

That every commit it is about to publish carries no `LI_PRIVATE_PATTERNS` match in any tracked file
or message (gate 1) and no secret per `gitleaks` (gate 2), and that the public branch is a
one-to-one, scrubbed, first-parent-only mirror of source `main`.

## Recovery

The dangerous window is between step 4 and step 7: the scrub tool updates `public-main` as it
goes, so a gate failure (step 5/6) or a tool refusal (step 4) can leave `public-main` ahead of the
mapping and never pushed. The mapping and the remote are then the source of truth, not the branch.

- **Nothing was pushed** (the usual case — a failed gate stops before step 8). Reset the branch to
  the last recorded sync and re-run:
  ```
  git -C "$LI_PUBLIC_CLONE" update-ref refs/heads/public-main <last_public-from-mapping>
  scripts/public-sync.sh        # re-do the re-make and both gates
  ```
- **A leaked string is already on the public remote.** Do not paper over it with another commit:
  the string is in public history and must be removed by re-making `public-main` from `base` with a
  fixed scrub tool, then force-updating the remote's `main` to the corrected tip. Fix the tool or
  extend `LI_PRIVATE_PATTERNS` first so the gate would now catch it, then re-run. Treat this as a
  rotation event: any value that leaked is public from then on.
- **The gate exits 2.** `scripts/local.env` is missing or lacks `LI_PRIVATE_PATTERNS` on this
  machine — set it (copy from the primary checkout or `local.env.example`) before merging or
  syncing. Exit 2 means "unconfigured", not "clean".
