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
scripts/check-private-strings.sh                 # the working tree's tracked files (default)
scripts/check-private-strings.sh <rev>           # tracked files at <rev> AND <rev>'s commit message
scripts/check-private-strings.sh --tree          # same as no argument
scripts/check-private-strings.sh --public <rev>  # the <rev> check, and refuse a ledger trailer
```

It greps the tracked content with `LI_PRIVATE_PATTERNS` and prints `file:line` for each hit.

- **exit 0** — nothing private found.
- **exit 1** — at least one private string, or (with `--public`) a ledger trailer; the offending
  `file:line` (and, for a `<rev>`, the matching message line) are listed on stderr.
- **exit 2** — `LI_PRIVATE_PATTERNS` is unset or empty, or the arguments do not make a scan.
  **The gate never passes silently:** a missing pattern is a hard stop, so an unconfigured machine
  cannot merge or sync green.

The pattern is read from `scripts/local.env`; in a git worktree that file lives in the primary
checkout, which the script resolves through the git common dir. Set `LI_PRIVATE_REPO=<path>` to
scan a *different* repository (the sync tool points it at the scrubbed clone to check the commits
it re-makes there); the pattern is always taken from this tree, never the scanned one.

This gate is wired into the local merge pipeline as a declared gate in
[`.lucina/pipeline.toml`](../../.lucina/pipeline.toml), the same kind of file LucinaOS uses to
declare its merge gates. Its `run` is `scripts/check-private-strings.sh --tree` with `paths = []`
(always runs), so any staged merge that would put a private string in a tracked file is refused
before it lands.

### Public mode: no ledger trailers

The private `main` **keeps** the ledger trailers: every worker commit carries the `Lead:`, `Run:`
and `Change:` lines the ledger attributes the work with, along with the lane's own `Lane-Run:` and
`Written-By:` stamps and any `Refs:`/`Fixes:` lines. A *public* commit must carry none of them. The
scrub tool drops those lines when it re-makes a commit; `--public` is the check that they went —
the second line of defence, because a tool that stopped dropping them would otherwise publish the
ledger into the public history.

With `--public <rev>` the gate refuses any message line starting with one of those keys (leading
whitespace is still a hit; a longer key that only begins with one of them — `Change-Id:`, `Runs:`,
`References:` — is not), lists each offending line, and exits 1 as any other finding does. The
trailer rule reads the message only, never file content, and the `<rev>` scan of tracked files it
comes with is exactly the one default mode already did. So default mode is unchanged and the merge
gate above (`--tree`, and a `<rev>` scan of the private history) keeps accepting the trailers where
they belong.

`crates/infr-testkit/tests/private_strings_gate.rs` is the test of both halves — an integration test
in the crate that already exists to be shared test support, so it drives the real shell gate through
`std::process::Command` rather than re-implementing its regexes. It builds throwaway git repositories
under a temporary directory that is cleaned up even when a case fails, hands the gate an invented
pattern of its own through `LI_PRIVATE_PATTERNS` rather than ever reading `local.env`, and checks
that a message with each refused trailer key fails in public mode and passes in default mode, that
look-alike keys such as `References:` are not refused, and that a clean message passes both. From
the repository root:

```
cargo test -p infr-testkit --test private_strings_gate
```

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
`LI_PUBLIC_REMOTE`, and `LI_PUBLIC_CHECK` (the third check below, needed only to push).

1. Confirm the clone is clean, has `public-main`, and its `origin` tracks the source `main`; fetch
   the latest source `main` into the clone.
2. Read the mapping file (create it as above if absent); assert the fork `base` has not moved.
3. Compute `git rev-list --reverse --first-parent <last_source>..main`. **If empty → "nothing to
   sync" and stop** (this is the everyday dry run, and it touches nothing on the remote).
4. For the new commits, re-make `public-main` with the scrub tool **exactly as the first sync did**:
   same author, dates and message per commit, each tree scrubbed (the ledger trailers included, which
   gate 1 then double-checks), and each merge collapsed to its single scrubbed mainline parent. The
   tool re-derives the whole `base..main` chain from the fork base, which is deterministic — commits
   already on `public-main` reproduce byte-for-byte and only the new ones append — so `public-main`
   stays a clean fast-forward. Before it runs, a `public-main` left ahead of `last_public` by a run
   that stopped is reset back to `last_public` (Recovery below).
5. **Gate 1** — run `check-private-strings.sh --public <rev>` on **every** re-made commit, in the
   clone: no private string in its files or message, and no ledger trailer left in its message.
6. **Gate 2** — run `gitleaks` over the new range (`<tip-before>..public-main`). The ignore file is
   the one at **this** repository's root: `--gitleaks-ignore-path=<this repo>/.gitleaksignore` when
   `.gitleaksignore` exists, because the scrubbed clone has no copy of it in its tree until a sync
   carries one over, and the run must not lose the known false positives in the meantime. With no
   such file the flag is simply not passed. Only the lead adds entries to it.
7. Only after both gates pass is the mapping file advanced to the new `source ↔ public` pair.
8. **Stop.** Without `--push` nothing leaves the machine: `public-main` holds the new scrubbed
   commits, ready for review, and the script prints the `--push` command to publish them.

9. **Gate 3** — only on the `--push` path, immediately before the push: the public-trailer check
   over the exact range about to be published,

   ```
   <check-tool> public-check <clone> <remote>/main..public-main
   ```

   where `<remote>` is `LI_PUBLIC_REMOTE` as named inside the clone, and `<check-tool>` is
   `LI_PUBLIC_CHECK` — a machine-local binary kept outside the repository like the scrub tool, so
   the repository never carries its path or its name. Exit `0` is clean. Exit `1` lists each commit
   in the range that still carries a `Lead:`/`Run:`/`Change:`/`Refs:`/`Fixes:`/`Lane-Run:` trailer
   or a lane-writer `Co-Authored-By`. Exit `2` means the range itself is bad. Any non-zero exit
   prints the tool's output and **refuses the push**; so does `--push` with `LI_PUBLIC_CHECK`
   unset, or pointing at something that is not executable. The clone is fetched from the remote
   first, so the range is measured against the remote's real `main`, and a remote-tracking ref the
   clone cannot see stops the push too rather than checking a made-up range.

`--push` publishes **only** `public-main → the remote's `main`** (a normal fast-forward; it never
force-pushes and never pushes any other ref). If gate 1 or gate 2 fails, or the scrub tool refuses,
the run stops with a non-zero exit, the mapping is not advanced, and no push happens. Gate 3 runs
after step 7, so a refusal there is the one case that leaves the mapping advanced with nothing
published — which is safe: `public-main` still holds the commits, and a later `--push` re-checks
all three over the range the remote actually still lacks.

A note on scratch-directory clones: if the clone was set up under a temporary, OS-wiped scratch
directory it can disappear between sessions. Move it somewhere durable — `work/public-sync/clone`
is the suggested home — and update `LI_PUBLIC_CLONE` to the new path. The mapping file travels with
the clone (it lives in the clone's `.git`), and `base`/`last_source`/`last_public` are plain object
IDs, so a move needs no re-sync.

## What it checks, in one line

That every commit it is about to publish carries no `LI_PRIVATE_PATTERNS` match in any tracked file
or message and no ledger trailer in its message (gate 1, public mode), and no secret per `gitleaks`
(gate 2, over this repository's ignore file), and — before a push — that the range it is about to
publish is trailer-clean per the public-trailer check (gate 3), and that the public branch is a
one-to-one, scrubbed, first-parent-only mirror of source `main`.

## Recovery

The dangerous window is between step 4 and step 7: the scrub tool updates `public-main` as it
goes, so a gate failure (step 5/6) or a tool refusal (step 4) can leave `public-main` ahead of the
mapping and never pushed. The mapping and the remote are then the source of truth, not the branch.

Exactly what such a run leaves behind, in order:

| stopped at | `public-main` | mapping file | remote |
| --- | --- | --- | --- |
| the sync tool refusing (step 4) | moved as far as the tool got | unchanged | untouched |
| gate 1 (step 5) | fully re-made, ahead of `last_public` | unchanged | untouched |
| gate 2 (step 6) | fully re-made, ahead of `last_public` | unchanged | untouched |
| after both gates (step 7/8) | fully re-made | advanced with them | untouched unless `--push` succeeded |
| gate 3 (step 9, `--push` only) | fully re-made | already advanced | untouched — the push refused |

So the branch can be ahead of the mapping while nothing was published, and the mapping is what the
next run reads. **The next run now repairs that itself:** before re-making it compares
`public-main` with `last_public`, and if `last_public` is an ancestor of the branch tip — the branch
is *ahead*, which is what a stopped run produces — it resets `public-main` back to `last_public`
with a plain `update-ref` in the clone, then re-makes the whole chain from `base` as usual. That
move is local to the clone and pushes nothing: `public-sync.sh` only ever writes the remote with
`--push`, and only ever `public-main → main`. Because the reset is to a recorded commit rather than
"whatever the tool just did", a stopped run is re-runnable as many times as needed and lands in the
same place each time.

If `public-main` is not ahead of `last_public` but *diverged* from it — `last_public` is not an
ancestor of the branch tip, so the tip carries commits the mapping has never recorded — the reset
would discard commits the sync did not make, and the run refuses instead and leaves the branch
alone. Resolve that by hand:

- **Nothing was pushed** (the usual case — a failed gate stops before step 8). Nothing to do but fix
  the cause and re-run; the reset above puts `public-main` back on its own:
  ```
  scripts/public-sync.sh        # resets if needed, re-does the re-make and both gates
  ```
  To do it by hand instead, or to see what the script would reset:
  ```
  git -C "$LI_PUBLIC_CLONE" update-ref refs/heads/public-main <last_public-from-mapping>
  scripts/public-sync.sh
  ```
- **A leaked string is already on the public remote.** Do not paper over it with another commit:
  the string is in public history and must be removed by re-making `public-main` from `base` with a
  fixed scrub tool, then force-updating the remote's `main` to the corrected tip. Fix the tool or
  extend `LI_PRIVATE_PATTERNS` first so the gate would now catch it, then re-run. Treat this as a
  rotation event: any value that leaked is public from then on.
- **The gate exits 2.** `scripts/local.env` is missing or lacks `LI_PRIVATE_PATTERNS` on this
  machine — set it (copy from the primary checkout or `local.env.example`) before merging or
  syncing. Exit 2 means "unconfigured", not "clean".
