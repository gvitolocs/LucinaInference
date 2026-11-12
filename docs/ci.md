# Continuous integration

Two GitHub Actions workflows, plus a set of **local merge gates** that are deliberately not in CI.

- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — "CI": build/lint/test/package/publish.
- [`.github/workflows/cron.yml`](../.github/workflows/cron.yml) — "Cron": the slow, deep checks.

**Triggers.** `ci.yml` runs on pushes to `main`, on tags `v*`, on every pull request, and on
manual dispatch. `cron.yml` runs weekly (Monday 06:00 UTC) and on manual dispatch.

**Runners.** Three hosted images, all used by the core matrix: `ubuntu-26.04`, `macos-15`,
`windows-2025`. `infr` builds on all three and needs `glslc` (the shaderc GLSL compiler) on every
runner, because `infr-vulkan`'s `build.rs` compiles the compute shaders at build time with no
prebuilt fallback; the per-platform install lives in [`.github/actions/install-glslc`](../.github/actions/install-glslc).

## What `ci.yml` runs

| Job | Runner(s) | What it does |
| --- | --- | --- |
| `fmt` | ubuntu-26.04 | `cargo fmt --all --check`. |
| `clippy` | ubuntu + macos + windows | `cargo clippy --workspace --all-targets --locked -D warnings`, `fail-fast: false` so one platform's lint failure never hides another's. |
| `docs` | ubuntu-26.04 | `cargo doc` with broken/private intra-doc links denied (`--keep-going` so the run is a total, not a partial). |
| `test` | ubuntu + macos + windows | `cargo nextest run` plus `cargo test --doc` across the workspace. **The CPU suite**: GPU tests are `#[ignore]`d and skipped (see below). |
| `cpu-goldens` | ubuntu-26.04 | Real-model CPU generation goldens. Downloads two small GGUFs into the Hugging Face cache and runs `infr-llama`'s `cpu_backend` tests in release, single-threaded, so the exact-token hashes actually assert. |
| `build` | ubuntu-26.04 | `cargo build --release --workspace` — a release-profile smoke build (debug is already covered by the matrix). |
| `metal-check` | ubuntu-26.04 | Cheap guard for the Apple target: cross-`clippy` to `aarch64-apple-darwin` (typecheck + lint, no link) including `infr-metal`, which the Linux jobs never build. |
| `test-macos` | macos-15 | Builds on an Apple-Silicon runner (a real Metal device) and runs the `infr-metal` parity suite with `--include-ignored`; `--nocapture` keeps the pipeline-cache measurement that is its only record. |
| `windows-smoke` | windows-2025 | `infr run` end to end on a real Windows filesystem: the HF cache location, a ranged pull, the blob hard-link fallback, a GGUF open, and one short CPU generation. |
| `dist` | ubuntu × 2, windows, macos × 2 | Release packaging for `x86_64/aarch64-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64/x86_64-apple-darwin`. A dry-run package on every `main` push; skipped on PRs. |
| `publish-github-release` | ubuntu-26.04 | Aggregate artifacts into a GitHub Release. **Tag-only** (`refs/tags/v*`). |
| `publish-crates` | ubuntu-26.04 | Publish the workspace crates to crates.io, topo order, idempotent. **Tag-only.** |
| `aur-bin` | ubuntu-26.04 | Push the `infr-bin` AUR package. **Tag-only.** |
| `brew-tap` | ubuntu-26.04 | Update the Homebrew formula. **Tag-only.** |
| `scoop-bucket` | ubuntu-26.04 | Update the Scoop manifest. **Tag-only.** |

## What `cron.yml` runs

| Job | Runner | What it does |
| --- | --- | --- |
| `deny` | ubuntu-26.04 | `cargo-deny` against a fresh advisory DB (licences, bans, sources, CVEs), pinned to the version `deny.toml` was written against. |
| `miri` | ubuntu-26.04 | The UB interpreter over `infr-cpu`'s `SpinPool`/`hostpager` and their raw-pointer plumbing — the part of the tree miri can actually execute (`-Zmiri-tree-borrows`, `-Zmiri-ignore-leaks`). |

These are weekly rather than per-push because both are slow and neither is a fast-feedback signal.

## Deliberately NOT run in CI

CI has no usable GPU, no model weights beyond two small goldens, and no access to the private
proof toolchain. These run **locally**:

- **Vulkan / discrete-GPU integration tests.** `ash` dlopens `libvulkan` at runtime and every GPU
  test is `#[ignore]`d, so CI runs the CPU suite. GPU work runs on the **build host's integrated
  GPU**, serially, in a GPU window, per [`AGENTS.md`](../AGENTS.md).
- **The llama.cpp parity oracle** ([`oracle/`](../oracle/)). Building `llama-logits`/`ggml-dequant`
  needs the pinned llama.cpp checkout and a GPU to compare against; it is run locally on the build
  host, not in CI.
- **Verus proofs** ([`verification/`](../verification/)). The `lucina-*` model-checked crates are
  verified **locally on the Verus host**, not in CI — GitHub has no Verus toolchain and the proofs
  are far too heavy for a per-push loop.
- **Real-model coverage beyond the CPU goldens.** `cpu-goldens` downloads only two small GGUFs. The
  multi-GB / MoE models are not fetched, and the FP-sensitive Qwen3.5 exact-token golden is not run
  there at all: its hash is not reproducible across CPU microarchitectures (same OS and ISA family,
  coherent output, different tokens), which is the whole reason it is pinned to one CPU and kept off
  the multi-platform matrix.

## Local merge gates

Some checks belong before a merge lands and cannot live in a public repository. They are declared
in [`.lucina/pipeline.toml`](../.lucina/pipeline.toml) — the same kind of file used to declare merge
gates across the Lucina projects — and run locally by the merge pipeline.

- **`no private strings in tracked files`** runs
  [`scripts/check-private-strings.sh --tree`](../scripts/check-private-strings.sh), which fails any
  staged change whose tracked content or (for a commit) message matches `LI_PRIVATE_PATTERNS`.

This gate is not, and cannot be, a CI job: its pattern lists host names, the home path, local ports
and private tool names, and that pattern is kept **out of the repository on purpose** (it lives only
in the gitignored `scripts/local.env`). A public runner therefore has nothing to check with. The
full procedure — how the private history becomes the public one under this and the secret gate — is
[docs/ci/PUBLIC-SYNC.md](ci/PUBLIC-SYNC.md).
