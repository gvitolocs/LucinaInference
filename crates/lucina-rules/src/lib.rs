//! Pure rules proved by Verus in the shipped files (see verification/lucina-*/rules.rs). No dependencies.
#![allow(unexpected_cfgs)]
// The modules named below are not a Rust-language crate so much as the input to six Verus proofs:
// `verification/*/rules.rs` include each file by `#[path]`, and `verification/*/mutants.sh` plant a
// bug by matching a FROM string against these files VERBATIM (see `try` in `verification/mutant-lib.sh`).
// So the proved code is not ours to restyle, and CI's two restylers must be told to keep out:
//
// * `#[rustfmt::skip]` on each `pub mod`: rustfmt re-wraps the long `assert_eq!` lines and reorders
//   module declarations, and a re-wrapped line is a mutant FROM string that no longer matches — the
//   proof still passes while the mutation harness silently stops finding its bugs.
// * the lints below: each is a rewrite that would change verified code. `question_mark` wants `?`
//   on `Option` in the verified `fn`s (they return `Option`, so `?` is not available to them),
//   `manual_is_multiple_of` wants `x.is_multiple_of(n)` where the specs state `x % n == 0`, and
//   `len_zero` wants `s.is_empty()` where the specs state `s.len() == 0`. The proof is of the text
//   as written; matching the spec's own spelling is the point.
#![allow(clippy::question_mark, clippy::manual_is_multiple_of, clippy::len_zero)]
#[rustfmt::skip]
pub mod gguf_bounds;
#[rustfmt::skip]
pub mod kv_rules;
#[rustfmt::skip]
pub mod split_rules;
#[rustfmt::skip]
pub mod gate_rules;
#[rustfmt::skip]
pub mod spec_rules;
#[rustfmt::skip]
pub mod dequant_index;
