#!/usr/bin/env bash
# Shows that verification/lucina-kv/rules.rs can fail. Run from the repo root on the Verus host (see scripts/local.env.example):
#   verus-remote <repo> bash verification/lucina-kv/mutants.sh
ENTRY=lucina-kv
source verification/mutant-lib.sh
F=kv_rules.rs
# Wrong block format.
try $F 'pub const Q4_0_BLOCK_BYTES: u64 = 18;' 'pub const Q4_0_BLOCK_BYTES: u64 = 16;'
try $F 'pub const Q4_0_BLOCK_ELEMS: u64 = 32;' 'pub const Q4_0_BLOCK_ELEMS: u64 = 64;'
# Partial blocks or a head dim that is not whole blocks, accepted.
try $F 'if elems % Q4_0_BLOCK_ELEMS != 0 {' 'if false {'
try $F 'if head_dim % Q4_0_BLOCK_ELEMS != 0 {' 'if false {'
# K without V.
try $F 'let a = match mul_u64(n_attn_layers, 2) {' 'let a = match mul_u64(n_attn_layers, 1) {'
# A position past the cache, or a ring row past it.
try $F '} else if pos < rows {' '} else if pos <= rows {'
try $F 'Some(pos % rows)' 'Some(pos)'
# A batch that overruns the context.
try $F 'p0 <= ctx && n <= ctx - p0' 'p0 <= ctx'
# The ring lemma with a ring one row too small.
try $F '        b - a < r,' '        b - a <= r,'
finish
