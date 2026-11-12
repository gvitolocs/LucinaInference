#!/usr/bin/env bash
# Shows that verification/lucina-split/rules.rs can fail. Run from the repo root on the Verus host (see scripts/local.env.example):
#   verus-remote <repo> bash verification/lucina-split/mutants.sh
ENTRY=lucina-split
source verification/mutant-lib.sh
F=split_rules.rs
# A gap left out, a match dropped, the tail dropped, the cursor not advanced.
try $F 'if cur < s {' 'if cur + 1 < s {'
try $F '        pieces.push((s, e));' '        if e > s + 1 { pieces.push((s, e)); }'
try $F 'if cur < len {' 'if cur + 1 < len {'
try $F '        cur = e;' '        cur = s;'
# The run-time check accepting an empty piece, a gap or overlap, or a short cover.
try $F 'if p[i].0 >= p[i].1 {' 'if p[i].0 > p[i].1 {'
try $F 'if i + 1 < p.len() && p[i].1 != p[i + 1].0 {' 'if false {'
try $F 'if len == 0 || p[0].0 != 0 || p[p.len() - 1].1 != len {' 'if len == 0 || p[0].0 != 0 {'
finish
