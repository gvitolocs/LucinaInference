#!/usr/bin/env bash
# Shows that verification/lucina-spec/rules.rs can fail. Run from the repo root on verus-host:
#   verus-remote <repo> bash verification/lucina-spec/mutants.sh
ENTRY=lucina-spec
source verification/mutant-lib.sh
F=spec_rules.rs
# Accept: one draft too many, the wrong next token, stop comparing early.
try $F "    (i, verify[i])" "    (i, verify[0])"
try $F "    while i < cand.len() && cand[i] == verify[i] {" "    while i < cand.len() {"
try $F "    while i < cand.len() && cand[i] == verify[i] {" "    while i < cand.len() && i < 1 && cand[i] == verify[i] {"
# Cycle: commit the wrong count, snapshot after a rejection, snapshot at the wrong place, ignore the context.
try $F "    let new_committed = committed + accepted + 1;" "    let new_committed = committed + accepted;"
try $F "    if accepted == cand_len {" "    if accepted + 1 >= cand_len {"
try $F "Some(Cycle { committed: new_committed, boundary: committed + cand_len, snapshot: true })" "Some(Cycle { committed: new_committed, boundary: committed + cand_len + 2, snapshot: true })"
try $F "    if new_committed > max_ctx {" "    if new_committed > max_ctx + 1 {"
# Chunking: a chunk longer than ub, a gap, the tail dropped.
try $F "    let rows = if rest < ub { rest } else { ub };" "    let rows = if rest < ub { rest } else { ub + 1 };"
try $F "    Some((p0 + start, rows))" "    Some((p0 + start + 1, rows))"
try $F "    let rows = if rest < ub { rest } else { ub };" "    let rows = if rest < ub { rest - 1 } else { ub };"
finish
