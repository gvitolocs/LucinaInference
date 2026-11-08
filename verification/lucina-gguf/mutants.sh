#!/usr/bin/env bash
# Shows that verification/lucina-gguf/rules.rs can fail: each line plants one bug in a copy of the shipped file, and Verus
# has to refuse it. Run from the repo root on verus-host: verus-remote <repo> bash verification/lucina-gguf/mutants.sh
set -uo pipefail
verus=${VERUS:-verus}
work=$(mktemp -d "${TMPDIR:-/tmp}/lucina-gguf-mutants.XXXXXX")
trap 'rm -rf "${work:?}"' EXIT
fail=0 refused=0
try() {
    local from=$1 to=$2
    mkdir -p "$work/crates/lucina-rules/src" "$work/verification/lucina-gguf"
    cp crates/lucina-rules/src/gguf_bounds.rs "$work/crates/lucina-rules/src/"
    cp verification/lucina-gguf/rules.rs "$work/verification/lucina-gguf/"
    local target=$work/crates/lucina-rules/src/gguf_bounds.rs
    local text; text=$(<"$target")
    case "$text" in
        *"$from"*) ;;
        *) echo "NOT FOUND: $from"; fail=1; return ;;
    esac
    printf "%s\n" "${text/"$from"/"$to"}" > "$target"
    if cmp -s "$target" crates/lucina-rules/src/gguf_bounds.rs; then echo "NOT CHANGED: $from"; fail=1; return; fi
    if "$verus" "$work/verification/lucina-gguf/rules.rs" > "$work/out" 2>&1; then
        echo "NOT REFUSED: $from -> $to"; fail=1
    else
        echo "refused: $from -> $to ($(grep -c '^error' "$work/out") errors)"; refused=$((refused + 1))
    fi
}

# A read that ends past the buffer.
try 'if n > room {' 'if n > room + 1 {'
# A count the file cannot hold, accepted.
try 'let ok = count <= remaining / min_bytes;' 'let ok = count <= remaining;'
# A tensor range past the end of the file, a misaligned offset, a partial block.
try 'if start > file_len || len > file_len - start {' 'if start > file_len {'
try 'if numel % blk_elems != 0 || offset % align != 0 {' 'if offset % align != 0 {'
try 'if numel % blk_elems != 0 || offset % align != 0 {' 'if numel % blk_elems != 0 {'
# Unchecked additions.
try 'if data_start > u64::MAX - offset {' 'if false {'
try 'if x > u64::MAX - add {' 'if false {'
# Overflowing products.
try 'if b > u64::MAX / a {' 'if b > u64::MAX / a + 1 {'
try 'Some(v) => acc = v,' 'Some(v) => acc = v + 1,'
# Too many dimensions.
try 'pub const MAX_DIMS: usize = 4;' 'pub const MAX_DIMS: usize = 5;'
try 'if dims.len() == 0 || dims.len() > MAX_DIMS {' 'if dims.len() > MAX_DIMS {'
# Alignment.
try 'let add = align - rem;' 'let add = align - rem + 1;'
echo "lucina-gguf mutants: $refused refused"
exit $fail
