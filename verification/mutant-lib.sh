# Shared mutant harness for verification/lucina-*/mutants.sh (sourced). Each `try FILE FROM TO` plants one bug in a
# copy of crates/lucina-rules/src and the entry, and Verus has to refuse it. Set ENTRY before sourcing.
set -uo pipefail
verus=${VERUS:-verus}
work=$(mktemp -d "${TMPDIR:-/tmp}/lucina-mutants.XXXXXX")
trap 'rm -rf "${work:?}"' EXIT
fail=0 refused=0
try() {
    local file=$1 from=$2 to=$3
    rm -rf "$work/crates" "$work/verification"
    mkdir -p "$work/crates/lucina-rules" "$work/verification/$ENTRY"
    cp -r crates/lucina-rules/src "$work/crates/lucina-rules/"
    cp verification/$ENTRY/rules.rs "$work/verification/$ENTRY/"
    local target=$work/crates/lucina-rules/src/$file
    local text; text=$(<"$target")
    case "$text" in
        *"$from"*) ;;
        *) echo "NOT FOUND: $file: $from"; fail=1; return ;;
    esac
    printf "%s\n" "${text/"$from"/"$to"}" > "$target"
    if cmp -s "$target" "crates/lucina-rules/src/$file"; then echo "NOT CHANGED: $from"; fail=1; return; fi
    if "$verus" "$work/verification/$ENTRY/rules.rs" > "$work/out" 2>&1; then
        echo "NOT REFUSED: $file: $from -> $to"; fail=1
    else
        echo "refused: $file: $from -> $to ($(grep -c '^error' "$work/out") errors)"; refused=$((refused + 1))
    fi
}
finish() {
    echo "$ENTRY mutants: $refused refused"
    exit $fail
}
