//! Pre-tokenizer split coverage. A regex pre-tokenizer with `Isolated` behaviour (HF `tokenizers` Split, llama.cpp
//! `unicode_regex_split`) turns the matches of its regex into pieces: every gap between matches becomes a piece, and
//! every match does too. [`isolated_pieces`] is that rule, proved here (`verification/lucina-split/rules.rs`) to cover
//! the input exactly once, with no gap and no overlap, and to terminate. [`is_cover`] checks the same property on any
//! piece list at run time; `crates/lucina-parity/tests/split_cover.rs` runs it on infr's real qwen35 pre-tokenizer.
//! Not proved: the regex engine itself.
#![allow(unexpected_cfgs)]

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Matches inside `[0, len)`, each non-empty, in order and not overlapping.
pub open spec fn valid_matches(len: int, m: Seq<(usize, usize)>) -> bool {
    &&& forall|i: int| 0 <= i < m.len() ==> (#[trigger] m[i]).0 < m[i].1 && m[i].1 <= len
    &&& forall|i: int| 0 <= i < m.len() - 1 ==> (#[trigger] m[i]).1 <= m[i + 1].0
}

/// `p` covers `[0, end)` exactly once: non-empty pieces, the first starting at 0, each starting where the last ended.
pub open spec fn covers(end: int, p: Seq<(usize, usize)>) -> bool {
    &&& p.len() == 0 <==> end == 0
    &&& forall|i: int| 0 <= i < p.len() ==> (#[trigger] p[i]).0 < p[i].1
    &&& p.len() > 0 ==> p[0].0 == 0 && p[p.len() - 1].1 == end
    &&& forall|i: int| 0 <= i < p.len() - 1 ==> (#[trigger] p[i]).1 == p[i + 1].0
}

} // verus!

/// The `Isolated` split of an input of `len` bytes given the regex's matches.
#[cfg_attr(verus_keep_ghost, verus_spec(pieces =>
    requires
        valid_matches(len as int, m@),
    ensures
        covers(len as int, pieces@),
))]
pub fn isolated_pieces(len: usize, m: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut cur: usize = 0;
    let mut i: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            i <= m.len(),
            valid_matches(len as int, m@),
            cur <= len,
            i == 0 ==> cur == 0,
            i > 0 ==> cur == m@[i - 1].1,
            covers(cur as int, pieces@),
        decreases m.len() - i,
    ))]
    while i < m.len() {
        let (s, e) = m[i];
        if cur < s {
            pieces.push((cur, s));
        }
        pieces.push((s, e));
        cur = e;
        i += 1;
    }
    if cur < len {
        pieces.push((cur, len));
    }
    pieces
}

/// Whether `p` covers `[0, len)` exactly once (the run-time twin of `covers`).
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r <==> covers(len as int, p@),
))]
pub fn is_cover(len: usize, p: &[(usize, usize)]) -> bool {
    if p.len() == 0 {
        return len == 0;
    }
    if len == 0 || p[0].0 != 0 || p[p.len() - 1].1 != len {
        return false;
    }
    let mut i: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            i <= p.len(),
            p.len() > 0,
            forall|j: int| 0 <= j < i ==> (#[trigger] p@[j]).0 < p@[j].1,
            forall|j: int| 0 <= j < i && j < p.len() - 1 ==> (#[trigger] p@[j]).1 == p@[j + 1].0,
        decreases p.len() - i,
    ))]
    while i < p.len() {
        if p[i].0 >= p[i].1 {
            return false;
        }
        if i + 1 < p.len() && p[i].1 != p[i + 1].0 {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_and_matches_become_pieces() {
        assert_eq!(isolated_pieces(10, &[(2, 4), (4, 7)]), vec![(0, 2), (2, 4), (4, 7), (7, 10)]);
        assert_eq!(isolated_pieces(3, &[(0, 3)]), vec![(0, 3)]);
        assert_eq!(isolated_pieces(0, &[]), vec![]);
        assert_eq!(isolated_pieces(5, &[]), vec![(0, 5)]);
    }

    #[test]
    fn cover_check() {
        assert!(is_cover(10, &[(0, 2), (2, 10)]));
        assert!(!is_cover(10, &[(0, 2), (3, 10)]), "gap");
        assert!(!is_cover(10, &[(0, 3), (2, 10)]), "overlap");
        assert!(!is_cover(10, &[(0, 2), (2, 9)]), "short");
        assert!(!is_cover(10, &[(0, 0), (0, 10)]), "empty piece");
        assert!(is_cover(0, &[]));
    }
}
