//! Speculative decoding (MTP draft head): the greedy accept rule and the commit/rollback bookkeeping of one cycle.
//! Verus proves these in this file (`verification/lucina-spec/rules.rs`); infr-llama's `spec_accept` calls
//! [`spec_accept`] here, and the MTP loop's cycle arithmetic is [`cycle`].
//!
//! Original: llama.cpp 3612199ed `common/speculative.cpp` (draft-mtp) and `common_sampler_sample_and_accept_n`
//! (greedy: keep the drafted tokens while they equal the target's choice, then take the target's own token).
#![allow(unexpected_cfgs)]

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Greedy accept. `verify` holds the target's argmax at each fed position: `verify[i]` is what the target predicts
/// after the committed tokens plus `cand[..i]`, and `verify[cand.len()]` is the bonus token. Returns how many
/// drafted tokens are kept and the token to commit after them.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        verify.len() == cand.len() + 1,
    ensures
        r.0 <= cand.len(),
        // every kept draft is exactly the target's own choice there (the stream equals greedy decoding)
        forall|i: int| 0 <= i < r.0 ==> cand@[i] == verify@[i],
        // and the first one not kept is not
        r.0 < cand.len() ==> cand@[r.0 as int] != verify@[r.0 as int],
        // the committed next token is the target's choice after the kept ones
        r.1 == verify@[r.0 as int],
))]
pub fn spec_accept(cand: &[u32], verify: &[u32]) -> (usize, u32) {
    let mut i: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            i <= cand.len(),
            verify.len() == cand.len() + 1,
            forall|j: int| 0 <= j < i ==> cand@[j] == verify@[j],
        decreases cand.len() - i,
    ))]
    while i < cand.len() && cand[i] == verify[i] {
        i += 1;
    }
    (i, verify[i])
}

/// The state one MTP cycle carries forward.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cycle {
    /// tokens committed so far (prompt + generated)
    pub committed: usize,
    /// where the recurrent (DeltaNet) snapshot stands: the next verify recomputes the state from here
    pub boundary: usize,
    /// whether this cycle took a new snapshot (every draft accepted) instead of restoring the old one
    pub snapshot: bool,
}

/// One cycle's bookkeeping after `accepted` of `cand_len` drafts were kept. The verify fed `committed + cand_len`
/// tokens; `accepted + 1` of them become committed. A fully accepted cycle snapshots the recurrent state at the end of
/// the fed tokens; otherwise the old snapshot is restored, because the state past it saw rejected tokens.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        boundary <= committed,
        accepted <= cand_len,
        committed + cand_len + 1 <= usize::MAX,
    ensures
        r matches Some(n) ==> n.committed == committed + accepted + 1,
        r matches Some(n) ==> (n.snapshot <==> accepted == cand_len),
        r matches Some(n) ==> n.snapshot ==> n.boundary == committed + cand_len,
        r matches Some(n) ==> !n.snapshot ==> n.boundary == boundary,
        // the rollback point never covers a token that is not committed (no rejected draft survives in the state)
        r matches Some(n) ==> n.boundary <= n.committed,
        // the context is never overrun
        r matches Some(n) ==> n.committed <= max_ctx,
        committed + accepted + 1 > max_ctx ==> r is None,
))]
pub fn cycle(committed: usize, boundary: usize, cand_len: usize, accepted: usize, max_ctx: usize) -> Option<Cycle> {
    let new_committed = committed + accepted + 1;
    if new_committed > max_ctx {
        return None;
    }
    if accepted == cand_len {
        Some(Cycle { committed: new_committed, boundary: committed + cand_len, snapshot: true })
    } else {
        Some(Cycle { committed: new_committed, boundary, snapshot: false })
    }
}

/// Prefill chunking: the `i`-th chunk of a prompt of `n` tokens starting at position `p0`, at most `ub` rows each.
/// Returns `(position, rows)`.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        ub > 0,
        p0 as int + n as int <= usize::MAX as int,
    ensures
        r matches Some(c) ==> 0 < c.1 <= ub,
        r matches Some(c) ==> c.0 as int == p0 as int + i as int * ub as int,
        r matches Some(c) ==> c.0 as int + c.1 as int <= p0 as int + n as int,
        // chunks follow each other with no gap: chunk i ends where chunk i+1 starts, the last ends at p0 + n
        r matches Some(c) ==> (c.1 < ub ==> c.0 as int + c.1 as int == p0 as int + n as int),
        i as int * ub as int >= n as int ==> r is None,
))]
pub fn prefill_chunk(p0: usize, n: usize, ub: usize, i: usize) -> Option<(usize, usize)> {
    if i > n / ub {
        #[cfg(verus_keep_ghost)]
        proof! {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(n as int, ub as int);
            assert(i as int * ub as int >= n as int) by (nonlinear_arith)
                requires i as int > n as int / ub as int, ub > 0, n as int == ub as int * (n as int / ub as int) + n as int % ub as int,
                    (n as int % ub as int) < (ub as int);
        }
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(i as int * ub as int <= n as int) by (nonlinear_arith)
            requires i as int <= n as int / ub as int, ub > 0;
    }
    let start = i * ub;
    if start >= n {
        return None;
    }
    let rest = n - start;
    let rows = if rest < ub { rest } else { ub };
    Some((p0 + start, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_keeps_the_matching_prefix() {
        assert_eq!(spec_accept(&[5, 6, 7], &[5, 6, 9, 1]), (2, 9));
        assert_eq!(spec_accept(&[5, 6], &[5, 6, 3]), (2, 3), "all accepted: bonus token");
        assert_eq!(spec_accept(&[5], &[4, 3]), (0, 4));
        assert_eq!(spec_accept(&[], &[8]), (0, 8));
    }

    #[test]
    fn cycle_bookkeeping() {
        assert_eq!(cycle(100, 90, 2, 2, 1000), Some(Cycle { committed: 103, boundary: 102, snapshot: true }));
        assert_eq!(cycle(100, 90, 2, 1, 1000), Some(Cycle { committed: 102, boundary: 90, snapshot: false }));
        assert_eq!(cycle(998, 990, 2, 2, 1000), None, "would overrun the context");
    }

    #[test]
    fn chunks_cover_the_prompt() {
        let (p0, n, ub) = (7usize, 1000usize, 256usize);
        let mut pos = p0;
        let mut i = 0;
        while let Some((s, r)) = prefill_chunk(p0, n, ub, i) {
            assert_eq!(s, pos);
            assert!(r > 0 && r <= ub);
            pos += r;
            i += 1;
        }
        assert_eq!(pos, p0 + n);
        assert_eq!(i, 4);
        assert_eq!(prefill_chunk(0, 512, 256, 2), None, "exactly two full chunks");
    }
}
