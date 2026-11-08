//! KV-cache and context accounting: how many bytes a q4_0 cache takes, which row a position lands on, and when a batch
//! fits the context. Verus proves these in this very file (`verification/lucina-kv/rules.rs`); the attributes are only
//! seen by Verus. Format facts: llama.cpp 3612199ed `block_q4_0` (32 elements in 18 bytes: one f16 scale, 16 nibble
//! bytes), and infr's `budget::kv_fmt_bytes`, which `crates/lucina-parity/tests/kv_accounting.rs` checks against these.
#![allow(unexpected_cfgs)]

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::gguf_bounds::mul_u64;

/// `QK4_0`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q4_0_BLOCK_ELEMS: u64 = 32;
/// `sizeof(block_q4_0)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q4_0_BLOCK_BYTES: u64 = 18;

/// Elements in the K and V caches of `n_attn_layers` attention layers: layers × 2 (K and V) × kv heads × head dim × rows.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r matches Some(n) ==> n as int == n_attn_layers as int * 2 * n_kv_heads as int * head_dim as int * rows as int,
))]
pub fn kv_elems(n_attn_layers: u64, n_kv_heads: u64, head_dim: u64, rows: u64) -> Option<u64> {
    let a = match mul_u64(n_attn_layers, 2) {
        Some(v) => v,
        None => return None,
    };
    let b = match mul_u64(a, n_kv_heads) {
        Some(v) => v,
        None => return None,
    };
    let c = match mul_u64(b, head_dim) {
        Some(v) => v,
        None => return None,
    };
    mul_u64(c, rows)
}

/// Bytes of `elems` q4_0 elements, refusing a count that is not whole blocks.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        // 32 and 18, not the constants, so that changing a constant is refused.
        r matches Some(b) ==> elems % 32 == 0 && b as int == (elems as int / 32) * 18,
        r matches Some(b) ==> b as int * 32 == elems as int * 18,
        elems % 32 != 0 ==> r is None,
))]
pub fn q4_0_bytes(elems: u64) -> Option<u64> {
    if elems % Q4_0_BLOCK_ELEMS != 0 {
        return None;
    }
    let blocks = elems / Q4_0_BLOCK_ELEMS;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(blocks as int * 18 <= u64::MAX as int) by (nonlinear_arith)
            requires blocks as int == elems as int / 32, elems as int <= u64::MAX as int;
        assert(blocks as int * 18 * 32 == elems as int * 18) by (nonlinear_arith)
            requires blocks as int == elems as int / 32, elems as int % 32 == 0;
    }
    Some(blocks * Q4_0_BLOCK_BYTES)
}

/// The whole q4_0 K+V cache of a model at `rows` rows per layer. `head_dim` must be a multiple of 32 so every row is
/// whole blocks (qwen35: 256).
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r matches Some(b) ==> b as int * 32
            == n_attn_layers as int * 2 * n_kv_heads as int * head_dim as int * rows as int * 18,
        head_dim % 32 != 0 ==> r is None,
))]
pub fn kv_q4_0_bytes(n_attn_layers: u64, n_kv_heads: u64, head_dim: u64, rows: u64) -> Option<u64> {
    if head_dim % Q4_0_BLOCK_ELEMS != 0 {
        return None;
    }
    let e = match kv_elems(n_attn_layers, n_kv_heads, head_dim, rows) {
        Some(v) => v,
        None => return None,
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        let k = head_dim as int / 32;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(head_dim as int, 32);
        let p = n_attn_layers as int * 2 * n_kv_heads as int * k * rows as int;
        assert(e as int == p * 32) by (nonlinear_arith)
            requires
                e as int == n_attn_layers as int * 2 * n_kv_heads as int * head_dim as int * rows as int,
                head_dim as int == 32 * k + head_dim as int % 32,
                head_dim as int % 32 == 0,
                p == n_attn_layers as int * 2 * n_kv_heads as int * k * rows as int;
        vstd::arithmetic::div_mod::lemma_mod_multiples_basic(p, 32);
    }
    q4_0_bytes(e)
}

/// The cache row that position `pos` uses. Without a ring, a position past the cache is refused; with a ring (sliding
/// window layers), the row is `pos % rows`. Either way the row is inside the cache.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        rows > 0,
    ensures
        r matches Some(x) ==> x < rows,
        r matches Some(x) ==> (!ring ==> x == pos),
        r matches Some(x) ==> (ring ==> x == pos % rows),
        !ring && pos >= rows ==> r is None,
        ring ==> r is Some,
))]
pub fn kv_row(pos: u64, rows: u64, ring: bool) -> Option<u64> {
    if ring {
        Some(pos % rows)
    } else if pos < rows {
        Some(pos)
    } else {
        None
    }
}

/// Whether a batch of `n` tokens starting at position `p0` fits a context of `ctx` positions: then every position it
/// writes is below `ctx` (no overrun, no overflow).
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r <==> p0 as int + n as int <= ctx as int,
))]
pub fn batch_fits(p0: u64, n: u64, ctx: u64) -> bool {
    p0 <= ctx && n <= ctx - p0
}

#[cfg(verus_keep_ghost)]
verus! {

/// A ring of `r` rows never puts two positions closer than `r` on the same row: the reason a sliding-window ring of
/// `window + ubatch` rows can hold every position a batch still reads.
pub proof fn ring_rows_distinct(a: int, b: int, r: int)
    requires
        0 <= a < b,
        b - a < r,
    ensures
        a % r != b % r,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(a, r);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(b, r);
    vstd::arithmetic::div_mod::lemma_mod_bound(a, r);
    vstd::arithmetic::div_mod::lemma_mod_bound(b, r);
    if a % r == b % r {
        assert(b - a == r * (b / r - a / r)) by (nonlinear_arith)
            requires a == r * (a / r) + a % r, b == r * (b / r) + b % r, a % r == b % r;
        assert(false) by (nonlinear_arith)
            requires b - a == r * (b / r - a / r), 0 < b - a, b - a < r, r > 0;
    }
}

} // verus!

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_cache_size() {
        // Qwen3.8-27B: 16 attention layers, 4 kv heads, head dim 256, ctx 122880, q4_0 K and V.
        assert_eq!(kv_q4_0_bytes(16, 4, 256, 122880), Some(2_264_924_160));
        assert_eq!(kv_q4_0_bytes(16, 4, 250, 122880), None, "head dim not whole blocks");
        assert_eq!(kv_q4_0_bytes(u64::MAX, 4, 256, 2), None, "overflow");
    }

    #[test]
    fn rows_and_batches() {
        assert_eq!(kv_row(5, 8, false), Some(5));
        assert_eq!(kv_row(8, 8, false), None);
        assert_eq!(kv_row(9, 8, true), Some(1));
        assert!(batch_fits(122_624, 256, 122_880));
        assert!(!batch_fits(122_625, 256, 122_880));
        assert!(!batch_fits(u64::MAX, 2, u64::MAX));
    }
}
