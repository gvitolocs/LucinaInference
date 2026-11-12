//! infr's own KV allocation size (`infr_core::budget::kv_fmt_bytes`, the number handed to `Backend::alloc`) against
//! the Verus-proved `lucina_rules::kv_rules::kv_q4_0_bytes`, for the production model and a sweep of contexts.
use infr_core::DType;
use lucina_rules::kv_rules::kv_q4_0_bytes;

/// Qwen3.8-27B: 16 attention layers (every 4th of 64), 4 KV heads, head dim 256.
const LAYERS: u64 = 16;
const HEADS: u64 = 4;
const HD: u64 = 256;

fn infr_bytes(rows: u64) -> u64 {
    // one K buffer and one V buffer per attention layer, each n_kv * head_dim * rows elements
    let side = (HEADS * HD * rows) as usize;
    (infr_core::budget::kv_fmt_bytes(DType::Q4_0, side) * 2) as u64 * LAYERS
}

#[test]
fn production_q4_0_kv_matches_the_proved_formula() {
    assert_eq!(infr_bytes(122_880), 2_264_924_160);
    for rows in [
        1u64, 31, 32, 256, 4096, 32_768, 65_536, 81_920, 122_880, 262_144,
    ] {
        assert_eq!(
            Some(infr_bytes(rows)),
            kv_q4_0_bytes(LAYERS, HEADS, HD, rows),
            "rows {rows}"
        );
    }
}
