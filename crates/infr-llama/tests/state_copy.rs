//! Quantized-KV / recurrent-state RANGE arithmetic, checked against llama.cpp v1.0.0.
//!
//! Upstream history this file pins (MIT, ggml/llama.cpp): at our baseline pin `3612199ed`
//! `src/llama-context.cpp:2764` sized an on-device state copy with
//! `n = winfo.size / ggml_element_size(winfo.tensor)`. For a ggml BLOCK type `ggml_element_size`
//! == `ggml_type_size` == BYTES PER BLOCK (q4_0: 18 B per 32 elements), so `n` is the number of
//! BLOCKS while `ggml_view_1d`/`ggml_new_tensor_1d` read it as a number of ELEMENTS — 32x too
//! small, and `e56f006ed` (#30541) records the consequence verbatim: "the restored state was
//! wrong, or the copy crashed". v1.0.0 carries the block-aware form at
//! `src/llama-context.cpp:3522-3655` (save at `:3522`, restore at `:3654`, range-to-range copy at
//! `:3727`): assert whole blocks, then `n = size / ggml_type_size(type) * ggml_blck_size(type)`.
//!
//! Every state range infr selects goes through [`infr_range`], so this file asks one question per
//! KV format: does infr pick the same BYTES v1.0.0 picks for a row range `[r0, r1)` of one
//! layer's cache side? CPU-only — no device, no model file.
//!
//! [`infr_range`] is `infr_core::budget::kv_state_range_bytes`, the one helper every state range
//! now routes through. [`infr_padded_prefix_range`] is the expression it replaced — a difference of
//! the allocation sizer `kv_fmt_bytes` — and it stays here, pinned as WRONG on q8_0, because the
//! pad it folds into a length is exactly the kind of arithmetic this slice is about.

use infr_core::backend::{Backend, BufferUsage};
use infr_core::budget::{kv_fmt_bytes, kv_state_copy_spans, kv_state_range_bytes};
use infr_core::tensor::DType;

/// The KV formats a session can actually run with, and their block geometry. Transcribed from
/// llama.cpp `ggml/src/ggml.c` `type_traits[]` (`.blck_size`, `.type_size`) — the same layouts
/// `infr_core::decode_spec::block_spec` documents from `ggml-quants.c`.
const KV_FORMATS: &[(DType, usize, usize)] = &[
    (DType::F16, 1, 2),
    (DType::Bf16, 1, 2),
    (DType::F32, 1, 4),
    (DType::Q8_0, 32, 34), // [half d][i8 qs[32]]
    (DType::Q4_0, 32, 18), // [half d][u8 qs[16]]
    (DType::Q4_1, 32, 20), // [half d][half m][u8 qs[16]]
    (DType::Q5_0, 32, 22), // [half d][u8 qh[4]][u8 qs[16]]
    (DType::Q5_1, 32, 24), // [half d][half m][u8 qh[4]][u8 qs[16]]
    (DType::Iq4Nl, 32, 18),
    (DType::Turbo2, 128, 34),
    (DType::Turbo3, 128, 50),
    (DType::Turbo4, 128, 66),
];

fn block_geom(dt: DType) -> (usize, usize) {
    KV_FORMATS
        .iter()
        .find(|(d, ..)| *d == dt)
        .map(|&(_, elems, bytes)| (elems, bytes))
        .expect("the roster covers every format this file iterates")
}

// ── the formulas under test ───────────────────────────────────────────────────────────────────

/// llama.cpp v1.0.0's state-copy element count (`llama-context.cpp:3522-3523`, restored at
/// `:3654-3655`): the whole-block assert, then bytes → ELEMENTS through the block size.
fn upstream_v1_elems(dt: DType, bytes: usize) -> usize {
    let (blk_elems, blk_bytes) = block_geom(dt);
    assert!(
        bytes.is_multiple_of(blk_bytes),
        "{dt:?} range is not whole blocks"
    );
    bytes / blk_bytes * blk_elems
}

/// The SAME expression at the pin (`llama-context.cpp:2764`) — what `e56f006ed` removed. Kept
/// here so the discrimination is executable: an element count that is really a block count.
fn upstream_pin_elems(dt: DType, bytes: usize) -> usize {
    let (_, blk_bytes) = block_geom(dt);
    bytes / blk_bytes
}

/// llama.cpp v1.0.0's answer to the question this slice is about: which BYTES hold rows
/// `[r0, r1)` of a cache whose row is `row_elems` elements. `llama-context.cpp:3727-3728` walks a
/// byte cursor over exactly this whole-block granularity.
fn upstream_v1_range(dt: DType, row_elems: usize, r0: usize, r1: usize) -> (usize, usize) {
    let (blk_elems, blk_bytes) = block_geom(dt);
    assert!(
        row_elems.is_multiple_of(blk_elems),
        "{dt:?} row splits a block"
    );
    let bytes_per_row = row_elems / blk_elems * blk_bytes;
    (r0 * bytes_per_row, (r1 - r0) * bytes_per_row)
}

/// INFR's state range for rows `[r0, r1)` — the helper `SeamKv::seed_from` now calls for the byte
/// length it hands `Backend::copy_buffer` (`crates/infr-llama/src/seam/weights.rs:658-672`).
fn infr_range(dt: DType, row_elems: usize, r0: usize, r1: usize) -> (usize, usize) {
    kv_state_range_bytes(dt, row_elems, r0, r1)
}

/// The expression the fix replaced: the same range as a DIFFERENCE of the allocation sizer.
/// `kv_fmt_bytes` is correct for what it sizes — a buffer big enough for a kernel's block-aligned
/// prefix — and wrong for what it is used for here, because q8_0's 34-byte block is rounded up to
/// 4 in it. Kept as a counter-example: [`state_copy_the_padded_prefix_difference_is_the_bug`]
/// pins where it disagrees with llama v1.0.0.
fn infr_padded_prefix_range(dt: DType, row_elems: usize, r0: usize, r1: usize) -> (usize, usize) {
    let off = kv_fmt_bytes(dt, r0 * row_elems);
    (off, kv_fmt_bytes(dt, r1 * row_elems) - off)
}

/// Bytes the PIN's element count actually ends up covering, once infr's sizer turns it back into a
/// buffer: `n` blocks re-sized as `n` elements.
fn pin_covers_bytes(dt: DType, range_bytes: usize) -> usize {
    kv_fmt_bytes(dt, upstream_pin_elems(dt, range_bytes))
}

// ── a KV region of known contents ─────────────────────────────────────────────────────────────

/// Bytes of a `rows` x `row_elems` cache side in `dt`, laid out so that ANY misalignment in a
/// range is visible: every block opens with its own index as a little-endian u16 at offset 0 —
/// where q4_0/q4_1/q5_0/q5_1/q8_0/iq4_nl/turbo all keep their `d` scale, so a range that starts
/// even two bytes off reads a neighbour's scale — and the rest of the block repeats that index.
fn kv_region(dt: DType, rows: usize, row_elems: usize) -> Vec<u8> {
    let (blk_elems, blk_bytes) = block_geom(dt);
    assert!(row_elems.is_multiple_of(blk_elems));
    let blocks = rows * row_elems / blk_elems;
    let mut b = vec![0u8; blocks * blk_bytes];
    for (i, blk) in b.chunks_mut(blk_bytes).enumerate() {
        blk[..2].copy_from_slice(&(i as u16).to_le_bytes());
        for byte in &mut blk[2..] {
            *byte = i as u8;
        }
    }
    b
}

/// The block index a region's bytes name at byte offset `off`.
fn block_index_at(bytes: &[u8], off: usize) -> usize {
    u16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize
}

/// A row range as block indices: rows `[r0, r1)` start at block `r0 * row_elems / blk_elems`.
fn first_block_of_row(dt: DType, row_elems: usize, r: usize) -> usize {
    r * row_elems / block_geom(dt).0
}

/// Every format's own block geometry, plus the one property the whole slice rests on: for a BLOCK
/// type, bytes and elements differ by the block factor, so any arithmetic that treats one as the
/// other is off by 32 (or 128, for TurboQuant).
#[test]
fn state_copy_block_formats_are_what_makes_bytes_and_elements_differ() {
    for &(dt, blk_elems, blk_bytes) in KV_FORMATS {
        assert!(blk_elems >= 1 && blk_bytes >= 1, "{dt:?} degenerate block");
        // `ggml_element_size` == `ggml_type_size` == bytes per block for a block type; that
        // identity is the whole bug, so state it here.
        let bytes = 8 * blk_bytes;
        assert_eq!(
            upstream_pin_elems(dt, bytes) * blk_elems,
            upstream_v1_elems(dt, bytes),
            "{dt:?}: the pin's element count is short by exactly the block factor"
        );
        if blk_elems > 1 {
            assert!(
                pin_covers_bytes(dt, bytes) < bytes,
                "{dt:?}: the pin's range must UNDER-cover a {bytes}-byte state copy"
            );
        } else {
            assert_eq!(pin_covers_bytes(dt, bytes), bytes, "{dt:?}");
        }
    }
}

/// The pin's arithmetic, made concrete on the production shape (a 1024-element KV row: 4 KV heads
/// x head_dim 256, the qwen35/qwen38 attention layers).
#[test]
fn state_copy_q4_0_the_pin_selects_a_32x_shorter_range_than_v1() {
    let (dt, row) = (DType::Q4_0, 1024);
    let (_off, len) = upstream_v1_range(dt, row, 3, 7);
    assert_eq!(len, 2304, "4 rows x 32 blocks x 18 B");
    // The pin sizes the same copy at 1/32 of that, because 2304/18 = 128 blocks read as 128
    // elements = 4 blocks = 72 bytes.
    assert_eq!(pin_covers_bytes(dt, len), 72);
    assert_eq!(upstream_v1_elems(dt, len), 4096);
    assert_eq!(upstream_pin_elems(dt, len), 128);
}

/// Rows `[r0, r1)` of one layer's cache side, for every KV format infr can run a session with:
/// infr must select the bytes v1.0.0 selects — same offset, same length — and those bytes must
/// really be the range's blocks.
#[test]
fn state_copy_subrange_selects_the_bytes_llama_v1_selects() {
    let (rows, r0, r1) = (12usize, 3usize, 7usize);
    for &(dt, blk_elems, _) in KV_FORMATS {
        // One odd block count per row as well as an even one: `blk_elems` per row is 1 block for
        // the dense types, and a row of an odd number of blocks is where an alignment pad folded
        // into a SIZE stops being the size of the DATA.
        for &row_elems in &[blk_elems, 3 * blk_elems, 32 * blk_elems] {
            let want = upstream_v1_range(dt, row_elems, r0, r1);
            let got = infr_range(dt, row_elems, r0, r1);
            let region = kv_region(dt, rows, row_elems);
            assert_eq!(
                got,
                want,
                "{dt:?} rows [{r0},{r1}) of a {row_elems}-elem row: v1.0.0 selects {want:?}, \
                 infr selects {got:?} (of a {}-byte region)",
                region.len()
            );
            // And the selected bytes are the range's own blocks.
            let (off, len) = got;
            assert!(
                off.is_multiple_of(block_geom(dt).1),
                "{dt:?}: range starts mid-block"
            );
            assert!(
                len.is_multiple_of(block_geom(dt).1),
                "{dt:?}: range ends mid-block"
            );
            assert_eq!(
                block_index_at(&region, off),
                first_block_of_row(dt, row_elems, r0),
                "{dt:?}: byte offset {off} is not row {r0}'s first block"
            );
            // The range's LAST block, not row r1-1's FIRST one: a row is `row_elems / blk_elems`
            // blocks wide, so at 3 blocks per row rows [3,7) end on block 20, not 18. (The first
            // version of this file compared against `first_block_of_row(r1 - 1)`, which is only
            // right at one block per row, and the run caught it — see the red record.)
            assert_eq!(
                block_index_at(&region, off + len - block_geom(dt).1),
                r1 * row_elems / block_geom(dt).0 - 1,
                "{dt:?}: the range does not end inside row {}",
                r1 - 1
            );
        }
    }
}

/// Where the pre-fix expression went wrong, kept as an executable counter-example. q8_0's block is
/// 34 bytes, and `kv_fmt_bytes` rounds a buffer up to a `u32` multiple so a GPU kernel addressing
/// the cache in `uint`s cannot run off the end. Inside a LENGTH that round-up is not headroom, it
/// is 2 extra bytes: at an ODD block count below the range the window starts mid-block and runs 2
/// bytes into the row after it.
#[test]
fn state_copy_the_padded_prefix_difference_is_the_bug() {
    let dt = DType::Q8_0;
    // 3 blocks (102 data bytes) per row, 340 bytes of stride below rows [3,7).
    assert_eq!(infr_range(dt, 96, 3, 7), upstream_v1_range(dt, 96, 3, 7));
    assert_eq!(infr_range(dt, 96, 3, 7), (306, 408));
    assert_eq!(
        infr_padded_prefix_range(dt, 96, 3, 7),
        (308, 408),
        "the differenced prefixes start 2 bytes into row 3's first block and end 2 bytes into row \
         7's"
    );
    // A prefix — the op `seed_from` performs — is short by the same 2 bytes.
    assert_eq!(infr_range(dt, 96, 0, 1), (0, 102));
    assert_eq!(infr_padded_prefix_range(dt, 96, 0, 1), (0, 104));
    // Even block counts hide it, which is why a production-shaped row can pass this test by luck:
    // 1024 elements = 32 blocks/row, so every row boundary is 4-aligned.
    assert_eq!(
        infr_padded_prefix_range(dt, 96, 2, 6),
        infr_range(dt, 96, 2, 6),
        "6 blocks below row 2: 204, already a multiple of 4"
    );
    assert_eq!(
        infr_padded_prefix_range(dt, 1024, 3, 7),
        infr_range(dt, 1024, 3, 7),
        "{dt:?} on the production row width is where the pad happens to be invisible"
    );
    // No other format has the pad branch, so the two expressions agree for all of them.
    for &(fmt, blk_elems, _) in KV_FORMATS {
        if fmt == dt {
            continue;
        }
        for &blocks_per_row in &[1usize, 3, 36] {
            let row_elems = blocks_per_row * blk_elems;
            assert_eq!(
                infr_padded_prefix_range(fmt, row_elems, 3, 7),
                upstream_v1_range(fmt, row_elems, 3, 7),
                "{fmt:?} x {blocks_per_row} blocks/row: the sizer has no pad to fold in"
            );
        }
    }
}

/// The op this arithmetic is FOR: `SeamKv::seed_from` seeds a forked slot from another slot's
/// shared prefix, which is rows `[0, p)` at offset 0. This test rebuilds that op from its two
/// pieces — the range helper and `Backend::copy_buffer` — and runs them on a real backend buffer.
/// It is NOT the live call: the live call site (`seed_kv_prefix`, which `seed_from` invokes per
/// side per layer) is driven by `seam::seed_tests::seed_kv_prefix_*` in the crate's own test
/// module, including the planar case below.
#[test]
fn state_copy_seed_prefix_copies_exactly_the_seeded_rows() {
    let be = infr_cpu::CpuBackend::new();
    let (rows, p) = (12usize, 5usize);
    for &(dt, blk_elems, _) in KV_FORMATS {
        let row_elems = 3 * blk_elems;
        let (off, len) = infr_range(dt, row_elems, 0, p);
        assert_eq!(off, 0, "{dt:?}: a prefix starts at 0 by construction");
        assert_eq!(
            (off, len),
            upstream_v1_range(dt, row_elems, 0, p),
            "{dt:?} prefix of {p} rows"
        );
        let region = kv_region(dt, rows, row_elems);
        let total = region.len();
        let src = be.alloc(total, BufferUsage::KvCache).unwrap();
        let dst = be.alloc(total, BufferUsage::KvCache).unwrap();
        be.upload(src.as_ref(), &region).unwrap();
        be.copy_buffer(src.as_ref(), dst.as_ref(), len).unwrap();
        let mut got = vec![0u8; total];
        be.download(dst.as_ref(), &mut got).unwrap();
        assert_eq!(&got[..len], &region[..len], "{dt:?}: seeded rows differ");
        assert!(
            got[len..].iter().all(|&b| b == 0),
            "{dt:?}: the copy ran past row {p}"
        );
    }
}

/// One packed length is not one COPY when the cache is PLANAR. `infr-vulkan`'s Q8_0 KV cache keeps
/// codes and scales in separate regions (`shaders/store_q8.comp:58-67`: code of element `e` at byte
/// `e`, scale of block `b` at byte `cap + b*2`), so a row range is TWO spans. Here the spans are
/// checked against a byte set rebuilt by walking the shader's own addressing per element and per
/// block, which is an independent derivation of the same layout.
#[test]
fn state_copy_planar_q8_spans_cover_exactly_the_bytes_the_shader_addresses() {
    // (row width in elements, cache capacity in elements). 4096 = 128 whole blocks, 4128 = 129:
    // an ODD block count, the shape that makes the allocation pad and the scale base disagree.
    for &(row_elems, cap_elems) in &[(32usize, 4096usize), (96, 4096), (576, 4128), (1024, 8192)] {
        let alloc = kv_fmt_bytes(DType::Q8_0, cap_elems);
        let blocks = cap_elems / 32;
        assert_eq!(
            alloc,
            (cap_elems + 2 * blocks).next_multiple_of(4),
            "the whole cache is codes then scales then at most a 3-byte pad: {row_elems}x{cap_elems}"
        );
        let rows = cap_elems / row_elems;
        for p in [1, 2, rows.min(3), rows] {
            if p == 0 || p > rows {
                continue;
            }
            let spans = kv_state_copy_spans(DType::Q8_0, row_elems, cap_elems, 0, p, true);
            assert_eq!(spans.len(), 2, "codes + scales for {row_elems}x{p} rows");
            // The shader's own byte list for rows [0, p).
            let mut want: Vec<usize> = Vec::new();
            for r in 0..p {
                for e in 0..row_elems {
                    want.push(r * row_elems + e);
                }
                let b0 = r * row_elems / 32;
                for b in b0..b0 + row_elems / 32 {
                    want.push(cap_elems + 2 * b);
                    want.push(cap_elems + 2 * b + 1);
                }
            }
            want.sort_unstable();
            let mut got: Vec<usize> = spans
                .iter()
                .flat_map(|&(off, _, len)| off..off + len)
                .collect();
            got.sort_unstable();
            assert_eq!(got, want, "{row_elems}x{cap_elems}: {p} rows");
            assert_eq!(
                got.len(),
                kv_state_range_bytes(DType::Q8_0, row_elems, 0, p).1,
                "the two spans are the packed byte count, split — not more, not less"
            );
            for &(off, _, len) in &spans {
                assert!(
                    off + len <= alloc,
                    "span [{off}, {}) leaves the {alloc}-byte allocation",
                    off + len
                );
            }
            // Packed is the same data as ONE span at the same length.
            let packed = kv_state_copy_spans(DType::Q8_0, row_elems, cap_elems, 0, p, false);
            assert_eq!(packed.len(), 1);
            assert_eq!(packed[0].2, got.len());
        }
    }
}

/// A restore re-assembles the state from segments, so the ranges of a partition must TILE the
/// buffer with no gap and no overlap — the invariant `llama-context.cpp:3727-3728`'s byte cursor
/// relies on when save and restore chunk a range differently.
#[test]
fn state_copy_ranges_tile_a_state_buffer_with_no_gap_or_overlap() {
    let rows = 12usize;
    for &(dt, blk_elems, blk_bytes) in KV_FORMATS {
        for &row_elems in &[blk_elems, 3 * blk_elems, 32 * blk_elems] {
            let data_bytes = rows * row_elems / blk_elems * blk_bytes;
            let mut cursor = 0usize;
            for &seg in &[2usize, 1, 3, 3, 3] {
                let (off, len) = infr_range(dt, row_elems, cursor, cursor + seg);
                assert_eq!(
                    off,
                    cursor * row_elems / blk_elems * blk_bytes,
                    "{dt:?}: segment after {cursor} rows starts at {off}"
                );
                assert_eq!(
                    len,
                    seg * row_elems / blk_elems * blk_bytes,
                    "{dt:?}: {seg}-row segment is {len} bytes"
                );
                assert_eq!(
                    off + len,
                    (cursor + seg) * row_elems / blk_elems * blk_bytes,
                    "{dt:?}: gap or overlap between the segment ending at row {cursor} and the \
                     next one"
                );
                cursor += seg;
            }
            assert_eq!(cursor, rows, "{dt:?}: the segments do not cover every row");
            assert_eq!(
                infr_range(dt, row_elems, 0, rows).1,
                data_bytes,
                "{dt:?}: the whole-state range is not {data_bytes} bytes"
            );
        }
    }
}

/// A range that splits a block cannot be copied byte-exactly by anything, so the helper refuses it
/// instead of quietly shortening it — the `GGML_ASSERT` discipline of
/// `llama-context.cpp:3521-3522`, which the pin's arithmetic had no equivalent of.
#[test]
fn state_copy_a_range_that_splits_a_block_is_refused_not_shortened() {
    let dt = DType::Q4_0;
    // 33 elements per row leaves one element of every row outside its block.
    let split_row = std::panic::catch_unwind(|| kv_state_range_bytes(dt, 33, 0, 4));
    assert!(
        split_row.is_err(),
        "a 33-element q4_0 row splits a 32-element block"
    );
    assert!(
        std::panic::catch_unwind(|| kv_state_range_bytes(dt, 64, 5, 4)).is_err(),
        "[5,4) is not a range"
    );
    // Whole blocks are accepted and give the exact data bytes: 64 elems = 2 blocks = 36 B per row.
    assert_eq!(kv_state_range_bytes(dt, 64, 0, 4), (0, 144));
    assert_eq!(kv_state_range_bytes(dt, 64, 2, 3), (72, 36));
    // An empty range is a length of 0, not a panic: `seed_from` returns before it ever asks.
    assert_eq!(kv_state_range_bytes(dt, 64, 3, 3), (108, 0));
}

/// The recurrent-state side of the same arithmetic: qwen35's DeltaNet conv/S snapshot is f32, so
/// one element IS one block and the whole-buffer copy `mtp_snapshot_delta`/`mtp_restore_delta`
/// make with `Buffer::len_bytes()` is already exact. Pinned, because it is why that path cannot
/// carry the pin's bug.
#[test]
fn state_copy_f32_recurrent_state_ranges_are_the_whole_buffer() {
    let be = infr_cpu::CpuBackend::new();
    for &elems in &[1usize, 7, 4096] {
        let (off, len) = infr_range(DType::F32, elems, 0, 1);
        assert_eq!((off, len), (0, elems * 4), "{elems} f32 state elements");
        // The checkpoint buffer is allocated from `len_bytes()` and copied at `len_bytes()`; for a
        // dense f32 state that is the same number the range arithmetic gives.
        let b = be.alloc(len, BufferUsage::KvCache).unwrap();
        assert_eq!(b.len_bytes(), len.max(4));
        assert_eq!(
            upstream_v1_elems(DType::F32, len),
            upstream_pin_elems(DType::F32, len),
            "f32: the pin and v1.0.0 agree, because ggml_blck_size(f32) == 1"
        );
    }
}
