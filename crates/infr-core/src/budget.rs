//! Shared KV-cache / VRAM-budget arithmetic.
//!
//! Deciding *where* a KV cache or a weight bank lives is two separable things: the device half
//! (`vkCreateBuffer`, the memory type, the arena, the staging ring) and a
//! device-INDEPENDENT half — how many bytes a KV format costs, which `INFR_*` knob spells what,
//! the cumulative-cap gate the VRAM-first spill applies, and the one-shot placement banner. The
//! second half was re-derived per backend (and, inside `infr-vulkan`, per multi-device wrapper)
//! and drifted. This module hosts it ONCE.
//!
//! What is here:
//!
//! - [`parse_kv_dtype`] — the `INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` name grammar (llama's
//!   `--cache-type-k` / `--cache-type-v`), the single spelling table.
//! - [`kv_fmt_bytes`] — the EXACT allocation size of a KV buffer of `elems` elements, and
//!   [`kv_bytes_per_elem`] — the same thing as a fractional per-element rate, which is what the
//!   context-fit ESTIMATES divide by. The two must agree; a test pins them against each other.
//! - [`flag_from`] / [`mib_bytes`] / [`reserve_bytes`] — the grammars and accessors the spill
//!   knobs share (`empty`/`0` = off; a MiB count → bytes; the 12%-of-VRAM-floored-at-2-GiB
//!   reserve). The VALUES come from the backend's [`Config`](crate::config::Config); nothing here
//!   reads the environment (R3).
//! - [`SpillTally`] / [`SpillCounts`] / [`spill_report_line`] — the VRAM-first placement
//!   bookkeeping and its banner. The counters and the message SKELETON are shared; every noun and
//!   every device-specific clause is passed in ([`SpillNouns`]), and so is the byte formatter,
//!   because a backend's MiB rounding is user-visible text that must not move.
//!
//! What is deliberately NOT here (device facts that only look shared):
//!
//! - The placement itself. Whether VRAM has room is a device question (Vulkan's live
//!   `VK_EXT_memory_budget` heap probe minus a guard headroom, in `vram_budget_fits`). Only the
//!   cumulative *cap* gate ([`SpillTally::admits`]), which is pure arithmetic over the tally, is
//!   shared.
//! - The human-readable byte formatter. Vulkan prints MiB to one decimal; unifying the spellings
//!   would move user-visible text for no behavioral gain, so [`spill_report_line`] takes the
//!   formatter as a parameter instead.
//! - The per-format GATES (`kv_align_ok`, which backends have a Q8 KV kernel, TurboQuant's
//!   `head_dim % 128`). Those are capabilities, and they stay at the seam that knows the backend.

use crate::decode_spec::block_layout;
use crate::tensor::DType;
use std::sync::atomic::{AtomicU64, Ordering};

// ── KV format sizing ─────────────────────────────────────────────────────────

/// The `INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` name grammar (llama.cpp's `--cache-type-k` /
/// `--cache-type-v`): a spelling → KV dtype, or `None` for unset/unrecognized, which every caller
/// treats as "fall through to the default ladder".
///
/// This is the ONLY table of KV format names. The runner turns the dtype into an allocation after
/// applying its backend/alignment gates; the placement estimator turns the same dtype into a
/// bytes-per-element rate via [`kv_bytes_per_elem`]. Both used to spell the names themselves, so a
/// format added to one and not the other was priced as f16 while being allocated as something
/// else.
pub fn parse_kv_dtype(name: &str) -> Option<DType> {
    Some(match name {
        // TurboQuant: WHT-rotated, 128-elem blocks.
        "turbo2" => DType::Turbo2,
        "turbo3" => DType::Turbo3,
        "turbo4" => DType::Turbo4,
        "q8_0" | "q8" | "Q8_0" => DType::Q8_0,
        // Mainline low-bit block quants (32-elem blocks).
        "q4_0" => DType::Q4_0,
        "q4_1" => DType::Q4_1,
        "q5_0" => DType::Q5_0,
        "q5_1" => DType::Q5_1,
        "iq4_nl" => DType::Iq4Nl,
        // Dense.
        "bf16" => DType::Bf16,
        "f32" => DType::F32,
        "f16" | "F16" => DType::F16,
        _ => return None,
    })
}

/// EXACT byte size of one KV-cache buffer holding `elems` elements in `dt` — the number the runner
/// hands `Backend::alloc`, so it is also what the VRAM budget guard sees.
///
/// Q8_0 additionally rounds up to a `u32` multiple: the GPU KV kernels address the cache as
/// `uint`s, and a 34-B-per-block cache is only 2-byte aligned at odd block counts. The other block
/// formats have even block sizes, so their products are already aligned.
///
/// Anything not a KV format falls to `elems * 4` (f32), which is what the seam has always done —
/// the KV dtype is chosen by [`parse_kv_dtype`], so the fallthrough is only ever reached for F32
/// itself.
pub fn kv_fmt_bytes(dt: DType, elems: usize) -> usize {
    match dt {
        DType::Q8_0 => (elems / 32 * 34).next_multiple_of(4),
        // TurboQuant 128-elem blocks: turbo2 = 34 B, turbo3 = 50 B, turbo4 = 66 B.
        DType::Turbo2 => elems / 128 * 34,
        DType::Turbo3 => elems / 128 * 50,
        DType::Turbo4 => elems / 128 * 66,
        // Mainline low-bit KV quants (32-elem blocks) + bf16.
        DType::Q4_0 | DType::Iq4Nl => elems / 32 * 18,
        DType::Q4_1 => elems / 32 * 20,
        DType::Q5_0 => elems / 32 * 22,
        DType::Q5_1 => elems / 32 * 24,
        DType::F16 | DType::Bf16 => elems * 2,
        _ => elems * 4, // F32
    }
}

/// The BYTES of a state RANGE — rows `[r0, r1)` of one cache side whose row is `row_elems`
/// elements — as `(offset, length)`: llama.cpp v1.0.0's arithmetic for saving, restoring and
/// copying KV/recurrent state (`src/llama-context.cpp:3727-3728` walks a byte cursor at exactly
/// this whole-block granularity; the same granularity is what its state save `:3522-3523` and
/// restore `:3654-3655` assert, MIT (c) ggml/llama.cpp contributors).
///
/// This is NOT [`kv_fmt_bytes`] differenced, which is all the seam could express before:
/// `kv_fmt_bytes` returns an ALLOCATION size, and q8_0's 34-byte block is rounded up to 4 there so
/// a `u32`-addressing GPU kernel never runs off the end. That pad is headroom, not data — the
/// block itself is still 34 bytes — so a range built from two padded prefixes lands 2 bytes
/// MID-BLOCK whenever the block count below it is odd: rows [3,7) of a 96-element q8_0 row start
/// at 308, where v1.0.0 says 306.
///
/// Nor may a byte count be turned back into an ELEMENT count to feed an element-indexed copy, which
/// is the bug this helper replaces: at our baseline pin `3612199ed` `src/llama-context.cpp:2764`
/// computed `n = winfo.size / ggml_element_size(winfo.tensor)`, and for a ggml BLOCK type
/// `ggml_element_size` == `ggml_type_size` == BYTES PER BLOCK, so `n` counted blocks while
/// `ggml_view_1d` read it as elements — 32x too short, until `e56f006ed` (#30541), whose commit
/// message records the consequence: "the restored state was wrong, or the copy crashed".
///
/// Whole blocks are asserted rather than rounded, as `GGML_ASSERT` does upstream: a range that
/// splits a block cannot be copied byte-exactly by anything, so a caller asking for one has a
/// geometry bug, and quietly shortening the range instead is how the pin lost state.
pub fn kv_state_range_bytes(dt: DType, row_elems: usize, r0: usize, r1: usize) -> (usize, usize) {
    let (blk_elems, blk_bytes) = block_layout(dt);
    assert!(blk_elems >= 1 && blk_bytes >= 1, "{dt:?}: degenerate block");
    assert!(
        row_elems.is_multiple_of(blk_elems),
        "{dt:?}: a {row_elems}-element state row splits a {blk_elems}-element block"
    );
    assert!(r0 <= r1, "{dt:?}: reversed state range [{r0},{r1})");
    // Upstream does this in `int64_t` (`src/llama-context.cpp:3728`); here the products are `usize`
    // and the multiply is checked, so a range past `usize` bounds is a panic rather than a wrapped
    // short length. No reachable context size comes close: the largest cache side here is
    // `max_ctx * row_elems` elements, i.e. terabytes before this could fire.
    let bytes_per_row = (row_elems / blk_elems)
        .checked_mul(blk_bytes)
        .unwrap_or_else(|| {
            panic!("{dt:?}: {row_elems}-element state row has no representable byte length")
        });
    let length = (r1 - r0).checked_mul(bytes_per_row).unwrap_or_else(|| {
        panic!("{dt:?}: state range [{r0},{r1}) has no representable byte length")
    });
    let offset = r0.checked_mul(bytes_per_row).unwrap_or_else(|| {
        panic!("{dt:?}: state range start {r0} has no representable byte offset")
    });
    (offset, length)
}

/// The byte SPANS that carry state rows `[r0, r1)` of one cache side, as
/// `(src_offset, dst_offset, length)` triples to be copied in order.
///
/// `cap_elems` is the side's ELEMENT capacity (total rows x `row_elems`) — the address the planar
/// layout puts its scale region behind. `planar` says this backend stores `dt` with codes and
/// scales in disjoint regions instead of one self-describing block per `blk_elems` elements.
///
/// * Packed (`planar == false`) — the CPU and Metal caches, Vulkan's low-bit GGUF-block quants and
///   Vulkan's Turbo cache (`shaders/quant_turbo.comp:39`, `bo = (off/128 + blk) * bb`) — is ONE
///   span, exactly [`kv_state_range_bytes`], which is what llama.cpp v1.0.0 copies as a single
///   region (`src/llama-context.cpp:3727-3728`).
/// * PLANAR Q8_0 — the Vulkan Q8_0 KV cache (`shaders/store_q8.comp:58-67`: the code byte of
///   element `e` is byte `e` of the region `[0, cap)`, and the f16 scale of block `b` is at byte
///   `cap + b*2`) — splits the same rows into TWO disjoint spans: codes `[P0, P1)` and scales
///   `[cap + P0/16, cap + P1/16)`, because a q8_0 block is 32 code bytes plus one 2-byte scale.
///   The two spans sum to `(P/32) * 34` = the packed length, which is what makes a single offset-0
///   prefix copy of that length LOOK right and be wrong here: it moves the seeded rows' codes plus
///   2 code bytes of the row AFTER them, and never touches the seeded rows' scales, which stay at
///   their zero fill — so attention reads the seeded prefix as codes x scale 0.
///
/// Zero-length spans are dropped: a 0-size `vkCmdCopyBuffer` region is not something a caller
/// should have to think about, and an empty range carries no bytes in either layout.
pub fn kv_state_copy_spans(
    dt: DType,
    row_elems: usize,
    cap_elems: usize,
    r0: usize,
    r1: usize,
    planar: bool,
) -> Vec<(usize, usize, usize)> {
    let (off, len) = kv_state_range_bytes(dt, row_elems, r0, r1);
    if !planar {
        return if len == 0 {
            Vec::new()
        } else {
            vec![(off, off, len)]
        };
    }
    assert!(
        matches!(dt, DType::Q8_0),
        "{dt:?}: Q8_0 is the only format with a planar cache; the low-bit quants and Turbo pack \
         each block's codes and scales together and are always copied as one span"
    );
    let blk = block_layout(dt).0;
    assert!(
        cap_elems.is_multiple_of(blk),
        "Q8_0: a {cap_elems}-element cache capacity splits a {blk}-element block, so its scale \
         region base is not a whole number of scales"
    );
    // One code byte per element, so a q8_0 element offset IS a codes-region byte offset.
    let (p0, p1) = (r0 * row_elems, r1 * row_elems);
    if p1 == p0 {
        return Vec::new();
    }
    vec![
        (p0, p0, p1 - p0),
        (
            cap_elems + p0 / blk * 2,
            cap_elems + p0 / blk * 2,
            (p1 - p0) / blk * 2,
        ),
    ]
}

/// KV bytes per ELEMENT in `dt`, as a fraction (e.g. Q8_0 = 34/32 = 1.0625).
///
/// The context-fit estimators need a rate, not a size: they divide a free-byte budget by
/// "bytes per token across all layers" to solve for a context length, which the exact
/// [`kv_fmt_bytes`] (with its per-buffer flooring and alignment rounding) cannot express. It is
/// nonetheless the SAME number — read straight off [`block_layout`], the one decode spec, instead
/// of a second hand-written table of ratios — and `kv_fmt_bytes_matches_the_per_elem_rate` pins
/// that the two agree on every whole block.
pub fn kv_bytes_per_elem(dt: DType) -> f64 {
    let (elems, bytes) = block_layout(dt);
    bytes as f64 / elems as f64
}

// ── grammars + accessors shared by the spill knobs ───────────────────────────

/// The boolean-flag grammar every overflow knob uses, over an explicit value: `Some(v)` with `v`
/// neither empty nor `"0"` ⇒ on; unset ⇒ off. (`INFR_KV_OVERFLOW`.)
/// NOT the same as the `is_ok()` presence grammar, where an empty
/// value is ON — `config::env`'s `flag` reader calls this so the difference survives the
/// migration; the read sites take the resolved `bool` off their backend's `Config`.
pub fn flag_from(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| !v.is_empty() && v != "0")
}

/// A MiB-valued knob's CONFIG field in bytes. (`Some(0)` stays `Some(0)` — the diagnostic caps use
/// it to mean "nothing resident", which is NOT the same as "no cap".)
///
/// The multiply SATURATES. The input is a `u64` MiB count straight off a config file or
/// `INFR_KV_OVERFLOW_VRAM_MB`, and anything above ~17.6 trillion MiB overflows the byte product —
/// a value a user can type and `parse::<u64>()` happily accepts. Wrapping is the dangerous
/// direction here, not panicking: in release the product wraps to a SMALL number, so "cap
/// placement at an absurdly high value" silently becomes "cap at a few bytes", which forces the
/// entire class to host and looks like a mysteriously slow run rather than a bad knob. Saturating
/// at `u64::MAX` keeps the only sane reading of a huge cap — effectively no cap — and it can never
/// be reached by real bytes, so the gate behaves exactly as "uncapped" downstream.
pub fn mib_bytes(mib: Option<u64>) -> Option<u64> {
    mib.map(|mb| mb.saturating_mul(1024 * 1024))
}

/// VRAM headroom a VRAM-first spill keeps free below the live free-byte figure, so the consumers
/// that allocate AFTER the spilled class (per-forward activation scratch, dequant caches, BLAS
/// workspace, a paged-MoE arena) still have room. Default 12% of total VRAM floored at 2 GiB —
/// enough for a several-hundred-row prefill on a large-vocab model — overridden by the knob's
/// config field (`kv.overflow_reserve_mb`, in MiB).
///
/// Over-reserving costs residency; under-reserving OOMs mid-forward in an allocator that is
/// infallible by contract, so the floor is deliberate.
pub fn reserve_bytes(total_vram: u64, override_mib: Option<u64>) -> u64 {
    if let Some(bytes) = mib_bytes(override_mib) {
        return bytes;
    }
    (total_vram / 100 * 12).max(2 * 1024 * 1024 * 1024)
}

// ── VRAM-first spill bookkeeping + banner ────────────────────────────────────

/// Resident/spilled counters for one VRAM-first placement class (the KV cache, or the dense
/// weight banks). Bumped from `alloc` on whichever arm won, drained once by the banner.
///
/// All-zero (the `Default`) means the flag was off or nothing of that class was allocated — both
/// print nothing.
#[derive(Debug, Default)]
pub struct SpillTally {
    vram_bufs: AtomicU64,
    vram_bytes: AtomicU64,
    host_bufs: AtomicU64,
    host_bytes: AtomicU64,
}

/// A [`SpillTally`] read at one instant — plain values, so the banner formats from a consistent
/// snapshot instead of four independent loads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpillCounts {
    pub vram_bufs: u64,
    pub vram_bytes: u64,
    pub host_bufs: u64,
    pub host_bytes: u64,
}

impl SpillCounts {
    /// Buffers placed either way. `0` ⇒ nothing to report.
    pub fn total_bufs(&self) -> u64 {
        self.vram_bufs + self.host_bufs
    }
}

impl SpillTally {
    /// Record one buffer that stayed resident in device-local VRAM.
    pub fn record_vram(&self, bytes: u64) {
        self.vram_bufs.fetch_add(1, Ordering::Relaxed);
        self.vram_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one buffer that spilled to host RAM.
    pub fn record_host(&self, bytes: u64) {
        self.host_bufs.fetch_add(1, Ordering::Relaxed);
        self.host_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Snapshot for the banner.
    pub fn counts(&self) -> SpillCounts {
        SpillCounts {
            vram_bufs: self.vram_bufs.load(Ordering::Relaxed),
            vram_bytes: self.vram_bytes.load(Ordering::Relaxed),
            host_bufs: self.host_bufs.load(Ordering::Relaxed),
            host_bytes: self.host_bytes.load(Ordering::Relaxed),
        }
    }

    /// The diagnostic CUMULATIVE-cap gate: may `bytes` more still be placed in VRAM?
    ///
    /// `cap` is `None` for the normal case (no cap — spill only when VRAM is genuinely full);
    /// `Some(0)` forces the whole class to host, which is how the whole-host path stays
    /// reproducible on a model that would otherwise fit entirely. The cap gates PLACEMENT only,
    /// never the real VRAM guard.
    ///
    /// The running total + `bytes` SATURATES, for the same reason [`mib_bytes`] does: a wrapped
    /// sum is a SMALL sum, and a small sum passes a cap it should have failed — the tally would
    /// keep admitting buffers into VRAM after the cap was blown. Saturating turns the degenerate
    /// input into a refusal (spill to host), which is the safe side of this gate. `u64::MAX` bytes
    /// is not a real allocation, so no honest caller is affected either way.
    pub fn admits(&self, cap: Option<u64>, bytes: u64) -> bool {
        match cap {
            None => true,
            Some(cap) => {
                self.vram_bytes
                    .load(Ordering::Relaxed)
                    .saturating_add(bytes)
                    <= cap
            }
        }
    }
}

/// The per-class nouns of the placement banner — everything [`spill_report_line`] cannot know.
#[derive(Debug, Clone, Copy)]
pub struct SpillNouns<'a> {
    /// Env var that enabled the spill, printed as the line's tag (`INFR_KV_OVERFLOW`).
    pub env: &'a str,
    /// Plural noun for the counted buffers (`"KV buffers"`, `"weight banks"`).
    pub noun: &'a str,
    /// Clause closing the all-resident line, after `"none spilled; "` (`"no PCIe KV reads."`).
    pub resident_note: &'a str,
    /// Clause closing the partial-spill line, after `"in "` — the placement AND who reads it over
    /// PCIe (`"SYSTEM RAM — attention reads those K/V over PCIe ..."`). Device-specific.
    pub spill_note: &'a str,
}

/// The one-shot VRAM-first placement banner, or `None` when nothing of the class was allocated.
///
/// Two shapes: everything stayed resident (worth saying — it means no PCIe reads at all), or the
/// cache split, in which case the counts on BOTH sides are named so a partial spill is visible
/// rather than looking like a slow run. Callers gate on their own enable flag and `eprintln!` the
/// result; keeping this a pure `String` builder is what lets a test pin the text.
pub fn spill_report_line(
    c: SpillCounts,
    n: &SpillNouns<'_>,
    fmt_bytes: impl Fn(u64) -> String,
) -> Option<String> {
    let total = c.total_bufs();
    if total == 0 {
        return None;
    }
    let (env, noun) = (n.env, n.noun);
    Some(if c.host_bufs == 0 {
        format!(
            "[infr] {env}: all {total} {noun} ({}) fit in VRAM — none spilled; {}",
            fmt_bytes(c.vram_bytes),
            n.resident_note,
        )
    } else {
        format!(
            "[infr] {env}: {} of {total} {noun} ({} resident) in VRAM, the remaining {} ({}) in {}",
            c.vram_bufs,
            fmt_bytes(c.vram_bytes),
            c.host_bufs,
            fmt_bytes(c.host_bytes),
            n.spill_note,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every KV spelling the two pre-hoist tables (the runner's `parse_kv_fmt` and the
    /// estimator's `kv_bytes_per_elem`) accepted, and the dtype each produced. A name dropped here
    /// silently reverts that format to f16 at runtime.
    #[test]
    fn kv_dtype_names_match_the_inline_tables() {
        for (name, want) in [
            ("f32", DType::F32),
            ("bf16", DType::Bf16),
            ("f16", DType::F16),
            ("F16", DType::F16),
            ("q8_0", DType::Q8_0),
            ("q8", DType::Q8_0),
            ("Q8_0", DType::Q8_0),
            ("q4_0", DType::Q4_0),
            ("q4_1", DType::Q4_1),
            ("q5_0", DType::Q5_0),
            ("q5_1", DType::Q5_1),
            ("iq4_nl", DType::Iq4Nl),
            ("turbo2", DType::Turbo2),
            ("turbo3", DType::Turbo3),
            ("turbo4", DType::Turbo4),
        ] {
            assert_eq!(parse_kv_dtype(name), Some(want), "{name}");
        }
        // Unset / misspelled / a weight-only quant / case variants that were never accepted all
        // fall through to the caller's default ladder.
        for name in ["", "q4_k", "Q4_0", "F32", "BF16", "turbo", "q8_1", "f 16"] {
            assert_eq!(parse_kv_dtype(name), None, "{name}");
        }
    }

    /// The per-element rates, verbatim from the pre-hoist `kv_bytes_per_elem` table in
    /// `infr-llama`'s `seam/model.rs`. These divide the free-VRAM budget in the default-ctx clamp,
    /// so a wrong one moves the context a user gets.
    #[test]
    fn kv_bytes_per_elem_matches_the_inline_table() {
        for (dt, want) in [
            (DType::F32, 4.0),
            (DType::Bf16, 2.0),
            (DType::F16, 2.0),
            (DType::Q8_0, 34.0 / 32.0),
            (DType::Q4_0, 18.0 / 32.0),
            (DType::Iq4Nl, 18.0 / 32.0),
            (DType::Q4_1, 20.0 / 32.0),
            (DType::Q5_0, 22.0 / 32.0),
            (DType::Q5_1, 24.0 / 32.0),
            (DType::Turbo2, 34.0 / 128.0),
            (DType::Turbo3, 50.0 / 128.0),
            (DType::Turbo4, 66.0 / 128.0),
        ] {
            // Bit-exact, not approximate: both sides are the same ratio of small integers, and an
            // estimator that drifts by an ULP is an estimator that was recomputed differently.
            assert_eq!(kv_bytes_per_elem(dt), want, "{dt:?}");
        }
    }

    /// The exact allocation sizes, verbatim from the pre-hoist `kv_fmt_bytes` in `infr-llama`'s
    /// `seam/mod.rs` (and the copy in `infr-metal`'s parity suite).
    #[test]
    fn kv_fmt_bytes_matches_the_inline_table() {
        let e = 4096usize;
        for (dt, want) in [
            (DType::Q8_0, (e / 32 * 34usize).next_multiple_of(4)),
            (DType::Turbo2, e / 128 * 34),
            (DType::Turbo3, e / 128 * 50),
            (DType::Turbo4, e / 128 * 66),
            (DType::Q4_0, e / 32 * 18),
            (DType::Iq4Nl, e / 32 * 18),
            (DType::Q4_1, e / 32 * 20),
            (DType::Q5_0, e / 32 * 22),
            (DType::Q5_1, e / 32 * 24),
            (DType::F16, e * 2),
            (DType::Bf16, e * 2),
            (DType::F32, e * 4),
        ] {
            assert_eq!(kv_fmt_bytes(dt, e), want, "{dt:?}");
        }
        // The Q8_0 u32 rounding is load-bearing at an odd block count: 33 blocks x 34 B = 1122,
        // which is not a u32 multiple.
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 33 * 32), 1124);
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 32 * 32), 32 * 34);
        // Sub-block element counts floor to zero bytes, as they always have (the runner never
        // allocates one — every KV row is gate-checked for block alignment first).
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 31), 0);
        assert_eq!(kv_fmt_bytes(DType::Turbo3, 127), 0);
    }

    /// A state RANGE is data, not an allocation: q8_0's `u32` pad must not enter it, and a row that
    /// splits a block is refused rather than rounded. Arithmetic ported from llama.cpp v1.0.0
    /// `src/llama-context.cpp:3521-3523` and `:3727-3728` (MIT).
    #[test]
    fn kv_state_range_bytes_counts_whole_blocks_not_strides() {
        // 3 blocks (102 B) of q8_0 per row: rows [3,7) start at byte 306, where differencing two
        // padded prefixes above (kv_fmt_bytes(Q8_0, 3*96) = 308) starts them mid-block.
        assert_eq!(kv_state_range_bytes(DType::Q8_0, 96, 3, 7), (306, 408));
        assert_eq!(kv_state_range_bytes(DType::Q8_0, 96, 0, 1), (0, 102));
        assert_ne!(
            kv_fmt_bytes(DType::Q8_0, 96),
            kv_state_range_bytes(DType::Q8_0, 96, 0, 1).1,
            "the stride is 104 and the data is 102: that gap is the bug this helper exists to avoid"
        );
        // Formats whose block is already 4-aligned agree with the sizer, and the dense ones are a
        // plain product — including the f32 recurrent state, whose whole-buffer copy is exact.
        assert_eq!(kv_state_range_bytes(DType::Q4_0, 1024, 3, 7), (1728, 2304));
        assert_eq!(
            kv_state_range_bytes(DType::Q4_0, 1024, 3, 7).1,
            kv_fmt_bytes(DType::Q4_0, 4 * 1024)
        );
        assert_eq!(kv_state_range_bytes(DType::F32, 4096, 1, 2), (16384, 16384));
        assert_eq!(kv_state_range_bytes(DType::Turbo2, 128, 5, 6), (170, 34));
        assert_eq!(kv_state_range_bytes(DType::Q4_1, 32, 0, 3), (0, 60));
        // A row that splits a block cannot be range-copied at all, so it panics instead of
        // returning a short length; an empty range is a 0-length answer, not a panic.
        assert!(std::panic::catch_unwind(|| kv_state_range_bytes(DType::Q4_0, 33, 0, 4)).is_err());
        assert_eq!(kv_state_range_bytes(DType::Q4_0, 64, 2, 2), (72, 0));
    }

    /// The PLANAR Q8_0 cache (the Vulkan KV layout) carries a row range in two disjoint regions, so
    /// the span list, not the packed length, is what a copy must follow. Offsets checked against
    /// `infr-vulkan/shaders/store_q8.comp:58-67`: code byte = element index, scale of block `b` at
    /// byte `cap + b*2`.
    #[test]
    fn kv_state_copy_spans_splits_a_planar_q8_range_into_codes_and_scales() {
        // cap 4096 elements = 128 whole blocks; a 32-element row is exactly one block.
        let cap = 4096;
        assert_eq!(
            kv_state_copy_spans(DType::Q8_0, 32, cap, 0, 3, true),
            vec![(0, 0, 96), (4096, 4096, 6)],
            "3 rows = 3 code blocks and 3 scales"
        );
        assert_eq!(
            kv_state_copy_spans(DType::Q8_0, 32, cap, 3, 7, true),
            vec![(96, 96, 128), (4102, 4102, 8)],
            "rows [3,7) start at scale byte cap + 3*2"
        );
        // An ODD block count below the range moves the scale base — the packed offset alone would
        // have said 4102 for rows [3,7) of any row width and been wrong for a different one.
        assert_eq!(
            kv_state_copy_spans(DType::Q8_0, 576, 4128, 0, 3, true),
            vec![(0, 0, 1728), (4128, 4128, 108)],
            "129 blocks in the cache: the scale region starts at byte 4128, not at a padded stride"
        );
        // The two spans are the SAME bytes as the packed range, never more and never less.
        for (row, r0, r1) in [
            (32, 0, 3),
            (32, 3, 7),
            (96, 3, 7),
            (576, 0, 3),
            (1024, 5, 40),
        ] {
            let cap = 4096 * row / 32;
            let spans = kv_state_copy_spans(DType::Q8_0, row, cap, r0, r1, true);
            let len = kv_state_range_bytes(DType::Q8_0, row, r0, r1).1;
            assert_eq!(
                spans[0].0,
                r0 * row,
                "the codes region is indexed by ELEMENT offset"
            );
            assert_eq!(
                spans.iter().map(|s| s.2).sum::<usize>(),
                len,
                "codes + scales must account for exactly the packed {len} bytes"
            );
            assert!(
                spans[1].0 + spans[1].2 <= cap + cap / 32 * 2,
                "the scale span must stay inside the cache's scale region"
            );
        }
        // Packed layouts get the single span back, unchanged — the q4_*, Turbo, f16/f32 and the
        // CPU/Metal Q8_0 seeds all still copy one contiguous range.
        assert_eq!(
            kv_state_copy_spans(DType::Q8_0, 96, 96 * 16, 3, 7, false),
            vec![(306, 306, 408)]
        );
        assert_eq!(
            kv_state_copy_spans(DType::Q4_0, 1024, 1024 * 8, 3, 7, false),
            vec![(1728, 1728, 2304)]
        );
        // An empty range carries no bytes in either layout.
        assert!(kv_state_copy_spans(DType::Q8_0, 96, 96 * 16, 4, 4, false).is_empty());
        assert!(kv_state_copy_spans(DType::Q8_0, 96, 96 * 16, 4, 4, true).is_empty());
        // Asking a packed-declared format to be split planar is a geometry lie, and it panics
        // rather than silently copying the wrong regions.
        assert!(std::panic::catch_unwind(|| kv_state_copy_spans(
            DType::Q4_0,
            1024,
            8192,
            0,
            2,
            true
        ))
        .is_err());
    }

    /// The exact sizer and the estimator rate are two views of one number: on any whole-block
    /// count they must agree (modulo Q8_0's u32 rounding, which only ever rounds UP).
    #[test]
    fn kv_fmt_bytes_matches_the_per_elem_rate() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::Q8_0,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Iq4Nl,
            DType::Turbo2,
            DType::Turbo3,
            DType::Turbo4,
        ] {
            for blocks in [1usize, 3, 128, 1024] {
                let elems = blocks * 128; // a multiple of both block sizes
                let exact = kv_fmt_bytes(dt, elems) as f64;
                let rate = kv_bytes_per_elem(dt) * elems as f64;
                assert!(
                    exact >= rate && exact - rate < 4.0,
                    "{dt:?} x {elems}: exact {exact} vs rate {rate}"
                );
            }
        }
    }

    /// `empty`/`0`/unset = off; anything else = on — over explicit values via [`flag_from`], so
    /// no test in this binary has to touch the process environment.
    #[test]
    fn flag_grammar() {
        assert!(!flag_from(None));
        for (raw, want) in [
            ("", false),
            ("0", false),
            ("1", true),
            ("00", true), // only the exact string "0" is off — matches the shipped grammar
            ("true", true),
            ("no", true),
        ] {
            assert_eq!(flag_from(Some(raw)), want, "raw={raw:?}");
        }
    }

    /// The same grammar as the CONFIG layer resolves it: `config::env`'s `flag` reader is
    /// `flag_from`, so a `Config` built from these spellings carries exactly the booleans above —
    /// which is what the spill sites now read.
    #[test]
    fn kv_overflow_config_field_follows_the_flag_grammar() {
        use crate::config::Config;
        for (raw, want) in [("", false), ("0", false), ("1", true), ("true", true)] {
            let key = "INFR_KV_OVERFLOW";
            let layer =
                crate::config::env::parse(&|k| (k == key).then(|| raw.to_string())).expect(key);
            let cfg = Config::load_from_layers(&[layer]);
            assert_eq!(cfg.kv.overflow, want, "{key}={raw:?}");
        }
        // Unset ⇒ the shipped default, off.
        assert!(!Config::default().kv.overflow);
    }

    /// MiB → bytes, with `0` distinct from unset (it forces whole-host placement) — over explicit
    /// values, no environment.
    ///
    /// This used to run over a `&str` façade (`mib_from`), which existed only so this sweep could
    /// avoid the process environment — something [`mib_bytes`] already allows. The STRING half of
    /// the grammar (trim, reject a float / a negative / an empty value) belongs to
    /// `config::env`'s `opt_mib` now and is pinned there; what is left here is the arithmetic.
    #[test]
    fn mib_grammar() {
        assert_eq!(mib_bytes(None), None);
        for (mib, want) in [
            (0u64, Some(0u64)),
            (1, Some(1024 * 1024)),
            (512, Some(512 * 1024 * 1024)),
        ] {
            assert_eq!(mib_bytes(Some(mib)), want, "mib={mib}");
        }
    }

    /// The MiB CAPS as the read sites now take them: a `Config` field in MiB, [`mib_bytes`] to
    /// bytes. `Some(0)` (whole-host placement) must survive as `Some(0)`, not collapse to "no cap".
    #[test]
    fn mib_caps_come_off_the_config_in_mib() {
        use crate::config::Config;
        assert_eq!(mib_bytes(None), None);
        assert_eq!(mib_bytes(Some(0)), Some(0));
        assert_eq!(mib_bytes(Some(512)), Some(512 * 1024 * 1024));
        // Unset ⇒ no cap on either class; set ⇒ the MiB count, byte-converted at the site.
        assert_eq!(Config::default().kv.overflow_vram_mb, None);
        let layer = crate::config::env::parse(&|k| match k {
            "INFR_KV_OVERFLOW_VRAM_MB" => Some("  512  ".to_string()),
            _ => None,
        })
        .expect("env layer");
        let cfg = Config::load_from_layers(&[layer]);
        assert_eq!(mib_bytes(cfg.kv.overflow_vram_mb), Some(512 * 1024 * 1024));
    }

    /// 12% of total, floored at 2 GiB, override in MiB — verbatim from the inline copy
    /// (`kv_overflow_vram_reserve`), over explicit values so the case sweep needs no environment.
    ///
    /// This used to run over a `&str` façade (`reserve_from`) purely to keep the environment out of
    /// the sweep, which [`reserve_bytes`] already does. An unparseable override is `None` by the
    /// time it reaches here — `config::env` decides that — and `None` is the first case below, so
    /// the deleted `Some("abc")` row is the same assertion spelled twice.
    #[test]
    fn reserve_bytes_matches_the_inline_formula() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Below the crossover (2 GiB / 0.12 ≈ 16.67 GiB) the floor wins.
        assert_eq!(reserve_bytes(8 * GIB, None), 2 * GIB);
        assert_eq!(reserve_bytes(16 * GIB, None), 2 * GIB);
        // Above it, 12% — computed as `total / 100 * 12`, the shipped (truncating) order.
        assert_eq!(reserve_bytes(24 * GIB, None), 24 * GIB / 100 * 12);
        assert_eq!(reserve_bytes(0, None), 2 * GIB);
        // The override wins outright, including below the floor and at zero.
        assert_eq!(reserve_bytes(24 * GIB, Some(128)), 128 * 1024 * 1024);
        assert_eq!(reserve_bytes(24 * GIB, Some(0)), 0);
    }

    /// The same policy driven off the CONFIG field (MiB) — the plumbing, where the test above is
    /// the formula.
    #[test]
    fn reserve_bytes_takes_the_override_from_the_config() {
        use crate::config::Config;
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(reserve_bytes(24 * GIB, None), 24 * GIB / 100 * 12);
        assert_eq!(reserve_bytes(8 * GIB, None), 2 * GIB);
        assert_eq!(reserve_bytes(24 * GIB, Some(128)), 128 * 1024 * 1024);
        assert_eq!(reserve_bytes(24 * GIB, Some(0)), 0);
        // Unset in the shipped config ⇒ the formula.
        let d = Config::default();
        assert_eq!(d.kv.overflow_reserve_mb, None);
        let layer = crate::config::env::parse(&|k| {
            (k == "INFR_KV_OVERFLOW_RESERVE_MB").then(|| "128".to_string())
        })
        .expect("env layer");
        let cfg = Config::load_from_layers(&[layer]);
        assert_eq!(
            reserve_bytes(24 * GIB, cfg.kv.overflow_reserve_mb),
            128 * 1024 * 1024
        );
    }

    /// A MiB count large enough to overflow the byte product must SATURATE, not wrap.
    ///
    /// `u64` MiB values this big are typeable and `parse::<u64>()`-able (a fat-fingered
    /// `INFR_KV_OVERFLOW_VRAM_MB`, a config file with a placeholder), and the wrapped result is
    /// actively harmful: `2^44` MiB wraps to exactly `0`, i.e. the LOUDEST possible cap turns into
    /// "nothing may stay resident" and the whole class goes to host. In debug the same expression
    /// panics instead. Both are worse than clamping to "effectively no cap".
    #[test]
    fn mib_bytes_saturates_instead_of_wrapping() {
        const MIB: u64 = 1024 * 1024;
        // The last MiB count that still fits, and the first that does not.
        let last_exact = u64::MAX / MIB; // 17_592_186_044_415
        assert_eq!(mib_bytes(Some(last_exact)), Some(last_exact * MIB));
        assert_eq!(mib_bytes(Some(last_exact + 1)), Some(u64::MAX));
        // `2^44` MiB is the pathological one: `2^44 * 2^20 == 2^64`, which WRAPS to 0.
        assert_eq!(mib_bytes(Some(1 << 44)), Some(u64::MAX));
        assert_eq!(mib_bytes(Some(u64::MAX)), Some(u64::MAX));
        // A saturated cap behaves as "no cap" at the gate it feeds — no real byte count reaches it.
        let t = SpillTally::default();
        t.record_vram(1 << 40);
        assert!(t.admits(mib_bytes(Some(u64::MAX)), 1 << 40));
        // And the ordinary values are untouched by the change.
        assert_eq!(mib_bytes(None), None);
        assert_eq!(mib_bytes(Some(0)), Some(0));
        assert_eq!(mib_bytes(Some(512)), Some(512 * MIB));
        // `reserve_bytes` reads the same helper, so its override saturates too rather than
        // wrapping to a tiny (or zero) reserve, which would under-reserve and OOM mid-forward.
        assert_eq!(reserve_bytes(24 * 1024 * MIB, Some(u64::MAX)), u64::MAX);
    }

    /// The cumulative cap admits up to and including the cap, and `Some(0)` admits nothing —
    /// the whole-host reproducibility case.
    #[test]
    fn spill_tally_cap_boundaries() {
        let t = SpillTally::default();
        assert!(t.admits(None, u64::MAX));
        assert!(t.admits(Some(0), 0));
        assert!(!t.admits(Some(0), 1));
        assert!(t.admits(Some(100), 100)); // exactly the cap fits
        assert!(!t.admits(Some(100), 101));

        t.record_vram(60);
        assert!(t.admits(Some(100), 40));
        assert!(!t.admits(Some(100), 41));
        t.record_vram(40);
        assert!(t.admits(Some(100), 0));
        assert!(!t.admits(Some(100), 1));
        // Host placements never consume the VRAM cap.
        t.record_host(1 << 40);
        assert!(t.admits(Some(200), 100));

        let c = t.counts();
        assert_eq!(
            c,
            SpillCounts {
                vram_bufs: 2,
                vram_bytes: 100,
                host_bufs: 1,
                host_bytes: 1 << 40,
            }
        );
        assert_eq!(c.total_bufs(), 3);
    }

    /// The cap gate's `resident + bytes` must SATURATE. A wrapped sum is a small sum, and a small
    /// sum passes a cap it should have failed — the tally would go on admitting buffers into VRAM
    /// after the cap was already blown (and in debug it panics outright inside a placement
    /// decision). Saturating refuses, which is the safe side: the buffer spills to host.
    #[test]
    fn spill_tally_cap_saturates_instead_of_wrapping() {
        let t = SpillTally::default();
        t.record_vram(1 << 40);
        // `(1<<40) + u64::MAX` wraps to `(1<<40) - 1`, which would sneak under a modest cap.
        assert!(!t.admits(Some(1 << 41), u64::MAX));
        assert!(!t.admits(Some(u64::MAX - 1), u64::MAX));
        // An uncapped tally never consults the sum at all, so it still admits anything.
        assert!(t.admits(None, u64::MAX));
        // A cap of exactly `u64::MAX` (what a saturated `mib_bytes` produces) still admits, since
        // the saturated sum compares equal to it.
        assert!(t.admits(Some(u64::MAX), u64::MAX));
        // Right at the boundary with no saturation involved: the honest arithmetic is unchanged.
        let u = SpillTally::default();
        u.record_vram(u64::MAX - 10);
        assert!(u.admits(Some(u64::MAX), 10));
    }

    /// The banner, byte-for-byte against the pre-hoist `eprintln!` (Vulkan KV).
    #[test]
    fn spill_report_line_reproduces_the_inline_banners() {
        /// Vulkan's `fmt_bytes` (MiB to one decimal).
        fn vk_fmt(b: u64) -> String {
            const KIB: u64 = 1 << 10;
            const MIB: u64 = 1 << 20;
            const GIB: u64 = 1 << 30;
            match b {
                b if b >= GIB => format!("{:.2} GiB", b as f64 / GIB as f64),
                b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
                b if b >= KIB => format!("{:.1} KiB", b as f64 / KIB as f64),
                b => format!("{b} B"),
            }
        }
        const VK_KV: SpillNouns = SpillNouns {
            env: "INFR_KV_OVERFLOW",
            noun: "KV buffers",
            resident_note: "no PCIe KV reads.",
            spill_note: "SYSTEM RAM — attention reads those K/V over PCIe by device address \
                         (PCIe-bound on the spilled layers). Spilled KV bytes are exempt from the \
                         VRAM budget guard.",
        };

        // Nothing allocated (or the flag off, which leaves the tally at zero) ⇒ no line.
        assert_eq!(
            spill_report_line(SpillCounts::default(), &VK_KV, vk_fmt),
            None
        );

        let all_resident = SpillCounts {
            vram_bufs: 56,
            vram_bytes: 3 << 30,
            host_bufs: 0,
            host_bytes: 0,
        };
        assert_eq!(
            spill_report_line(all_resident, &VK_KV, vk_fmt).unwrap(),
            "[infr] INFR_KV_OVERFLOW: all 56 KV buffers (3.00 GiB) fit in VRAM — none spilled; \
             no PCIe KV reads."
        );

        let split = SpillCounts {
            vram_bufs: 40,
            vram_bytes: 3 << 30,
            host_bufs: 16,
            host_bytes: 1 << 30,
        };
        assert_eq!(
            spill_report_line(split, &VK_KV, vk_fmt).unwrap(),
            "[infr] INFR_KV_OVERFLOW: 40 of 56 KV buffers (3.00 GiB resident) in VRAM, the \
             remaining 16 (1.00 GiB) in SYSTEM RAM — attention reads those K/V over PCIe by \
             device address (PCIe-bound on the spilled layers). Spilled KV bytes are exempt from \
             the VRAM budget guard."
        );
    }
}
