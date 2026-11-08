//! Dequant index rules: for element `i` of one block of each ggml type the reference dequantizes
//! (`lucina-ref::quant::dequant_row`), the byte offsets it reads and the integer parts it extracts (4/5/6-bit
//! quants, 6-bit scales and mins, IQ3_S grid indices and sign bits). Verus proves every offset stays inside the
//! block, every offset follows ggml's layout formula, and every part is in its format's range. The f32 arithmetic
//! is not proved here; `crates/lucina-ref/tests/dequant_parts.rs` checks bit for bit that the shipped
//! `dequant_row` equals these parts combined in ggml's f32 order.
//!
//! Verus proves this very file (`verification/lucina-dequant/rules.rs`; the attributes are only seen by Verus).
//! Original: llama.cpp 3612199ed `ggml/src/ggml-quants.c` (`dequantize_row_q8_0`, `_q3_K`, `_q4_K`, `_q5_K`,
//! `_q6_K`, `_iq4_nl`, `_iq4_xs`, `_iq3_s`, `get_scale_min_k4`). Drafted by Qwen, rewritten 2026-11-08.
#![allow(unexpected_cfgs)]

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const QK8_0: usize = 32;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const QK_K: usize = 256;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const QK4_NL: usize = 32;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q8_0_BYTES: usize = 34;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q3_K_BYTES: usize = 110;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q4_K_BYTES: usize = 144;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q5_K_BYTES: usize = 176;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const Q6_K_BYTES: usize = 210;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const IQ4_NL_BYTES: usize = 18;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const IQ3_S_BYTES: usize = 110;
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const IQ4_XS_BYTES: usize = 136;

/// (elements per block, bytes per block) for the ggml type ids the reference dequantizes, `None` for the rest.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        t == 0 ==> r == Some((1usize, 4usize)),
        (t == 1 || t == 30) ==> r == Some((1usize, 2usize)),
        t == 8 ==> r == Some((32usize, 34usize)),
        t == 11 ==> r == Some((256usize, 110usize)),
        t == 12 ==> r == Some((256usize, 144usize)),
        t == 13 ==> r == Some((256usize, 176usize)),
        t == 14 ==> r == Some((256usize, 210usize)),
        t == 20 ==> r == Some((32usize, 18usize)),
        t == 21 ==> r == Some((256usize, 110usize)),
        t == 23 ==> r == Some((256usize, 136usize)),
        !(t == 0 || t == 1 || t == 30 || t == 8 || t == 11 || t == 12 || t == 13 || t == 14 || t == 20
            || t == 21 || t == 23) ==> r is None,
))]
pub fn block_layout(t: u32) -> Option<(usize, usize)> {
    match t {
        0 => Some((1, 4)),
        1 | 30 => Some((1, 2)),
        8 => Some((QK8_0, Q8_0_BYTES)),
        11 => Some((QK_K, Q3_K_BYTES)),
        12 => Some((QK_K, Q4_K_BYTES)),
        13 => Some((QK_K, Q5_K_BYTES)),
        14 => Some((QK_K, Q6_K_BYTES)),
        20 => Some((QK4_NL, IQ4_NL_BYTES)),
        21 => Some((QK_K, IQ3_S_BYTES)),
        23 => Some((QK_K, IQ4_XS_BYTES)),
        _ => None,
    }
}

/// `get_scale_min_k4`: the 6-bit scale and min of sub-block `j` from the 12 packed bytes at `b[off..off + 12]`.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        j < 8,
        off + 12 <= b.len(),
    ensures
        r.0 < 64,
        r.1 < 64,
        j < 4 ==> r.0 == (b@[off + j] & 63u8) && r.1 == (b@[off + j + 4] & 63u8),
))]
pub fn scale_min_k4(b: &[u8], off: usize, j: usize) -> (u8, u8) {
    if j < 4 {
        let (x, y) = (b[off + j], b[off + j + 4]);
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((x & 63u8) < 64u8) by (bit_vector);
            assert((y & 63u8) < 64u8) by (bit_vector);
        }
        (x & 63, y & 63)
    } else {
        let (a, lo, hi) = (b[off + j + 4], b[off + j - 4], b[off + j]);
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(((a & 0xFu8) | ((lo >> 6u8) << 4u8)) < 64u8) by (bit_vector);
            assert(((a >> 4u8) | ((hi >> 6u8) << 4u8)) < 64u8) by (bit_vector);
        }
        ((a & 0xF) | ((lo >> 6) << 4), (a >> 4) | ((hi >> 6) << 4))
    }
}

/// Element `i` of a Q8_0 block: `y = d * quant`, `d` = the half at byte 0.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == Q8_0_BYTES,
        i < QK8_0,
    ensures
        -128 <= r && r < 128,
        // the element's own int8, right after the 2-byte scale
        r == b@[2 + i] as i8 as i32,
))]
pub fn q8_0_quant(b: &[u8], i: usize) -> i32 {
    b[2 + i] as i8 as i32
}

/// Element `i` of a K4-style block (Q4_K, Q5_K): `y = d * scale * quant - dmin * min`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K4Parts {
    pub quant: u8,
    pub scale: u8,
    pub min: u8,
    /// the byte holding the element's 4 low bits
    pub qs_off: usize,
    /// sub-block 0..8 (index into the packed scales)
    pub sub: usize,
}

/// Element `i` of a Q4_K block { half d; half dmin; u8 scales[12]; u8 qs[128]; }.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == Q4_K_BYTES,
        i < QK_K,
    ensures
        r.quant < 16,
        r.scale < 64,
        r.min < 64,
        r.sub == i / 32,
        r.qs_off == 16 + 32 * (i / 64) + i % 32,
        r.qs_off < b.len(),
))]
pub fn q4_k_parts(b: &[u8], i: usize) -> K4Parts {
    let t = i / 64;
    let r = i % 64;
    let sub = i / 32;
    let qs_off = 16 + 32 * t + r % 32;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(t < 4 && sub < 8 && sub == 2 * t + r / 32 && r % 32 == i % 32) by (nonlinear_arith)
            requires t == i / 64, r == i % 64, sub == i / 32, i < 256;
    }
    let (scale, min) = scale_min_k4(b, 4, sub);
    let q = b[qs_off];
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((q & 0xFu8) < 16u8) by (bit_vector);
        assert((q >> 4u8) < 16u8) by (bit_vector);
    }
    let quant = if r < 32 { q & 0xF } else { q >> 4 };
    K4Parts { quant, scale, min, qs_off, sub }
}

/// Element `i` of a Q5_K block { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }. The fifth bit of
/// element `i` is bit `i / 32` of `qh[i % 32]`.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == Q5_K_BYTES,
        i < QK_K,
    ensures
        r.0.quant < 32,
        r.0.scale < 64,
        r.0.min < 64,
        r.0.sub == i / 32,
        r.0.qs_off == 48 + 32 * (i / 64) + i % 32,
        r.0.qs_off < b.len(),
        r.1 == 16 + i % 32,
        r.1 < b.len(),
))]
pub fn q5_k_parts(b: &[u8], i: usize) -> (K4Parts, usize) {
    let t = i / 64;
    let r = i % 64;
    let sub = i / 32;
    let l = r % 32;
    let qs_off = 48 + 32 * t + l;
    let qh_off = 16 + l;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(t < 4 && sub < 8 && l == i % 32 && l < 32) by (nonlinear_arith)
            requires t == i / 64, r == i % 64, sub == i / 32, l == r % 32, i < 256;
    }
    let (scale, min) = scale_min_k4(b, 4, sub);
    let (q, h) = (b[qs_off], b[qh_off]);
    let sh = sub as u8;
    let u: u8 = 1u8 << sh;
    let hi: u8 = if h & u != 0 { 16 } else { 0 };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((q & 0xFu8) < 16u8) by (bit_vector);
        assert((q >> 4u8) < 16u8) by (bit_vector);
    }
    let quant = (if r < 32 { q & 0xF } else { q >> 4 }) + hi;
    (K4Parts { quant, scale, min, qs_off, sub }, qh_off)
}

/// Element `i` of a Q6_K block { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d (at 208); }:
/// `y = d * scale * quant`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q6KParts {
    pub quant: i32,
    pub scale: i32,
    pub ql_off: usize,
    pub qh_off: usize,
    pub sc_off: usize,
}

#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == Q6_K_BYTES,
        i < QK_K,
    ensures
        -32 <= r.quant && r.quant < 32,
        -128 <= r.scale && r.scale < 128,
        r.ql_off == 64 * (i / 128) + i % 32 + (if (i % 128 / 32) % 2 == 1 { 32int } else { 0int }),
        r.qh_off == 128 + 32 * (i / 128) + i % 32,
        r.sc_off == 192 + 8 * (i / 128) + (i % 32) / 16 + 2 * (i % 128 / 32),
        r.ql_off < 128,
        128 <= r.qh_off && r.qh_off < 192,
        192 <= r.sc_off && r.sc_off < 208,
))]
pub fn q6_k_parts(b: &[u8], i: usize) -> Q6KParts {
    let h = i / 128;
    let rem = i % 128;
    let g = rem / 32;
    let l = i % 32;
    let ql_off = 64 * h + l + if g % 2 == 1 { 32 } else { 0 };
    let qh_off = 128 + 32 * h + l;
    let sc_off = 192 + 8 * h + l / 16 + 2 * g;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(h < 2 && g < 4 && l < 32 && l / 16 < 2) by (nonlinear_arith)
            requires h == i / 128, rem == i % 128, g == rem / 32, l == i % 32, i < 256;
    }
    let (q, hb) = (b[ql_off], b[qh_off]);
    let low: u8 = if g < 2 { q & 0xF } else { q >> 4 };
    let sh = (2 * g) as u8;
    let v: u8 = low | (((hb >> sh) & 3) << 4);
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((q & 0xFu8) < 16u8) by (bit_vector);
        assert((q >> 4u8) < 16u8) by (bit_vector);
        assert((low | (((hb >> sh) & 3u8) << 4u8)) < 64u8) by (bit_vector) requires low < 16u8;
    }
    Q6KParts { quant: v as i32 - 32, scale: b[sc_off] as i8 as i32, ql_off, qh_off, sc_off }
}

/// Element `i` of a Q3_K block { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d (at 108); }:
/// `y = d * (scale - 32) * quant`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q3KParts {
    pub quant: i32,
    pub scale: u8,
    pub qs_off: usize,
    /// bit of `qs[qs_off]` the 2-bit quant starts at
    pub shift: u8,
    /// hmask byte and bit of the element
    pub hm_off: usize,
    pub hm_bit: u8,
    /// which of the 16 expanded scales
    pub group: usize,
}

#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == Q3_K_BYTES,
        i < QK_K,
    ensures
        -4 <= r.quant && r.quant < 4,
        r.scale < 64,
        r.qs_off == 32 + 32 * (i / 128) + i % 32,
        r.shift == 2 * (i % 128 / 32),
        r.hm_off == i % 32,
        r.hm_bit == 4 * (i / 128) + i % 128 / 32,
        r.group == i / 16,
        32 <= r.qs_off && r.qs_off < 96,
        r.hm_bit < 8,
        r.group < 16,
))]
pub fn q3_k_parts(b: &[u8], i: usize) -> Q3KParts {
    let n = i / 128;
    let j = i % 128 / 32;
    let l = i % 32;
    let group = i / 16;
    let qs_off = 32 + 32 * n + l;
    let hm_bit = (4 * n + j) as u8;
    let shift = (2 * j) as u8;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(n < 2 && j < 4 && l < 32 && group < 16 && group == 8 * n + 2 * j + l / 16) by (nonlinear_arith)
            requires n == i / 128, j == i % 128 / 32, l == i % 32, group == i / 16, i < 256;
    }
    // the 16 scales of ggml's aux[] expansion: group k = 4*hi + lo
    let (hi, lo) = (group / 4, group % 4);
    let (s0, s4, s8) = (b[96 + lo], b[100 + lo], b[104 + lo]);
    let scale: u8 = if hi == 0 {
        (s0 & 0x0F) | ((s8 & 0x03) << 4)
    } else if hi == 1 {
        (s4 & 0x0F) | (((s8 >> 2) & 0x03) << 4)
    } else if hi == 2 {
        (s0 >> 4) | (((s8 >> 4) & 0x03) << 4)
    } else {
        (s4 >> 4) | (((s8 >> 6) & 0x03) << 4)
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(((s0 & 0x0Fu8) | ((s8 & 0x03u8) << 4u8)) < 64u8) by (bit_vector);
        assert(((s4 & 0x0Fu8) | (((s8 >> 2u8) & 0x03u8) << 4u8)) < 64u8) by (bit_vector);
        assert(((s0 >> 4u8) | (((s8 >> 4u8) & 0x03u8) << 4u8)) < 64u8) by (bit_vector);
        assert(((s4 >> 4u8) | (((s8 >> 6u8) & 0x03u8) << 4u8)) < 64u8) by (bit_vector);
        let q = b@[qs_off as int];
        assert(((q >> shift) & 3u8) < 4u8) by (bit_vector);
    }
    let qv = ((b[qs_off] >> shift) & 3) as i32;
    let m: u8 = 1u8 << hm_bit;
    let adj: i32 = if b[l] & m != 0 { 0 } else { 4 };
    Q3KParts { quant: qv - adj, scale, qs_off, shift, hm_off: l, hm_bit, group }
}

/// Element `i` of an IQ4_NL block { half d; u8 qs[16]; }: `y = d * KVALUES_IQ4NL[code]`. Element `j < 16` is the
/// low nibble of `qs[j]`, element `j >= 16` the high nibble of `qs[j - 16]`.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == IQ4_NL_BYTES,
        i < QK4_NL,
    ensures
        r.0 < 16,
        r.1 == 2 + i % 16,
        r.1 < b.len(),
))]
pub fn iq4_nl_code(b: &[u8], i: usize) -> (u8, usize) {
    let off = 2 + i % 16;
    let q = b[off];
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((q & 0xFu8) < 16u8) by (bit_vector);
        assert((q >> 4u8) < 16u8) by (bit_vector);
    }
    (if i < 16 { q & 0xF } else { q >> 4 }, off)
}

/// Element `i` of an IQ4_XS block { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }:
/// `y = d * scale * KVALUES_IQ4NL[code]` with the 6-bit scale signed by `- 32`.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == IQ4_XS_BYTES,
        i < QK_K,
    ensures
        r.0 < 16,
        -32 <= r.1 && r.1 < 32,
        r.2 == 8 + 16 * (i / 32) + i % 16,
        r.2 < b.len(),
))]
pub fn iq4_xs_parts(b: &[u8], i: usize) -> (u8, i32, usize) {
    let ib = i / 32;
    let j = i % 32;
    let qs_off = 8 + 16 * ib + j % 16;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(ib < 8 && j < 32 && j % 16 == i % 16 && ib / 2 < 4) by (nonlinear_arith)
            requires ib == i / 32, j == i % 32, i < 256;
    }
    let sl = b[4 + ib / 2] as u32;
    let sh = (b[2] as u32) | ((b[3] as u32) << 8);
    let s1 = (4 * (ib % 2)) as u32;
    let s2 = (2 * ib) as u32;
    let ls = ((sl >> s1) & 0xF) | (((sh >> s2) & 3) << 4);
    let q = b[qs_off];
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((((sl >> s1) & 0xFu32) | (((sh >> s2) & 3u32) << 4u32)) < 64u32) by (bit_vector);
        assert((q & 0xFu8) < 16u8) by (bit_vector);
        assert((q >> 4u8) < 16u8) by (bit_vector);
    }
    let code = if j < 16 { q & 0xF } else { q >> 4 };
    (code, ls as i32 - 32, qs_off)
}

/// Element `i` of an IQ3_S block { half d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]; }:
/// `y = d * (1 + 2 * scale_nib) * byte(IQ3S_GRID[grid], nib) * (sign ? -1 : 1)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IQ3SParts {
    pub grid: u32,
    pub nib: u32,
    pub sign: bool,
    pub scale_nib: u8,
    pub qs_off: usize,
    pub qh_off: usize,
    pub qh_bit: u8,
    pub sign_off: usize,
    pub sign_bit: u8,
}

#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        b.len() == IQ3_S_BYTES,
        i < QK_K,
    ensures
        r.grid < 512,
        r.nib < 4,
        r.scale_nib < 16,
        r.qs_off == 2 + 8 * (i / 32) + 2 * (i % 32 / 8) + i % 8 / 4,
        r.qh_off == 66 + i / 32,
        r.qh_bit == 2 * (i % 32 / 8) + i % 8 / 4,
        r.sign_off == 74 + 4 * (i / 32) + i % 32 / 8,
        r.sign_bit == i % 8,
        r.nib == i % 4,
        2 <= r.qs_off && r.qs_off < 66,
        66 <= r.qh_off && r.qh_off < 74,
        74 <= r.sign_off && r.sign_off < 106,
))]
pub fn iq3_s_parts(b: &[u8], i: usize) -> IQ3SParts {
    // ggml walks 32-wide halves; half k (0..8) owns qs[8k..8k+8], qh[k], signs[4k..4k+4] and the low (even k)
    // or high (odd k) nibble of scales[k / 2]
    let k = i / 32;
    let w = i % 32;
    let l = w / 8;
    let e = i % 8;
    let g2 = e / 4;
    let qs_off = 2 + 8 * k + 2 * l + g2;
    let qh_off = 66 + k;
    let qh_bit = (2 * l + g2) as u8;
    let sign_off = 74 + 4 * k + l;
    let sign_bit = e as u8;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(k < 8 && l < 4 && e < 8 && g2 < 2 && w == i % 32 && e == w % 8 && i % 4 == e % 4 && k / 2 < 4)
            by (nonlinear_arith)
            requires k == i / 32, w == i % 32, l == w / 8, e == i % 8, g2 == e / 4, i < 256;
    }
    let qh = b[qh_off];
    let high: u32 = if (qh >> qh_bit) & 1 != 0 { 256 } else { 0 };
    let base = b[qs_off] as u32;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((base | 256u32) < 512u32) by (bit_vector) requires base < 256u32;
        assert((base | 0u32) < 512u32) by (bit_vector) requires base < 256u32;
    }
    let grid = base | high;
    let sign = b[sign_off] & (1u8 << sign_bit) != 0;
    let sb = b[106 + k / 2];
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((sb & 0xFu8) < 16u8) by (bit_vector);
        assert((sb >> 4u8) < 16u8) by (bit_vector);
    }
    let scale_nib = if k % 2 == 0 { sb & 0xF } else { sb >> 4 };
    IQ3SParts { grid, nib: (e % 4) as u32, sign, scale_nib, qs_off, qh_off, qh_bit, sign_off, sign_bit }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    #[test]
    fn layout_names_every_supported_type() {
        assert_eq!(block_layout(0), Some((1, 4)));
        assert_eq!(block_layout(1), Some((1, 2)));
        assert_eq!(block_layout(30), Some((1, 2)));
        assert_eq!(block_layout(8), Some((32, 34)));
        assert_eq!(block_layout(11), Some((256, 110)));
        assert_eq!(block_layout(12), Some((256, 144)));
        assert_eq!(block_layout(13), Some((256, 176)));
        assert_eq!(block_layout(14), Some((256, 210)));
        assert_eq!(block_layout(20), Some((32, 18)));
        assert_eq!(block_layout(21), Some((256, 110)));
        assert_eq!(block_layout(23), Some((256, 136)));
        assert_eq!(block_layout(2), None);
        assert_eq!(block_layout(255), None);
    }

    /// The Q3_K scale of every element equals ggml's explicit 16-entry aux[] expansion.
    #[test]
    fn q3_k_scale_matches_the_aux_expansion() {
        for seed in 1..20 {
            let b = bytes(Q3_K_BYTES, seed);
            let s = &b[96..108];
            let mut sc = [0u8; 16];
            for k in 0..4 {
                sc[k] = s[k] & 0x0f | (s[8 + k] & 0x03) << 4;
                sc[4 + k] = s[4 + k] & 0x0f | ((s[8 + k] >> 2) & 0x03) << 4;
                sc[8 + k] = (s[k] >> 4) | ((s[8 + k] >> 4) & 0x03) << 4;
                sc[12 + k] = (s[4 + k] >> 4) | ((s[8 + k] >> 6) & 0x03) << 4;
            }
            for i in 0..QK_K {
                let p = q3_k_parts(&b, i);
                assert_eq!(p.scale, sc[i / 16], "i {i}");
            }
        }
    }

    /// IQ3_S: the grid index from the parts equals ggml's shift-and-mask form.
    #[test]
    fn iq3_s_grid_matches_ggml_shift_form() {
        for seed in 1..20 {
            let b = bytes(IQ3_S_BYTES, seed);
            for i in 0..QK_K {
                let p = iq3_s_parts(&b, i);
                let (k, l, e) = (i / 32, (i % 32) / 8, i % 8);
                let qs = &b[2 + 8 * k..];
                let qh = b[66 + k] as u32;
                let g = if e < 4 {
                    qs[2 * l] as u32 | ((qh << (8 - 2 * l)) & 256)
                } else {
                    qs[2 * l + 1] as u32 | ((qh << (7 - 2 * l)) & 256)
                };
                assert_eq!(p.grid, g, "i {i}");
                assert_eq!(p.sign, b[74 + 4 * k + l] & (1 << e) != 0);
            }
        }
    }
}
