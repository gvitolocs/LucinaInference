//! Row dequantization, ported from llama.cpp ggml/src/ggml-quants.c at 3612199ed (MIT, (c) the ggml
//! authors): `dequantize_row_q8_0`, `_q4_K`, `_q5_K`, `_q6_K`, `get_scale_min_k4`, `_q3_K`,
//! `_iq4_nl`, `_iq4_xs`, `_iq3_s` (block structs and lookup tables from ggml/src/ggml-common.h, see
//! `iq_tables.rs`). The arithmetic is kept in f32 exactly as ggml does it, so a row matches ggml's
//! `to_float` bit for bit; the reference forward widens to f64 afterwards.
use anyhow::{bail, Result};
use half::{bf16, f16};

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ4_NL: u32 = 20;
pub const IQ3_S: u32 = 21;
pub const IQ4_XS: u32 = 23;
pub const BF16: u32 = 30;

const QK_K: usize = 256;
const QK4_NL: usize = 32;

/// (elements per block, bytes per block)
pub fn block_layout(t: u32) -> Result<(usize, usize)> {
    Ok(match t {
        F32 => (1, 4),
        F16 | BF16 => (1, 2),
        Q8_0 => (32, 34),
        Q3_K => (QK_K, 110),
        Q4_K => (QK_K, 144),
        Q5_K => (QK_K, 176),
        Q6_K => (QK_K, 210),
        IQ4_NL => (QK4_NL, 18),
        IQ3_S => (QK_K, 110),
        IQ4_XS => (QK_K, 136),
        _ => bail!("ggml type {t} not supported by the reference (needs a port)"),
    })
}

fn h(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize one row of `n` elements from `src` (exactly the row's bytes) into `y`.
pub fn dequant_row(t: u32, src: &[u8], y: &mut [f32]) -> Result<()> {
    let n = y.len();
    match t {
        F32 => {
            for (i, c) in src.as_chunks::<4>().0.iter().take(n).enumerate() {
                y[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        F16 => {
            for (i, c) in src.as_chunks::<2>().0.iter().take(n).enumerate() {
                y[i] = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        BF16 => {
            for (i, c) in src.as_chunks::<2>().0.iter().take(n).enumerate() {
                y[i] = bf16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        Q8_0 => {
            for (bi, b) in src.as_chunks::<34>().0.iter().take(n / 32).enumerate() {
                let d = h(&b[0..2]);
                for j in 0..32 {
                    y[bi * 32 + j] = (b[2 + j] as i8) as f32 * d;
                }
            }
        }
        Q3_K => {
            // block_q3_K { u8 hmask[QK_K/8]; u8 qs[QK_K/4]; u8 scales[12]; half d; }
            for (bi, b) in src.as_chunks::<110>().0.iter().take(n / QK_K).enumerate() {
                let d_all = h(&b[108..110]);
                let hm = &b[0..32];
                let mut q = &b[32..96];
                let s = &b[96..108];
                // the 12 scale bytes expand to the 16 int8 scales of aux[0..4] in ggml-quants.c
                let mut scales = [0i32; 16];
                for k in 0..4 {
                    scales[k] = (s[k] & 0x0f | (s[8 + k] & 0x03) << 4) as i32;
                    scales[4 + k] = (s[4 + k] & 0x0f | ((s[8 + k] >> 2) & 0x03) << 4) as i32;
                    scales[8 + k] = ((s[k] >> 4) | ((s[8 + k] >> 4) & 0x03) << 4) as i32;
                    scales[12 + k] = ((s[4 + k] >> 4) | ((s[8 + k] >> 6) & 0x03) << 4) as i32;
                }
                let y = &mut y[bi * QK_K..(bi + 1) * QK_K];
                let (mut is, mut m, mut o) = (0usize, 1u32, 0usize);
                for _ in (0..QK_K).step_by(128) {
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let dl = d_all * (scales[is] - 32) as f32;
                        is += 1;
                        for l in 0..16 {
                            let qv = ((q[l] >> shift) & 3) as i32
                                - if hm[l] as u32 & m != 0 { 0 } else { 4 };
                            y[o + l] = dl * qv as f32;
                        }
                        let dl = d_all * (scales[is] - 32) as f32;
                        is += 1;
                        for l in 0..16 {
                            let qv = ((q[l + 16] >> shift) & 3) as i32
                                - if hm[l + 16] as u32 & m != 0 { 0 } else { 4 };
                            y[o + 16 + l] = dl * qv as f32;
                        }
                        shift += 2;
                        m <<= 1;
                        o += 32;
                    }
                    q = &q[32..];
                }
            }
        }
        Q4_K => {
            // block_q4_K { half d; half dmin; u8 scales[12]; u8 qs[128]; }
            for (bi, b) in src.as_chunks::<144>().0.iter().take(n / QK_K).enumerate() {
                let d = h(&b[0..2]);
                let min = h(&b[2..4]);
                let scales = &b[4..16];
                let mut q = &b[16..144];
                let y = &mut y[bi * QK_K..(bi + 1) * QK_K];
                let (mut is, mut o) = (0usize, 0usize);
                for _ in (0..QK_K).step_by(64) {
                    let (sc, m) = scale_min_k4(is, scales);
                    let (d1, m1) = (d * sc as f32, min * m as f32);
                    let (sc, m) = scale_min_k4(is + 1, scales);
                    let (d2, m2) = (d * sc as f32, min * m as f32);
                    for l in 0..32 {
                        y[o + l] = d1 * (q[l] & 0xF) as f32 - m1;
                    }
                    for l in 0..32 {
                        y[o + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                    q = &q[32..];
                    is += 2;
                    o += 64;
                }
            }
        }
        Q5_K => {
            // block_q5_K { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }
            for (bi, b) in src.as_chunks::<176>().0.iter().take(n / QK_K).enumerate() {
                let d = h(&b[0..2]);
                let min = h(&b[2..4]);
                let scales = &b[4..16];
                let qh = &b[16..48];
                let mut ql = &b[48..176];
                let y = &mut y[bi * QK_K..(bi + 1) * QK_K];
                let (mut is, mut o) = (0usize, 0usize);
                let (mut u1, mut u2) = (1u8, 2u8);
                for _ in (0..QK_K).step_by(64) {
                    let (sc, m) = scale_min_k4(is, scales);
                    let (d1, m1) = (d * sc as f32, min * m as f32);
                    let (sc, m) = scale_min_k4(is + 1, scales);
                    let (d2, m2) = (d * sc as f32, min * m as f32);
                    for l in 0..32 {
                        let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                        y[o + l] = d1 * ((ql[l] & 0xF) + hi) as f32 - m1;
                    }
                    for l in 0..32 {
                        let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[o + 32 + l] = d2 * ((ql[l] >> 4) + hi) as f32 - m2;
                    }
                    ql = &ql[32..];
                    is += 2;
                    o += 64;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        Q6_K => {
            // block_q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; }
            for (bi, b) in src.as_chunks::<210>().0.iter().take(n / QK_K).enumerate() {
                let d = h(&b[208..210]);
                let y = &mut y[bi * QK_K..(bi + 1) * QK_K];
                for half_ in 0..2 {
                    let ql = &b[half_ * 64..half_ * 64 + 64];
                    let qh = &b[128 + half_ * 32..128 + half_ * 32 + 32];
                    let sc = &b[192 + half_ * 8..192 + half_ * 8 + 8];
                    let y = &mut y[half_ * 128..half_ * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        IQ4_NL => {
            // block_iq4_nl { half d; u8 qs[QK4_NL/2]; }
            for (bi, b) in src.as_chunks::<18>().0.iter().take(n / QK4_NL).enumerate() {
                let d = h(&b[0..2]);
                let qs = &b[2..18];
                let y = &mut y[bi * QK4_NL..(bi + 1) * QK4_NL];
                for j in 0..QK4_NL / 2 {
                    y[j] = d * crate::iq_tables::KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
                    y[j + QK4_NL / 2] =
                        d * crate::iq_tables::KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
        }
        IQ3_S => {
            // block_iq3_s { half d; u8 qs[QK_K/4]; u8 qh[QK_K/32]; u8 signs[QK_K/8]; u8 scales[QK_K/64]; }
            for (bi, b) in src.as_chunks::<110>().0.iter().take(n / QK_K).enumerate() {
                let d = h(&b[0..2]);
                let mut qs = &b[2..66];
                let mut qh = &b[66..74];
                let mut signs = &b[74..106];
                let scales = &b[106..110];
                let y = &mut y[bi * QK_K..(bi + 1) * QK_K];
                let mut o = 0usize;
                for ib32 in (0..QK_K / 32).step_by(2) {
                    let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf)) as f32;
                    let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4)) as f32;
                    for l in 0..4 {
                        // grid index: qs byte | bit 8 from qh; the grid entry's 4 bytes are the values
                        let g1 = crate::iq_tables::IQ3S_GRID[(qs[2 * l] as usize)
                            | (((qh[0] as u32) << (8 - 2 * l)) & 256) as usize];
                        let g2 = crate::iq_tables::IQ3S_GRID[(qs[2 * l + 1] as usize)
                            | (((qh[0] as u32) << (7 - 2 * l)) & 256) as usize];
                        for j in 0..4 {
                            let s1 = if signs[l] & crate::iq_tables::KMASK_IQ2XS[j] != 0 {
                                -1.0f32
                            } else {
                                1.0
                            };
                            let s2 = if signs[l] & crate::iq_tables::KMASK_IQ2XS[j + 4] != 0 {
                                -1.0f32
                            } else {
                                1.0
                            };
                            y[o + j] = db1 * ((g1 >> (8 * j)) & 0xff) as f32 * s1;
                            y[o + 4 + j] = db1 * ((g2 >> (8 * j)) & 0xff) as f32 * s2;
                        }
                        o += 8;
                    }
                    qs = &qs[8..];
                    signs = &signs[4..];
                    for l in 0..4 {
                        let g1 = crate::iq_tables::IQ3S_GRID[(qs[2 * l] as usize)
                            | (((qh[1] as u32) << (8 - 2 * l)) & 256) as usize];
                        let g2 = crate::iq_tables::IQ3S_GRID[(qs[2 * l + 1] as usize)
                            | (((qh[1] as u32) << (7 - 2 * l)) & 256) as usize];
                        for j in 0..4 {
                            let s1 = if signs[l] & crate::iq_tables::KMASK_IQ2XS[j] != 0 {
                                -1.0f32
                            } else {
                                1.0
                            };
                            let s2 = if signs[l] & crate::iq_tables::KMASK_IQ2XS[j + 4] != 0 {
                                -1.0f32
                            } else {
                                1.0
                            };
                            y[o + j] = db2 * ((g1 >> (8 * j)) & 0xff) as f32 * s1;
                            y[o + 4 + j] = db2 * ((g2 >> (8 * j)) & 0xff) as f32 * s2;
                        }
                        o += 8;
                    }
                    qh = &qh[2..];
                    qs = &qs[8..];
                    signs = &signs[4..];
                }
            }
        }
        IQ4_XS => {
            // block_iq4_xs { half d; u16 scales_h; u8 scales_l[QK_K/64]; u8 qs[QK_K/2]; }
            for (bi, b) in src.as_chunks::<136>().0.iter().take(n / QK_K).enumerate() {
                let d = h(&b[0..2]);
                let scales_h = u16::from_le_bytes([b[2], b[3]]);
                let scales_l = &b[4..8];
                let mut qs = &b[8..136];
                let y = &mut y[bi * QK_K..(bi + 1) * QK_K];
                let mut o = 0usize;
                for ib in 0..QK_K / 32 {
                    let ls = ((scales_l[ib / 2] as i32 >> (4 * (ib % 2))) & 0xf)
                        | (((scales_h as i32 >> (2 * ib)) & 3) << 4);
                    let dl = d * (ls - 32) as f32;
                    for j in 0..16 {
                        y[o + j] =
                            dl * crate::iq_tables::KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
                        y[o + 16 + j] =
                            dl * crate::iq_tables::KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                    }
                    o += 32;
                    qs = &qs[16..];
                }
            }
        }
        _ => bail!("ggml type {t} not supported by the reference"),
    }
    Ok(())
}

/// ggml-cpu's activation rounding before an integer dot (diagnostic emulation, see `Model::act_quant`).
/// `quantize_row_q8_K_ref`: per 256, iscale = -127 / (signed value of max |x|), q = nearest(iscale * x)
/// capped at 127, d = 1 / iscale (f32). `quantize_row_q8_0_ref`: per 32, d = amax / 127 stored as f16,
/// q = roundf(x / d_f32). Returns q * d in f64. F32/F16/BF16 weights: unchanged (F16 rounding ignored).
pub fn ggml_act_quant(weight_type: u32, x: &[f64]) -> Vec<f64> {
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let mut out = vec![0f64; x.len()];
    match weight_type {
        Q3_K | Q4_K | Q5_K | Q6_K | IQ3_S | IQ4_XS => {
            for (blk, o) in xf.chunks(QK_K).zip(out.chunks_mut(QK_K)) {
                let (mut amax, mut max) = (0f32, 0f32);
                for &v in blk {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = v;
                    }
                }
                if amax == 0.0 {
                    continue;
                }
                let iscale = -127.0f32 / max;
                let d = 1.0f32 / iscale;
                for (oj, &v) in o.iter_mut().zip(blk) {
                    let q = ((iscale * v).round_ties_even() as i32).min(127);
                    *oj = q as f64 * d as f64;
                }
            }
        }
        Q8_0 | IQ4_NL => {
            for (blk, o) in xf.chunks(32).zip(out.chunks_mut(32)) {
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dh = f16::from_f32(d).to_f32();
                for (oj, &v) in o.iter_mut().zip(blk) {
                    *oj = (v * id).round() as f64 * dh as f64;
                }
            }
        }
        _ => return x.to_vec(),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block_is_scale_times_int() {
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        for j in 0..32 {
            b[2 + j] = (j as i8 - 16) as u8;
        }
        let mut y = vec![0f32; 32];
        dequant_row(Q8_0, &b, &mut y).unwrap();
        for (j, &v) in y.iter().enumerate() {
            assert_eq!(v, (j as f32 - 16.0) * 0.5);
        }
    }

    #[test]
    fn q4_k_first_subblock_uses_low_nibbles_and_scale_0() {
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes()); // d
        b[2..4].copy_from_slice(&f16::from_f32(1.0).to_le_bytes()); // dmin
        b[4] = 3; // scale 0 = 3
        b[8] = 5; // min 0 = 5
        b[16] = 0x7A; // q[0]: low nibble 10, high nibble 7
        let mut y = vec![0f32; 256];
        dequant_row(Q4_K, &b, &mut y).unwrap();
        assert_eq!(y[0], 2.0 * 3.0 * 10.0 - 5.0);
        // element 32 uses the high nibble with scale/min index 1 (both zero here)
        assert_eq!(y[32], 0.0);
    }

    #[test]
    fn q6_k_zero_quant_is_minus_32() {
        let mut b = vec![0u8; 210];
        b[192] = 1; // scale for elements 0..16
        b[208..210].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
        let mut y = vec![0f32; 256];
        dequant_row(Q6_K, &b, &mut y).unwrap();
        assert_eq!(y[0], 0.25 * -32.0);
    }
}
