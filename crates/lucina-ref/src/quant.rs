//! Row dequantization, ported from llama.cpp ggml/src/ggml-quants.c at 3612199ed (MIT, (c) the ggml
//! authors): `dequantize_row_q8_0`, `_q4_K`, `_q5_K`, `_q6_K`, `get_scale_min_k4`. The arithmetic is
//! kept in f32 exactly as ggml does it, so a row matches ggml's `to_float` bit for bit; the reference
//! forward widens to f64 afterwards.
use anyhow::{bail, Result};
use half::{bf16, f16};

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const BF16: u32 = 30;

const QK_K: usize = 256;

/// (elements per block, bytes per block)
pub fn block_layout(t: u32) -> Result<(usize, usize)> {
    Ok(match t {
        F32 => (1, 4),
        F16 | BF16 => (1, 2),
        Q8_0 => (32, 34),
        Q4_K => (QK_K, 144),
        Q5_K => (QK_K, 176),
        Q6_K => (QK_K, 210),
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
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantize one row of `n` elements from `src` (exactly the row's bytes) into `y`.
pub fn dequant_row(t: u32, src: &[u8], y: &mut [f32]) -> Result<()> {
    let n = y.len();
    match t {
        F32 => {
            for (i, c) in src.chunks_exact(4).take(n).enumerate() {
                y[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        F16 => {
            for (i, c) in src.chunks_exact(2).take(n).enumerate() {
                y[i] = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        BF16 => {
            for (i, c) in src.chunks_exact(2).take(n).enumerate() {
                y[i] = bf16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        Q8_0 => {
            for (bi, b) in src.chunks_exact(34).take(n / 32).enumerate() {
                let d = h(&b[0..2]);
                for j in 0..32 {
                    y[bi * 32 + j] = (b[2 + j] as i8) as f32 * d;
                }
            }
        }
        Q4_K => {
            // block_q4_K { half d; half dmin; u8 scales[12]; u8 qs[128]; }
            for (bi, b) in src.chunks_exact(144).take(n / QK_K).enumerate() {
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
            for (bi, b) in src.chunks_exact(176).take(n / QK_K).enumerate() {
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
            for (bi, b) in src.chunks_exact(210).take(n / QK_K).enumerate() {
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
        _ => bail!("ggml type {t} not supported by the reference"),
    }
    Ok(())
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
        for j in 0..32 {
            assert_eq!(y[j], (j as f32 - 16.0) * 0.5);
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
