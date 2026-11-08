//! Ties the Verus-proved dequant index rules (`lucina_rules::dequant_index`) to the shipped reference
//! dequantizer: for pseudo-random blocks of every block type, each element `dequant_row` produces equals, bit for
//! bit, the element rebuilt from the proved parts in ggml's f32 order. So the proved offsets and ranges are the
//! ones the reference actually uses.
use half::f16;
use lucina_ref::iq_tables::{IQ3S_GRID, KVALUES_IQ4NL};
use lucina_ref::quant::{self, dequant_row};
use lucina_rules::dequant_index as di;

const BLOCKS: usize = 64;

struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 24) as u8
    }
}

/// `n` blocks of `size` random bytes; each half scale at `halves` is finite and positive (high byte < 0x3C).
fn blocks(seed: u64, size: usize, halves: &[usize]) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut v: Vec<u8> = (0..size * BLOCKS).map(|_| r.byte()).collect();
    for b in v.chunks_exact_mut(size) {
        for &o in halves {
            b[o + 1] &= 0x3B;
        }
    }
    v
}

fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

fn check(t: u32, size: usize, elems: usize, halves: &[usize], rebuild: impl Fn(&[u8], usize) -> f32) {
    for seed in [1u64, 0x9E37_79B9_7F4A_7C15, 42] {
        let src = blocks(seed, size, halves);
        let mut y = vec![0f32; elems * BLOCKS];
        dequant_row(t, &src, &mut y).unwrap();
        for (bi, b) in src.chunks_exact(size).enumerate() {
            for i in 0..elems {
                let want = y[bi * elems + i];
                let got = rebuild(b, i);
                assert_eq!(got.to_bits(), want.to_bits(), "type {t} block {bi} element {i}: parts {got} vs dequant_row {want}");
            }
        }
    }
}

#[test]
fn q8_0() {
    check(quant::Q8_0, di::Q8_0_BYTES, di::QK8_0, &[0], |b, i| di::q8_0_quant(b, i) as f32 * h(b, 0));
}

#[test]
fn q3_k() {
    check(quant::Q3_K, di::Q3_K_BYTES, di::QK_K, &[108], |b, i| {
        let p = di::q3_k_parts(b, i);
        let dl = h(b, 108) * (p.scale as i32 - 32) as f32;
        dl * p.quant as f32
    });
}

#[test]
fn q4_k() {
    check(quant::Q4_K, di::Q4_K_BYTES, di::QK_K, &[0, 2], |b, i| {
        let p = di::q4_k_parts(b, i);
        let (d1, m1) = (h(b, 0) * p.scale as f32, h(b, 2) * p.min as f32);
        d1 * p.quant as f32 - m1
    });
}

#[test]
fn q5_k() {
    check(quant::Q5_K, di::Q5_K_BYTES, di::QK_K, &[0, 2], |b, i| {
        let (p, _) = di::q5_k_parts(b, i);
        let (d1, m1) = (h(b, 0) * p.scale as f32, h(b, 2) * p.min as f32);
        d1 * p.quant as f32 - m1
    });
}

#[test]
fn q6_k() {
    check(quant::Q6_K, di::Q6_K_BYTES, di::QK_K, &[208], |b, i| {
        let p = di::q6_k_parts(b, i);
        h(b, 208) * p.scale as f32 * p.quant as f32
    });
}

#[test]
fn iq4_nl() {
    check(quant::IQ4_NL, di::IQ4_NL_BYTES, di::QK4_NL, &[0], |b, i| {
        let (code, _) = di::iq4_nl_code(b, i);
        h(b, 0) * KVALUES_IQ4NL[code as usize] as f32
    });
}

#[test]
fn iq4_xs() {
    check(quant::IQ4_XS, di::IQ4_XS_BYTES, di::QK_K, &[0], |b, i| {
        let (code, scale, _) = di::iq4_xs_parts(b, i);
        let dl = h(b, 0) * scale as f32;
        dl * KVALUES_IQ4NL[code as usize] as f32
    });
}

#[test]
fn iq3_s() {
    check(quant::IQ3_S, di::IQ3_S_BYTES, di::QK_K, &[0], |b, i| {
        let p = di::iq3_s_parts(b, i);
        let db = h(b, 0) * (1 + 2 * p.scale_nib) as f32;
        let s = if p.sign { -1.0f32 } else { 1.0 };
        db * ((IQ3S_GRID[p.grid as usize] >> (8 * p.nib)) & 0xff) as f32 * s
    });
}

#[test]
fn block_layouts_agree() {
    for t in 0..=40u32 {
        assert_eq!(quant::block_layout(t).ok(), di::block_layout(t), "type {t}");
    }
}
