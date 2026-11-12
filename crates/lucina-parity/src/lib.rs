//! Parity metrics between two logits matrices (`n_tokens x n_vocab`, row = position).
//!
//! The reference is always the pinned llama.cpp (see `oracle/` and `docs/upstreams.toml`). A raw
//! "max |diff|" is a poor test for a quantized model: llama.cpp itself moves logits by several
//! tenths between its own batch sizes, because its CPU matmuls quantize the activations to Q8_K.
//! So every comparison reports the distribution-level numbers that decide what a sampler sees:
//!
//! - `kl`: KL(p_ref || p_test) in nats over the full vocabulary, per position;
//! - `top1`: whether the argmax agrees (only counted where the reference's top-2 margin is not a
//!   near-tie, see [`Report::top1_decisive`]);
//! - `rel_rms`: rms(test - ref) / rms(ref - mean(ref)), i.e. the error relative to the logit spread;
//! - `max_abs`: the largest single logit difference.
//!
//! The tolerance is not a guess: [`Tolerance::from_self_spread`] derives it from how far llama.cpp
//! moves against itself across configurations it treats as equivalent (batch size, threads, KV
//! precision). See docs/lucina/stage1-parity.md for the measured spread.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// One logits matrix as written by `oracle/llama-logits` or `lucina-parity infr-logits`.
pub struct Logits {
    pub n_vocab: usize,
    pub data: Vec<f32>,
}

impl Logits {
    pub fn read(path: &Path, n_vocab: usize) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        if bytes.len() % (4 * n_vocab) != 0 {
            bail!(
                "{}: {} bytes is not a whole number of {n_vocab}-wide rows",
                path.display(),
                bytes.len()
            );
        }
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self { n_vocab, data })
    }

    pub fn write(path: &Path, data: &[f32]) -> Result<()> {
        let mut out = Vec::with_capacity(data.len() * 4);
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
    }

    pub fn n_rows(&self) -> usize {
        self.data.len() / self.n_vocab
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.n_vocab..(i + 1) * self.n_vocab]
    }
}

/// Read a token file: one id per line (logits mode) or ids separated by spaces.
pub fn read_tokens(path: &Path) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    s.split_whitespace()
        .map(|t| {
            t.parse::<u32>()
                .with_context(|| format!("bad token id {t:?}"))
        })
        .collect()
}

/// First index where two token-id sequences differ; `None` when they are equal. When one is a
/// prefix of the other, the shorter length is the first difference.
pub fn first_token_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    match (0..n).find(|&i| a[i] != b[i]) {
        Some(i) => Some(i),
        None if a.len() != b.len() => Some(n),
        None => None,
    }
}

/// Per-position comparison.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowStats {
    pub kl: f64,
    pub max_abs: f32,
    pub rel_rms: f64,
    pub top1_ref: usize,
    pub top1_test: usize,
    /// logit gap between the reference's best and second-best token
    pub ref_margin: f32,
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = row
        .iter()
        .map(|&x| ((x as f64) - m).exp())
        .sum::<f64>()
        .ln()
        + m;
    row.iter().map(|&x| x as f64 - lse).collect()
}

fn top2(row: &[f32]) -> (usize, f32, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > v1 {
            v2 = v1;
            v1 = v;
            i1 = i;
        } else if v > v2 {
            v2 = v;
        }
    }
    (i1, v1, v2)
}

pub fn compare_row(r: &[f32], t: &[f32]) -> RowStats {
    assert_eq!(r.len(), t.len());
    let lr = log_softmax(r);
    let lt = log_softmax(t);
    let kl = lr
        .iter()
        .zip(&lt)
        .map(|(&a, &b)| a.exp() * (a - b))
        .sum::<f64>()
        .max(0.0);
    let mean = r.iter().map(|&x| x as f64).sum::<f64>() / r.len() as f64;
    let spread =
        (r.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
    let err = (r
        .iter()
        .zip(t)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / r.len() as f64)
        .sqrt();
    let max_abs = r
        .iter()
        .zip(t)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (i_r, v1, v2) = top2(r);
    let (i_t, _, _) = top2(t);
    RowStats {
        kl,
        max_abs,
        rel_rms: err / spread.max(1e-12),
        top1_ref: i_r,
        top1_test: i_t,
        ref_margin: v1 - v2,
    }
}

/// Aggregate over all positions.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub rows: Vec<RowStats>,
}

/// Positions whose reference top-2 margin is below this many logits are near-ties: an argmax flip
/// there is precision, not a bug (llama.cpp flips them against itself too).
pub const NEAR_TIE: f32 = 0.5;

impl Report {
    pub fn new(r: &Logits, t: &Logits) -> Result<Self> {
        if r.n_vocab != t.n_vocab || r.n_rows() != t.n_rows() {
            bail!(
                "shape mismatch: ref {}x{} vs test {}x{}",
                r.n_rows(),
                r.n_vocab,
                t.n_rows(),
                t.n_vocab
            );
        }
        use rayon::prelude::*;
        let rows = (0..r.n_rows())
            .into_par_iter()
            .map(|i| compare_row(r.row(i), t.row(i)))
            .collect();
        Ok(Self { rows })
    }

    pub fn kl_mean(&self) -> f64 {
        self.rows.iter().map(|s| s.kl).sum::<f64>() / self.rows.len() as f64
    }
    pub fn kl_max(&self) -> f64 {
        self.rows.iter().map(|s| s.kl).fold(0.0, f64::max)
    }
    pub fn rel_rms_max(&self) -> f64 {
        self.rows.iter().map(|s| s.rel_rms).fold(0.0, f64::max)
    }
    pub fn max_abs(&self) -> f32 {
        self.rows.iter().map(|s| s.max_abs).fold(0.0, f32::max)
    }
    pub fn top1_all(&self) -> (usize, usize) {
        (
            self.rows
                .iter()
                .filter(|s| s.top1_ref == s.top1_test)
                .count(),
            self.rows.len(),
        )
    }
    /// argmax agreement on positions where the reference is not a near-tie
    pub fn top1_decisive(&self) -> (usize, usize) {
        let d: Vec<_> = self
            .rows
            .iter()
            .filter(|s| s.ref_margin >= NEAR_TIE)
            .collect();
        (
            d.iter().filter(|s| s.top1_ref == s.top1_test).count(),
            d.len(),
        )
    }

    pub fn json(&self) -> serde_json::Value {
        let (a1, n1) = self.top1_all();
        let (a2, n2) = self.top1_decisive();
        serde_json::json!({
            "positions": self.rows.len(),
            "kl_mean": self.kl_mean(), "kl_max": self.kl_max(),
            "rel_rms_max": self.rel_rms_max(), "max_abs": self.max_abs(),
            "top1_all": [a1, n1], "top1_decisive": [a2, n2],
        })
    }
}

/// Pass/fail bounds for a parity run.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub kl_mean: f64,
    pub kl_max: f64,
    pub rel_rms_max: f64,
    /// decisive argmax flips allowed: as many as the original shows against the same reference
    /// (0 against llama.cpp itself; on the 27B llama.cpp flips one decisive position against exact
    /// math, at a massive-activation token where its Q8 activation rounding loses the most)
    pub decisive_flips: usize,
}

impl Tolerance {
    /// `k` times the largest spread llama.cpp shows against itself, per metric. The multiplier
    /// covers the fact that our engine is a third "configuration" with its own rounding order;
    /// docs/lucina/stage1-parity.md records the spreads and the chosen `k`.
    pub fn from_self_spread(spreads: &[&Report], k: f64) -> Self {
        let f = |g: &dyn Fn(&Report) -> f64| spreads.iter().map(|r| g(r)).fold(0.0, f64::max) * k;
        let flips = spreads
            .iter()
            .map(|r| r.top1_decisive().1 - r.top1_decisive().0)
            .max()
            .unwrap_or(0);
        Self {
            kl_mean: f(&|r| r.kl_mean()),
            kl_max: f(&|r| r.kl_max()),
            rel_rms_max: f(&|r| r.rel_rms_max()),
            decisive_flips: flips,
        }
    }

    pub fn check(&self, r: &Report) -> Result<()> {
        let (a, n) = r.top1_decisive();
        let mut bad = vec![];
        if r.kl_mean() > self.kl_mean {
            bad.push(format!(
                "kl_mean {:.3e} > {:.3e}",
                r.kl_mean(),
                self.kl_mean
            ));
        }
        if r.kl_max() > self.kl_max {
            bad.push(format!("kl_max {:.3e} > {:.3e}", r.kl_max(), self.kl_max));
        }
        if r.rel_rms_max() > self.rel_rms_max {
            bad.push(format!(
                "rel_rms_max {:.3e} > {:.3e}",
                r.rel_rms_max(),
                self.rel_rms_max
            ));
        }
        if n - a > self.decisive_flips {
            bad.push(format!(
                "top1 on decisive positions {a}/{n} (allowed flips {})",
                self.decisive_flips
            ));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            bail!("parity outside tolerance: {}", bad.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rows_have_zero_error() {
        let r = vec![1.0f32, 2.0, 3.0, -1.0];
        let s = compare_row(&r, &r);
        assert_eq!(s.max_abs, 0.0);
        assert!(s.kl.abs() < 1e-12);
        assert_eq!(s.top1_ref, 2);
        assert_eq!(s.top1_test, 2);
        assert!((s.ref_margin - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_constant_shift_is_invisible_to_kl_but_not_to_max_abs() {
        let r = vec![0.5f32, -2.0, 4.0, 1.0];
        let t: Vec<f32> = r.iter().map(|x| x + 3.0).collect();
        let s = compare_row(&r, &t);
        assert!(s.kl < 1e-12, "softmax is shift invariant");
        assert!((s.max_abs - 3.0).abs() < 1e-6);
    }

    #[test]
    fn kl_matches_a_hand_computed_value() {
        // p = softmax([0, ln 3]) = [0.25, 0.75]; q = [0.5, 0.5]
        let r = vec![0.0f32, (3.0f32).ln()];
        let t = vec![0.0f32, 0.0];
        let want = 0.25 * (0.25f64 / 0.5).ln() + 0.75 * (0.75f64 / 0.5).ln();
        assert!((compare_row(&r, &t).kl - want).abs() < 1e-7);
    }

    #[test]
    fn near_tie_flips_do_not_count_as_decisive_failures() {
        let n_vocab = 3;
        let r = Logits {
            n_vocab,
            data: vec![1.0, 1.1, -5.0, 9.0, 0.0, 0.0],
        };
        let t = Logits {
            n_vocab,
            data: vec![1.1, 1.0, -5.0, 9.0, 0.0, 0.0],
        };
        let rep = Report::new(&r, &t).unwrap();
        assert_eq!(rep.top1_all(), (1, 2));
        assert_eq!(rep.top1_decisive(), (1, 1));
    }

    #[test]
    fn tolerance_rejects_a_decisive_flip() {
        let n_vocab = 2;
        let r = Logits {
            n_vocab,
            data: vec![5.0, 0.0],
        };
        let t = Logits {
            n_vocab,
            data: vec![0.0, 5.0],
        };
        let rep = Report::new(&r, &t).unwrap();
        let tol = Tolerance {
            kl_mean: 1e9,
            kl_max: 1e9,
            rel_rms_max: 1e9,
            decisive_flips: 0,
        };
        assert!(tol.check(&rep).is_err());
    }

    #[test]
    fn first_token_diff_equal() {
        assert_eq!(first_token_diff(&[1, 2, 3], &[1, 2, 3]), None);
    }

    #[test]
    fn first_token_diff_at_index_zero() {
        assert_eq!(first_token_diff(&[9, 2, 3], &[1, 2, 3]), Some(0));
    }

    #[test]
    fn first_token_diff_in_the_middle() {
        assert_eq!(first_token_diff(&[1, 2, 3, 4], &[1, 2, 9, 4]), Some(2));
    }

    #[test]
    fn first_token_diff_prefix_both_ways() {
        assert_eq!(first_token_diff(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(first_token_diff(&[1, 2, 3], &[1, 2]), Some(2));
    }

    #[test]
    fn first_token_diff_both_empty() {
        assert_eq!(first_token_diff(&[], &[]), None);
    }
}
