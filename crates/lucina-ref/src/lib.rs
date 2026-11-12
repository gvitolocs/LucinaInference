//! f64 reference forward for the `qwen35` architecture (Qwen3.5 / 3.6 / 3.8 dense hybrids).
//!
//! Purpose: an exact-arithmetic yardstick. Both llama.cpp (the original) and the engine are measured
//! against it, so the parity tolerance rests on numbers instead of a guess. It is slow by design:
//! weights are dequantized row by row (in f32, as ggml does) and every product, sum, norm and
//! exponential is f64. The recurrence runs token by token (the exact form of the gated delta rule).
//!
//! Behaviour follows llama.cpp 3612199ed (MIT): src/models/qwen35.cpp (graph),
//! src/models/delta-net-base.cpp and ggml-cpu `gated_delta_net` (recurrence, head mapping
//! `iv1 % n_k_heads`), src/models/models.h `build_gdn_l2_norm`, ggml-cpu `ssm_conv` and
//! `ggml_mrope_cache_init` (IMROPE with text positions copied into all sections).
pub mod gguf;
pub mod iq_tables;
pub mod quant;

use anyhow::{bail, Context, Result};
use gguf::{Gguf, TensorInfo};
use rayon::prelude::*;

pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    pub eps: f64,
    pub full_attn_interval: usize,
    pub conv_kernel: usize,
    pub d_state: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub d_inner: usize,
    pub n_vocab: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.s("general.architecture")?;
        if arch != "qwen35" {
            bail!("reference supports qwen35 only, file is {arch}");
        }
        let k = |s: &str| format!("qwen35.{s}");
        let n_layer_all = g.u(&k("block_count"))? as usize;
        let nextn =
            g.kv.get(&k("nextn_predict_layers"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
        let sections = match g.kv.get(&k("rope.dimension_sections")) {
            Some(gguf::Value::Arr(a)) => a.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>(),
            _ => vec![],
        };
        let n_rot = g.u(&k("rope.dimension_count"))? as usize;
        // IMROPE reduces to plain NEOX rope for text only when the three sections cover every pair.
        if sections.len() == 4
            && (sections[0] + sections[1] + sections[2] + sections[3]) as usize != n_rot / 2
        {
            bail!(
                "rope sections {sections:?} do not cover n_rot/2 = {}",
                n_rot / 2
            );
        }
        let n_vocab = g.info("token_embd.weight")?.dims[1] as usize;
        Ok(Self {
            n_layer: n_layer_all - nextn,
            n_embd: g.u(&k("embedding_length"))? as usize,
            n_head: g.u(&k("attention.head_count"))? as usize,
            n_head_kv: g.u(&k("attention.head_count_kv"))? as usize,
            head_dim: g.u(&k("attention.key_length"))? as usize,
            n_rot,
            rope_base: g.f(&k("rope.freq_base"))?,
            eps: g.f(&k("attention.layer_norm_rms_epsilon"))?,
            full_attn_interval: g
                .kv
                .get(&k("full_attention_interval"))
                .and_then(|v| v.as_u64())
                .unwrap_or(4) as usize,
            conv_kernel: g.u(&k("ssm.conv_kernel"))? as usize,
            d_state: g.u(&k("ssm.state_size"))? as usize,
            n_k_heads: g.u(&k("ssm.group_count"))? as usize,
            n_v_heads: g.u(&k("ssm.time_step_rank"))? as usize,
            d_inner: g.u(&k("ssm.inner_size"))? as usize,
            n_vocab,
        })
    }

    pub fn is_recurrent(&self, il: usize) -> bool {
        !(il + 1).is_multiple_of(self.full_attn_interval)
    }
}

pub struct Model {
    pub g: Gguf,
    pub cfg: Config,
    /// Deliberate bug for the gate's sensitivity check (docs/lucina/stage1-parity.md), never in normal
    /// runs: `LUCINA_REF_MUTANT=head_div` maps DeltaNet v-head j to k-head j / (n_v / n_k) instead of
    /// j % n_k; `rope_adjacent` rotates adjacent pairs (2i, 2i+1) instead of NEOX pairs (i, i + n_rot/2).
    pub mutant: Option<String>,
    /// `LUCINA_REF_ACT_QUANT=ggml`: round every matmul input the way ggml-cpu does before its integer
    /// dot (Q8_K blocks of 256 for K/IQ weights, Q8_0 blocks of 32 with an f16 scale for Q8_0/IQ4_NL).
    /// Diagnostic only: it shows how much of llama.cpp's distance from exact math is that rounding.
    pub act_quant: bool,
}

type Mat = Vec<Vec<f64>>; // [token][feature]

fn rows_of(g: &Gguf, t: &TensorInfo) -> Result<(usize, usize, usize)> {
    let n_in = t.dims[0] as usize;
    let n_out = t.dims.get(1).copied().unwrap_or(1) as usize;
    let (blk, sz) = quant::block_layout(t.ggml_type)?;
    let _ = g;
    Ok((n_in, n_out, n_in / blk * sz))
}

impl Model {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let cfg = Config::from_gguf(&g)?;
        let mutant = std::env::var("LUCINA_REF_MUTANT")
            .ok()
            .filter(|m| !m.is_empty());
        if let Some(m) = &mutant {
            if m != "head_div" && m != "rope_adjacent" {
                bail!("unknown LUCINA_REF_MUTANT {m}");
            }
            tracing::warn!(
                "lucina-ref: MUTANT {m} (sensitivity check, results are wrong on purpose)"
            );
        }
        let act_quant = std::env::var("LUCINA_REF_ACT_QUANT").as_deref() == Ok("ggml");
        if act_quant {
            tracing::info!("lucina-ref: emulating ggml-cpu activation quantization (diagnostic)");
        }
        Ok(Self {
            g,
            cfg,
            mutant,
            act_quant,
        })
    }

    fn rope(&self, x: &mut [f64], pos: f64) {
        if self.mutant.as_deref() == Some("rope_adjacent") {
            for i in 0..self.cfg.n_rot / 2 {
                let theta = pos
                    * self
                        .cfg
                        .rope_base
                        .powf(-2.0 * i as f64 / self.cfg.n_rot as f64);
                let (s, c) = theta.sin_cos();
                let (x0, x1) = (x[2 * i], x[2 * i + 1]);
                x[2 * i] = x0 * c - x1 * s;
                x[2 * i + 1] = x0 * s + x1 * c;
            }
        } else {
            rope_neox(x, pos, self.cfg.n_rot, self.cfg.rope_base);
        }
    }

    /// A 1-D f32 tensor (norm weights, biases, ssm_a) as f64.
    fn vec(&self, name: &str) -> Result<Vec<f64>> {
        let t = self.g.info(name)?;
        let n: u64 = t.dims.iter().product();
        let mut y = vec![0f32; n as usize];
        quant::dequant_row(t.ggml_type, self.g.bytes(t)?, &mut y)?;
        Ok(y.into_iter().map(|v| v as f64).collect())
    }

    /// One dequantized row (as f64) of a 2-D tensor.
    pub fn row(&self, name: &str, r: usize) -> Result<Vec<f64>> {
        let t = self.g.info(name)?;
        let (n_in, _n_out, rb) = rows_of(&self.g, t)?;
        let b = self.g.bytes(t)?;
        let mut y = vec![0f32; n_in];
        quant::dequant_row(t.ggml_type, &b[r * rb..(r + 1) * rb], &mut y)?;
        Ok(y.into_iter().map(|v| v as f64).collect())
    }

    /// y[t] = W x[t] for every token: out features = rows of W. f64 dot products.
    fn linear(&self, name: &str, xs: &Mat) -> Result<Mat> {
        let t = self.g.info(name)?;
        let (n_in, n_out, rb) = rows_of(&self.g, t)?;
        let b = self.g.bytes(t)?;
        if xs.iter().any(|x| x.len() != n_in) {
            bail!("{name}: input width {} != {n_in}", xs[0].len());
        }
        let ty = t.ggml_type;
        let emulated;
        let xs = if self.act_quant {
            emulated = xs
                .iter()
                .map(|x| quant::ggml_act_quant(ty, x))
                .collect::<Mat>();
            &emulated
        } else {
            xs
        };
        // cols[r][tok]
        let cols: Vec<Vec<f64>> = (0..n_out)
            .into_par_iter()
            .map_init(
                || vec![0f32; n_in],
                |buf, r| {
                    quant::dequant_row(ty, &b[r * rb..(r + 1) * rb], buf).expect("dequant");
                    xs.iter()
                        .map(|x| buf.iter().zip(x).map(|(&w, &v)| w as f64 * v).sum::<f64>())
                        .collect()
                },
            )
            .collect();
        Ok((0..xs.len())
            .map(|tok| cols.iter().map(|c| c[tok]).collect())
            .collect())
    }

    fn has(&self, name: &str) -> bool {
        self.g.tensors.contains_key(name)
    }

    /// Logits for every position, `[n_tokens][n_vocab]`.
    pub fn forward(&self, tokens: &[u32]) -> Result<Mat> {
        let c = &self.cfg;
        let mut x: Mat = tokens
            .iter()
            .map(|&t| self.row("token_embd.weight", t as usize))
            .collect::<Result<_>>()?;
        for il in 0..c.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let xn = rms_norm_rows(&x, &self.vec(&p("attn_norm.weight"))?, c.eps);
            let a = if c.is_recurrent(il) {
                self.delta_net(il, &xn)?
            } else {
                self.attention(il, &xn)?
            };
            for (xt, at) in x.iter_mut().zip(&a) {
                add(xt, at);
            }
            let hn = rms_norm_rows(&x, &self.vec(&p("post_attention_norm.weight"))?, c.eps);
            let gate = self.linear(&p("ffn_gate.weight"), &hn)?;
            let up = self.linear(&p("ffn_up.weight"), &hn)?;
            let act: Mat = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g.iter().zip(u).map(|(&g, &u)| silu(g) * u).collect())
                .collect();
            let down = self.linear(&p("ffn_down.weight"), &act)?;
            for (xt, dt) in x.iter_mut().zip(&down) {
                add(xt, dt);
            }
            // `debug!`, like every other per-layer line: a whole forward is minutes, not hours, so the
            // default `info` filter does not need it — `RUST_LOG=debug` brings it back on a run that
            // looks stuck.
            tracing::debug!(
                "lucina-ref: layer {il} done ({})",
                if c.is_recurrent(il) {
                    "delta net"
                } else {
                    "attention"
                }
            );
        }
        let xn = rms_norm_rows(&x, &self.vec("output_norm.weight")?, c.eps);
        let head = if self.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.linear(head, &xn)
    }

    fn attention(&self, il: usize, xn: &Mat) -> Result<Mat> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{il}.{s}");
        let hd = c.head_dim;
        let qfull = self.linear(&p("attn_q.weight"), xn)?; // per head: [q (hd) | gate (hd)]
        let kf = self.linear(&p("attn_k.weight"), xn)?;
        let vf = self.linear(&p("attn_v.weight"), xn)?;
        let qn = self.vec(&p("attn_q_norm.weight"))?;
        let kn = self.vec(&p("attn_k_norm.weight"))?;
        let n = xn.len();
        let mut q = vec![vec![vec![0f64; hd]; c.n_head]; n];
        let mut gate = vec![vec![vec![0f64; hd]; c.n_head]; n];
        let mut k = vec![vec![vec![0f64; hd]; c.n_head_kv]; n];
        let mut v = vec![vec![vec![0f64; hd]; c.n_head_kv]; n];
        for t in 0..n {
            for h in 0..c.n_head {
                let base = h * 2 * hd;
                q[t][h] = rms_norm(&qfull[t][base..base + hd], &qn, c.eps);
                gate[t][h] = qfull[t][base + hd..base + 2 * hd].to_vec();
                self.rope(&mut q[t][h], t as f64);
            }
            for h in 0..c.n_head_kv {
                k[t][h] = rms_norm(&kf[t][h * hd..(h + 1) * hd], &kn, c.eps);
                self.rope(&mut k[t][h], t as f64);
                v[t][h] = vf[t][h * hd..(h + 1) * hd].to_vec();
            }
        }
        let scale = 1.0 / (hd as f64).sqrt();
        let group = c.n_head / c.n_head_kv;
        let out: Mat = (0..n)
            .into_par_iter()
            .map(|t| {
                let mut o = vec![0f64; c.n_head * hd];
                for h in 0..c.n_head {
                    let kv = h / group; // ggml mul_mat broadcast: q head h reads kv head h / (n_head / n_head_kv)
                    let s: Vec<f64> = (0..=t).map(|j| dot(&q[t][h], &k[j][kv]) * scale).collect();
                    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for d in 0..hd {
                        let acc: f64 = (0..=t).map(|j| e[j] * v[j][kv][d]).sum::<f64>() / z;
                        o[h * hd + d] = acc * sigmoid(gate[t][h][d]);
                    }
                }
                o
            })
            .collect();
        self.linear(&p("attn_output.weight"), &out)
    }

    fn delta_net(&self, il: usize, xn: &Mat) -> Result<Mat> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{il}.{s}");
        let n = xn.len();
        let (sk, nk, nv) = (c.d_state, c.n_k_heads, c.n_v_heads);
        let dv = c.d_inner / nv; // head_v_dim
        if dv != sk {
            bail!("head_v_dim {dv} != head_k_dim {sk}");
        }
        let key_dim = sk * nk;
        let conv_dim = 2 * key_dim + c.d_inner;

        let qkv = self.linear(&p("attn_qkv.weight"), xn)?;
        let z = self.linear(&p("attn_gate.weight"), xn)?;
        let beta: Mat = self
            .linear(&p("ssm_beta.weight"), xn)?
            .into_iter()
            .map(|r| r.into_iter().map(sigmoid).collect())
            .collect();
        let alpha = self.linear(&p("ssm_alpha.weight"), xn)?;
        let dt = self.vec(&p("ssm_dt.bias"))?;
        let a = self.vec(&p("ssm_a"))?;
        let g: Mat = alpha
            .iter()
            .map(|r| (0..nv).map(|j| softplus(r[j] + dt[j]) * a[j]).collect())
            .collect();

        // causal conv over [conv_kernel-1 zero columns (fresh state) | qkv tokens], then SiLU
        let w = self.g.info(&p("ssm_conv1d.weight"))?;
        if w.dims[0] as usize != c.conv_kernel || w.dims[1] as usize != conv_dim {
            bail!("ssm_conv1d shape {:?}", w.dims);
        }
        let wk: Vec<Vec<f64>> = (0..conv_dim)
            .map(|ch| self.row(&p("ssm_conv1d.weight"), ch))
            .collect::<Result<_>>()?;
        let kw = c.conv_kernel;
        let conv: Mat = (0..n)
            .map(|t| {
                (0..conv_dim)
                    .map(|ch| {
                        let mut s = 0f64;
                        for (kk, &wv) in wk[ch].iter().enumerate().take(kw) {
                            // window position kk covers input index t + kk - (kw - 1)
                            let src = t as isize + kk as isize - (kw as isize - 1);
                            if src >= 0 {
                                s += qkv[src as usize][ch] * wv;
                            }
                        }
                        silu(s)
                    })
                    .collect()
            })
            .collect();

        let ssm_norm = self.vec(&p("ssm_norm.weight"))?;
        let scale = 1.0 / (sk as f64).sqrt();
        // per v-head recurrence, heads are independent
        let per_head: Vec<Vec<Vec<f64>>> = (0..nv)
            .into_par_iter()
            .map(|j| {
                // ggml repeat / fused kernel: v head j reads k/q head j % n_k_heads
                let kh = if self.mutant.as_deref() == Some("head_div") {
                    j / (nv / nk)
                } else {
                    j % nk
                };
                let mut s = vec![vec![0f64; sk]; dv]; // S[b = v index][a = k index]
                let mut outs = Vec::with_capacity(n);
                for t in 0..n {
                    let qv = l2_norm(&conv[t][kh * sk..(kh + 1) * sk], c.eps);
                    let kv = l2_norm(&conv[t][key_dim + kh * sk..key_dim + (kh + 1) * sk], c.eps);
                    let vv = &conv[t][2 * key_dim + j * dv..2 * key_dim + (j + 1) * dv];
                    let decay = g[t][j].exp();
                    for row in s.iter_mut() {
                        for x in row.iter_mut() {
                            *x *= decay;
                        }
                    }
                    for b in 0..dv {
                        let d = (vv[b] - dot(&s[b], &kv)) * beta[t][j];
                        for (sa, ka) in s[b].iter_mut().zip(&kv) {
                            *sa += ka * d;
                        }
                    }
                    let o: Vec<f64> = (0..dv).map(|b| dot(&s[b], &qv) * scale).collect();
                    // gated RMS norm: norm(o) * w * silu(z)
                    let on = rms_norm(&o, &ssm_norm, c.eps);
                    let zh = &z[t][j * dv..(j + 1) * dv];
                    outs.push(on.iter().zip(zh).map(|(&v, &zz)| v * silu(zz)).collect());
                }
                outs
            })
            .collect();
        let flat: Mat = (0..n)
            .map(|t| {
                (0..nv)
                    .flat_map(|j| per_head[j][t].iter().cloned())
                    .collect()
            })
            .collect();
        self.linear(&p("ssm_out.weight"), &flat).context("ssm_out")
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}
pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
pub fn softplus(x: f64) -> f64 {
    // log(1 + e^x) without overflow
    if x > 30.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
/// ggml_rms_norm then multiply by the weight: x / sqrt(mean(x^2) + eps) * w
pub fn rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}
fn rms_norm_rows(x: &Mat, w: &[f64], eps: f64) -> Mat {
    x.iter().map(|r| rms_norm(r, w, eps)).collect()
}
/// build_gdn_l2_norm: scale(rms_norm(x, eps/n), 1/sqrt(n)) = x / sqrt(sum(x^2) + eps)
pub fn l2_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let s = x.iter().map(|v| v * v).sum::<f64>();
    let r = 1.0 / (s + eps).sqrt();
    x.iter().map(|v| v * r).collect()
}
/// NEOX-paired rotation of the first `n_rot` dims: `(x[i], x[i + n_rot/2])`, `theta_i = pos * base^(-2i/n_rot)`.
/// This is IMROPE for text tokens (all sections share the position; sections cover all n_rot/2 pairs).
pub fn rope_neox(x: &mut [f64], pos: f64, n_rot: usize, base: f64) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos * base.powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_norm_matches_the_ggml_rms_norm_identity() {
        let x = [3.0, 4.0];
        let eps = 1e-6;
        let n = x.len() as f64;
        let via_rms: Vec<f64> = rms_norm(&x, &[1.0, 1.0], eps / n)
            .iter()
            .map(|v| v / n.sqrt())
            .collect();
        let l2 = l2_norm(&x, eps);
        for (a, b) in via_rms.iter().zip(&l2) {
            assert!((a - b).abs() < 1e-15);
        }
        assert!((l2[0] - 0.6).abs() < 1e-7);
    }

    #[test]
    fn rope_at_position_zero_is_identity_and_preserves_norm() {
        let mut x: Vec<f64> = (0..8).map(|i| i as f64 + 1.0).collect();
        let orig = x.clone();
        rope_neox(&mut x, 0.0, 4, 1e7);
        assert_eq!(x, orig);
        rope_neox(&mut x, 37.0, 4, 1e7);
        let n0: f64 = orig[..4].iter().map(|v| v * v).sum();
        let n1: f64 = x[..4].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-9);
        assert_eq!(&x[4..], &orig[4..], "dims past n_rot are not rotated");
    }

    #[test]
    fn softplus_is_stable() {
        assert!((softplus(0.0) - 2f64.ln()).abs() < 1e-15);
        assert_eq!(softplus(100.0), 100.0);
        assert!(softplus(-100.0) > 0.0);
    }
}
