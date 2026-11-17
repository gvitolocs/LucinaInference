//! `kl-compare`: next-token DISTRIBUTION parity over llama.cpp's log-probability files.
//!
//! The benchmark's correctness oracle is a comparison of what two runs predict at the same token
//! prefixes, position by position, with a tolerance that is not a guess but a noise floor MEASURED
//! from the reference against itself. Both the reference (llama.cpp v1.0.0 `llama-perplexity`) and
//! infr's dump write the same file format, so this module reads that format and compares the files
//! to each other.
//!
//! # The file format
//!
//! Everything below is `tools/perplexity/perplexity.cpp` at llama.cpp v1.0.0 (the writer is
//! `eval_perf`, the reader is `kl_divergence`):
//!
//! | bytes | contents | source |
//! |---|---|---|
//! | 8 | the magic `_logits_` | `:466` |
//! | 4 | `int32 n_ctx` | `:467` |
//! | 4 | `int32 n_vocab` | `:521` |
//! | 4 | `int32 n_chunk` | `:522` |
//! | 4 · n_chunk · n_ctx | `int32 tokens[]`, the chunk rows | `:523` |
//! | then | `n_chunk` chunks × `n_ctx - 1 - n_ctx/2` records, chunk-major | `:540`, `:613` |
//!
//! Chunk `c` holds tokens `c*n_ctx .. (c+1)*n_ctx`; the window is only scored from row `n_ctx/2`
//! because the first half is context (`first = n_ctx/2`, `:540`), and the record for row `r` is the
//! distribution that predicts `tokens[r + 1]` (`:613` passes `n_ctx - 1 - first` tokens starting at
//! `first`, and `log_softmax` scores row `i` against `tokens[i+1]`, `:160`).
//!
//! A record is `nv = 2*((n_vocab+1)/2) + 4` uint16 (`:147`): two floats — `f32 scale`,
//! `f32 min_log_prob` — followed by one uint16 per vocabulary entry, padded to an even count. The
//! writer clamps the logit range to 16 (`min_logit = max(min_logit, max_logit - 16)`, `:86`),
//! quantises the log-probabilities into the uint16 grid (`:93-101`) and the reader decodes them
//! back exactly as llama.cpp does:
//!
//! ```text
//! log_p = scale * q + min_log_prob                     // :205-208, :225
//! ```
//!
//! # The KL, and why it is llama's and not the textbook one
//!
//! `--kl-divergence-base` sums, per position (`:220-233`):
//!
//! ```text
//! for i in 0..n_vocab:
//!     p_log_base = scale*base[i] + min_log_prob
//!     if p_log_base > -16:                              // :230 — the truncation
//!         sum += expf(p_log_base) * (p_log_base - log_p(i))
//! ```
//!
//! which is `KL(base || cand)` over the entries the base itself considers non-negligible. This
//! module keeps that truncation, keeps the ascending-index accumulation into an `f64`, and keeps
//! llama's argmax convention (strictly-greater scan, so the lowest id wins a tie, `:193-199`,
//! `:226`), so its per-position numbers can be cross-checked against `llama-perplexity
//! --kl-divergence` itself. Two deliberate differences: the candidate's `log_p` here also comes from
//! a quantised record rather than from live logits, and the term is formed as
//! `p_base * (log_p_base - log_p_cand)` instead of llama's `p_base * (log_p_base - logit + max_logit)`
//! (the same quantity, one rounding step rather than two).
//!
//! The reverse `KL(cand || base)` is the identical formula with the roles swapped, so it truncates
//! on the CANDIDATE's entries instead; `--top` decides how many entries a failure listing shows.
//!
//! # Streaming
//!
//! These files are 1-2 GB. Only the header and the token array are held (both are needed to prove
//! the files describe the same tokens, and to name a token in a failure), and after that exactly one
//! record per file is in memory at a time — the files are read in lockstep, position by position.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// The magic every file starts with (`perplexity.cpp:466`).
const MAGIC: &[u8; 8] = b"_logits_";

/// A base entry whose log-probability is not above this contributes nothing to the KL
/// (`perplexity.cpp:230`).
pub const TRUNC: f32 = -16.0;

/// uint16 slots in one record: the two floats of the header, then one per vocabulary entry rounded
/// up to an even count (`perplexity.cpp:147`).
pub fn record_slots(n_vocab: usize) -> usize {
    2 * n_vocab.div_ceil(2) + 4
}

/// The first scored chunk row: the first half of the window is context, not measurement
/// (`perplexity.cpp:540`).
pub fn first_scored_row(n_ctx: usize) -> usize {
    n_ctx / 2
}

/// Records stored for one chunk: the rows from [`first_scored_row`] up to `n_ctx - 2`, each of
/// which predicts the token after it (`perplexity.cpp:613`).
pub fn records_per_chunk(n_ctx: usize) -> usize {
    n_ctx
        .saturating_sub(1)
        .saturating_sub(first_scored_row(n_ctx))
}

fn i32_at(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The fixed part of a file, which every file in one comparison must agree on exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub n_ctx: u32,
    pub n_vocab: u32,
    pub n_chunk: u32,
}

impl Header {
    fn text(&self) -> String {
        format!(
            "n_ctx {} n_vocab {} n_chunk {}",
            self.n_ctx, self.n_vocab, self.n_chunk
        )
    }
}

/// One record of one file at one position: the quantised log-probabilities and their scale.
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    pub scale: f32,
    pub min_log_prob: f32,
    q: &'a [u8],
    n_vocab: usize,
}

impl Record<'_> {
    /// Split a record's bytes (`record_slots(n_vocab) * 2` of them) into its header floats and its
    /// quantised entries.
    pub fn new(bytes: &[u8], n_vocab: usize) -> Record<'_> {
        debug_assert_eq!(bytes.len(), record_slots(n_vocab) * 2);
        Record {
            scale: f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            min_log_prob: f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            q: &bytes[8..],
            n_vocab,
        }
    }

    /// `log_p` of entry `i`, exactly as llama.cpp decodes it (`:225`).
    #[inline]
    pub fn log_p(&self, i: usize) -> f32 {
        let q = u16::from_le_bytes([self.q[2 * i], self.q[2 * i + 1]]);
        self.scale * q as f32 + self.min_log_prob
    }

    /// The `k` largest entries, highest log-probability first, lowest id first among equals — the
    /// order llama.cpp's argmax scan would produce.
    pub fn top_k(&self, k: usize) -> Vec<(u32, f32)> {
        let cap = k.min(self.n_vocab);
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(cap);
        if cap == 0 {
            return top;
        }
        for i in 0..self.n_vocab {
            let lp = self.log_p(i);
            if top.len() == cap {
                let (last_id, last_lp) = top[top.len() - 1];
                if lp < last_lp || (lp == last_lp && i as u32 >= last_id) {
                    continue;
                }
                top.pop();
            }
            let at = top
                .iter()
                .position(|&(eid, elp)| elp < lp || (elp == lp && eid > i as u32))
                .unwrap_or(top.len());
            top.insert(at, (i as u32, lp));
        }
        top
    }
}

/// One file, opened for streaming: header and tokens in memory, records read one at a time.
pub struct KlFile {
    pub path: PathBuf,
    pub header: Header,
    /// `n_chunk * n_ctx` ids, the chunk rows exactly as the file stores them.
    pub tokens: Vec<i32>,
    file: File,
    record_bytes: usize,
    records_left: usize,
}

impl KlFile {
    /// Read and validate the header and the token array. The file length is checked against them so
    /// a truncated or concatenated dump is refused here rather than at record 4,000.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let mut head = [0u8; 8];
        file.read_exact(&mut head)
            .with_context(|| format!("read magic of {}", path.display()))?;
        if &head != MAGIC {
            bail!(
                "{}: does not look like a file containing log-probabilities (magic {:?}, expected \
                 `_logits_`)",
                path.display(),
                String::from_utf8_lossy(&head)
            );
        }
        let mut ints = [0u8; 12];
        file.read_exact(&mut ints)
            .with_context(|| format!("read header of {}", path.display()))?;
        let (n_ctx, n_vocab, n_chunk) = (
            i32_at(&ints[0..4]),
            i32_at(&ints[4..8]),
            i32_at(&ints[8..12]),
        );
        for (name, v) in [("n_ctx", n_ctx), ("n_vocab", n_vocab), ("n_chunk", n_chunk)] {
            if v <= 0 {
                bail!(
                    "{}: {name} is {v}, expected a positive count",
                    path.display()
                );
            }
        }
        let (n_ctx, n_vocab, n_chunk) = (n_ctx as usize, n_vocab as usize, n_chunk as usize);
        if records_per_chunk(n_ctx) == 0 {
            bail!(
                "{}: n_ctx {n_ctx} scores no position (the first {} rows are context)",
                path.display(),
                first_scored_row(n_ctx)
            );
        }
        let n_tokens = n_ctx
            .checked_mul(n_chunk)
            .with_context(|| format!("{}: n_ctx * n_chunk overflows", path.display()))?;
        let mut tb = vec![0u8; n_tokens * 4];
        file.read_exact(&mut tb)
            .with_context(|| format!("read {} tokens of {}", n_tokens, path.display()))?;
        let tokens: Vec<i32> = tb.as_chunks::<4>().0.iter().map(|c| i32_at(c)).collect();
        let record_bytes = record_slots(n_vocab) * 2;
        let records_left = n_chunk * records_per_chunk(n_ctx);
        let expect = 8 + 12 + n_tokens * 4 + records_left * record_bytes;
        if len as usize != expect {
            bail!(
                "{}: {} bytes, but its header says {} ({} records of {} bytes)",
                path.display(),
                len,
                expect,
                records_left,
                record_bytes
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            header: Header {
                n_ctx: n_ctx as u32,
                n_vocab: n_vocab as u32,
                n_chunk: n_chunk as u32,
            },
            tokens,
            file,
            record_bytes,
            records_left,
        })
    }

    /// Scored positions in this file.
    pub fn positions(&self) -> usize {
        self.header.n_chunk as usize * records_per_chunk(self.header.n_ctx as usize)
    }

    /// Read the next record into `buf`, which is reused position after position.
    pub fn next_record(&mut self, buf: &mut Vec<u8>) -> Result<()> {
        if self.records_left == 0 {
            bail!(
                "{}: past the last of its {} records",
                self.path.display(),
                self.positions()
            );
        }
        self.records_left -= 1;
        buf.clear();
        buf.resize(self.record_bytes, 0);
        self.file
            .read_exact(buf)
            .with_context(|| format!("read a record of {}", self.path.display()))?;
        Ok(())
    }
}

/// The reference's decoded record at the current position, kept so the candidate passes read it
/// without re-decoding it `n_files` times.
#[derive(Default)]
pub struct BaseState {
    lp: Vec<f32>,
    /// `exp(lp)` where the entry survives [`TRUNC`], 0 where it does not: multiplying by it is
    /// llama's truncated sum, term for term.
    p: Vec<f32>,
}

/// What the reference itself says at one position.
#[derive(Clone, Debug)]
pub struct BaseInfo {
    pub top1: u32,
    /// log-probability gap between the reference's best and second-best entry — its own margin, in
    /// the same units as the KL
    pub gap: f32,
    pub lp_top1: f32,
}

impl BaseState {
    fn ensure(&mut self, n_vocab: usize) {
        self.lp.clear();
        self.p.clear();
        self.lp.resize(n_vocab, 0.0);
        self.p.resize(n_vocab, 0.0);
    }

    fn fill(&mut self, base: &Record<'_>) -> BaseInfo {
        let mut top1 = 0u32;
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for i in 0..base.n_vocab {
            let lp = base.log_p(i);
            self.lp[i] = lp;
            self.p[i] = if lp > TRUNC { lp.exp() } else { 0.0 };
            if lp > best {
                second = best;
                best = lp;
                top1 = i as u32;
            } else if lp > second {
                second = lp;
            }
        }
        BaseInfo {
            top1,
            gap: best - second,
            lp_top1: best,
        }
    }
}

/// One non-reference file's numbers at one position.
#[derive(Clone, Copy, Debug, Default)]
pub struct PosStats {
    /// `KL(base || x)` with llama.cpp's truncation of the base's entries
    pub kl_fwd: f64,
    /// `KL(x || base)`, the same sum with the roles swapped (so it truncates on x's entries)
    pub kl_rev: f64,
    pub top1: u32,
    /// |p_base - p_x| at the reference's own top-1
    pub p_diff: f64,
}

fn pos_stats(base: &BaseState, info: &BaseInfo, x: &Record<'_>) -> PosStats {
    let mut fwd = 0.0f64;
    let mut rev = 0.0f64;
    let mut top1 = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, (&p_b, &lp_b)) in base.p.iter().zip(&base.lp).enumerate() {
        let lp_x = x.log_p(i);
        fwd += (p_b * (lp_b - lp_x)) as f64;
        if lp_x > TRUNC {
            rev += (lp_x.exp() * (lp_x - lp_b)) as f64;
        }
        if lp_x > best {
            best = lp_x;
            top1 = i as u32;
        }
    }
    let t = info.top1 as usize;
    let p_b = info.lp_top1.exp();
    let p_x = x.log_p(t).exp();
    PosStats {
        kl_fwd: fwd,
        kl_rev: rev,
        top1,
        p_diff: (p_x - p_b).abs() as f64,
    }
}

/// Which side of a comparison a file is on. Only candidates are judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// the first `--ref`: the base every distribution is compared against
    Base,
    /// another `--ref`: the reference against itself, which sets the noise floor
    Ref,
    /// a `--cand`: the file the pass rule applies to
    Cand,
}

impl Role {
    pub fn text(self) -> &'static str {
        match self {
            Role::Base => "base",
            Role::Ref => "ref",
            Role::Cand => "cand",
        }
    }
}

/// A pooled statistic over all positions of one per-position quantity.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dist {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    /// nearest-rank percentile: the smallest value at or above this fraction of the positions
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    /// positions where the value is exactly 0 — the reference files were bit-identical there
    pub zeros: usize,
}

impl Dist {
    fn of(mut v: Vec<f64>) -> Dist {
        let n = v.len();
        if n == 0 {
            return Dist::default();
        }
        let mean = v.iter().sum::<f64>() / n as f64;
        let zeros = v.iter().filter(|&&x| x == 0.0).count();
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let rank = |q: f64| {
            let r = (q * n as f64).ceil() as usize;
            v[r.saturating_sub(1).min(n - 1)]
        };
        Dist {
            n,
            mean,
            median: rank(0.5),
            p90: rank(0.9),
            p99: rank(0.99),
            max: v[n - 1],
            zeros,
        }
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "n": self.n, "mean": self.mean, "median": self.median,
            "p90": self.p90, "p99": self.p99, "max": self.max, "zeros": self.zeros,
        })
    }
}

/// Everything one non-base file showed, pooled over the positions.
#[derive(Clone, Debug)]
pub struct FileStats {
    pub path: PathBuf,
    pub role: Role,
    pub positions: usize,
    pub kl_fwd_mean: f64,
    pub kl_fwd_max: f64,
    pub kl_rev_max: f64,
    pub p_diff_mean: f64,
    pub p_diff_max: f64,
    pub top1_agrees: usize,
    /// candidates only
    pub passes: usize,
    /// candidates only
    pub failures: usize,
}

impl FileStats {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "file": self.path.to_string_lossy(),
            "role": self.role.text(),
            "positions": self.positions,
            "kl_fwd_mean": self.kl_fwd_mean,
            "kl_fwd_max": self.kl_fwd_max,
            "kl_rev_max": self.kl_rev_max,
            "p_diff_mean": self.p_diff_mean,
            "p_diff_max": self.p_diff_max,
            "top1_agrees": [self.top1_agrees, self.positions],
            "passes": self.passes,
            "failures": self.failures,
        })
    }
}

/// One position at which a candidate is outside the reference's own spread.
#[derive(Clone, Debug)]
pub struct Failure {
    pub chunk: usize,
    /// the chunk row whose logits this record holds; it predicts `tok`
    pub row: usize,
    pub tok: i32,
    /// the scored token and the ids before it, for naming the position in a corpus
    pub around: Vec<i32>,
    /// index into [`KlSummary::files`]
    pub file: usize,
    pub kl: f64,
    pub floor: f64,
    pub base_top1: u32,
    pub cand_top1: u32,
    /// the other reference files did not all agree with the base's top-1 here, so a flip of the
    /// candidate's is not by itself a failure
    pub top1_stable: bool,
    pub base_top: Vec<(u32, f64)>,
    pub cand_top: Vec<(u32, f64)>,
}

impl Failure {
    /// Which half of the pass rule broke. Both are possible at one position.
    fn why(&self) -> String {
        let mut v = Vec::new();
        if self.kl > self.floor {
            v.push(format!("kl {:.3e} > floor {:.3e}", self.kl, self.floor));
        }
        if self.top1_stable && self.cand_top1 != self.base_top1 {
            v.push(format!(
                "top-1 is {} but the base's is {} and every --ref agrees with the base there",
                self.cand_top1, self.base_top1
            ));
        }
        v.join("; ")
    }

    fn top_text(top: &[(u32, f64)]) -> String {
        top.iter()
            .map(|(id, p)| format!("{id} {p:.6}"))
            .collect::<Vec<_>>()
            .join("  ")
    }

    /// The listing lines for this failure: the position, what broke, and both top-k.
    pub fn lines(&self, file: &FileStats, top: usize) -> Vec<String> {
        vec![
            format!(
                "FAIL {} chunk {} row {} predicts tok {} (ids {})",
                file.path.display(),
                self.chunk,
                self.row,
                self.tok,
                self.around
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            format!("  {}", self.why()),
            format!("  base top-{top}: {}", Self::top_text(&self.base_top)),
            format!("  cand top-{top}: {}", Self::top_text(&self.cand_top)),
        ]
    }

    fn json(&self, file: &FileStats, top: usize) -> serde_json::Value {
        let pairs = |t: &[(u32, f64)]| {
            t.iter()
                .map(|(id, p)| serde_json::json!([id, p]))
                .collect::<Vec<_>>()
        };
        serde_json::json!({
            "file": file.path.to_string_lossy(),
            "chunk": self.chunk,
            "row": self.row,
            "token": self.tok,
            "ids": self.around,
            "kl": self.kl,
            "floor": self.floor,
            "base_top1": self.base_top1,
            "cand_top1": self.cand_top1,
            "top1_stable": self.top1_stable,
            "base_top": pairs(&self.base_top),
            "cand_top": pairs(&self.cand_top),
            "top": top,
        })
    }
}

/// The whole comparison.
#[derive(Clone, Debug)]
pub struct KlSummary {
    pub header: Header,
    pub base: PathBuf,
    pub positions: usize,
    /// the `--ref` files other than the base, i.e. how many set the floor
    pub floor_refs: usize,
    pub files: Vec<FileStats>,
    /// pooled statistics of the per-position noise floor
    pub floor_kl: Dist,
    pub floor_kl_rev: Dist,
    pub floor_p_diff: Dist,
    /// positions at which the reference files did not all agree on the top-1
    pub top1_unstable: usize,
    pub failures: Vec<Failure>,
    pub top: usize,
}

impl KlSummary {
    /// Total failing (candidate, position) pairs.
    pub fn failure_count(&self) -> usize {
        self.files.iter().map(|f| f.failures).sum()
    }

    /// The human summary. Returned rather than printed: stdout is the CLI's contract and lives in
    /// `main.rs`, which owns the printing.
    pub fn report(&self) -> Vec<String> {
        let mut o = Vec::new();
        o.push(format!(
            "kl-compare: {} scored positions, {}",
            self.positions,
            self.header.text()
        ));
        o.push(format!(
            "  base  {}{}",
            self.base.display(),
            if self.floor_refs == 0 {
                "  (the only --ref: the floor is 0 everywhere)"
            } else {
                ""
            }
        ));
        for f in &self.files {
            o.push(format!(
                "  {:4} {}  positions {}{}  kl max {:.3e} mean {:.3e}  rev max {:.3e}  top1 {}/{}  \
                 |dp| max {:.3e} mean {:.3e}",
                f.role.text(),
                f.path.display(),
                f.positions,
                if f.role == Role::Cand {
                    format!(", pass {}, fail {}", f.passes, f.failures)
                } else {
                    String::new()
                },
                f.kl_fwd_max,
                f.kl_fwd_mean,
                f.kl_rev_max,
                f.top1_agrees,
                f.positions,
                f.p_diff_max,
                f.p_diff_mean,
            ));
        }
        let d = |n: &str, x: &Dist| {
            format!(
                "  floor {n}: mean {:.3e} median {:.3e} p90 {:.3e} p99 {:.3e} max {:.3e} zero {}/{}",
                x.mean, x.median, x.p90, x.p99, x.max, x.zeros, x.n
            )
        };
        o.push(format!(
            "noise floor = max over the {} other --ref file(s), per position:",
            self.floor_refs
        ));
        o.push(d("kl", &self.floor_kl));
        o.push(d("kl_rev", &self.floor_kl_rev));
        o.push(d("p_diff", &self.floor_p_diff));
        o.push(format!(
            "  top-1 not stable across refs: {}/{} positions",
            self.top1_unstable, self.positions
        ));
        if !self.failures.is_empty() {
            o.push(format!("failures ({}):", self.failures.len()));
            for x in &self.failures {
                o.extend(x.lines(&self.files[x.file], self.top));
            }
        }
        o.push(if self.failure_count() == 0 {
            "kl-compare: PASS".to_string()
        } else {
            format!("kl-compare: FAIL ({} positions)", self.failure_count())
        });
        o
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "base": self.base.to_string_lossy(),
            "header": {"n_ctx": self.header.n_ctx, "n_vocab": self.header.n_vocab, "n_chunk": self.header.n_chunk},
            "positions": self.positions,
            "top": self.top,
            "floor_refs": self.floor_refs,
            "floor_kl": self.floor_kl.json(),
            "floor_kl_rev": self.floor_kl_rev.json(),
            "floor_p_diff": self.floor_p_diff.json(),
            "top1_unstable": self.top1_unstable,
            "files": self.files.iter().map(|f| f.json()).collect::<Vec<_>>(),
            "failures": self.failures.iter().map(|f| f.json(&self.files[f.file], self.top)).collect::<Vec<_>>(),
            "verdict": if self.failure_count() == 0 { "PASS" } else { "FAIL" },
        })
    }
}

/// What to compare. `refs[0]` is the base; the other refs only set the floor.
pub struct KlOptions {
    pub refs: Vec<PathBuf>,
    pub cands: Vec<PathBuf>,
    pub top: usize,
}

/// Open every file and refuse the run unless they all describe the same tokens.
fn open_all(refs: &[PathBuf], cands: &[PathBuf]) -> Result<Vec<(KlFile, Role)>> {
    if refs.is_empty() {
        bail!("kl-compare: at least one --ref (the base) is required");
    }
    let mut out = Vec::with_capacity(refs.len() + cands.len());
    for (i, p) in refs.iter().enumerate() {
        out.push((
            KlFile::open(p)?,
            if i == 0 { Role::Base } else { Role::Ref },
        ));
    }
    for p in cands {
        out.push((KlFile::open(p)?, Role::Cand));
    }
    let base = &out[0].0;
    for (f, _) in &out[1..] {
        if f.header != base.header {
            bail!(
                "{}: header differs from the base {} — this file is {}, the base is {}",
                f.path.display(),
                base.path.display(),
                f.header.text(),
                base.header.text()
            );
        }
        if let Some(i) = f.tokens.iter().zip(&base.tokens).position(|(a, b)| a != b) {
            bail!(
                "{}: token array differs from the base {} at index {} ({} vs {}), so the files do \
                 not score the same positions",
                f.path.display(),
                base.path.display(),
                i,
                f.tokens[i],
                base.tokens[i]
            );
        }
    }
    Ok(out)
}

/// Compare the files, position by position. With `tsv`, one row per (position, file) is streamed to
/// it as the positions are read.
pub fn kl_compare(opts: &KlOptions, tsv: Option<&Path>) -> Result<KlSummary> {
    let mut files = open_all(&opts.refs, &opts.cands)?;
    let header = files[0].0.header;
    let n_vocab = header.n_vocab as usize;
    let n_ctx = header.n_ctx as usize;
    let n_chunk = header.n_chunk as usize;
    let first = first_scored_row(n_ctx);
    let per_chunk = records_per_chunk(n_ctx);
    let positions = n_chunk * per_chunk;
    let base_path = files[0].0.path.clone();

    let mut acc: Vec<FileAgg> = files[1..]
        .iter()
        .map(|(f, role)| FileAgg::new(f.path.clone(), *role))
        .collect();
    // which accumulated files are refs (they set the floor) and which are candidates
    let is_ref: Vec<bool> = files[1..].iter().map(|(_, r)| *r == Role::Ref).collect();
    let floor_refs = is_ref.iter().filter(|&&b| b).count();

    let mut tsv = match tsv {
        Some(p) => Some(BufWriter::new(
            File::create(p).with_context(|| format!("create {}", p.display()))?,
        )),
        None => None,
    };
    if let Some(w) = &mut tsv {
        writeln!(
            w,
            "chunk\trow\trec\ttoken\tfile\trole\tp_base\tp_x\tgap_base\tbase_top1\tkl_fwd\tkl_rev\ttop1\ttop1_agrees\tp_diff\tfloor_kl\tfloor_p_diff\tresult"
        )
        .context("write tsv header")?;
    }

    let mut bufs: Vec<Vec<u8>> = files
        .iter()
        .map(|_| Vec::with_capacity(record_slots(n_vocab) * 2))
        .collect();
    let mut base_state = BaseState::default();
    base_state.ensure(n_vocab);
    let mut floor_kl = Vec::with_capacity(positions);
    let mut floor_rev = Vec::with_capacity(positions);
    let mut floor_pd = Vec::with_capacity(positions);
    let mut failures = Vec::new();
    let mut top1_unstable = 0usize;
    let mut stats = vec![PosStats::default(); files.len() - 1];

    for chunk in 0..n_chunk {
        for rec in 0..per_chunk {
            for (i, f) in files.iter_mut().enumerate() {
                f.0.next_record(&mut bufs[i])?;
            }
            let views: Vec<Record<'_>> = bufs.iter().map(|b| Record::new(b, n_vocab)).collect();
            let (base_view, x_views) = views.split_first().expect("at least one file");
            let info = base_state.fill(base_view);
            debug_assert_eq!(x_views.len(), stats.len());
            stats
                .par_iter_mut()
                .zip(x_views)
                .for_each(|(out, x)| *out = pos_stats(&base_state, &info, x));

            // the floor: how far the reference moved against itself at THIS position
            let mut f_kl = 0.0f64;
            let mut f_rev = 0.0f64;
            let mut f_pd = 0.0f64;
            let mut stable = true;
            for (i, s) in stats.iter().enumerate() {
                if is_ref[i] {
                    f_kl = f_kl.max(s.kl_fwd);
                    f_rev = f_rev.max(s.kl_rev);
                    f_pd = f_pd.max(s.p_diff);
                    if s.top1 != info.top1 {
                        stable = false;
                    }
                }
            }
            floor_kl.push(f_kl);
            floor_rev.push(f_rev);
            floor_pd.push(f_pd);
            if !stable {
                top1_unstable += 1;
            }

            let row = first + rec;
            let gtoken = chunk * n_ctx + row + 1;
            let tok = files[0].0.tokens[gtoken];
            let around = files[0].0.tokens[gtoken.saturating_sub(3)..=gtoken].to_vec();

            for (i, s) in stats.iter().enumerate() {
                let a = &mut acc[i];
                a.n += 1;
                a.kl += s.kl_fwd;
                a.kl_max = a.kl_max.max(s.kl_fwd);
                a.kl_rev_max = a.kl_rev_max.max(s.kl_rev);
                a.pd += s.p_diff;
                a.pd_max = a.pd_max.max(s.p_diff);
                a.top1 += usize::from(s.top1 == info.top1);
                let result = if a.role == Role::Cand {
                    let ok_kl = s.kl_fwd <= f_kl;
                    let ok_top = s.top1 == info.top1 || !stable;
                    if ok_kl && ok_top {
                        a.passes += 1;
                        "pass"
                    } else {
                        a.failures += 1;
                        failures.push(Failure {
                            chunk,
                            row,
                            tok,
                            around: around.clone(),
                            file: i,
                            kl: s.kl_fwd,
                            floor: f_kl,
                            base_top1: info.top1,
                            cand_top1: s.top1,
                            top1_stable: stable,
                            base_top: base_view
                                .top_k(opts.top)
                                .into_iter()
                                .map(|(id, lp)| (id, f64::from(lp.exp())))
                                .collect(),
                            cand_top: x_views[i]
                                .top_k(opts.top)
                                .into_iter()
                                .map(|(id, lp)| (id, f64::from(lp.exp())))
                                .collect(),
                        });
                        "fail"
                    }
                } else {
                    "ref"
                };
                if let Some(w) = &mut tsv {
                    writeln!(
                        w,
                        "{chunk}\t{row}\t{rec}\t{tok}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{}\t{:.6e}\t{:.6e}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{result}",
                        tsv_path(&a.path),
                        a.role.text(),
                        info.lp_top1.exp(),
                        x_views[i].log_p(info.top1 as usize).exp(),
                        info.gap,
                        info.top1,
                        s.kl_fwd,
                        s.kl_rev,
                        s.top1,
                        u8::from(s.top1 == info.top1),
                        s.p_diff,
                        f_kl,
                        f_pd,
                    )
                    .context("write tsv row")?;
                }
            }
        }
    }
    if let Some(w) = &mut tsv {
        w.flush().context("flush tsv")?;
    }

    let files_json = acc
        .into_iter()
        .map(|a| FileStats {
            path: a.path,
            role: a.role,
            positions: a.n,
            kl_fwd_mean: a.kl / a.n.max(1) as f64,
            kl_fwd_max: a.kl_max,
            kl_rev_max: a.kl_rev_max,
            p_diff_mean: a.pd / a.n.max(1) as f64,
            p_diff_max: a.pd_max,
            top1_agrees: a.top1,
            passes: a.passes,
            failures: a.failures,
        })
        .collect();
    Ok(KlSummary {
        header,
        base: base_path,
        positions,
        floor_refs,
        files: files_json,
        floor_kl: Dist::of(floor_kl),
        floor_kl_rev: Dist::of(floor_rev),
        floor_p_diff: Dist::of(floor_pd),
        top1_unstable,
        failures,
        top: opts.top,
    })
}

/// A path as a TSV field: a tab or newline in a file name must not shift the columns.
fn tsv_path(p: &Path) -> String {
    p.to_string_lossy()
        .chars()
        .map(|c| {
            if c == '\t' || c == '\n' || c == '\r' {
                '?'
            } else {
                c
            }
        })
        .collect()
}

/// Running totals for one non-base file, folded into [`FileStats`] at the end.
struct FileAgg {
    path: PathBuf,
    role: Role,
    n: usize,
    kl: f64,
    kl_max: f64,
    kl_rev_max: f64,
    pd: f64,
    pd_max: f64,
    top1: usize,
    passes: usize,
    failures: usize,
}

impl FileAgg {
    fn new(path: PathBuf, role: Role) -> Self {
        Self {
            path,
            role,
            n: 0,
            kl: 0.0,
            kl_max: 0.0,
            kl_rev_max: 0.0,
            pd: 0.0,
            pd_max: 0.0,
            top1: 0,
            passes: 0,
            failures: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_slots_pads_the_vocabulary_to_even() {
        // :147 — 2*((n_vocab+1)/2) quantised entries plus the two floats
        assert_eq!(record_slots(4), 4 + 4);
        assert_eq!(record_slots(3), 4 + 4);
        assert_eq!(record_slots(151_936), 151_936 + 4);
        assert_eq!(record_slots(151_937), 151_938 + 4);
    }

    #[test]
    fn the_scored_rows_are_llamas() {
        // :540 first = n_ctx/2, :613 n_ctx - 1 - first records per chunk
        assert_eq!(first_scored_row(512), 256);
        assert_eq!(records_per_chunk(512), 255);
        assert_eq!(first_scored_row(5), 2);
        assert_eq!(records_per_chunk(5), 2);
        assert_eq!(records_per_chunk(2), 0);
    }

    fn rec(scale: f32, min: f32, q: &[u16]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&scale.to_le_bytes());
        b.extend_from_slice(&min.to_le_bytes());
        for &x in q {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.resize(record_slots(q.len()) * 2, 0);
        b
    }

    #[test]
    fn a_record_decodes_scale_q_plus_min() {
        let bytes = rec(0.25, -20.0, &[80, 16, 15, 0]);
        let r = Record::new(&bytes, 4);
        assert_eq!(r.log_p(0), 0.0);
        assert_eq!(r.log_p(1), -16.0);
        assert_eq!(r.log_p(2), -16.25);
        assert_eq!(r.log_p(3), -20.0);
    }

    #[test]
    fn top_k_prefers_the_log_prob_then_the_lower_id() {
        let bytes = rec(1.0, -3.0, &[0, 3, 1, 3]);
        let r = Record::new(&bytes, 4);
        let top = r.top_k(3);
        assert_eq!(
            top.iter().map(|t| t.0).collect::<Vec<_>>(),
            vec![1, 3, 2],
            "ties go to the lower id, as llama.cpp's strictly-greater scan does"
        );
        assert!(
            top[0].1 == top[1].1,
            "ids 1 and 3 hold the same log-probability"
        );
        assert!(top[1].1 > top[2].1);
        assert_eq!(r.top_k(0).len(), 0);
        assert_eq!(r.top_k(9).len(), 4);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let d = Dist::of((1..=100).map(|i| i as f64).collect());
        assert_eq!(
            (d.n, d.mean, d.median, d.p90, d.p99, d.max, d.zeros),
            (100, 50.5, 50.0, 90.0, 99.0, 100.0, 0)
        );
        assert_eq!(Dist::of(Vec::new()).n, 0);
        assert_eq!(Dist::of(vec![0.0, 0.0, 1.0]).zeros, 2);
    }
}
