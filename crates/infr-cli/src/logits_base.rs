//! The `llama-perplexity --kl-divergence-base` / `--save-all-logits` file format (the correctness
//! oracle's reference dump), and the quantisation that turns a raw logits row into one record of
//! that format.
//!
//! Source of truth: llama.cpp `v1.0.0` `tools/perplexity/perplexity.cpp`. Every number below is
//! quoted to a line of that file so a reader can diff this module against it rather than against
//! prose. The writer is `perplexity()` (`:466`, `:521-524`, `:540`, `:610-617`); the reader is
//! `kl_divergence()` (`:1691`, `:1698-1713`, `:1732`); the per-row quantisation is the `uint16_t`
//! overload of `log_softmax` (`:79-107`) driven by the `ostream` overload of `process_logits`
//! (`:147`, `:164`).
//!
//! Layout of the whole file, byte-exact:
//! - `"_logits_"` — the 8-byte magic (`:466 logits_stream.write("_logits_", 8);`).
//! - `i32 n_ctx` — the window every chunk is evaluated at (`:467`).
//! - `i32 n_vocab` (`:521`), `i32 n_chunk` (`:522`).
//! - `i32 tokens[n_chunk * n_ctx]` — the token ids, `n_ctx` per chunk, verbatim as tokenized
//!   (`:523`). This is the prefix both engines must run on, so we echo these bytes UNCHANGED into
//!   the output. The forward is NOT always these exact bytes: for `add_bos` vocabularies the
//!   reference swaps BOS into chunk position 0 at eval time and restores the stored id afterwards
//!   (`:563-569`, KL reader `:1798-1802`), which `next-probs --like-base` mirrors — see
//!   `base_chunk_feed_ids` in `main.rs`.
//! - then, per chunk, `n_ctx - 1 - first` records, `first = n_ctx / 2` (`:540`, `:613`), each record
//!   `nv = 2 * ((n_vocab + 1) / 2) + 4` `uint16` (`:147`), written back-to-back with no separators.
//!
//! A record is one quantised log-softmax distribution (see [`quantise_record`]). The chunk's record
//! `r` is the distribution AFTER consuming the token at chunk-local position `first + r`, i.e.
//! predicting position `first + r + 1` — the reference reads `llama_get_logits_ith(ctx, ... + first)`
//! and scores against `tokens[... + 1]` (`:608`, `:613`). Positions are CHUNK-LOCAL: each chunk is an
//! independent context (`:551` clears the KV, `:572` restarts the position counter at 0), so our
//! writer takes rows relative to each chunk's own start.

use std::io::{Read, Write};
use std::path::Path;

/// The magic `llama-perplexity` writes first and `kl_divergence` checks (`:466`, `:1691`).
pub const MAGIC: [u8; 8] = *b"_logits_";

/// The three `int32` header fields after the magic, in file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseHeader {
    /// Window every chunk is evaluated at. llama.cpp writes `int32_t n_ctx` (`:467`) and reads it
    /// back as `uint32_t` (`:1699`) — the same four little-endian bytes; a chunk never exceeds
    /// 2^31-1 tokens, so we carry it as `i32` and index with `usize`.
    pub n_ctx: i32,
    /// Vocabulary size. Must equal the engine's own; the reference refuses otherwise
    /// (`:1715 if (n_vocab != llama_vocab_n_tokens(vocab))`).
    pub n_vocab: i32,
    /// How many chunks follow (`:522`).
    pub n_chunk: i32,
}

/// A parsed base file: the header, the token ids it was computed over, and the exact prefix bytes
/// (magic + three `i32` + the whole `i32 tokens[]` array) so the output can echo them verbatim.
pub struct BaseFile {
    pub header: BaseHeader,
    /// `n_chunk * n_ctx` ids, the concatenation of each chunk's window, as the file stores them.
    pub tokens: Vec<i32>,
    /// The bytes from offset 0 through the end of the token array — the header + token array the
    /// output file must reproduce byte-for-byte.
    pub prefix: Vec<u8>,
}

/// `first = n_ctx/2` (`:540`): the chunk-local position of the FIRST scored row.
pub const fn first(n_ctx: i32) -> usize {
    (n_ctx / 2) as usize
}

/// The chunk-local position of the LAST scored row: `n_ctx - 2`. The reference writes
/// `n_ctx - 1 - first` rows starting at `first` (`:613`), so the last is
/// `first + (n_ctx - 1 - first) - 1 = n_ctx - 2`.
pub const fn last_row(n_ctx: i32) -> usize {
    n_ctx as usize - 2
}

/// Records written per chunk: `n_ctx - 1 - first` (`:613`).
pub const fn records_per_chunk(n_ctx: i32) -> usize {
    (n_ctx - 1) as usize - first(n_ctx)
}

/// `nv = 2 * ((n_vocab + 1) / 2) + 4` — the number of `uint16` slots in one record (`:147`). The `+4`
/// is the two little-endian `f32`s (`scale`, `min_log_prob`) that lead every record; the
/// `2 * ((n_vocab + 1) / 2)` is the quantised log-probs, with one trailing zero when `n_vocab` is
/// odd so the next record's `f32` pair stays 4-byte aligned.
// The `(n_vocab + 1) / 2` shape is the reference's arithmetic VERBATIM (`perplexity.cpp:147`), so a
// reader diffing this file against that line sees the same expression. `div_ceil` would compute the
// same thing but hides the correspondence.
#[allow(clippy::manual_div_ceil)]
pub const fn record_slots(n_vocab: usize) -> usize {
    2 * ((n_vocab + 1) / 2) + 4
}

/// One record's byte size: `nv * sizeof(uint16_t)`.
pub const fn record_bytes(n_vocab: usize) -> usize {
    record_slots(n_vocab) * 2
}

/// llama.cpp `nearest_int` (`:71-77`), verbatim: the magic-number round-half-to-even trick.
///
/// ```cpp
/// float val = fval + 12582912.f;        // :74
/// int i; memcpy(&i, &val, sizeof(int)); // :75
/// return (i & 0x007fffff) - 0x00400000; // :76
/// ```
pub fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// Quantise one raw logits row into a `nv`-slot record, replicating the `uint16_t` overload of
/// `log_softmax` (`:79-107`) EXACTLY, including its float/double choices:
/// - `max_logit`, `min_logit`: f32 running max/min (`:80-85`);
/// - `min_logit = max(min_logit, max_logit - 16)` (`:86`) — a 16-nat floor so the quantisation
///   never spends bits below the range any plausible softmax reaches;
/// - `sum_exp`: a DOUBLE accumulator of `expf(logits[i] - max_logit)` (`:87-90`, `expf` is the
///   single-precision exp, then widened to double for the sum);
/// - `log_sum_exp`, `min_log_prob`, `scale`: FLOAT (`:91-93`) — `log(sum_exp)` is truncated back to
///   `f32`, which is why we mirror the promotion and do not compute the whole chain in `f64`;
/// - `d[0] = scale`, `d[1] = min_log_prob` (`:94-96`) — the first four `uint16` slots reused as two
///   little-endian `f32`s;
/// - entries (`:98-105`): `scale != 0` ⇒ `nearest_int(inv_scale * (logit - min_logit))` for every
///   `logit > min_logit`, else `0` (so entries AT OR BELOW the clamped floor store `0`, and `argmax`
///   sits at `65535`); `scale == 0` ⇒ the whole row is zeroed.
///
/// Returns the `record_bytes(logits.len())` payload. The reference's `double` return (`:106`, the NLL
/// against the true next token) is a perplexity bookkeeping number the file never stores, so it is
/// not computed here. Panics on an empty row (there is no distribution to quantise); callers feed a
/// vocabulary-wide row.
pub fn quantise_record(logits: &[f32]) -> Vec<u8> {
    let n_vocab = logits.len();
    assert!(
        n_vocab >= 1,
        "quantise_record: an empty logits row has no distribution"
    );
    // Zero-initialised, so the odd-vocab padding slot (`:147`'s `(n_vocab + 1)/2`) and the
    // `scale == 0` memset case (`:104`) are already correct without extra writes.
    let mut rec = vec![0u8; record_bytes(n_vocab)];
    let mut max_logit = logits[0];
    let mut min_logit = logits[0];
    for &l in &logits[1..] {
        max_logit = max_logit.max(l);
        min_logit = min_logit.min(l);
    }
    // `:86` — clamp the floor to `max_logit - 16` (float subtraction, float max).
    min_logit = min_logit.max(max_logit - 16.0f32);
    // `:87-90` — double accumulator of single-precision exps.
    let mut sum_exp = 0.0f64;
    for &l in logits {
        sum_exp += (l - max_logit).exp() as f64;
    }
    // `:91-93` — `log()` is over the double sum but the reference stores the result in `float`.
    let log_sum_exp = sum_exp.ln() as f32;
    let min_log_prob = min_logit - max_logit - log_sum_exp;
    let scale = (max_logit - min_logit) / 65535.0f32;
    // `:94-96` — the two `f32`s into the first four `uint16` slots (little-endian, as the file is).
    rec[0..4].copy_from_slice(&scale.to_le_bytes());
    rec[4..8].copy_from_slice(&min_log_prob.to_le_bytes());
    // `:98-105` — the entries. `scale == 0` (a constant row) leaves them all zero (the memset).
    if scale != 0.0 {
        let inv_scale = 1.0f32 / scale;
        for (i, &l) in logits.iter().enumerate() {
            // `:101`, with the C `int`→`uint16_t` narrowing reproduced by `as u16`.
            let v: u16 = if l > min_logit {
                nearest_int(inv_scale * (l - min_logit)) as u16
            } else {
                0
            };
            rec[8 + i * 2..8 + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    rec
}

/// Reads a base file. Refuses a wrong magic (as `kl_divergence` does, `:1691`) and a truncated
/// header/token array.
pub fn read_base(path: &Path) -> anyhow::Result<BaseFile> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open base file {}: {e}", path.display()))?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)
        .map_err(|e| anyhow::anyhow!("{}: read magic: {e}", path.display()))?;
    if magic != MAGIC {
        // Render the 8 magic bytes as a Latin-1 string so a reader sees the actual bytes at the file
        // head (`_logits_` if the file is close, garbage if it is not) — `{:?}` would print them as
        // decimal integers, which is useless next to the C++ magic literal.
        let shown: String = magic.iter().map(|&b| b as char).collect();
        anyhow::bail!(
            "{} does not look like a file containing log-probabilities (magic {shown:?})",
            path.display()
        );
    }
    let mut head = [0u8; 12];
    f.read_exact(&mut head)
        .map_err(|e| anyhow::anyhow!("{}: read n_ctx/n_vocab/n_chunk: {e}", path.display()))?;
    let header = BaseHeader {
        n_ctx: i32::from_le_bytes(head[0..4].try_into().unwrap()),
        n_vocab: i32::from_le_bytes(head[4..8].try_into().unwrap()),
        n_chunk: i32::from_le_bytes(head[8..12].try_into().unwrap()),
    };
    let want = (header.n_ctx as usize)
        .checked_mul(header.n_chunk as usize)
        .ok_or_else(|| anyhow::anyhow!("{}: n_ctx*n_chunk overflows", path.display()))?;
    let mut raw = vec![0u8; want * 4];
    f.read_exact(&mut raw)
        .map_err(|e| anyhow::anyhow!("{}: read {} tokens: {e}", path.display(), want))?;
    let tokens: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    // Rebuild the exact bytes to echo: the file's first `20 + 4*want` bytes.
    let mut prefix = Vec::with_capacity(20 + raw.len());
    prefix.extend_from_slice(&MAGIC);
    prefix.extend_from_slice(&head);
    prefix.extend_from_slice(&raw);
    Ok(BaseFile {
        header,
        tokens,
        prefix,
    })
}

/// Streams the output file: it reproduces the input's header + token array byte-for-byte, then
/// appends records chunk by chunk, in the reference's order, so a `kl_divergence`-style reader can
/// interleave the two files 1:1.
pub struct BaseWriter {
    out: std::io::BufWriter<std::fs::File>,
    n_vocab: usize,
    first: usize,
    last: usize,
    recs_per_chunk: usize,
    emitted: usize,
}

impl BaseWriter {
    /// Create `path` and write the header + token array prefix (the bytes [`read_base`] captured from
    /// the input, so the output's first `20 + 4*n_chunk*n_ctx` bytes are identical to the input's).
    pub fn create(path: &Path, header: BaseHeader, prefix: &[u8]) -> anyhow::Result<Self> {
        let file = std::fs::File::create(path)
            .map_err(|e| anyhow::anyhow!("create base file {}: {e}", path.display()))?;
        let mut out = std::io::BufWriter::new(file);
        out.write_all(prefix)
            .map_err(|e| anyhow::anyhow!("write header {}: {e}", path.display()))?;
        Ok(BaseWriter {
            out,
            n_vocab: header.n_vocab as usize,
            first: first(header.n_ctx),
            last: last_row(header.n_ctx),
            recs_per_chunk: records_per_chunk(header.n_ctx),
            emitted: 0,
        })
    }

    /// Start a new chunk: its records begin again at the chunk-local position [`first`].
    pub fn begin_chunk(&mut self) {
        self.emitted = 0;
    }

    /// Consume one tapped prefill dispatch's rows. `row_start` is the CHUNK-LOCAL position of row 0
    /// (a fresh session's absolute position IS its chunk-local position, since each chunk restarts
    /// the counter at 0); `rows` is `n_rows` vocabulary-wide f32 rows laid out row-major. Rows whose
    /// position is outside `[first, last]` are dropped; in-range rows must arrive in ascending order.
    pub fn on_rows(&mut self, row_start: usize, rows: &[f32]) -> anyhow::Result<()> {
        let n_rows = rows.len() / self.n_vocab;
        for j in 0..n_rows {
            let pos = row_start + j;
            if pos < self.first {
                continue;
            }
            if pos > self.last {
                break; // rows ascend; nothing in range follows.
            }
            let expected = self.first + self.emitted;
            if pos != expected {
                anyhow::bail!(
                    "base writer: expected the record for chunk-local position {expected}, got \
                     {pos} (rows must arrive in ascending order within a chunk)"
                );
            }
            let row = &rows[j * self.n_vocab..(j + 1) * self.n_vocab];
            self.out
                .write_all(&quantise_record(row))
                .map_err(|e| anyhow::anyhow!("write record at position {pos}: {e}"))?;
            self.emitted += 1;
        }
        Ok(())
    }

    /// Close a chunk. Errors if the dispatches did not cover every record position `[first, last]` —
    /// a missing record would silently shift every later chunk's reads in the comparer.
    pub fn end_chunk(&mut self) -> anyhow::Result<()> {
        if self.emitted != self.recs_per_chunk {
            anyhow::bail!(
                "base writer: chunk produced {} records, expected {}",
                self.emitted,
                self.recs_per_chunk
            );
        }
        Ok(())
    }

    /// Flush the buffered records (the file stays open until the writer is dropped).
    pub fn finish(&mut self) -> anyhow::Result<()> {
        self.out
            .flush()
            .map_err(|e| anyhow::anyhow!("flush base file: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row of `n` distinct ascending logits `0, 1, ..., n-1` (no clamping, no ties), for hand
    /// checks against the reference's formula.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    /// Read the entry slots (past the two leading `f32`s) back as little-endian `u16`s.
    fn entries(rec: &[u8], n_vocab: usize) -> Vec<u16> {
        rec[8..8 + n_vocab * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect()
    }

    /// Hand-computed 5-entry record. With logits `[0,1,2,3,4]`: `max_logit = 4`, `min_logit = 0`
    /// (clamped to `max(0, 4-16) = 0`), `scale = 4/65535`, `inv_scale = 65535/4`; entry `i` is the
    /// rounded `inv_scale * i`; `min_log_prob = 0 - 4 - ln(sum(exp(i-4)))`.
    #[test]
    fn quantise_record_matches_a_hand_computed_five_entry_example() {
        let logits = ramp(5);
        let rec = quantise_record(&logits);
        assert_eq!(rec.len(), record_bytes(5));
        let scale = f32::from_le_bytes(rec[0..4].try_into().unwrap());
        let min_log_prob = f32::from_le_bytes(rec[4..8].try_into().unwrap());
        assert_eq!(scale, 4.0f32 / 65535.0f32);
        let mut sum = 0.0f64;
        for &l in &logits {
            sum += (l - 4.0f32).exp() as f64;
        }
        let lse = sum.ln() as f32;
        assert_eq!(min_log_prob, 0.0f32 - 4.0f32 - lse);

        let ent = entries(&rec, 5);
        let inv = 65535.0f32 / 4.0f32;
        for (i, &e) in ent.iter().enumerate() {
            assert_eq!(
                e,
                nearest_int(inv * (i as f32 - 0.0f32)) as u16,
                "entry {i}"
            );
        }
        // argmax (i = 4) lands on the top code; i = 0 sits exactly at min_logit so it is NOT > and
        // stores 0.
        assert_eq!(ent[4], 65535);
        assert_eq!(ent[0], 0);
        // n_vocab = 5 is odd, so one padding slot after the five entries stays zero.
        let pad = &rec[8 + 5 * 2..];
        assert!(
            pad.iter().all(|&b| b == 0),
            "the odd-vocab padding slot must be zero"
        );
    }

    #[test]
    fn nearest_int_rounds_half_to_even_like_the_reference() {
        assert_eq!(nearest_int(0.0), 0);
        assert_eq!(nearest_int(0.4), 0);
        assert_eq!(nearest_int(0.6), 1);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(65535.0), 65535);
    }

    /// An INDEPENDENT transcription of the reference `log_softmax` (`:79-107`) that writes the record
    /// straight into a byte buffer (no slot vector), as a second opinion over pseudo-random rows.
    fn reference_record_independent(logits: &[f32]) -> Vec<u8> {
        let n = logits.len();
        let nv = 2 * n.div_ceil(2) + 4;
        let mut bytes = vec![0u8; nv * 2];
        let mut max_logit = logits[0];
        let mut min_logit = logits[0];
        for &l in logits.iter().skip(1) {
            if l > max_logit {
                max_logit = l;
            }
            if l < min_logit {
                min_logit = l;
            }
        }
        if min_logit < max_logit - 16.0f32 {
            min_logit = max_logit - 16.0f32;
        }
        let mut sum_exp = 0.0f64;
        for &l in logits {
            sum_exp += (l - max_logit).exp() as f64;
        }
        let log_sum_exp = sum_exp.ln() as f32;
        let min_log_prob = min_logit - max_logit - log_sum_exp;
        let scale = (max_logit - min_logit) / 65535.0f32;
        bytes[0..4].copy_from_slice(&scale.to_le_bytes());
        bytes[4..8].copy_from_slice(&min_log_prob.to_le_bytes());
        if scale != 0.0 {
            let inv_scale = 1.0f32 / scale;
            for i in 0..n {
                let v: u16 = if logits[i] > min_logit {
                    nearest_int(inv_scale * (logits[i] - min_logit)) as u16
                } else {
                    0
                };
                bytes[8 + i * 2..8 + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    /// A small deterministic LCG so the "random rows" test is reproducible and CI-stable.
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (*seed >> 11) as f64; // 53 bits
        ((u / (1u64 << 53) as f64) * 20.0 - 10.0) as f32
    }

    #[test]
    fn quantise_record_matches_the_reference_formula_on_random_rows() {
        for n_vocab in [1usize, 2, 3, 7, 64, 65, 257, 1000] {
            for trial in 0..200 {
                let mut seed = (n_vocab as u64) * 1000 + trial as u64;
                let logits: Vec<f32> = (0..n_vocab).map(|_| lcg(&mut seed)).collect();
                assert_eq!(
                    quantise_record(&logits),
                    reference_record_independent(&logits),
                    "n_vocab={n_vocab} trial={trial}"
                );
            }
        }
    }

    #[test]
    fn quantise_record_honours_the_floor_and_the_constant_row() {
        // A row wider than 16 nats: everything at/below `max_logit - 16` stores 0.
        let logits = vec![0.0f32, 5.0, 30.0];
        let rec = quantise_record(&logits);
        let ent = entries(&rec, 3);
        assert_eq!(ent[0], 0); // 0.0 is not > the clamped floor 14
        assert_eq!(ent[1], 0); // 5.0 likewise
        assert_eq!(ent[2], 65535); // argmax
                                   // A constant row: `max == min => scale == 0`, so the reference memsets the ENTRY slots
                                   // (`:104 memset(log_prob, 0, n_vocab*sizeof(uint16_t))`) AFTER writing the two `f32` header
                                   // words UNCONDITIONALLY (`:94-96 d[0]=scale; d[1]=min_log_prob`). So the leading `f32`s are
                                   // NOT all zero for this case — `min_log_prob = min-max-lse = -ln(n_vocab)`.
        let flat = quantise_record(&[7.0f32; 9]);
        let scale = f32::from_le_bytes(flat[0..4].try_into().unwrap());
        let min_log_prob = f32::from_le_bytes(flat[4..8].try_into().unwrap());
        assert_eq!(scale, 0.0f32);
        assert_eq!(min_log_prob, -(9.0f64).ln() as f32);
        assert!(
            flat[8..].iter().all(|&b| b == 0),
            "the entry slots AND the padding slot stay zero when scale is 0"
        );
    }

    #[test]
    fn header_sizes_and_counts_match_the_reference_arithmetic() {
        assert_eq!(first(512), 256);
        assert_eq!(last_row(512), 510);
        assert_eq!(records_per_chunk(512), 512 - 1 - 256);
        assert_eq!(record_slots(248320), 248320 + 4);
        assert_eq!(record_slots(3), 2 * 2 + 4);
        assert_eq!(record_bytes(248320), (248320 + 4) * 2);
    }

    /// Build the reference's header + token array bytes directly (no `read_base`), for round-trips.
    fn build_prefix(n_ctx: i32, n_vocab: i32, n_chunk: i32, tokens: &[i32]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&n_ctx.to_le_bytes());
        v.extend_from_slice(&n_vocab.to_le_bytes());
        v.extend_from_slice(&n_chunk.to_le_bytes());
        for &t in tokens {
            v.extend_from_slice(&t.to_le_bytes());
        }
        v
    }

    #[test]
    fn header_and_token_array_round_trip_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!("infr-logits-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let in_path = dir.join("in.logits");

        // A small well-formed IN file: n_ctx=4, n_vocab=3, n_chunk=2.
        let (n_ctx, n_vocab, n_chunk) = (4i32, 3i32, 2i32);
        let tokens: Vec<i32> = (0..n_ctx * n_chunk).map(|i| i * 7 + 1).collect();
        let in_bytes = build_prefix(n_ctx, n_vocab, n_chunk, &tokens);
        std::fs::write(&in_path, &in_bytes).unwrap();

        let parsed = read_base(&in_path).unwrap();
        assert_eq!(
            parsed.header,
            BaseHeader {
                n_ctx,
                n_vocab,
                n_chunk
            }
        );
        assert_eq!(parsed.tokens, tokens);
        assert_eq!(parsed.prefix, in_bytes);

        // Write OUT: same prefix, then one record per scored row per chunk.
        let out_path = dir.join("out.logits");
        let mut ow = BaseWriter::create(&out_path, parsed.header, &parsed.prefix).unwrap();
        for c in 0..n_chunk {
            ow.begin_chunk();
            // Positions 0..3 (last scored row is n_ctx-2 = 2, first = 2), so three rows.
            let rows: Vec<f32> = (0..3)
                .flat_map(|j| [(j as f32) * 0.5, 1.0 + j as f32, 2.0 * j as f32 + c as f32])
                .collect();
            ow.on_rows(0, &rows).unwrap();
            ow.end_chunk().unwrap();
        }
        ow.finish().unwrap();

        let out_all = std::fs::read(&out_path).unwrap();
        // Header + token prefix is byte-identical.
        assert_eq!(&out_all[..in_bytes.len()], &in_bytes[..]);
        // Then exactly n_chunk * records_per_chunk * record_bytes payload.
        assert_eq!(
            out_all.len(),
            in_bytes.len()
                + (n_chunk as usize) * records_per_chunk(n_ctx) * record_bytes(n_vocab as usize)
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn base_writer_refuses_a_gap_within_a_chunk() {
        let dir = std::env::temp_dir().join(format!("infr-logits-ooo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.logits");
        // n_ctx=6 -> first=3, last=4, records=6-1-3=2.
        let header = BaseHeader {
            n_ctx: 6,
            n_vocab: 2,
            n_chunk: 1,
        };
        let prefix = build_prefix(6, 2, 1, &[0i32; 6]);
        let mut w = BaseWriter::create(&p, header, &prefix).unwrap();
        w.begin_chunk();
        // Rows for positions 0,1,2 (all below first=3) are skipped; then jump straight to position 4.
        w.on_rows(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let err = w.on_rows(4, &[1.0, 2.0, 3.0, 4.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("expected the record for chunk-local position 3"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// LI-R019 (non-blocker: "no test ties the runner's row index to the writer's record index"):
    /// the tap hands the writer ONE DISPATCH at a time and the dispatch boundaries are the
    /// placement's ubatch splits, which have nothing to do with the scored window. Feed the two
    /// split shapes the runner produces for a 1024-token window — four even 256-row dispatches
    /// (256+256+256+255 = 1023 rows, since `pf_end = prompt.len()-1` drops the frontier row) and an
    /// uneven 67+200+246+1+509 sweep — and require EXACTLY `n_ctx - 1 - first` records per chunk,
    /// one per position `first..=n_ctx-2`, ascending, none duplicated or dropped.
    ///
    /// How a record is fingerprinted: the quantisation is SHIFT-INVARIANT (adding a constant to a
    /// row leaves `scale` and every entry unchanged — `min_logit` clamps to `max_logit - 16` and the
    /// entries are differences from it), so a row cannot be tagged by an offset. Its SPREAD tags it:
    /// row `[0, 0, p*0.001]` has `max_logit - min_logit = p*0.001` (well under the 16.0 clamp) and
    /// therefore `scale = p*0.001/65535`, unique per position.
    #[test]
    fn base_writer_covers_every_scored_position_across_multi_dispatch_splits() {
        let dir =
            std::env::temp_dir().join(format!("infr-logits-multi-dispatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("out.logits");

        let (n_ctx, n_vocab, n_chunk) = (1024i32, 3i32, 2i32);
        let first = first(n_ctx);
        let last = last_row(n_ctx);
        let recs = records_per_chunk(n_ctx);
        assert_eq!((first, last, recs), (512, 1022, 511));
        let header = BaseHeader {
            n_ctx,
            n_vocab,
            n_chunk,
        };
        let stored_tokens = vec![7i32; (n_ctx * n_chunk) as usize];
        let prefix = build_prefix(n_ctx, n_vocab, n_chunk, &stored_tokens);
        let mut w = BaseWriter::create(&p, header, &prefix).unwrap();

        // The rows of one dispatch: `k` positions starting at `row_start`, each tagged by its spread.
        fn block(row_start: usize, k: usize, n_vocab: usize) -> Vec<f32> {
            (0..k)
                .flat_map(|j| {
                    let p = (row_start + j) as f32 * 0.001f32;
                    let mut row = vec![0.0f32; n_vocab];
                    *row.last_mut().unwrap() = p;
                    row
                })
                .collect()
        }

        // Chunk 0: the even 256-height sweep, PLUS a dispatch at the frontier row 1023 that the
        // batched prefill never produces (`pf_end = prompt.len() - 1`) and the writer must drop.
        w.begin_chunk();
        let mut row_start = 0usize;
        for k in [256usize, 256, 256, 255, 1] {
            w.on_rows(row_start, &block(row_start, k, n_vocab as usize))
                .unwrap();
            row_start += k;
        }
        assert_eq!(row_start, n_ctx as usize);
        w.end_chunk().unwrap();

        // Chunk 1: an uneven sweep that crosses `first` in the middle of a dispatch and contains a
        // 1-row dispatch (67+200+246+1+509 = 1023 rows, i.e. positions 0..=1022 exactly).
        w.begin_chunk();
        row_start = 0;
        for k in [67usize, 200, 246, 1, 509] {
            w.on_rows(row_start, &block(row_start, k, n_vocab as usize))
                .unwrap();
            row_start += k;
        }
        assert_eq!(row_start, (n_ctx - 1) as usize);
        w.end_chunk().unwrap();
        w.finish().unwrap();

        let out = std::fs::read(&p).unwrap();
        assert_eq!(&out[..prefix.len()], &prefix[..], "the header echo");
        let rec_bytes = record_bytes(n_vocab as usize);
        let n_recs = n_chunk as usize * recs;
        assert_eq!(
            out.len(),
            prefix.len() + n_recs * rec_bytes,
            "{n_chunk} chunks x {recs} records"
        );
        // Decode the fingerprints: 512..=1022, then 512..=1022 again — in order, none duplicated or
        // dropped, and `begin_chunk` back at the window start for the second chunk.
        let got: Vec<usize> = (0..n_recs)
            .map(|r| {
                let rec = &out[prefix.len() + r * rec_bytes..];
                let scale = f32::from_le_bytes(rec[0..4].try_into().unwrap());
                (scale * 65535.0f32 / 0.001f32).round() as usize
            })
            .collect();
        let want: Vec<usize> = (0..n_chunk).flat_map(|_| first..=last).collect();
        assert_eq!(got, want);
        // And every record is a real 3-entry distribution: floor entry 0, argmax entry at the top.
        for r in 0..n_recs {
            let off = prefix.len() + r * rec_bytes;
            assert_eq!(
                entries(&out[off..off + rec_bytes], 3),
                vec![0, 0, 65535],
                "record {r}"
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// LI-R019 (optional item): a GOLDEN record — a fixed 6-entry row's entry slots as bytes written
    /// out by hand from the reference formula, not produced by the code under test.
    ///
    /// `[0, 1, 2, 3, 4, 5]`: `max_logit = 5`, `min_logit = max(0, 5 - 16) = 0` (`:86`), so
    /// `scale = 5/65535` (`:93`) and `inv_scale = 65535/5 = 13107` EXACTLY. Entry 0 is not `>` the
    /// floor ⇒ 0 (`:101`); entry `i` is `nearest_int(13107 * i)` of an exact integer, and the
    /// `+12582912` trick returns that integer exactly (`:71-76`) — 13107 = `0x3333`, 26214 =
    /// `0x6666`, 39321 = `0x9999`, 52428 = `0xcccc`, 65535 = `0xffff`. `nv = 2*((6+1)/2)+4 = 10`
    /// slots (`:147`): the two `f32`s, six entries, no padding (even vocab).
    #[test]
    fn golden_record_bytes_for_a_six_entry_ramp() {
        let rec = quantise_record(&ramp(6));
        assert_eq!(rec.len(), record_bytes(6));
        assert_eq!(rec.len(), 20, "10 uint16 slots");
        assert_eq!(
            &rec[8..],
            &[
                0x00, 0x00, // entry 0: at the floor, stores 0
                0x33, 0x33, // 13107
                0x66, 0x66, // 26214
                0x99, 0x99, // 39321
                0xcc, 0xcc, // 52428
                0xff, 0xff, // 65535 = argmax
            ][..],
            "little-endian entry slots, hand-derived from `perplexity.cpp:98-105`"
        );
        let scale = f32::from_le_bytes(rec[0..4].try_into().unwrap());
        assert_eq!(scale, 5.0f32 / 65535.0f32, "`:93`'s scale");
    }
}
