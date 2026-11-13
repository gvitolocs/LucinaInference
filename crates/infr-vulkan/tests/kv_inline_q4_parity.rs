//! Measures the INLINE q4_0 KV decode in `attn_partial` (`-DKMAINLINE -DVMAINLINE -DFMT_Q4_0`, the
//! `dqv4` decoder) and the PREPASS path (`dequant_kv_f16` q4_0 → f16 scratch, then the f16 split-K
//! `attn_partial` decode) against EXACT attention over the same q4_0 data, computed in f64 on the
//! CPU: ggml q4_0 dequantisation (`y = d * (nibble - 8)`, `d` the block's f16 scale; element `j < 16`
//! of a block is the LOW nibble of byte `j`, `j >= 16` the HIGH nibble of byte `j - 16`), the f16
//! query widened to f64, GQA `kv_head = h / (nh / nkv)`, scale `1/sqrt(hd)`, full causal softmax in
//! f64.
//!
//! Each leg is asserted to stay within twice the other's distance from exact, beyond the
//! `2^-11 * max |exact|` f16-rounding floor: neither may be materially worse than the other. The
//! shader comment's stronger claim — that the inline vec4 decoder reproduces the prepass's f16
//! scratch BYTE-FOR-BYTE — does NOT hold on RADV (iGPU, measured 2026-11-08):
//! `kv_inline_q4_matches_prepass_at_head_dim_256` in this file measures it and reports the result
//! instead of asserting it. Measured there, the inline leg is in fact the CLOSER one to exact (its
//! `mlround` is not observed to round; the prepass is the leg that lands on the f16 scratch).
//!
//! Shapes (Qwen3.8-27B attention layers): nh=24 query heads, nkv=4 kv heads, hd=256, rows=1, at
//! kv_len in {1, 31, 32, 33, 257, 4096}, plus a smaller nh=8/nkv=2/hd=256/kv_len=64 case covering
//! chunk-count > 1 at a short context.
//!
//! `kv_prefill_q4_0_head_dim_256_nc_fa_matches_nonfa` extends the same oracle to PREFILL (rows=256,
//! kv_len in {4096, 32768}) and to the two tiers the adapter's ladder chooses between there:
//! `attention_prefill_nonfa` (coopmat QK → row softmax → coopmat PV through a materialized
//! `[nh, mpad, kv_pad]` f16 scores matrix) and `attention_prefill_nc_fa` (the fused online-softmax
//! tile that the `kernels.vulkan.nc_fa_wide` reroute picks for hd > 128, which allocates no pool
//! scratch). The reroute DEFAULTS OFF (2.2x slower prefill at 32k depth — see the knob's doc), but
//! this test records both kernels directly, so its coverage of the nc_fa route does not depend on
//! the switch and the kernel cannot rot with it off. Both read the SAME q4_0 cache through the
//! SAME `dequant_kv_f16` prepass, so the exact
//! reference is attention over that prepass output — the two legs' shared f16 KV rounding cancels
//! and the assertion is about the attention tier. It is not symmetric between them: nonfa keeps
//! its scores in f16 (a 2^-10 · max|s| · max|v| term the test derives and allows for), nc_fa keeps
//! them in f32 registers, so nc_fa is the CLOSER leg by construction and the reroute is not an
//! accuracy trade. Because nonfa stores scores in f16, the prefill cases use
//! `q4_0_side_bytes_bounded` — see its doc.
//!
//! SAFETY: the discrete card (RX 7900 XTX / NAVI31) may be serving production traffic. The default
//! for both tests is the integrated GPU (1002:164e) via `MESA_VK_DEVICE_SELECT=1002:164e
//! MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1`; a "7900"/"NAVI31" device is refused before any
//! buffer is touched (see [`device_gate`]).
//!
//! That refusal is also what makes the ONE card that can build the coopmat `nonfa` pipeline unable
//! to report anything: gfx10-class RADV aborts the process when that pipeline compiles (see
//! [`coopmat_assembles`]), so on the integrated GPU the prefill comparison runs half its legs.
//! Set `INFR_PARITY_DISCRETE_LEASE` to the id of the GPU lock/lease you hold on the discrete card
//! (however your machine arbitrates it) and the refusal lifts; the value is also printed in the
//! device line, so a log says which lease the run was under.
//!
//! Run (integrated GPU):
//! ```text
//! MESA_VK_DEVICE_SELECT=1002:164e MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 \
//!   cargo test --release -p infr-vulkan --test kv_inline_q4_parity -- --nocapture
//! ```
//! Run (discrete card, ONLY while holding its lease — `<lease-id>` is yours, whatever tool grants
//! it), and give the lease a memory budget that covers the 32k nonfa leg's 384 MiB scores matrix on
//! top of the caches (4000 MB was enough):
//! ```text
//! <your-gpu-lock> … -- env INFR_PARITY_DISCRETE_LEASE=<lease-id> \
//!   MESA_VK_DEVICE_SELECT=1002:744c MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 \
//!   cargo test --release -p infr-vulkan --test kv_inline_q4_parity -- --nocapture
//! ```
//!
//! Either way a refused/absent device is not a pass: a test whose cases all skipped prints
//! `SKIPPED: no case ran`, and `INFR_PARITY_REQUIRE_RUN=1` turns that into a failure.
use half::f16;
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

/// Pseudo-random bytes in `0x00..=0xFF` (nibble bodies and scale low bytes).
fn synth_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h & 0xFF) as u8
        })
        .collect()
}

/// `n` f16 elements whose bits are masked to `0x3FFF`: sign + top exponent bit clear → every value
/// is finite, non-negative, < 2.0. Never NaN/Inf, so a bitwise output compare is never vacuous.
fn synth_f16(n: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let h = ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 5) as u16;
        out.extend_from_slice(&(h & 0x3FFF).to_le_bytes());
    }
    out
}

fn bits(v: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(v).to_vec()
}

/// Bytes for one q4_0 KV side holding `elems` cache elements. Layout per 32 elements:
/// `[f16 d][u8 qs[16]]` = 18 bytes. The scale is a valid finite f16 with high byte < 0x3C (sign
/// clear, exponent < 15 → finite and normal); the 16 code bytes are random (random nibbles).
fn q4_0_side_bytes(elems: usize, seed: usize) -> Vec<u8> {
    assert_eq!(
        elems % 32,
        0,
        "q4_0 cache must be a whole number of 32-elem blocks"
    );
    let mut b = synth_bytes((elems / 32) * 18, seed);
    for (bi, blk) in b.chunks_mut(18).enumerate() {
        // f16 scale: low byte random, high byte in 0x04..=0x3B (< 0x3C → finite, non-zero exponent).
        let h = (bi.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
        blk[0] = (h & 0xFF) as u8;
        blk[1] = (((h >> 8) % 0x38) + 0x04) as u8;
        // blk[2..18] stays random nibble body from synth_bytes.
    }
    b
}

/// 256-byte garbage prefix (aligned) for the inline pointer read, mirroring `kv_addr_parity.rs`.
const OFF: usize = 256;

/// Like [`q4_0_side_bytes`], but with every block scale pinned to a narrow exponent band (13 or 14,
/// zero high mantissa bits → `d` in [0.25, 1.0)), so dequantised values stay inside [-8, 7] and the
/// scores of a 256-row × 32k-key prefill stay well inside f16 range. The PREFILL cases need this:
/// `attention_prefill_nonfa` stores its scores matrix as f16, and `q4_0_side_bytes`' full exponent
/// spread (0x04..=0x3B, i.e. block scales over ~2^15 of range) pushes |scale·QKᵀ| past 65504 =
/// f16 max, which would compare two overflowing legs rather than two attention tiers. Real KV rows
/// are O(1), which is what this models. The decode cases keep the wide spread — `attn_partial`
/// holds its scores in f32 registers, where nothing overflows.
fn q4_0_side_bytes_bounded(elems: usize, seed: usize) -> Vec<u8> {
    assert_eq!(
        elems % 32,
        0,
        "q4_0 cache must be a whole number of 32-elem blocks"
    );
    let mut b = synth_bytes((elems / 32) * 18, seed);
    for (bi, blk) in b.chunks_mut(18).enumerate() {
        // Little-endian f16: byte 0 is the mantissa's low byte, byte 1 holds bits 15..8, so the
        // exponent (bits 14..10) sits in byte 1 at `exp << 2`. Exponent 13 or 14, mantissa bits
        // 9..8 clear; byte 0 and the nibble body blk[2..18] stay as `synth_bytes` drew them.
        blk[1] = ((13 + (bi % 2)) << 2) as u8;
    }
    b
}

fn f16_at(bytes: &[u8], idx: usize) -> f64 {
    let lo = bytes[2 * idx] as u16;
    let hi = bytes[2 * idx + 1] as u16;
    f16::from_bits(lo | (hi << 8)).to_f64()
}

/// f16 cache bytes widened to f32 — losslessly, since f32 represents every f16 value.
fn widen_f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| f16::from_bits(u16::from_le_bytes([w[0], w[1]])).to_f32())
        .collect()
}

/// Exact ggml q4_0 dequantisation of flat cache element `e`:
/// `y = d * (nibble - 8)`; element `j < 16` of a block is the LOW nibble of byte `j`, `j >= 16` the
/// HIGH nibble of byte `j - 16`.
fn q4_0_dequant(bytes: &[u8], e: usize) -> f64 {
    let blk = e / 32;
    let p = e % 32;
    let off = blk * 18;
    let d = f16::from_bits(u16::from_le_bytes([bytes[off], bytes[off + 1]])).to_f64();
    let byte = bytes[off + 2 + if p < 16 { p } else { p - 16 }];
    let nib = if p < 16 { byte & 0x0F } else { byte >> 4 };
    d * (nib as f64 - 8.0)
}

/// EXACT decode attention in f64 over the same q4_0 K/V and f16 Q, for the single (last) query row:
/// GQA `kv_head = h / (nh / nkv)`, scale `1/sqrt(hd)`, full causal softmax over all `kv_len` keys.
fn exact_attention(
    q: &[u8],
    kd: &[u8],
    vd: &[u8],
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_len: usize,
) -> Vec<f64> {
    let g = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0.0f64; nh * hd];
    for h in 0..nh {
        let kvh = h / g;
        let qbase = h * hd;
        let mut sc = vec![0.0f64; kv_len];
        let mut mx = f64::NEG_INFINITY;
        for (j, s) in sc.iter_mut().enumerate() {
            let kbase = (j * nkv + kvh) * hd;
            let mut dot = 0.0f64;
            for x in 0..hd {
                dot += f16_at(q, qbase + x) * q4_0_dequant(kd, kbase + x);
            }
            *s = dot * scale;
            mx = mx.max(*s);
        }
        let mut l = 0.0f64;
        for s in &sc {
            l += (s - mx).exp();
        }
        for x in 0..hd {
            let mut acc = 0.0f64;
            for (j, s) in sc.iter().enumerate() {
                let p = (s - mx).exp() / l;
                acc += p * q4_0_dequant(vd, (j * nkv + kvh) * hd + x);
            }
            out[qbase + x] = acc;
        }
    }
    out
}

/// Max |leg - exact| over every output, in f64.
fn max_abs_vs_exact(leg: &[u32], exact: &[f64]) -> f64 {
    leg.iter()
        .zip(exact)
        .map(|(&b, &e)| (f32::from_bits(b) as f64 - e).abs())
        .fold(0.0f64, f64::max)
}

/// Max |a - b| between two f32 output buffers (bit patterns), in f64.
fn max_abs_between_bits(a: &[u32], b: &[u32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f32::from_bits(x) as f64 - f32::from_bits(y) as f64).abs())
        .fold(0.0f64, f64::max)
}

/// EXACT causal PREFILL attention in f64 over the shared prepass output (K/V the f16 cache widened
/// losslessly to f32 — f32 represents every f16 — and Q the f16 bytes it was uploaded from), for
/// the given `(row, head)` pairs of one chunk: query row `r` sits at absolute position `pos + r`
/// and attends keys `<= pos + r`; GQA `kv_head = h / (nh / nkv)`; scale `1/sqrt(hd)`; softmax over
/// that row's full causal span. Returns `(output base index, hd exact values, max |score|)` per
/// pair, base = `(r * nh + h) * hd` in the `[rows, nh, hd]` f32 layout both prefill tiers write.
/// The `max |score|` is for the tolerance below: it is what `attn_qk` rounds to f16.
///
/// The pairs are SAMPLED (see `run_prefill_case`) because a full 256-row × 24-head oracle at
/// kv_len 32768 is ~10^11 f64 MACs. The pairs are spread over threads for the same reason.
#[allow(clippy::too_many_arguments)]
fn exact_prefill(
    q: &[u8],
    kd: &[f32],
    vd: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_len: usize,
    pos: usize,
    pairs: &[(usize, usize)],
) -> Vec<(usize, Vec<f64>, f64)> {
    let g = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(pairs.len().max(1));
    let per = pairs.len().div_ceil(nthreads);
    let mut out: Vec<(usize, Vec<f64>, f64)> = Vec::with_capacity(pairs.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = pairs
            .chunks(per)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for &(r, h) in chunk {
                        let kvh = h / g;
                        let nkeys = (pos + r + 1).min(kv_len);
                        let qrow: Vec<f64> =
                            (0..hd).map(|x| f16_at(q, (r * nh + h) * hd + x)).collect();
                        let mut sc = vec![0.0f64; nkeys];
                        let mut mx = f64::NEG_INFINITY;
                        let mut amax = 0.0f64;
                        for (j, s) in sc.iter_mut().enumerate() {
                            let kbase = (j * nkv + kvh) * hd;
                            let mut dot = 0.0f64;
                            for x in 0..hd {
                                dot += qrow[x] * kd[kbase + x] as f64;
                            }
                            *s = dot * scale;
                            mx = mx.max(*s);
                            amax = amax.max(s.abs());
                        }
                        let mut l = 0.0f64;
                        for s in &sc {
                            l += (s - mx).exp();
                        }
                        // V scanned key-major (stride-1 over the row's hd dims) into one
                        // accumulator: the alternative — dim-major with the key loop inside —
                        // re-walks the whole cache once per dim.
                        let mut acc = vec![0.0f64; hd];
                        for (j, s) in sc.iter().enumerate() {
                            let p = (s - mx).exp() / l;
                            let vbase = (j * nkv + kvh) * hd;
                            for x in 0..hd {
                                acc[x] += p * vd[vbase + x] as f64;
                            }
                        }
                        local.push(((r * nh + h) * hd, acc, amax));
                    }
                    local
                })
            })
            .collect();
        for handle in handles {
            out.extend(handle.join().unwrap());
        }
    });
    out
}

/// [`max_abs_vs_exact`] restricted to the sampled `(base, values, max|score|)` blocks: returns the
/// leg's max absolute distance from exact AND the max |exact| over the same entries (the f16
/// floor's scale).
fn max_abs_vs_exact_at(leg: &[u32], parts: &[(usize, Vec<f64>, f64)]) -> (f64, f64) {
    let mut d = 0.0f64;
    let mut mx = 0.0f64;
    for (base, want, _) in parts {
        for (x, &e) in want.iter().enumerate() {
            let got = f32::from_bits(leg[base + x]) as f64;
            d = d.max((got - e).abs());
            mx = mx.max(e.abs());
        }
    }
    (d, mx)
}

/// Compares one inline leg against the prepass reference, returning the first differing index, both
/// values, and the max absolute difference. This is a measured REPORT only (see the module docs:
/// the shader's byte-identity claim does not hold on RADV).
fn compare_bit_identical(name: &str, reference: &[u32], got: &[u32]) -> Result<(), String> {
    if reference.len() != got.len() {
        return Err(format!(
            "{name}: output length mismatch ({} vs {})",
            reference.len(),
            got.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut first: Option<(usize, u32, u32)> = None;
    for (i, (&r, &p)) in reference.iter().zip(got.iter()).enumerate() {
        let (rf, pf) = (f32::from_bits(r), f32::from_bits(p));
        let d = (rf - pf).abs();
        if d.is_finite() {
            max_abs = max_abs.max(d);
        }
        if first.is_none() && r != p {
            first = Some((i, r, p));
        }
    }
    match first {
        None => Ok(()),
        Some((i, r, p)) => Err(format!(
            "{name}: INLINE differs from prepass at out {i}: {} vs {} (bits {r:#010x} vs {p:#010x}); \
             max |diff| = {max_abs:e} over {} outputs",
            f32::from_bits(r),
            f32::from_bits(p),
            reference.len()
        )),
    }
}

/// Runs one decode case: prepass (q4_0 → f16 scratch → f16 split-K attention) and the inline q4_0
/// split-K attention at base offset 0 and behind a garbage 256-byte prefix. Prints both legs'
/// distance from exact attention and the prepass↔inline bit-identity report; returns the failed
/// accuracy comparisons so the caller can report every case before failing.
fn run_case(
    be: &VulkanBackend,
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_len: usize,
) -> Vec<Result<(), String>> {
    let rows = 1usize;
    let pos = kv_len - rows;
    let scale = 0.0f32; // default 1/sqrt(hd)
    let window = 0usize; // full causal
    let cap = 0usize; // full-context (RROW identity)
                      // Same ~32-chunk policy the adapter uses; kv_len <= 64 then lands on a single chunk (legal).
    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk).max(1);
    let elems = kv_len * nkv * hd; // full-context cache, one side

    let q = synth_f16(rows * nh * hd, 101);
    let q_buf = be.alloc(q.len(), BufferUsage::Activations).unwrap();
    be.upload(q_buf.as_ref(), &q).unwrap();
    let o_bytes = rows * nh * hd * 4;

    let kd = q4_0_side_bytes(elems, 11);
    let vd = q4_0_side_bytes(elems, 23);

    let pm = be
        .alloc(rows * nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(rows * nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(rows * nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();

    // ── Leg A: PREPASS. dequant_kv_f16 both sides to a concrete f16 cache (download + re-upload
    // severs the cross-recorder scratch hazard, exactly like kv_addr_parity's mainline reference),
    // then the f16 split-K decode. ─────────────────────────────────────────────────────────────
    let kq = be.alloc(kd.len(), BufferUsage::KvCache).unwrap();
    let vq = be.alloc(vd.len(), BufferUsage::KvCache).unwrap();
    be.upload(kq.as_ref(), &kd).unwrap();
    be.upload(vq.as_ref(), &vd).unwrap();
    let kf16 = be.alloc(elems * 2, BufferUsage::Activations).unwrap();
    let vf16 = be.alloc(elems * 2, BufferUsage::Activations).unwrap();
    {
        let rec = be.recorder().unwrap();
        rec.dequant_kv_f16(infr_core::DType::Q4_0, kq.as_ref(), kf16.as_ref(), elems);
        rec.dequant_kv_f16(infr_core::DType::Q4_0, vq.as_ref(), vf16.as_ref(), elems);
        rec.finish().unwrap();
    }
    let mut kf16_bytes = vec![0u8; elems * 2];
    let mut vf16_bytes = vec![0u8; elems * 2];
    be.download(kf16.as_ref(), &mut kf16_bytes).unwrap();
    be.download(vf16.as_ref(), &mut vf16_bytes).unwrap();
    let kf16c = be.alloc(elems * 2, BufferUsage::KvCache).unwrap();
    let vf16c = be.alloc(elems * 2, BufferUsage::KvCache).unwrap();
    be.upload(kf16c.as_ref(), &kf16_bytes).unwrap();
    be.upload(vf16c.as_ref(), &vf16_bytes).unwrap();
    let o_ref = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    {
        let rec = be.recorder().unwrap();
        rec.attention_kv_split(
            q_buf.as_ref(),
            kf16c.as_ref(),
            vf16c.as_ref(),
            o_ref.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            rows,
            pos,
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            None,
            false,
            false,
            0,
            false,
        );
        rec.finish().unwrap();
    }
    let mut ref_out = vec![0u8; o_bytes];
    be.download(o_ref.as_ref(), &mut ref_out).unwrap();
    let reference = bits(&ref_out);

    // ── Leg B: INLINE. attention_kv_split_at with `kv_ml = Some(Q4_0)` reads the q4_0 cache in
    // kernel via `dqv4` — the exact recorder entry the adapter's `decode_inline` arm calls. ──────
    let inline_leg = |prefix: usize| -> Vec<u32> {
        let mut kback = synth_bytes(prefix, 0xBAD);
        kback.extend_from_slice(&kd);
        let mut vback = synth_bytes(prefix, 0xBEEF);
        vback.extend_from_slice(&vd);
        let kbuf = be.alloc(kback.len(), BufferUsage::KvCache).unwrap();
        let vbuf = be.alloc(vback.len(), BufferUsage::KvCache).unwrap();
        be.upload(kbuf.as_ref(), &kback).unwrap();
        be.upload(vbuf.as_ref(), &vback).unwrap();
        let ka = kbuf.device_addr().unwrap() + prefix as u64;
        let va = vbuf.device_addr().unwrap() + prefix as u64;
        let o_buf = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split_at(
            q_buf.as_ref(),
            kbuf.as_ref(),
            vbuf.as_ref(),
            ka,
            va,
            o_buf.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            rows,
            pos,
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            None,
            false,
            false,
            cap,
            false,
            Some(infr_core::DType::Q4_0),
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; o_bytes];
        be.download(o_buf.as_ref(), &mut out).unwrap();
        bits(&out)
    };
    let inline0 = inline_leg(0);
    let inlineoff = inline_leg(OFF);

    let name = format!("q4_0 inline kv={kv_len} nh={nh} nkv={nkv} hd={hd}");
    if !reference.iter().any(|&b| b != 0) {
        return vec![Err(format!(
            "{name}: prepass reference is all zeros — not exercising the kernel"
        ))];
    }

    // ── Exact f64 reference over the same q4_0 data, and both legs' distance from it. ───────────
    let exact = exact_attention(&q, &kd, &vd, nh, nkv, hd, kv_len);
    let exact_max = exact.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let d_pre = max_abs_vs_exact(&reference, &exact);
    let d_in0 = max_abs_vs_exact(&inline0, &exact);
    let d_inoff = max_abs_vs_exact(&inlineoff, &exact);
    // Half an f16 ulp: the rounding floor of the f16 path, scaled by the output magnitude.
    let floor = (2.0f64).powi(-11) * exact_max;
    let rel = |d: f64| if exact_max > 0.0 { d / exact_max } else { d };

    println!(
        "{name}: max|exact| = {exact_max:e}; prepass max|prepass-exact| = {d_pre:e} (rel {:e})",
        rel(d_pre)
    );
    for (label, got, d_in) in [("@0", &inline0, d_in0), ("@offset", &inlineoff, d_inoff)] {
        // Keep the shader's byte-identity claim as a measured report, not an assert.
        match compare_bit_identical(label, &reference, got) {
            Ok(()) => println!("{name} {label}: bit-identical to prepass"),
            Err(e) => println!("{name} {label}: {e}"),
        }
        let d_between = max_abs_between_bits(got, &reference);
        println!(
            "{name} {label}: inline max|inline-exact| = {d_in:e} (rel {:e}), \
             max|inline-prepass| = {d_between:e} (rel {:e})",
            rel(d_in),
            rel(d_between),
        );
    }

    let mut failures = Vec::new();
    for (label, d_in) in [("@0", d_in0), ("@offset", d_inoff)] {
        // Neither leg may be more than twice as far from exact as the other, beyond f16 rounding.
        let bound_inline = 2.0 * d_pre.max(floor);
        if d_in > bound_inline {
            failures.push(Err(format!(
                "{name} {label}: inline is {d_in:e} from exact, bound {bound_inline:e} \
                 (2 * max(prepass {d_pre:e}, f16 floor {floor:e}))"
            )));
        }
        let bound_prepass = 2.0 * d_in.max(floor);
        if d_pre > bound_prepass {
            failures.push(Err(format!(
                "{name} {label}: prepass is {d_pre:e} from exact, bound {bound_prepass:e} \
                 (2 * max(inline {d_in:e}, f16 floor {floor:e}))"
            )));
        }
    }
    if failures.is_empty() {
        println!("ok: {name} (chunk={chunk}, n_chunks={n_chunks})");
    }
    failures
}

/// One side of the q4_0 cache → `dequant_kv_f16` → a concrete f16 cache, plus that cache widened
/// exactly to f32 for the oracle.
///
/// Sides are processed ONE AT A TIME and the quantised bytes are released as soon as they are on
/// the device: an earlier version of this test that held both sides' q4_0 bytes, both device caches
/// and both host copies at once for the 32k case threw `std::bad_alloc` out of the driver's
/// `vkAllocateMemory` inside the job's memory slot (measured 2026-11-12).
///
/// The returned cache is a fresh `KvCache` buffer rather than the prepass's own output buffer:
/// downloading and re-uploading severs the cross-recorder scratch hazard (see the decode legs), and
/// both prefill tiers then read the very same bytes.
fn prefill_prepass(
    be: &VulkanBackend,
    elems: usize,
    seed: usize,
) -> (Box<dyn infr_core::backend::Buffer>, Vec<f32>) {
    let bytes = q4_0_side_bytes_bounded(elems, seed);
    let q4 = be.alloc(bytes.len(), BufferUsage::KvCache).unwrap();
    be.upload(q4.as_ref(), &bytes).unwrap();
    drop(bytes);
    let f16 = be.alloc(elems * 2, BufferUsage::Activations).unwrap();
    {
        let rec = be.recorder().unwrap();
        rec.dequant_kv_f16(infr_core::DType::Q4_0, q4.as_ref(), f16.as_ref(), elems);
        rec.finish().unwrap();
    }
    let mut h = vec![0u8; elems * 2];
    be.download(f16.as_ref(), &mut h).unwrap();
    drop(q4);
    drop(f16);
    // The oracle reads this f16 cache, not the q4_0 bytes: the prepass's own rounding to f16 is
    // then common-mode between the two legs and the exact reference, so the tolerance below is a
    // statement about the ATTENTION tier. That the prepass reproduces exact q4_0 values is what
    // the decode test above pins.
    let exact = widen_f16_to_f32(&h);
    let concrete = be.alloc(elems * 2, BufferUsage::KvCache).unwrap();
    be.upload(concrete.as_ref(), &h).unwrap();
    drop(h);
    (concrete, exact)
}

/// Whether this device can build the non-FA coopmat pipeline at all — leg 1 of the prefill test.
///
/// `attn_qk_warp`/`attn_qk`/`attn_pv` are `GL_KHR_cooperative_matrix : require`, and gfx10-class
/// RADV (RAPHAEL_MENDOCINO, measured 2026-11-12) ENUMERATES the 16x16x16 f16 config but ABORTS the
/// process in `aco_assembler.cpp:1291` — "Unsupported opcode:
/// v_wmma_f32_16x16x16_f16" — when the pipeline is compiled. An abort is not catchable, so the leg
/// is refused by device rather than attempted and recovered. RDNA1/RDNA2 iGPU targets are listed;
/// an unrecognised device is allowed — that is the 7900 XTX and every discrete coopmat part, which
/// is where this leg has to run at all (see the module header for the lease that unlocks it).
fn coopmat_assembles(name: &str) -> bool {
    const PRE_RDNA3: [&str; 9] = [
        "RAPHAEL",
        "MENDOCINO",
        "YELLOW_CARP",
        "VAN_GOGH",
        "NAVI10",
        "NAV10",
        "NAVI12",
        "NAVI14",
        "NAVI21",
    ];
    !PRE_RDNA3.iter().any(|t| name.contains(t))
}

/// Runs one hd>128 PREFILL case (rows=256) through BOTH tiers the ladder chooses between there, on
/// the same q4_0 cache and the same `dequant_kv_f16` prepass the adapter's `deq_k`/`deq_v` arms
/// record: `attention_prefill_nonfa` (coopmat QK → row softmax → coopmat PV, through a materialized
/// `[nh, mpad, kv_pad]` f16 scores matrix) and `attention_prefill_nc_fa` (the fused online-softmax
/// tile the `nc_fa_wide` reroute picks, which allocates no pool scratch at all). Each leg is scored
/// against the exact f64 oracle on the sampled `(row, head)` pairs under this file's rule — no leg
/// may be more than twice as far from exact as the other, beyond the 2^-11 f16 floor — with the
/// floor widened for `nonfa` by its own f16 scores-matrix term (see `scores_term` below), and the
/// pair's bit-identity REPORTED, not asserted (different accumulation order by design).
///
/// `run_nonfa` false (see [`coopmat_assembles`]) drops leg 1 and its mutual bound; what remains is
/// the absolute bound on nc_fa plus the scratch accounting, which is the point of the reroute.
///
/// Rows are sampled across the 32-row `nc_fa` tile and 64-row `nonfa` tile boundaries and the
/// causal end of the chunk; heads are sampled to cover every GQA kv group (both the first and the
/// last q head of each), since a whole 256×24-head oracle at kv_len 32768 is ~10^11 f64 MACs.
///
/// `rows_per_call` is the nc_fa leg's dispatch size — 256 is the whole chunk in one call, the shape
/// the adapter issues; a smaller value splits it into independent row pieces (same `BM` tile, same
/// key loop, nothing shared between pieces) so that one submission stays inside the device's
/// command timeout. See the leg-2 comment.
///
/// The oracle runs over the prepass's f16 cache rather than over the raw q4_0 nibbles: both legs
/// read those f16 bytes, so the prepass's own rounding is common-mode and the two distances measure
/// the ATTENTION tier, which is what this reroute changes. The prepass reproducing exact q4_0 values
/// is what the decode test above pins.
#[allow(clippy::too_many_arguments)]
fn run_prefill_case(
    be: &VulkanBackend,
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_len: usize,
    rows_sample: &[usize],
    heads_sample: &[usize],
    run_nonfa: bool,
    rows_per_call: usize,
) -> Vec<Result<(), String>> {
    let rows = 256usize;
    let pos = kv_len - rows;
    let scale = 0.0f32; // default 1/sqrt(hd)
    let window = 0usize; // full causal
    let cap = 0usize; // full-context cache (RROW identity)
    let mpad = rows.div_ceil(64) * 64;
    let kv_pad = kv_len.div_ceil(256) * 256;
    let elems = kv_len * nkv * hd;
    let o_bytes = mpad * nh * hd * 4;
    let mib = |b: usize| b as f64 / (1 << 20) as f64;
    let name = format!(
        "q4_0 prefill rows={rows}/{rows_per_call}percall kv={kv_len} nh={nh} nkv={nkv} hd={hd}"
    );
    // Per-stage timings: on a small iGPU an fma-only flash tile over 32k keys is not free, and
    // without these a slow case and a hung one look identical in the log.
    let clock = std::time::Instant::now;
    let mut t_nonfa = None::<std::time::Duration>;

    let q = synth_f16(rows * nh * hd, 101);
    let q_buf = be.alloc(q.len(), BufferUsage::Activations).unwrap();
    be.upload(q_buf.as_ref(), &q).unwrap();

    // ── the shared prepass: q4_0 → f16 through the adapter's `dequant_kv_f16`, one side at a time
    // (see [`prefill_prepass`]); both tiers below read the two caches it hands back. ────────────
    let lap = clock();
    let (kf16, kd32) = prefill_prepass(be, elems, 11);
    let (vf16, vd32) = prefill_prepass(be, elems, 23);
    let t_prepass = lap.elapsed();

    // ── Leg 1: `attention_prefill_nonfa`, the tier hd 256 took before the reroute. ─────────────
    // Its scratch is the cost the reroute removes, so it is accounted for even when the leg
    // cannot run (see [`coopmat_assembles`]).
    let s_bytes = nh * mpad * kv_pad * 2;
    let pv_bytes = 8 * mpad * nh * hd * 4;
    println!(
        "{name}: nonfa scratch = {s:.1} MiB scores [nh, mpad, kv_pad] f16 + {pv:.1} MiB PV \
         partials; nc_fa = 0 MiB",
        s = mib(s_bytes),
        pv = mib(pv_bytes),
    );
    let nonfa_bits: Option<Vec<u32>> = if run_nonfa {
        let lap = clock();
        let s = match be.alloc(s_bytes, BufferUsage::Activations) {
            Ok(b) => b,
            // The scores matrix is exactly what `nc_fa_wide` exists to avoid; if the pool refuses
            // it, that refusal IS the finding, so report it rather than skipping quietly.
            Err(e) => panic!(
                "{name}: the nonfa leg needs the {:.1} MiB scores matrix that `nc_fa_wide` exists \
                 to avoid, and the device refused it: {e}",
                mib(s_bytes)
            ),
        };
        let pv = be.alloc(pv_bytes, BufferUsage::Activations).unwrap();
        let o_nonfa = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        {
            let rec = be.recorder().unwrap();
            rec.attention_prefill_nonfa(
                q_buf.as_ref(),
                kf16.as_ref(),
                vf16.as_ref(),
                o_nonfa.as_ref(),
                s.as_ref(),
                pv.as_ref(),
                rows,
                kv_len,
                nh,
                nkv,
                hd,
                pos,
                window,
                scale,
            );
            rec.finish().unwrap();
        }
        let mut out = vec![0u8; o_bytes];
        be.download(o_nonfa.as_ref(), &mut out).unwrap();
        t_nonfa = Some(lap.elapsed());
        let leg = Some(bits(&out));
        drop(s);
        drop(pv);
        drop(o_nonfa);
        leg
    } else {
        println!(
            "{name}: nonfa leg SKIPPED — the driver on this device cannot assemble the coopmat \
             pipeline (see `coopmat_assembles`), so only the nc_fa leg runs"
        );
        None
    };

    // ── Leg 2: `attention_prefill_nc_fa`, the reroute's target. Same q, same f16 KV, same scale. ─
    // Issued in `rows_per_call`-row pieces; the tile is `BM` rows and the pieces are independent, so
    // `rows_per_call == rows` is one dispatch of exactly the shape the adapter issues — which is
    // what the 4k case does, on every device. The 32k case runs the SPLIT form (4 × 64 rows) on
    // every device too: the single 256-row × 32k-key dispatch has been measured unsafe on exactly
    // one card — RAPHAEL_MENDOCINO, where it takes the GPU down every attempt with "The CS has been
    // cancelled because the context is lost. This context is guilty of a hard recovery" (measured
    // 2026-11-12) while the four dispatches that cover the identical work finish in 3.0 s — and has
    // never been asked of the 7900. Flipping this case to 256 is the remaining coverage gap; it
    // needs a lease, not a guess.
    let mut ncfa_bits = vec![0u32; mpad * nh * hd];
    let lap = clock();
    for ch in 0..rows.div_ceil(rows_per_call) {
        let r0 = ch * rows_per_call;
        let rc = rows_per_call.min(rows - r0);
        let qb = be
            .alloc(rc * nh * hd * 2, BufferUsage::Activations)
            .unwrap();
        let qbytes = nh * hd * 2;
        be.upload(qb.as_ref(), &q[r0 * qbytes..(r0 + rc) * qbytes])
            .unwrap();
        let oc = be
            .alloc(rc * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        {
            let rec = be.recorder().unwrap();
            rec.attention_prefill_nc_fa(
                qb.as_ref(),
                kf16.as_ref(),
                vf16.as_ref(),
                oc.as_ref(),
                rc,
                kv_len,
                nh,
                nkv,
                hd,
                pos + r0,
                window,
                scale,
                cap,
            );
            rec.finish().unwrap();
        }
        let mut out = vec![0u8; rc * nh * hd * 4];
        be.download(oc.as_ref(), &mut out).unwrap();
        let n = rc * nh * hd;
        ncfa_bits[r0 * nh * hd..r0 * nh * hd + n].copy_from_slice(&bits(&out));
    }
    let t_ncfa = lap.elapsed();

    let mut failures = Vec::new();
    for (label, leg) in [
        ("nc_fa", ncfa_bits.as_slice()),
        ("nonfa", nonfa_bits.as_deref().unwrap_or(&[])),
    ] {
        if !leg.is_empty() && !leg.iter().any(|&b| b != 0) {
            failures.push(Err(format!(
                "{name} {label}: output is all zeros — the leg is not exercising the kernel"
            )));
        }
    }
    if !failures.is_empty() {
        return failures;
    }

    // ── Exact f64 attention over the f16 KV both legs read, on the sampled (row, head) pairs. ──
    // Both legs are done with the device side of the cache; the oracle reads the host copies.
    drop(q_buf);
    drop(kf16);
    drop(vf16);
    let pairs: Vec<(usize, usize)> = rows_sample
        .iter()
        .flat_map(|&r| heads_sample.iter().map(move |&h| (r, h)))
        .collect();
    let lap = clock();
    let parts = exact_prefill(&q, &kd32, &vd32, nh, nkv, hd, kv_len, pos, &pairs);
    let t_oracle = lap.elapsed();
    let ms = |d: std::time::Duration| format!("{:.0}", d.as_secs_f64() * 1e3);
    println!(
        "{name}: timings (ms) — prepass {}, nonfa {}, nc_fa {}, oracle {}",
        ms(t_prepass),
        t_nonfa.map_or_else(|| "n/a".to_string(), ms),
        ms(t_ncfa),
        ms(t_oracle)
    );
    // The largest |v| any row can average over, for the scores-rounding term below.
    let v_max = vd32.iter().fold(0.0f32, |a, x| a.max(x.abs())) as f64;
    drop(kd32);
    drop(vd32);
    let (d_ncfa, exact_max) = max_abs_vs_exact_at(&ncfa_bits, &parts);
    // Half an f16 ulp of the output magnitude — the same rounding floor the decode legs use.
    let floor = (2.0f64).powi(-11) * exact_max;
    // `attn_qk` writes its scores matrix as f16, so each score reaches `exp` with a relative error
    // of up to 2^-11 * |s|; with `sum|dp| <= 2 * 2^-11 * max|s|` (the softmax normalisation cancels
    // the common part) that puts the nonfa leg's OWN rounding at up to 2^-10 * max|s| * max|v|
    // above the output floor. `attn_nc_fa` keeps its scores in f32 registers and has no such term.
    // The mutual 2x rule below is therefore applied against `floor + scores_term` for the nonfa
    // leg; without that term the rule would be asserting that an f16-scores tier rounds like a
    // f32-scores tier, which is not true and is not what the reroute is about.
    let scores_term =
        (2.0f64).powi(-10) * parts.iter().fold(0.0f64, |a, &(_, _, s)| a.max(s)) * v_max;
    let rel = |d: f64| if exact_max > 0.0 { d / exact_max } else { d };
    println!(
        "{name}: max|exact| = {exact_max:e} over {} sampled (row, head) blocks; \
         nc_fa max|nc_fa-exact| = {d_ncfa:e} (rel {:e}); f16 floor = {floor:e}, nonfa \
         scores-rounding term = {scores_term:e} (max|s| {}, max|v| {v_max:e})",
        pairs.len(),
        rel(d_ncfa),
        parts.iter().fold(0.0f64, |a, &(_, _, s)| a.max(s)),
    );

    // The reroute must not degrade accuracy: nc_fa may be no further from exact than twice the
    // nonfa leg's distance (beyond f16 rounding), and no further than twice the floor itself.
    let bound_ncfa_abs = 4.0 * floor;
    if d_ncfa > bound_ncfa_abs {
        failures.push(Err(format!(
            "{name}: nc_fa is {d_ncfa:e} from exact, absolute bound {bound_ncfa_abs:e} \
             (4 * f16 floor {floor:e})"
        )));
    }
    if let Some(nonfa_bits) = &nonfa_bits {
        let d_nonfa = max_abs_vs_exact_at(nonfa_bits, &parts).0;
        println!(
            "{name}: nonfa max|nonfa-exact| = {d_nonfa:e} (rel {:e})",
            rel(d_nonfa)
        );
        match compare_bit_identical("nc_fa vs nonfa", nonfa_bits, &ncfa_bits) {
            Ok(()) => {
                println!("{name}: the two prefill legs are BIT-IDENTICAL (reported, not asserted)")
            }
            Err(e) => println!("{name}: {e} (reported, not asserted)"),
        }
        let d_between = max_abs_between_bits(nonfa_bits, &ncfa_bits);
        println!(
            "{name}: max|nc_fa-nonfa| = {d_between:e} (rel {:e})",
            rel(d_between)
        );
        let bound_ncfa = 2.0 * d_nonfa.max(floor);
        if d_ncfa > bound_ncfa {
            failures.push(Err(format!(
                "{name}: nc_fa is {d_ncfa:e} from exact, bound {bound_ncfa:e} \
                 (2 * max(nonfa {d_nonfa:e}, f16 floor {floor:e}))"
            )));
        }
        let bound_nonfa = 2.0 * d_ncfa.max(floor + scores_term);
        if d_nonfa > bound_nonfa {
            failures.push(Err(format!(
                "{name}: nonfa is {d_nonfa:e} from exact, bound {bound_nonfa:e} \
                 (2 * max(nc_fa {d_ncfa:e}, floor {floor:e} + its f16 scores term \
                 {scores_term:e}))"
            )));
        }
    } else {
        println!(
            "{name}: no nonfa leg on this device — nc_fa is checked against exact alone, against \
             the absolute {} * f16-floor bound",
            bound_ncfa_abs / floor
        );
    }
    if failures.is_empty() {
        println!("ok: {name} (pos={pos}, mpad={mpad}, kv_pad={kv_pad})");
    }
    failures
}

/// Both test fns in this file drive the integrated GPU; the repo runs GPU workloads serially, and
/// the harness would otherwise run them on two threads at once.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device selection for both tests. Returns the label to print, or `None` when the caller must
/// skip — which the caller reports through [`no_case_ran`], never as a silent pass.
///
/// A discrete card (an RX 7900 XTX, or a NAVI31 reference part) is refused unless
/// `INFR_PARITY_DISCRETE_LEASE` names the GPU lock/lease the caller holds on it — the value is both
/// the proof of the lease and what shows up in the log, so a run on that card cannot be mistaken for
/// one on the integrated GPU. It is also the only card whose driver assembles the coopmat pipeline
/// the prefill test's `nonfa` leg needs (see [`coopmat_assembles`]).
fn device_gate(name: &str) -> Option<String> {
    let discrete = name.contains("7900") || name.contains("NAVI31");
    let lease = std::env::var("INFR_PARITY_DISCRETE_LEASE").unwrap_or_default();
    if discrete && lease.is_empty() {
        eprintln!(
            "skip: refusing discrete GPU ({name}) — run it under a lease and name the lease in \
             INFR_PARITY_DISCRETE_LEASE (see the module header)"
        );
        return None;
    }
    Some(if discrete {
        format!("{name} (discrete, under lease {lease})")
    } else {
        name.to_string()
    })
}

/// One call site per test fn, on every path that ran no case. The lease holder has to be able to
/// tell "the discrete card agreed with the iGPU" from "the discrete card refused to run", so a
/// fully-skipped test says so out loud and fails when the run was demanded.
fn no_case_ran(test: &str) {
    println!("{test}: SKIPPED: no case ran");
    if std::env::var("INFR_PARITY_REQUIRE_RUN").is_ok_and(|v| v == "1") {
        panic!("{test}: INFR_PARITY_REQUIRE_RUN=1 but no case ran (device refused or absent)");
    }
}

#[test]
fn kv_inline_q4_matches_prepass_at_head_dim_256() {
    const THIS: &str = "kv_inline_q4_matches_prepass_at_head_dim_256";
    let _serial = gpu_serial();
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        no_case_ran(THIS);
        return;
    };
    let Some(device) = device_gate(&be.capabilities().name) else {
        no_case_ran(THIS);
        return;
    };
    println!("device: {device}");

    // Production model shapes (Qwen3.8-27B attention layers) plus one smaller case.
    let cases: [(usize, usize, usize, usize); 7] = [
        (24, 4, 256, 1),
        (24, 4, 256, 31),
        (24, 4, 256, 32),
        (24, 4, 256, 33),
        (24, 4, 256, 257),
        (24, 4, 256, 4096),
        (8, 2, 256, 64),
    ];
    let mut failures = Vec::new();
    for (nh, nkv, hd, kv_len) in cases {
        for r in run_case(&be, nh, nkv, hd, kv_len) {
            if let Err(e) = r {
                eprintln!("FAIL: {e}");
                failures.push(e);
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} leg(s) are more than 2x further from exact than the other leg (f16 floor allowed):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

/// The reroute's numeric claim: at the Qwen3.8-27B prefill shape (nh=24, nkv=4, hd=256, q4_0 KV,
/// rows=256, causal) the fused non-coopmat flash tile `attention_prefill_nc_fa` — which the ladder
/// now prefers over `attention_prefill_nonfa` for hd > 128 — is as close to exact attention as the
/// tier it replaces (and, where that tier cannot run at all, within the f16 output floor on its
/// own), at 4k and at 32k depth. See `run_prefill_case` for the legs, the sampling and the
/// bounded-scale blocks.
///
/// Both cases run on both cards: the discrete lease (`INFR_PARITY_DISCRETE_LEASE`, module header)
/// is what enables the `nonfa` comparison leg, and the integrated GPU runs the `nc_fa` leg plus the
/// same oracle with that leg skipped by device.
#[test]
fn kv_prefill_q4_0_head_dim_256_nc_fa_matches_nonfa() {
    const THIS: &str = "kv_prefill_q4_0_head_dim_256_nc_fa_matches_nonfa";
    let _serial = gpu_serial();
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        no_case_ran(THIS);
        return;
    };
    let name = be.capabilities().name;
    let Some(device) = device_gate(&name) else {
        no_case_ran(THIS);
        return;
    };
    let run_nonfa = coopmat_assembles(&name);
    println!(
        "device: {device}; coopmat nonfa leg: {}",
        if run_nonfa { "on" } else { "off" }
    );

    // The f64 oracle dominates the runtime (~1.7e8 MACs/s measured in a DEBUG build on the job's 2
    // cores; the --release build the header command uses is faster), so rows and heads are sampled
    // — see `run_prefill_case`. The 4k case (cheap keys) walks eight rows straddling both tiers' row
    // tiles and EVERY q head; the 32k case keeps the four rows that bound the tiles (chunk start,
    // end of nc_fa's 32-row tile 0, start of tile 2 = nonfa's 64-row boundary, chunk end) and cuts
    // the heads to each GQA group's first and last. On the iGPU the two cases together measured
    // 20-30 s, about half of it the oracle.
    let rows: &[usize] = &[0, 1, 31, 32, 63, 64, 127, 255];
    let tile_bounds: &[usize] = &[0, 31, 64, 255];
    let group_heads: &[usize] = &[0, 5, 6, 11, 12, 17, 18, 23];
    let all_heads: Vec<usize> = (0..24).collect();
    // (kv_len, sampled rows, sampled heads, nc_fa rows per dispatch)
    let cases: [(usize, &[usize], &[usize], usize); 2] = [
        (4096, rows, &all_heads, 256),
        (32768, tile_bounds, group_heads, 64),
    ];

    let mut failures = Vec::new();
    for (kv_len, rows_sample, heads_sample, rows_per_call) in cases {
        for leg in run_prefill_case(
            &be,
            24,
            4,
            256,
            kv_len,
            rows_sample,
            heads_sample,
            run_nonfa,
            rows_per_call,
        ) {
            if let Err(e) = leg {
                eprintln!("FAIL: {e}");
                failures.push(e);
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} prefill leg(s) fell outside the tolerance (2x the other leg's distance from \
             exact, beyond the f16 floor and nonfa's f16 scores term):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
