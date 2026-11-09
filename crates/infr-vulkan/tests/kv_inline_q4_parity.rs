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
//! SAFETY: the RX 7900 XTX runs a production server. Everything here must run on the integrated
//! GPU (1002:164e) via `MESA_VK_DEVICE_SELECT=1002:164e
//! MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1`; the test additionally refuses a "7900"/"NAVI31"
//! device before touching any buffer.
//!
//! Run: `MESA_VK_DEVICE_SELECT=1002:164e MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 \
//!   cargo test --release -p infr-vulkan --test kv_inline_q4_parity -- --nocapture`
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

fn f16_at(bytes: &[u8], idx: usize) -> f64 {
    let lo = bytes[2 * idx] as u16;
    let hi = bytes[2 * idx + 1] as u16;
    f16::from_bits(lo | (hi << 8)).to_f64()
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

#[test]
fn kv_inline_q4_matches_prepass_at_head_dim_256() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let device = be.capabilities().name;
    println!("device: {device}");
    if device.contains("7900") || device.contains("NAVI31") {
        eprintln!("skip: refusing discrete GPU ({device})");
        return;
    }

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
