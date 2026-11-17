//! `kl-compare` on hand-made log-probability files: the format, llama.cpp's truncated KL, the
//! noise floor the reference sets for itself, and the pass rule. CPU only, no model, no engine —
//! every distribution below is written into the file by hand, so the expected numbers are
//! arithmetic rather than measurements.
//!
//! The records are built from integers on a scale that is exact in binary, so a decoded log-prob
//! `scale * q + min_log_prob` is the value the test names: either `ln 2` (every entry is then a
//! dyadic probability, and a normalised distribution is one whose probabilities sum to 1) or `1/4`
//! (used for the tests that put an entry exactly on the `-16` truncation boundary).

use lucina_parity::kl::{kl_compare, KlOptions, Role};
use std::path::{Path, PathBuf};

/// A scratch directory under `TMPDIR` (or the platform temp dir), removed on drop — including when
/// the test panics, because `cargo test` unwinds and drops its locals.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lucina-kl-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the scratch directory");
        Self(dir)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One record: the two header floats, then one uint16 per entry, padded to an even count.
fn rec(scale: f32, min: f32, q: &[u16]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&scale.to_le_bytes());
    b.extend_from_slice(&min.to_le_bytes());
    for &x in q {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b.resize(lucina_parity::kl::record_slots(q.len()) * 2, 0);
    b
}

/// `ln 2` as the writer would have stored it, and the offset that makes `q` count the powers of two
/// below it: `dyadic(q) = 2^-(q0 - q[i])`, so `q = [4,3,2,1,1]` is `[1/2, 1/4, 1/8, 1/16, 1/16]`.
fn dyadic(q: &[u16]) -> Vec<u8> {
    rec(LN2, -5.0 * LN2, q)
}

/// The vocabulary of every file below: five entries, enough for a normalised dyadic distribution.
const V: usize = 5;
const LN2: f32 = std::f32::consts::LN_2;

/// Write a `_logits_` file: magic, `n_ctx`, `n_vocab`, `n_chunk`, the tokens, then `records`
/// (chunk-major, `records_per_chunk(n_ctx)` per chunk).
fn write_logits(path: &Path, n_ctx: i32, tokens: &[i32], records: &[Vec<u8>]) {
    assert_eq!(
        records[0].len(),
        lucina_parity::kl::record_slots(V) * 2,
        "every record is a {}-entry distribution",
        V
    );
    let n_chunk = (tokens.len() / n_ctx as usize) as i32;
    let mut b = Vec::new();
    b.extend_from_slice(b"_logits_");
    for x in [n_ctx, V as i32, n_chunk] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for t in tokens {
        b.extend_from_slice(&t.to_le_bytes());
    }
    for r in records {
        b.extend_from_slice(r);
    }
    std::fs::write(path, b).expect("write the logits file");
}

/// Two chunks of `n_ctx = 5` → rows 2 and 3 of each are scored → 4 positions; the ids are distinct
/// so a failure listing names its position unambiguously.
fn tokens4() -> Vec<i32> {
    (101..111).collect()
}

fn opts<P: AsRef<Path>>(refs: &[P], cands: &[P], top: usize) -> KlOptions {
    KlOptions {
        refs: refs.iter().map(|p| p.as_ref().to_path_buf()).collect(),
        cands: cands.iter().map(|p| p.as_ref().to_path_buf()).collect(),
        top,
    }
}

#[test]
fn identical_files_give_zero_kl_and_a_zero_floor() {
    let s = Scratch::new("identical");
    // every position of every file the same distribution: the reference cannot move against itself
    let one = dyadic(&[4, 3, 2, 1, 1]);
    let toks = tokens4();
    let mut files = Vec::new();
    for name in ["base", "ref2", "cand"] {
        let p = s.at(name);
        write_logits(
            &p,
            5,
            &toks,
            &vec![one.clone(); 2 * lucina_parity::kl::records_per_chunk(5)],
        );
        files.push(p);
    }
    let sum = kl_compare(&opts(&files[..2], &files[2..], 3), None).expect("compare");
    assert_eq!(sum.positions, 4);
    assert_eq!(sum.floor_refs, 1);
    assert_eq!(sum.files.len(), 2);
    assert_eq!(sum.files[0].role, Role::Ref);
    assert_eq!(sum.files[1].role, Role::Cand);
    for f in &sum.files {
        assert_eq!(f.kl_fwd_max, 0.0, "identical records must give an exact 0");
        assert_eq!(f.kl_rev_max, 0.0);
        assert_eq!(f.p_diff_max, 0.0);
        assert_eq!(f.top1_agrees, 4);
    }
    assert_eq!(sum.files[1].passes, 4);
    assert_eq!(sum.files[1].failures, 0);
    // the zero floor is reported as what it is, not hidden
    assert_eq!(sum.floor_kl.zeros, 4);
    assert_eq!(sum.floor_kl.max, 0.0);
    assert_eq!(sum.failures.len(), 0);
    assert_eq!(sum.failure_count(), 0);
    assert!(sum.report().iter().any(|l| l.contains("kl-compare: PASS")));
}

#[test]
fn a_known_shift_gives_the_hand_computed_kl() {
    let s = Scratch::new("known");
    // base [1/2, 1/4, 1/8, 1/16, 1/16] vs cand [1/4, 1/2, 1/8, 1/16, 1/16]: the two top entries
    // swap, so KL(base || cand) = 1/2 ln 2 - 1/4 ln 2 = ln 2 / 4, and the reverse is the same
    // number because the swap is its own inverse.
    let toks = tokens4();
    let per = 2 * lucina_parity::kl::records_per_chunk(5);
    let base = s.at("base");
    let cand = s.at("cand");
    write_logits(&base, 5, &toks, &vec![dyadic(&[4, 3, 2, 1, 1]); per]);
    write_logits(&cand, 5, &toks, &vec![dyadic(&[3, 4, 2, 1, 1]); per]);
    let sum = kl_compare(&opts(&[&base], &[&cand], 10), None).expect("compare");
    let want = 0.25f64 * 2.0f64.ln();
    let f = &sum.files[0];
    assert!(
        (f.kl_fwd_max - want).abs() < 1e-6,
        "kl_fwd {} != ln 2 / 4 = {want}",
        f.kl_fwd_max
    );
    assert!(
        (f.kl_rev_max - want).abs() < 1e-6,
        "kl_rev {} != ln 2 / 4 = {want}",
        f.kl_rev_max
    );
    assert!(
        (f.p_diff_max - 0.25).abs() < 1e-6,
        "|dp| at the base's top-1 is |1/4 - 1/2| = 0.25, got {}",
        f.p_diff_max
    );
    assert_eq!(f.top1_agrees, 0);
    // no second --ref, so the floor is 0 everywhere and every position of the candidate fails
    assert_eq!(sum.floor_refs, 0);
    assert_eq!(sum.floor_kl.max, 0.0);
    assert_eq!(sum.floor_kl.zeros, 4);
    assert_eq!(f.failures, 4);
    assert_eq!(sum.failures[0].base_top1, 0);
    assert_eq!(sum.failures[0].cand_top1, 1);
    let top = &sum.failures[0].base_top;
    assert_eq!(top[0].0, 0);
    assert!(
        (top[0].1 - 0.5).abs() < 1e-6,
        "the base's top-1 probability is 1/2"
    );
    assert_eq!(sum.header.n_vocab as usize, V);
}

#[test]
fn the_truncation_at_minus_16_is_llamas() {
    let s = Scratch::new("trunc");
    // scale 1/4 and min -20 put the entries exactly on the boundary. The base's log-probabilities
    // are [0, -16, -16.25, -20, -20] over 5 entries; llama.cpp sums `p_log_base > -16` only, so
    // ONLY entry 0 counts, and the candidate agrees with the base there: the forward KL is exactly 0
    // even though three other entries differ. The reverse KL truncates on the CANDIDATE's entries
    // instead, and its entry 1 (-15) survives: KL(cand || base) = exp(-15) * 1.
    let toks = tokens4();
    let per = 2 * lucina_parity::kl::records_per_chunk(5);
    let b = [80u16, 16, 15, 0, 0];
    let c = [80u16, 20, 15, 4, 0];
    let base = s.at("base");
    let cand = s.at("cand");
    write_logits(&base, 5, &toks, &vec![rec(0.25, -20.0, &b); per]);
    write_logits(&cand, 5, &toks, &vec![rec(0.25, -20.0, &c); per]);
    let sum = kl_compare(&opts(&[&base], &[&cand], 5), None).expect("compare");
    let f = &sum.files[0];
    assert_eq!(
        f.kl_fwd_max, 0.0,
        "entries at or below -16 contribute nothing"
    );
    let want = (-15.0f64).exp();
    assert!(
        (f.kl_rev_max - want).abs() < 1e-12,
        "reverse kl {} != exp(-15) = {want}",
        f.kl_rev_max
    );
    // the base's top-1 is entry 0 in both, and its probability is exp(0) = 1 in this hand-made file
    assert_eq!(f.top1_agrees, 4);
    assert_eq!(f.p_diff_max, 0.0);
    assert_eq!(f.failures, 0, "kl 0 <= floor 0 and the top-1 agrees");
}

#[test]
fn a_failing_position_is_listed_with_its_numbers() {
    let s = Scratch::new("fail");
    // one chunk of n_ctx 5 → rows 2 and 3 are scored → 2 positions. Position 0 (chunk 0, row 2):
    // the reference moves a little, the candidate moves a lot and flips the top-1. Position 1
    // (chunk 0, row 3): all three files hold the same record.
    let toks: Vec<i32> = (101..106).collect();
    let p0_base = dyadic(&[4, 3, 2, 1, 1]);
    let p0_ref = dyadic(&[4, 2, 3, 1, 1]);
    let p0_cand = dyadic(&[3, 4, 2, 1, 1]);
    let same = dyadic(&[4, 3, 2, 1, 1]);
    let mk = |a: Vec<u8>, b: Vec<u8>, path: &Path| write_logits(path, 5, &toks, &[a, b]);
    let (base, r2, cand, tsv, json) = (
        s.at("base"),
        s.at("ref2"),
        s.at("cand"),
        s.at("out.tsv"),
        s.at("out.json"),
    );
    mk(p0_base.clone(), same.clone(), &base);
    mk(p0_ref, same.clone(), &r2);
    mk(p0_cand, same, &cand);
    let sum = kl_compare(&opts(&[&base, &r2], &[&cand], 3), Some(&tsv)).expect("compare");
    assert_eq!(sum.positions, 2);
    assert_eq!(
        sum.floor_kl.zeros, 1,
        "one position where the refs were bit-identical"
    );
    let floor = 0.125f64 * 2.0f64.ln();
    let want = 0.25f64 * 2.0f64.ln();
    assert!(
        (sum.floor_kl.max - floor).abs() < 1e-6,
        "the floor is the reference's own spread, got {}",
        sum.floor_kl.max
    );
    let c = &sum.files[1];
    assert_eq!((c.passes, c.failures), (1, 1));
    assert_eq!(c.top1_agrees, 1);
    assert_eq!(sum.failures.len(), 1);
    let x = &sum.failures[0];
    assert_eq!((x.chunk, x.row), (0, 2));
    assert_eq!(x.tok, toks[3], "row 2 predicts the token after it");
    assert_eq!(
        x.around,
        vec![101, 102, 103, 104],
        "the scored token and the three ids before it"
    );
    assert_eq!(
        x.file, 1,
        "index into the compared files, which excludes the base"
    );
    assert!((x.kl - want).abs() < 1e-6);
    assert!((x.floor - floor).abs() < 1e-6);
    assert!((x.base_top1, x.cand_top1) == (0, 1));
    assert!(
        x.top1_stable,
        "the one other ref kept the base's top-1 here"
    );
    assert_eq!(x.base_top[0].0, 0);
    assert_eq!(x.cand_top[0].0, 1);
    assert_eq!(x.base_top.len(), 3);
    let text = sum.report().join("\n");
    for needle in [
        "FAIL",
        "chunk 0 row 2",
        &format!("{:.3e}", x.kl),
        &format!("{:.3e}", x.floor),
    ] {
        assert!(
            text.contains(needle),
            "the summary omits {needle:?}:\n{text}"
        );
    }
    // the TSV: a header plus one row per (position, file), the files in ref-then-cand order
    let t = std::fs::read_to_string(&tsv).expect("read the tsv");
    let rows: Vec<&str> = t.lines().collect();
    assert_eq!(rows.len(), 1 + 2 * 2);
    assert_eq!(rows[0].split('\t').count(), 18);
    let last = rows[2].split('\t').collect::<Vec<_>>();
    assert_eq!(last[0], "0");
    assert_eq!(last[1], "2");
    assert_eq!(last[4], cand.to_string_lossy());
    assert_eq!(last[5], "cand");
    assert_eq!(last[17], "fail");
    assert_eq!(rows[1].split('\t').next_back().unwrap(), "ref");
    assert_eq!(rows[3].split('\t').next_back().unwrap(), "ref");
    assert_eq!(rows[4].split('\t').next_back().unwrap(), "pass");
    let j: serde_json::Value =
        serde_json::from_str(&sum.json().to_string()).expect("the summary is valid JSON");
    assert_eq!(j["verdict"], "FAIL");
    assert_eq!(j["positions"], 2);
    assert_eq!(j["failures"].as_array().unwrap().len(), 1);
    assert_eq!(j["failures"][0]["chunk"], 0);
    assert_eq!(j["failures"][0]["row"], 2);
    assert_eq!(j["floor_refs"], 1);
    assert_eq!(j["files"].as_array().unwrap().len(), 2);
    assert!(!json.exists(), "the tsv run wrote no json");
}

#[test]
fn an_unstable_reference_top1_excuses_the_candidates_flip() {
    let s = Scratch::new("unstable");
    let toks = tokens4();
    // the other ref flips the top-1 too, and the candidate copies it exactly: the pass rule's
    // second half lets the flip through, and `kl == floor` satisfies the first half.
    let per = 2 * lucina_parity::kl::records_per_chunk(5);
    let base = s.at("base");
    let r2 = s.at("ref2");
    let cand = s.at("cand");
    write_logits(&base, 5, &toks, &vec![dyadic(&[4, 3, 2, 1, 1]); per]);
    write_logits(&r2, 5, &toks, &vec![dyadic(&[3, 4, 2, 1, 1]); per]);
    write_logits(&cand, 5, &toks, &vec![dyadic(&[3, 4, 2, 1, 1]); per]);
    let sum = kl_compare(&opts(&[&base, &r2], &[&cand], 2), None).expect("compare");
    assert_eq!(
        sum.top1_unstable, per,
        "no position has a stable reference top-1"
    );
    assert_eq!(sum.files[1].top1_agrees, 0);
    assert_eq!(
        sum.files[1].failures, 0,
        "the flip is what the reference does too"
    );
    assert_eq!(sum.files[1].passes, per);
}

#[test]
fn a_lone_candidate_is_held_to_a_zero_floor() {
    let s = Scratch::new("lone");
    let toks = tokens4();
    let per = 2 * lucina_parity::kl::records_per_chunk(5);
    let base = s.at("base");
    let cand = s.at("cand");
    write_logits(&base, 5, &toks, &vec![dyadic(&[4, 3, 2, 1, 1]); per]);
    // a candidate that keeps the base's top-1 and still moves the distribution: with a single --ref
    // there is nothing to measure the floor against, so any positive KL is a failure
    write_logits(&cand, 5, &toks, &vec![dyadic(&[4, 2, 3, 1, 1]); per]);
    let sum = kl_compare(&opts(&[&base], &[&cand], 5), None).expect("compare");
    assert_eq!(sum.floor_kl.max, 0.0);
    assert!(sum.files[0].kl_fwd_max > 0.0);
    assert_eq!(sum.files[0].top1_agrees, per);
    assert_eq!(
        sum.files[0].failures, per,
        "with one --ref the floor is 0, so any KL fails"
    );
    assert!(sum
        .report()
        .iter()
        .any(|l| l.contains("the floor is 0 everywhere")));
}

#[test]
fn a_header_or_token_mismatch_is_refused_by_name() {
    let s = Scratch::new("mismatch");
    let toks = tokens4();
    let per = 2 * lucina_parity::kl::records_per_chunk(5);
    let one = dyadic(&[4, 3, 2, 1, 1]);
    let base = s.at("base");
    write_logits(&base, 5, &toks, &vec![one.clone(); per]);
    // a different n_chunk: the same tokens, one chunk fewer
    let short = s.at("short");
    write_logits(
        &short,
        5,
        &toks[..5],
        &vec![one.clone(); lucina_parity::kl::records_per_chunk(5)],
    );
    let e = kl_compare(&opts(&[&base, &short], &[], 3), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains(&short.to_string_lossy().to_string()), "{e}");
    assert!(e.contains("header differs"), "{e}");
    assert!(e.contains("n_chunk 1") && e.contains("n_chunk 2"), "{e}");
    // the same shape, one token changed at a known index
    let other = s.at("other");
    let mut bad = toks.clone();
    bad[7] = 999;
    write_logits(&other, 5, &bad, &vec![one.clone(); per]);
    let e = kl_compare(&opts(&[&base, &other], &[], 3), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains(&other.to_string_lossy().to_string()), "{e}");
    assert!(
        e.contains("token array differs") && e.contains("index 7"),
        "{e}"
    );
    // a different context length
    let wide = s.at("wide");
    let t8: Vec<i32> = (101..117).collect();
    write_logits(
        &wide,
        8,
        &t8,
        &vec![one.clone(); 2 * lucina_parity::kl::records_per_chunk(8)],
    );
    let e = kl_compare(&opts(&[&base, &wide], &[], 3), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("n_ctx 8") && e.contains("n_ctx 5"), "{e}");
    // a file that is not a log-probability file at all
    let junk = s.at("junk");
    std::fs::write(&junk, b"not a logits dump, not even close").unwrap();
    let e = kl_compare(&opts(&[&base, &junk], &[], 3), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("log-probabilities"), "{e}");
    // a truncated record stream is caught at open time, not 4,000 records in
    let cut = s.at("cut");
    let mut bytes = std::fs::read(&base).unwrap();
    bytes.truncate(bytes.len() - 3);
    std::fs::write(&cut, bytes).unwrap();
    let e = kl_compare(&opts(&[&cut], &[], 3), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("bytes") && e.contains("records"), "{e}");
}

#[test]
fn the_cli_writes_its_report_tsv_and_json() {
    let s = Scratch::new("cli");
    let toks = tokens4();
    let per = 2 * lucina_parity::kl::records_per_chunk(5);
    let base = s.at("base");
    let r2 = s.at("ref2");
    let cand = s.at("cand");
    let tsv = s.at("out.tsv");
    let json = s.at("out.json");
    write_logits(&base, 5, &toks, &vec![dyadic(&[4, 3, 2, 1, 1]); per]);
    write_logits(&r2, 5, &toks, &vec![dyadic(&[4, 2, 3, 1, 1]); per]);
    write_logits(&cand, 5, &toks, &vec![dyadic(&[3, 4, 2, 1, 1]); per]);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lucina-parity"))
        .args([
            "kl-compare",
            "--ref",
            &base.to_string_lossy(),
            "--ref",
            &r2.to_string_lossy(),
            "--cand",
            &cand.to_string_lossy(),
            "--top",
            "2",
            "--tsv",
            &tsv.to_string_lossy(),
            "--json",
            &json.to_string_lossy(),
        ])
        .output()
        .expect("run lucina-parity");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        !out.status.success(),
        "a failing candidate must be a non-zero exit: {stdout}"
    );
    assert!(stdout.contains("kl-compare: FAIL"), "{stdout}");
    assert!(stdout.contains("noise floor"), "{stdout}");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("outside the reference's own floor"), "{err}");
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read the json file"))
            .expect("the json file is valid");
    assert_eq!(j["verdict"], "FAIL");
    assert_eq!(j["header"]["n_vocab"], V as u64);
    assert_eq!(
        j["failures"].as_array().unwrap().len(),
        4,
        "all four positions fail"
    );
    let rows = std::fs::read_to_string(&tsv).expect("read the tsv");
    assert_eq!(rows.lines().count(), 1 + 4 * 2);
    // a run that passes exits zero
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lucina-parity"))
        .args([
            "kl-compare",
            "--ref",
            &base.to_string_lossy(),
            "--cand",
            &base.to_string_lossy(),
        ])
        .output()
        .expect("run lucina-parity");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("kl-compare: PASS"));
}
