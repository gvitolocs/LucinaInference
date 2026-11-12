//! `lucina-parity`: run the engine on the oracle's tokens and compare against llama.cpp.
//!
//!   lucina-parity infr-logits --model M.gguf --tokens work/ref/x.tokens.txt --out work/infr/x
//!   lucina-parity compare --n-vocab V --ref work/ref/x.logits.f32 --test work/infr/x.logits.f32
//!   lucina-parity gate --n-vocab V --ref R --self S1,S2,.. --test T [--k 2]
//!   lucina-parity tokenize --model M.gguf --lines corpus.txt --out work/infr/tok
//!   lucina-parity tokens-diff --ref R.tokens.txt --test T.tokens.txt [--lines corpus.txt]
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use lucina_parity::{first_token_diff, read_tokens, Logits, Report, Tolerance};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// All-position logits from the engine's CPU reference forward.
    InfrLogits {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the comparison of two logits files as JSON.
    Compare {
        #[arg(long)]
        n_vocab: usize,
        #[arg(long = "ref")]
        reference: PathBuf,
        #[arg(long)]
        test: PathBuf,
        /// also print one line per position
        #[arg(long)]
        rows: bool,
    },
    /// Fail unless `test` is within k times llama.cpp's own spread (ref vs each `self` file).
    Gate {
        #[arg(long)]
        n_vocab: usize,
        #[arg(long = "ref")]
        reference: PathBuf,
        #[arg(long = "self", value_delimiter = ',')]
        selves: Vec<PathBuf>,
        #[arg(long)]
        test: PathBuf,
        #[arg(long, default_value_t = 2.0)]
        k: f64,
    },
    /// Tokenize each line (with \n, \t, \\ escapes) like `llama-logits --tokenize-lines`.
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        lines: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compare two tokenized-corpus files line by line (one line per corpus line, ids space-separated).
    /// GPU window probe on the Vulkan backend, one model load: open a session at `ctx` with the given
    /// KV type (weights + KV resident) and report VRAM, drop it, then last-token logits for each
    /// prefix length (written as `<out>-p<N>.logits.f32`, one row each).
    VulkanWindow {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long, value_delimiter = ',')]
        prefixes: Vec<usize>,
        /// session context to probe VRAM with; 0 skips the session probe
        #[arg(long, default_value_t = 0)]
        ctx: usize,
        #[arg(long, default_value = "q4_0")]
        kv: String,
        /// also write the logits of every row (`<out>.logits.f32`) from one batched forward
        #[arg(long)]
        all_rows: bool,
        #[arg(long)]
        out: PathBuf,
    },
    TokensDiff {
        #[arg(long = "ref")]
        reference: PathBuf,
        #[arg(long)]
        test: PathBuf,
        /// also print the corpus line (1-based) for each mismatch, truncated to 120 chars
        #[arg(long)]
        lines: Option<PathBuf>,
    },
}

fn load_cpu(model: &Path) -> Result<infr_llama::SeamModel> {
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = Some("cpu".into());
    infr_llama::SeamModel::load_with(model, None, Arc::new(cfg))
}

fn unescape(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.peek() {
                Some('n') => {
                    out.push('\n');
                    it.next();
                }
                Some('t') => {
                    out.push('\t');
                    it.next();
                }
                Some('\\') => {
                    out.push('\\');
                    it.next();
                }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Read a tokenized-corpus file: one line per corpus line, token ids separated by single spaces
/// (an empty line means the input tokenized to zero tokens).
fn read_token_lines(path: &Path) -> Result<Vec<Vec<u32>>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    s.lines()
        .map(|l| {
            l.split_whitespace()
                .map(|t| {
                    t.parse::<u32>()
                        .with_context(|| format!("bad token id {t:?} in {}", path.display()))
                })
                .collect()
        })
        .collect()
}

/// Up to 3 ids before index `k` plus `k` itself and up to 5 after it, space-separated.
fn token_window(ids: &[u32], k: usize) -> String {
    let start = k.saturating_sub(3);
    let end = (k + 6).min(ids.len());
    ids[start..end]
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Diagnostics through `tracing`, on STDERR: stdout is this tool's contract — `check-oracle.sh`
/// collapses `compare`'s JSON into one log line — and a subscriber left on its stdout default would
/// corrupt it, the same trap `infr` documents at its own `main`. `RUST_LOG` tunes the levels; the
/// default keeps every line as visible as it was.
fn init_logging() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
}

fn main() -> Result<()> {
    init_logging();
    match Cli::parse().cmd {
        Cmd::InfrLogits { model, tokens, out } => {
            let toks = read_tokens(&tokens)?;
            let m = load_cpu(&model)?;
            let t0 = std::time::Instant::now();
            let (logits, _h) = m.verify_logits_and_h_cpu(&toks)?;
            tracing::info!(
                "infr cpu forward: {} tokens in {:.1} s",
                toks.len(),
                t0.elapsed().as_secs_f64()
            );
            let mut p = out.into_os_string();
            p.push(".logits.f32");
            Logits::write(Path::new(&p), &logits)?;
        }
        Cmd::Compare {
            n_vocab,
            reference,
            test,
            rows,
        } => {
            let rep = Report::new(
                &Logits::read(&reference, n_vocab)?,
                &Logits::read(&test, n_vocab)?,
            )?;
            if rows {
                for (i, s) in rep.rows.iter().enumerate() {
                    // print-ok: `compare --rows` — the per-position table on stdout the user asked for.
                    println!(
                        "{i:4} kl {:.2e} rel_rms {:.3} max_abs {:6.3} top1 {} {} margin {:.2}",
                        s.kl, s.rel_rms, s.max_abs, s.top1_ref, s.top1_test, s.ref_margin
                    );
                }
            }
            // print-ok: `compare`'s JSON result is this subcommand's whole output; scripts pipe it.
            println!("{}", serde_json::to_string_pretty(&rep.json())?);
        }
        Cmd::Gate {
            n_vocab,
            reference,
            selves,
            test,
            k,
        } => {
            let r = Logits::read(&reference, n_vocab)?;
            let spreads: Vec<Report> = selves
                .iter()
                .map(|s| Report::new(&r, &Logits::read(s, n_vocab)?))
                .collect::<Result<_>>()?;
            let tol = Tolerance::from_self_spread(&spreads.iter().collect::<Vec<_>>(), k);
            let rep = Report::new(&r, &Logits::read(&test, n_vocab)?)?;
            // print-ok: `gate`'s JSON record — the tolerance, the llama.cpp self-spread, the test
            // report. `PASS` and the exit code only summarise it; this is what gets kept.
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "tolerance": {"kl_mean": tol.kl_mean, "kl_max": tol.kl_max, "rel_rms_max": tol.rel_rms_max, "decisive_flips": tol.decisive_flips, "k": k},
                    "self_spread": spreads.iter().map(|s| s.json()).collect::<Vec<_>>(),
                    "test": rep.json(),
                }))?
            );
            tol.check(&rep)?;
            // print-ok: the gate's verdict line, on stderr so the JSON above stays parseable.
            eprintln!("PASS");
        }
        Cmd::Tokenize { model, lines, out } => {
            let m = load_cpu(&model)?;
            let text = std::fs::read_to_string(&lines)
                .with_context(|| format!("read {}", lines.display()))?;
            let mut o = String::new();
            for line in text.lines() {
                let ids = m.encode(&unescape(line))?;
                o.push_str(
                    &ids.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(" "),
                );
                o.push('\n');
            }
            let mut p = out.into_os_string();
            p.push(".tokens.txt");
            std::fs::write(&p, o)?;
        }
        Cmd::VulkanWindow {
            model,
            tokens,
            prefixes,
            ctx,
            kv,
            all_rows,
            out,
        } => {
            let toks = read_tokens(&tokens)?;
            let mut cfg = infr_llama::EngineConfig::default();
            let dt = infr_core::budget::parse_kv_dtype(&kv)
                .with_context(|| format!("unknown kv type {kv}"))?;
            cfg.kv.type_k = Some(dt);
            cfg.kv.type_k_specified = true;
            cfg.kv.type_v = Some(dt);
            cfg.kv.type_v_specified = true;
            let t0 = std::time::Instant::now();
            let m = infr_llama::SeamModel::load_with(&model, None, Arc::new(cfg))?;
            tracing::info!("WINDOW model opened in {:.1} s", t0.elapsed().as_secs_f64());
            if ctx > 0 {
                let t = std::time::Instant::now();
                let s = m.vulkan_session(ctx)?;
                let v = s.vram();
                // print-ok: the probe's REPORT — the VRAM figures are the measurement `vulkan-window`
                // exists to take; the logits it also writes go to files.
                eprintln!(
                    "WINDOW session ctx {ctx} kv {kv} on {}: ready in {:.1} s; device total {} MiB, available {} MiB, used {} MiB",
                    s.device_name(),
                    t.elapsed().as_secs_f64(),
                    v.total >> 20,
                    v.available >> 20,
                    (v.total - v.available) >> 20
                );
            }
            if all_rows {
                let t = std::time::Instant::now();
                let rows = m.verify_logits_vulkan(&toks)?;
                tracing::info!(
                    "WINDOW vulkan all-row logits, {} tokens: {:.1} s",
                    toks.len(),
                    t.elapsed().as_secs_f64()
                );
                let mut o = out.clone().into_os_string();
                o.push(".logits.f32");
                Logits::write(Path::new(&o), &rows)?;
            }
            for p in prefixes {
                let t = std::time::Instant::now();
                let row = m.prefill_logits_vulkan(&toks[..p.min(toks.len())])?;
                tracing::info!(
                    "WINDOW vulkan logits prefix {p}: {:.1} s",
                    t.elapsed().as_secs_f64()
                );
                let mut o = out.clone().into_os_string();
                o.push(format!("-p{p}.logits.f32"));
                Logits::write(Path::new(&o), &row)?;
            }
        }
        Cmd::TokensDiff {
            reference,
            test,
            lines,
        } => {
            let ref_lines = read_token_lines(&reference)?;
            let test_lines = read_token_lines(&test)?;
            if ref_lines.len() != test_lines.len() {
                // print-ok: the tokens-diff REPORT — this subcommand's output IS these stdout lines.
                println!(
                    "line counts differ: ref {} lines, test {} lines",
                    ref_lines.len(),
                    test_lines.len()
                );
            }
            let corpus = match &lines {
                Some(p) => Some(
                    std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?,
                ),
                None => None,
            };
            let n = ref_lines.len().max(test_lines.len());
            let mut mismatches = 0usize;
            for i in 0..n {
                let a: &[u32] = ref_lines.get(i).map_or(&[], |v| v.as_slice());
                let b: &[u32] = test_lines.get(i).map_or(&[], |v| v.as_slice());
                // a line present on one side only is a mismatch even when the other side tokenized to nothing
                let missing = i >= ref_lines.len() || i >= test_lines.len();
                if let Some(k) = first_token_diff(a, b).or(missing.then_some(0)) {
                    mismatches += 1;
                    // print-ok: the differing line, with the token window around the first difference.
                    println!(
                        "line {}: first difference at token index {}: ref [{}] test [{}]",
                        i + 1,
                        k,
                        token_window(a, k),
                        token_window(b, k)
                    );
                    if let Some(c) = &corpus {
                        if let Some(line) = c.lines().nth(i) {
                            let short: String = line.chars().take(120).collect();
                            // print-ok: the corpus text the ids above came from, under its diff line.
                            println!("  corpus line {}: {}", i + 1, short);
                        }
                    }
                }
            }
            // print-ok: the report's tally, the line every tokens-diff run ends with.
            println!("tokens-diff: {} of {} lines match", n - mismatches, n);
            if mismatches > 0 {
                bail!("tokens-diff: {} of {} lines differ", mismatches, n);
            }
        }
    }
    Ok(())
}
