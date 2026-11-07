//! `lucina-parity`: run the engine on the oracle's tokens and compare against llama.cpp.
//!
//!   lucina-parity infr-logits --model M.gguf --tokens work/ref/x.tokens.txt --out work/infr/x
//!   lucina-parity compare --n-vocab V --ref work/ref/x.logits.f32 --test work/infr/x.logits.f32
//!   lucina-parity gate --n-vocab V --ref R --self S1,S2,.. --test T [--k 2]
//!   lucina-parity tokenize --model M.gguf --lines corpus.txt --out work/infr/tok
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lucina_parity::{read_tokens, Logits, Report, Tolerance};
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

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::InfrLogits { model, tokens, out } => {
            let toks = read_tokens(&tokens)?;
            let m = load_cpu(&model)?;
            let t0 = std::time::Instant::now();
            let (logits, _h) = m.verify_logits_and_h_cpu(&toks)?;
            eprintln!("infr cpu forward: {} tokens in {:.1} s", toks.len(), t0.elapsed().as_secs_f64());
            let mut p = out.into_os_string();
            p.push(".logits.f32");
            Logits::write(Path::new(&p), &logits)?;
        }
        Cmd::Compare { n_vocab, reference, test, rows } => {
            let rep = Report::new(&Logits::read(&reference, n_vocab)?, &Logits::read(&test, n_vocab)?)?;
            if rows {
                for (i, s) in rep.rows.iter().enumerate() {
                    println!(
                        "{i:4} kl {:.2e} rel_rms {:.3} max_abs {:6.3} top1 {} {} margin {:.2}",
                        s.kl, s.rel_rms, s.max_abs, s.top1_ref, s.top1_test, s.ref_margin
                    );
                }
            }
            println!("{}", serde_json::to_string_pretty(&rep.json())?);
        }
        Cmd::Gate { n_vocab, reference, selves, test, k } => {
            let r = Logits::read(&reference, n_vocab)?;
            let spreads: Vec<Report> =
                selves.iter().map(|s| Report::new(&r, &Logits::read(s, n_vocab)?)).collect::<Result<_>>()?;
            let tol = Tolerance::from_self_spread(&spreads.iter().collect::<Vec<_>>(), k);
            let rep = Report::new(&r, &Logits::read(&test, n_vocab)?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "tolerance": {"kl_mean": tol.kl_mean, "kl_max": tol.kl_max, "rel_rms_max": tol.rel_rms_max, "k": k},
                    "self_spread": spreads.iter().map(|s| s.json()).collect::<Vec<_>>(),
                    "test": rep.json(),
                }))?
            );
            tol.check(&rep)?;
            eprintln!("PASS");
        }
        Cmd::Tokenize { model, lines, out } => {
            let m = load_cpu(&model)?;
            let text = std::fs::read_to_string(&lines).with_context(|| format!("read {}", lines.display()))?;
            let mut o = String::new();
            for line in text.lines() {
                let ids = m.encode(&unescape(line))?;
                o.push_str(&ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
                o.push('\n');
            }
            let mut p = out.into_os_string();
            p.push(".tokens.txt");
            std::fs::write(&p, o)?;
        }
    }
    Ok(())
}
