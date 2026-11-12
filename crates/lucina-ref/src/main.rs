//! lucina-ref-logits --model M.gguf --tokens work/ref/x.tokens.txt --out work/f64/x
//! Writes `<out>.logits.f32` (n_tokens x n_vocab, f64 results rounded to f32 once at the end).
use anyhow::{bail, Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    // `lucina_ref`'s diagnostics are `tracing` now; a subscriber is what makes them visible again.
    // STDERR, where every one of these lines has always been, and `RUST_LOG` tunes them.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let (mut model, mut tokens, mut out) = (None, None, None);
    while let Some(a) = args.next() {
        let v = args
            .next()
            .with_context(|| format!("missing value for {a}"))?;
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(v)),
            "--tokens" => tokens = Some(PathBuf::from(v)),
            "--out" => out = Some(v),
            _ => bail!("unknown argument {a}"),
        }
    }
    let (model, tokens, out) = (
        model.context("--model")?,
        tokens.context("--tokens")?,
        out.context("--out")?,
    );
    let toks: Vec<u32> = std::fs::read_to_string(&tokens)?
        .split_whitespace()
        .map(|t| t.parse().with_context(|| format!("bad token {t}")))
        .collect::<Result<_>>()?;
    let m = lucina_ref::Model::open(&model)?;
    let t0 = std::time::Instant::now();
    let logits = m.forward(&toks)?;
    tracing::info!(
        "lucina-ref: {} tokens in {:.1} s",
        toks.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut bytes = Vec::with_capacity(toks.len() * m.cfg.n_vocab * 4);
    for row in &logits {
        for &v in row {
            bytes.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    std::fs::write(format!("{out}.logits.f32"), bytes)?;
    Ok(())
}
