//! End-to-end logits parity: the engine's CPU reference forward on the `q35-9b-chat` prompt
//! versus the pinned llama.cpp oracle (`work/ref/q35-9b-chat-b512.*`).
//!
//! The pass bar is not an absolute number: it is `k = 2` times the spread llama.cpp shows
//! against itself across configurations it treats as equivalent (`--threads 8` vs `--threads 4`
//! vs `--one-by-one` vs `--kv f32`, all batch 512), per the metrics in [`lucina_parity::Report`].
//! The test also requires argmax agreement on every position whose reference top-2 margin is
//! not a near-tie and a row count equal to the prompt token count. Self-skips when the model
//! GGUF or any of the oracle files is missing.
//!
//! To regenerate the oracle reference:
//!
//! ```sh
//! bash oracle/build.sh
//! work/bin/llama-logits --model "$LI_MODELS_DIR/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf" \
//!   --prompt-file oracle/prompts/chat.txt --out work/ref/q35-9b-chat-b512 --ubatch 512 --threads 8
//! # same but --threads 4                -> --out work/ref/q35-9b-chat-b512t4
//! # same but --one-by-one               -> --out work/ref/q35-9b-chat-seq
//! # same but --kv f32                   -> --out work/ref/q35-9b-chat-b512f32
//! ```
//!
//! (Each invocation also writes the matching `.tokens.txt` next to the `.logits.f32`.)
//! Slow: a CPU forward over the full prompt takes about a minute in release.

use std::path::{Path, PathBuf};
use std::sync::Arc;

const N_VOCAB: usize = 248320;

/// Default 9B model path: under `LI_MODELS_DIR` (see scripts/local.env.example). Unset gives a path
/// that cannot exist, which takes the normal self-skip below.
fn q35_9b_default() -> String {
    let dir = std::env::var("LI_MODELS_DIR").unwrap_or_default();
    format!("{dir}/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf")
}

#[test]
fn qwen35_9b_cpu_logits_match_llama_cpp() {
    let model = std::env::var("LUCINA_PARITY_Q35_9B").unwrap_or_else(|_| q35_9b_default());
    let ref_dir: PathBuf = std::env::var("LUCINA_PARITY_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("work/ref")
        });

    let tokens_path = ref_dir.join("q35-9b-chat-b512.tokens.txt");
    let ref_path = ref_dir.join("q35-9b-chat-b512.logits.f32");
    let self_paths = [
        ref_dir.join("q35-9b-chat-seq.logits.f32"),
        ref_dir.join("q35-9b-chat-b512t4.logits.f32"),
        ref_dir.join("q35-9b-chat-b512f32.logits.f32"),
    ];
    let mut missing = vec![];
    if !Path::new(&model).exists() {
        missing.push(format!("model {}", model));
    }
    if !tokens_path.exists() {
        missing.push(format!("{}", tokens_path.display()));
    }
    if !ref_path.exists() {
        missing.push(format!("{}", ref_path.display()));
    }
    missing.extend(
        self_paths
            .iter()
            .filter(|p| !p.exists())
            .map(|p| p.display().to_string()),
    );
    if !missing.is_empty() {
        eprintln!("skip: missing {}", missing.join(", "));
        return;
    }

    let tokens = lucina_parity::read_tokens(&tokens_path).unwrap();
    let reference = lucina_parity::Logits::read(&ref_path, N_VOCAB).unwrap();
    let spreads = self_paths
        .iter()
        .map(|p| {
            lucina_parity::Report::new(
                &reference,
                &lucina_parity::Logits::read(p, N_VOCAB).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let tol = lucina_parity::Tolerance::from_self_spread(&spreads.iter().collect::<Vec<_>>(), 2.0);

    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = Some("cpu".into());
    let m = infr_llama::SeamModel::load_with(Path::new(&model), None, Arc::new(cfg)).unwrap();
    let (logits, _h) = m.verify_logits_and_h_cpu(&tokens).unwrap();
    let test = lucina_parity::Logits {
        n_vocab: N_VOCAB,
        data: logits,
    };

    let rep = lucina_parity::Report::new(&reference, &test).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&rep.json()).unwrap());
    eprintln!(
        "tolerance: kl_mean {:.3e} kl_max {:.3e} rel_rms_max {:.3e}",
        tol.kl_mean, tol.kl_max, tol.rel_rms_max
    );
    tol.check(&rep).unwrap();

    let (agree, total) = rep.top1_decisive();
    assert_eq!(
        agree, total,
        "argmax flips on decisive positions: {agree}/{total}"
    );
    assert_eq!(rep.rows.len(), tokens.len(), "one row per prompt token");
}

/// The stronger gate (docs/lucina/stage1-parity.md): distances are measured from the f64 reference
/// (`target/release/lucina-ref-logits ... --out work/f64/q35-9b-chat`), and the engine may be at most
/// 2x as far from exact arithmetic as llama.cpp itself is (batch and token-by-token runs).
/// Self-skips when the f64 file or the oracle files are missing.
#[test]
fn qwen35_9b_cpu_logits_within_2x_of_llama_cpp_distance_from_f64() {
    let model = std::env::var("LUCINA_PARITY_Q35_9B").unwrap_or_else(|_| q35_9b_default());
    let work = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../work");
    let f64_path = work.join("f64/q35-9b-chat.logits.f32");
    let tokens_path = work.join("ref/q35-9b-chat-b512.tokens.txt");
    let llama = [
        work.join("ref/q35-9b-chat-b512.logits.f32"),
        work.join("ref/q35-9b-chat-seq.logits.f32"),
    ];
    let missing: Vec<String> = std::iter::once(PathBuf::from(&model))
        .chain([f64_path.clone(), tokens_path.clone()])
        .chain(llama.iter().cloned())
        .filter(|p| !p.exists())
        .map(|p| p.display().to_string())
        .collect();
    if !missing.is_empty() {
        eprintln!("skip: missing {}", missing.join(", "));
        return;
    }
    let exact = lucina_parity::Logits::read(&f64_path, N_VOCAB).unwrap();
    let spreads: Vec<_> = llama
        .iter()
        .map(|p| {
            lucina_parity::Report::new(&exact, &lucina_parity::Logits::read(p, N_VOCAB).unwrap())
                .unwrap()
        })
        .collect();
    let tol = lucina_parity::Tolerance::from_self_spread(&spreads.iter().collect::<Vec<_>>(), 2.0);
    let tokens = lucina_parity::read_tokens(&tokens_path).unwrap();
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = Some("cpu".into());
    let m = infr_llama::SeamModel::load_with(Path::new(&model), None, Arc::new(cfg)).unwrap();
    let (logits, _h) = m.verify_logits_and_h_cpu(&tokens).unwrap();
    let rep = lucina_parity::Report::new(
        &exact,
        &lucina_parity::Logits {
            n_vocab: N_VOCAB,
            data: logits,
        },
    )
    .unwrap();
    eprintln!("engine vs f64: {}", rep.json());
    tol.check(&rep).unwrap();
}
