//! Tokenizer parity with llama.cpp for the `qwen35` vocabulary (Qwen3.5 / 3.8).
//!
//! The golden file is llama.cpp 3612199ed's own tokenization of `oracle/prompts/tokenizer-corpus.txt`
//! (`work/bin/llama-logits --model <27B> --tokenize-lines oracle/prompts/tokenizer-corpus.txt --out x`,
//! special tokens parsed). The Qwen3.5-9B and Qwen3.8-27B GGUFs share the vocabulary and produced the
//! same file. Self-skips when no model file is present.
use std::path::Path;

fn unescape(line: &str) -> String {
    line.replace("\\\\", "\u{0}").replace("\\n", "\n").replace("\\t", "\t").replace('\u{0}', "\\")
}

#[test]
fn qwen35_tokenizer_matches_llama_cpp_on_the_corpus() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        std::env::var("LUCINA_PARITY_QWEN35_MODEL").unwrap_or_default(),
        "~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf".into(),
        "~/models/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf".into(),
    ];
    let Some(model) = candidates.iter().find(|p| !p.is_empty() && Path::new(p).exists()) else {
        eprintln!("skip: no qwen35 GGUF found");
        return;
    };
    let corpus = std::fs::read_to_string(root.join("oracle/prompts/tokenizer-corpus.txt")).unwrap();
    let golden = std::fs::read_to_string(root.join("oracle/golden/qwen35-tokenizer-corpus.tokens.txt")).unwrap();
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = Some("cpu".into());
    let m = infr_llama::SeamModel::load_with(Path::new(model), None, std::sync::Arc::new(cfg)).unwrap();
    let mut bad = vec![];
    let (corpus_lines, golden_lines): (Vec<_>, Vec<_>) = (corpus.lines().collect(), golden.lines().collect());
    assert_eq!(corpus_lines.len(), golden_lines.len(), "golden file does not match the corpus");
    for (i, (line, want)) in corpus_lines.iter().zip(&golden_lines).enumerate() {
        let got = m.encode(&unescape(line)).unwrap();
        let want: Vec<u32> = want.split_whitespace().map(|t| t.parse().unwrap()).collect();
        if got != want {
            bad.push(format!("line {}: {line:?}\n  llama.cpp {want:?}\n  engine    {got:?}", i + 1));
        }
    }
    assert!(bad.is_empty(), "{} of {} lines differ from llama.cpp:\n{}", bad.len(), corpus_lines.len(), bad.join("\n"));
}
