//! infr's real qwen35 pre-tokenizer split (the `tokenizers` Split with `QWEN35_PRE_RE`, Isolated, exactly as
//! infr-llama builds it) checked with the Verus-proved `lucina_rules::split_rules::is_cover` on the tokenizer corpus and
//! on generated strings: every split covers its input exactly once, with no gap and no overlap.
use lucina_rules::split_rules::is_cover;
use std::path::Path;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};

fn unescape(line: &str) -> String {
    line.replace("\\\\", "\u{0}").replace("\\n", "\n").replace("\\t", "\t").replace('\u{0}', "\\")
}

fn pieces(split: &Split, text: &str) -> Vec<(usize, usize)> {
    let mut pts = PreTokenizedString::from(text);
    split.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetReferential::Original, OffsetType::Byte).into_iter().map(|(_, o, _)| o).collect()
}

#[test]
fn qwen35_split_covers_every_input_exactly_once() {
    let split = Split::new(SplitPattern::Regex(infr_llama::QWEN35_PRE_RE.to_string()), SplitDelimiterBehavior::Isolated, false)
        .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let corpus = std::fs::read_to_string(root.join("oracle/prompts/tokenizer-corpus.txt")).unwrap();
    let mut inputs: Vec<String> = corpus.lines().map(unescape).collect();
    // generated strings mixing every class the regex distinguishes
    let alphabet: Vec<char> = "aZé\u{301}ß5٣ ,.;'’\"\t\n\r!?-_/\\(){}<>|中語한ๆऩ😀👍🏽\u{200d}🇩🇰\u{3000}x".chars().collect();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..2000 {
        let mut s = String::new();
        let n = {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % 40
        };
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            s.push(alphabet[(state % alphabet.len() as u64) as usize]);
        }
        inputs.push(s);
    }
    inputs.push(String::new());
    let mut bad = vec![];
    for t in &inputs {
        let p = pieces(&split, t);
        if !is_cover(t.len(), &p) {
            bad.push(format!("{t:?} -> {p:?}"));
        }
    }
    assert!(bad.is_empty(), "{} of {} inputs not covered exactly once:\n{}", bad.len(), inputs.len(), bad.join("\n"));
}
