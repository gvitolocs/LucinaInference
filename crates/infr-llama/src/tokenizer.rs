//! GGUF-embedded tokenizer construction (byte-level BPE + SentencePiece).
//! Mechanically split out of `lib.rs` (no logic change).
use crate::{
    DEEPSEEK_CODER_PRE_RES, DEEPSEEK_LLM_PRE_RES, DEEPSEEK_V3_PRE_RES, LLAMA4_PRE_RE, QWEN2_PRE_RE,
    QWEN35_PRE_RE,
};
use anyhow::{anyhow, bail, Context, Result};
use infr_core::loader::{MetaValue, Metadata};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, DecoderWrapper, SplitDelimiterBehavior, Tokenizer};

/// The `tokenizer.ggml.merges` line grammar: each entry is `"left right"`, ONE space, and the
/// right half may itself contain spaces (SPM pieces are `▁`-prefixed, but a byte-level BPE piece
/// can be a literal space) — hence `splitn(2, ' ')`, not `split`. Entries that are not strings, or
/// that have no space at all, are dropped rather than erroring: a merge we cannot read is one the
/// BPE simply never applies, which degrades tokenization instead of refusing to load the model.
///
/// Single-sources the grammar for both tokenizer families. `build_tokenizer` (gpt2 BPE) and
/// `build_spm_tokenizer` (gemma4's explicit merges) each spelled this expression out; a fix to one
/// copy — the `splitn` bound in particular — silently did not reach the other, and the two families
/// would then disagree about what a merge line means while both claiming to read the same key.
fn parse_merges(arr: &[MetaValue]) -> Vec<(String, String)> {
    arr.iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut it = s.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect()
}

/// Register `tokenizer.ggml.token_type` 3 (CONTROL) and 4 (USER_DEFINED) tokens on `tok`, matching
/// HF: control tokens are SPECIAL (`AddedToken::from(s, true)`), user-defined ones are normal added
/// tokens. Both encode atomically; only the special ones are dropped by
/// `decode(.., skip_special=true)`, which is why `<think>`/`</think>` (type 4) must NOT be special —
/// the reasoning block has to stay visible in the output for the reasoning split to find it.
///
/// Absent `token_type` ⇒ nothing to register. The empty-vec guards are kept because `add_tokens` /
/// `add_special_tokens` on an empty slice still bump the tokenizer's added-vocab bookkeeping, and
/// the ORDER (added before specials) is load-bearing: a string appearing in both lists must end up
/// special, and the later call wins.
///
/// This is the single copy of a policy that used to be written out verbatim in both
/// [`build_tokenizer`] and [`build_spm_tokenizer`]. The two copies were already drifting in their
/// comments about which type meant what, and any new token_type the GGUF spec grows would have had
/// to be added twice — with the SPM family (which loads llama/gemma) the easy one to miss.
fn register_token_types(md: &Metadata, toks: &[MetaValue], tok: &mut Tokenizer) {
    let Some(types) = md
        .get("tokenizer.ggml.token_type")
        .and_then(MetaValue::as_arr)
    else {
        return;
    };
    let mut specials = Vec::new();
    let mut added = Vec::new();
    for (i, ty) in types.iter().enumerate() {
        let Some(s) = toks.get(i).and_then(MetaValue::as_str) else {
            continue;
        };
        match ty.as_u64() {
            Some(3) => specials.push(AddedToken::from(s.to_string(), true)),
            Some(4) => added.push(AddedToken::from(s.to_string(), false)),
            _ => {}
        }
    }
    if !added.is_empty() {
        tok.add_tokens(&added);
    }
    if !specials.is_empty() {
        tok.add_special_tokens(&specials);
    }
}

/// Build an HF `Tokenizer` from the GGUF's embedded vocab (`tokenizer.ggml.*`). Supports the
/// GPT-2 byte-level BPE family (Qwen/Llama-3/SmolLM etc., `tokenizer.ggml.model == "gpt2"`):
/// vocab from `.tokens`, merges from `.merges`, ByteLevel pre-tokenizer + decoder, and control /
/// user-defined tokens (token_type 3/4, e.g. `<|im_start|>`) registered as special so they encode
/// atomically. SentencePiece (`model == "llama"`) isn't built here — pass a `tokenizer.json`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    let md = g.metadata();
    let model = md.str("tokenizer.ggml.model").unwrap_or("");
    match model {
        "gpt2" => {}
        // SentencePiece (llama/gemma3/gemma4): a byte-fallback BPE with a Metaspace (▁) word-boundary
        // scheme. gemma4 ships explicit merges; llama/gemma3 reconstruct them from the token scores.
        "llama" | "gemma4" => return build_spm_tokenizer(g),
        other => bail!(
            "can't derive a tokenizer from tokenizer.ggml.model={other:?} \
             (only gpt2 BPE / llama SPM); pass a tokenizer.json sidecar instead"
        ),
    }
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.tokens")?;
    let vocab: std::collections::HashMap<String, u32> = toks
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.as_str().map(|s| (s.to_string(), i as u32)))
        .collect();
    let merges = parse_merges(
        md.get("tokenizer.ggml.merges")
            .and_then(MetaValue::as_arr)
            .context("gguf missing tokenizer.ggml.merges")?,
    );
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("build bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    let add_prefix = matches!(
        md.get("tokenizer.ggml.add_space_prefix"),
        Some(MetaValue::Bool(true))
    );
    let pre = md.str("tokenizer.ggml.pre").unwrap_or("default");
    // qwen2/llama4 both split on a SINGLE regex before ByteLevel (matching HF); only the pattern
    // differs. DeepSeek pre-types apply a LIST of successive Split pre-tokenizers (see §0.2 of
    // docs/deepseek.md, and [`build_multi_split_seq`] for the token-id evidence that N successive
    // Isolated splits reproduce llama.cpp's sequential application).
    // The sequences below follow llama.cpp's `src/llama-vocab.cpp` verbatim.
    //
    // llama.cpp additionally sets `clean_spaces = false` for all three DeepSeek pre-types. Nothing
    // is needed here to match that: `clean_spaces` is read in exactly one place, the DETOKENIZER
    // (`llama_vocab::impl::detokenize`), where it drops the space before `?!.,`, strips a lone
    // apostrophe between spaces, and closes up `'s`/`'m`/`'re`/`'ve`. It never runs during
    // encoding. llama.cpp turns it ON for every BPE vocab and the DeepSeek arms turn it back off;
    // infr detokenizes through `tokenizers::Tokenizer::decode`, which has no such pass at all, so
    // infr is unconditionally in the `clean_spaces = false` state the DeepSeek arms ask for.
    match pre {
        "deepseek-llm" => {
            let seq = build_multi_split_seq(DEEPSEEK_LLM_PRE_RES.as_slice())?;
            tok.with_pre_tokenizer(Some(seq));
        }
        "deepseek-coder" => {
            let seq = build_multi_split_seq(DEEPSEEK_CODER_PRE_RES.as_slice())?;
            tok.with_pre_tokenizer(Some(seq));
        }
        "deepseek-v3" => {
            let seq = build_multi_split_seq(DEEPSEEK_V3_PRE_RES.as_slice())?;
            tok.with_pre_tokenizer(Some(seq));
        }
        "qwen2" | "qwen35" | "llama4" => {
            let re = match pre {
                "qwen2" => QWEN2_PRE_RE,
                "qwen35" => QWEN35_PRE_RE,
                _ => LLAMA4_PRE_RE,
            };
            // Sequence[ Split(pre regex, Isolated), ByteLevel(use_regex=false) ] — matches HF.
            let split = Split::new(
                SplitPattern::Regex(re.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| anyhow!("split pretokenizer: {e}"))?;
            let seq = PreSequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
            ]);
            tok.with_pre_tokenizer(Some(seq));
        }
        _ => {
            tok.with_pre_tokenizer(Some(ByteLevel::new(add_prefix, true, true)));
        }
    }
    tok.with_decoder(Some(ByteLevelDecoder::default()));
    // Control (type 3, e.g. <|im_end|>) as SPECIAL, user-defined (type 4, e.g. <think>) as normal
    // added tokens — see [`register_token_types`].
    register_token_types(md, toks, &mut tok);
    Ok(tok)
}

/// The global ordering of reconstructed SPM merge candidates, keyed by `(score, id_l, id_r)`:
/// score DESCENDING (a higher GGUF score is an EARLIER merge), ties broken by `(id_l, id_r)`
/// ASCENDING. That exact ordering is load-bearing — it is what reproduces HF `SpmConverter`'s
/// merge ranking, and flipping either half silently re-tokenizes every prompt on a llama/gemma3
/// GGUF (no merges array, so the ranks come from here). Factored out of
/// [`build_spm_tokenizer`] only so a test can pin it without a model file.
///
/// Scores compare with [`f64::total_cmp`], NOT `partial_cmp().unwrap_or(Equal)`. A NaN score — a
/// corrupt or hand-edited `tokenizer.ggml.scores` — makes the `unwrap_or(Equal)` comparator
/// non-transitive (NaN is "equal" to everything, but the things it ties with are not equal to each
/// other), and Rust's sort DETECTS that and panics with "user-provided comparison function does not
/// correctly implement a total order". So a bad score used to abort model load inside the sort with
/// an error about our comparator instead of about the file. `total_cmp` is a genuine total order
/// over every f64 including NaN, so the sort completes and the corrupt entry merely ranks
/// consistently (NaN sorts to one end) rather than taking the process down.
fn spm_merge_order(a: (f64, u32, u32), b: (f64, u32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2)))
}

/// Build a `PreSequence` for a list of successive `Split` pre-tokenizers (DeepSeek's multi-regex
/// approach — see docs/deepseek.md §0.2). Each regex splits the output of the previous one in
/// order, followed by `ByteLevel`; mirrors llama.cpp's `unicode_regex_split` pipeline.
///
/// That N successive `Isolated` splits reproduce llama.cpp's sequential `unicode_regex_split` is
/// no longer an assumption: it was checked against `llama-tokenize` token ids on the V2-Lite-Chat
/// Q4_K_M GGUF (`tokenizer.ggml.pre == "deepseek-llm"`, so the six-regex list), agreeing exactly on
/// every text of a battery covering digits, CJK, Hangul, Greek, Hebrew, Arabic, Devanagari,
/// punctuation runs, mixed scripts, emoji, CRLF and non-ASCII whitespace. The comparison has teeth:
/// re-breaking `DEEPSEEK_LLM_PRE_RES[2]` made it disagree with llama.cpp. `deepseek-coder` and
/// `deepseek-v3` are structurally identical but were NOT checked against ids — no GGUF with either
/// pre-type was available. [`build_tokenizer`]'s tests pin all three at the boundary level.
fn build_multi_split_seq(patterns: &[&str]) -> Result<PreSequence> {
    let mut parts: Vec<PreTokenizerWrapper> = patterns
        .iter()
        .map(|re| {
            Split::new(
                SplitPattern::Regex(re.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map(PreTokenizerWrapper::Split)
            .map_err(|e| anyhow!("split pretokenizer: {e}"))
        })
        .collect::<Result<_, _>>()?;
    parts.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
        false, false, false,
    )));
    Ok(PreSequence::new(parts))
}

/// Build a SentencePiece (Unigram) tokenizer from a GGUF's embedded vocab (`tokenizer.ggml.model
/// == "llama"`, used by llama/gemma). The token strings + `scores` become the Unigram lattice;
/// `<0xXX>` byte tokens (token_type 6) are handled by Unigram byte-fallback; CONTROL tokens
/// (type 3, e.g. `<bos>`/`<start_of_turn>`) register as special so they encode atomically. The
/// Metaspace replacement (▁) maps spaces; `add_space_prefix` controls the leading dummy space.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_spm_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    let md = g.metadata();
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.tokens")?;
    let token_strs: Vec<String> = toks
        .iter()
        .map(|t| t.as_str().unwrap_or("").to_string())
        .collect();
    let vocab: std::collections::HashMap<String, u32> = token_strs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();
    // Merge list for the greedy BPE. gemma4 ships explicit `merges` ("left right", ▁ for spaces);
    // llama/gemma3 don't, so reconstruct them from the token scores the same way HF's SpmConverter
    // builds `LlamaTokenizerFast` (the GGUF scores are negative merge RANKS, not unigram log-probs —
    // a Unigram model would maximize their sum and wrongly split common words). For every piece, each
    // split into two existing pieces is a candidate merge, globally ordered by the merged piece's
    // score (descending = earliest), ties broken by piece ids; greedy BPE over these reproduces SPM.
    let merges: Vec<(String, String)> =
        if let Some(arr) = md.get("tokenizer.ggml.merges").and_then(MetaValue::as_arr) {
            parse_merges(arr)
        } else {
            let scores = md
                .get("tokenizer.ggml.scores")
                .and_then(MetaValue::as_arr)
                .context("gguf needs tokenizer.ggml.merges or .scores for the SPM tokenizer")?;
            // (score, id_l, id_r, l, r) per candidate — global sort by score desc, then (id_l, id_r).
            let mut cand: Vec<(f64, u32, u32, &str, &str)> = Vec::new();
            for (i, piece) in token_strs.iter().enumerate() {
                if piece.len() < 2 {
                    continue;
                }
                let score = scores.get(i).and_then(MetaValue::as_f64).unwrap_or(0.0);
                for (b, _) in piece.char_indices().skip(1) {
                    let (l, r) = piece.split_at(b);
                    if let (Some(&il), Some(&ir)) = (vocab.get(l), vocab.get(r)) {
                        cand.push((score, il, ir, l, r));
                    }
                }
            }
            cand.sort_by(|a, b| spm_merge_order((a.0, a.1, a.2), (b.0, b.1, b.2)));
            cand.into_iter()
                .map(|(_, _, _, l, r)| (l.to_string(), r.to_string()))
                .collect()
        };
    let unk = md
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(MetaValue::as_u64)
        .and_then(|i| token_strs.get(i as usize).cloned())
        .unwrap_or_else(|| "<unk>".to_string());
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token(unk)
        .byte_fallback(true)
        .fuse_unk(true)
        .build()
        .map_err(|e| anyhow!("build spm bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    // SPM: spaces → ▁. add_space_prefix=true prepends a dummy ▁ (PrependScheme::First); gemma3
    // sets it false. `split` keeps Metaspace's word splitting on the replacement char.
    let add_prefix = matches!(
        md.get("tokenizer.ggml.add_space_prefix"),
        Some(MetaValue::Bool(true))
    );
    let scheme = if add_prefix {
        PrependScheme::First
    } else {
        PrependScheme::Never
    };
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', scheme, true)));
    // Decode: reassemble byte-fallback bytes, fuse, then map ▁→space (Metaspace decoder).
    let dec = DecoderSequence::new(vec![
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Fuse(Fuse::default()),
        DecoderWrapper::Metaspace(Metaspace::new('▁', scheme, true)),
    ]);
    tok.with_decoder(Some(dec));
    // CONTROL tokens (type 3, e.g. <bos>/<start_of_turn>/<end_of_turn>) encode atomically as
    // special; USER_DEFINED (type 4) as normal added tokens — see [`register_token_types`].
    register_token_types(md, toks, &mut tok);
    Ok(tok)
}

/// Tokenize `text` the way llama.cpp's `/completion` tokenizes a prompt — the tokenization a parity
/// test against `llama-server` / `llama-tokenize` has to reproduce, and deliberately not the chat
/// path: no template, no `think` tags.
///
/// The reference is llama.cpp `3612199ed`: `/completion` calls
/// `tokenize_input_prompts(vocab, mctx, prompt, /*add_special*/ true, /*parse_special*/ true, ..)`
/// (`tools/server/server-context.cpp:4327`; the CLI path `:2239` is identical), which reaches
/// `common_tokenize(vocab, s, add_special, parse_special)` at `tools/server/server-common.cpp:880`.
/// The `llama-tokenize` tool defaults `parse_special` to true "like the old tokenize tool"
/// (`common/arg.cpp:1407`), so it is the same reference for a bare file.
///
/// * **BOS iff the GGUF says so.** `add_special` reaches a bare prompt as BOS and nothing else:
///   `src/llama-vocab.cpp:2616` loads `add_bos` from `tokenizer.ggml.add_bos_token` (key name
///   `src/llama-arch.cpp:391`; a GGUF without the key leaves it `false` — `:1824`), and every vocab
///   family's `tokenize` prepends `special_bos_id` on `add_special && add_bos` — SPM
///   `src/llama-vocab.cpp:3412`, UGM `:3523`, and the BPE arm's `append_bos` `:3470`, which applies
///   the same key (`:579`). So: prepend the id from `tokenizer.ggml.bos_token_id`, exactly once.
///   llama.cpp would also append EOS / SEP for `add_special` when `tokenizer.ggml.add_eos_token` /
///   `add_sep_token` are set (`:3449`, `:3547`); the GGUFs these runs target set neither, and
///   appending EOS would end the prompt before a token is generated — this implements the BOS half,
///   which is what the key names.
/// * **Specials ARE parsed out of the text.** `parse_special = true` makes `tokenizer_st_partition`
///   (`src/llama-vocab.cpp:3225`) cut every literal occurrence of a CONTROL or USER_DEFINED vocab
///   entry out of the prompt and emit its id, so `<|im_start|>` written in a prompt is the
///   `<|im_start|>` token, not the characters it is spelled with. The shared tokenizer already does
///   exactly that: [`register_token_types`] adds type 3 as special and type 4 as added tokens, and
///   `encode` with the default `encode_special_tokens = false` matches both atomically. No second
///   pass is needed.
///
/// The second argument of `encode` is NOT the `add_special` flag: in `tokenizers` it only reaches the
/// post-processor, and a GGUF-derived tokenizer has none (`build_tokenizer` never sets one), so
/// BOS is added here from the GGUF keys instead.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn tokenize_raw(g: &Gguf, tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let md = g.metadata();
    let encoded = tok
        .encode(text, false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let mut ids: Vec<u32> = encoded.get_ids().to_vec();
    if matches!(
        md.get("tokenizer.ggml.add_bos_token"),
        Some(MetaValue::Bool(true))
    ) {
        let bos = md.u64("tokenizer.ggml.bos_token_id").ok_or_else(|| {
            anyhow!("tokenizer.ggml.add_bos_token is true and the GGUF has no tokenizer.ggml.bos_token_id")
        })?;
        let bos =
            u32::try_from(bos).map_err(|_| anyhow!("invalid tokenizer.ggml.bos_token_id {bos}"))?;
        ids.insert(0, bos);
    }
    Ok(ids)
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    use crate::sampling::{sample_logits, Sampler};

    /// The chunk boundaries a DeepSeek pre-tokenizer sequence yields, as slices of the ORIGINAL
    /// text. [`build_multi_split_seq`] ends the sequence with `ByteLevel`, which rewrites each
    /// chunk's CONTENT (a space becomes `\u{0120}`, every non-ASCII byte a printable stand-in),
    /// so this reads the original-referential offsets instead of the chunk strings. Those offsets
    /// are the boundaries the BPE then merges within, and they are exactly what a wrong character
    /// class moves — which is the thing under test.
    fn pre_chunks(patterns: &[&str], text: &str) -> Vec<String> {
        use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
        let seq = build_multi_split_seq(patterns).expect("pattern list must compile");
        let mut pre = PreTokenizedString::from(text);
        seq.pre_tokenize(&mut pre).expect("pre-tokenize");
        pre.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, (start, end), _)| text[start..end].to_string())
            .collect()
    }

    /// The three DeepSeek pre-tokenizer regex lists, pinned by the chunk boundaries they produce.
    ///
    /// These constants are transcribed from llama.cpp's `src/llama-vocab.cpp`, and a wrong
    /// character inside a class does not fail to compile, does not error and does not crash — it
    /// silently moves a chunk boundary, which silently changes token ids, which silently changes
    /// the model's output. Two such slips shipped (`docs/backlog.md` B43), and the `deepseek-llm`
    /// table below has rows that go red under each of them:
    ///
    /// * The Hebrew rows catch `DEEPSEEK_LLM_PRE_RES[2]` opening its quote range with `\u{0027}`
    ///   instead of `\u{2018}`. That runs the class from U+0027 to U+201F — most of the BMP,
    ///   including Hebrew, Arabic, Devanagari and the ASCII digits — so split #3 swallowed the
    ///   Hebrew word together with the full stop after it. Nothing after split #3 matches Hebrew,
    ///   so no later split puts that boundary back.
    /// * The `\u{2139}\u{213A}\u{213F}` row catches `DEEPSEEK_LLM_PRE_RES[1]` having
    ///   `\u{2139}-\u{213F}` where the reference has `\u{2139}\u{213C}-\u{213F}`.
    ///   The two extra letters made split #2 run all three characters together as one word.
    ///
    /// The `"a \u{00A0}. b"` row is the one that was shown to move real TOKEN IDS, not just
    /// boundaries: with the broken class, `\u{00A0}` (NO-BREAK SPACE, and NBSP before punctuation
    /// is ordinary French typography and ordinary scraped HTML) is pulled into split #3 together
    /// with the space before it and the full stop after it. On the V2-Lite vocab that turns
    /// llama.cpp's `[64, 207, 1202, 13, 270]` into `[64, 30683, 13, 270]`. The Hebrew and
    /// letterlike rows above move boundaries but happen NOT to move ids on that vocab, because it
    /// has no merge spanning the boundary they lose — so they guard the pre-tokenizer, while this
    /// row also guards the observable end of it.
    ///
    /// `deepseek-coder` and `deepseek-v3` were byte-identical to the reference when this was
    /// written (verified codepoint by codepoint against `llama-vocab.cpp`); their tables are here
    /// so the next slip lands on a red test rather than on a user. The differences between the
    /// three tables are real behaviour, not noise: `deepseek-coder` splits digits one at a time
    /// (`\p{N}`, no `+`) and matches Hebrew through `\p{L}`, while `deepseek-v3` keeps
    /// `(x` together because its final pattern binds an opening delimiter to the letters after it.
    #[test]
    fn deepseek_pre_split_boundaries_match_the_reference_lists() {
        fn check(name: &str, patterns: &[&str], cases: &[(&str, &[&str])]) {
            for (text, want) in cases {
                let got = pre_chunks(patterns, text);
                assert_eq!(
                    got, *want,
                    "{name} chunk boundaries moved for {text:?}\n  got  {got:?}\n  want {want:?}"
                );
            }
        }

        check(
            "deepseek-llm",
            DEEPSEEK_LLM_PRE_RES.as_slice(),
            &[
                ("a \u{00A0}. b", &["a", " ", "\u{00A0}.", " b"]),
                ("שלום.", &["שלום", "."]),
                ("עברית 42 test.", &["עברית ", "42", " test", "."]),
                ("ℹ℺ℿ", &["ℹ", "℺", "ℿ"]),
                ("3.14", &["3", ".", "14"]),
                (
                    "50% off, 2 items?",
                    &["50", "%", " off", ",", " ", "2", " items", "?"],
                ),
                ("中文 汉字。", &["中文", " ", "汉字", "。"]),
                (
                    "def f(x): return x*2\n",
                    &["def", " f", "(", "x", "):", " return", " x", "*", "2", "\n"],
                ),
            ],
        );
        check(
            "deepseek-coder",
            DEEPSEEK_CODER_PRE_RES.as_slice(),
            &[
                ("a \u{00A0}. b", &["a", " ", "\u{00A0}.", " b"]),
                ("שלום.", &["שלום", "."]),
                ("עברית 42 test.", &["עברית", " ", "4", "2", " test", "."]),
                ("ℹ℺ℿ", &["ℹ", "℺", "ℿ"]),
                ("3.14", &["3", ".", "1", "4"]),
                (
                    "50% off, 2 items?",
                    &["5", "0", "%", " off", ",", " ", "2", " items", "?"],
                ),
                ("中文 汉字。", &["中文", " ", "汉字", "。"]),
                (
                    "def f(x): return x*2\n",
                    &["def", " f", "(", "x", "):", " return", " x", "*", "2", "\n"],
                ),
            ],
        );
        check(
            "deepseek-v3",
            DEEPSEEK_V3_PRE_RES.as_slice(),
            &[
                ("a \u{00A0}. b", &["a", " ", "\u{00A0}", ".", " b"]),
                ("שלום.", &["שלום", "."]),
                ("עברית 42 test.", &["עברית", " ", "42", " test", "."]),
                ("ℹ℺ℿ", &["ℹ", "℺", "ℿ"]),
                ("3.14", &["3", ".", "14"]),
                (
                    "50% off, 2 items?",
                    &["50", "%", " off", ",", " ", "2", " items", "?"],
                ),
                ("中文 汉字。", &["中文", " ", "汉字", "。"]),
                (
                    "def f(x): return x*2\n",
                    &["def", " f", "(x", "):", " return", " x", "*", "2", "\n"],
                ),
            ],
        );
    }

    /// The SPM merge ranking, pinned in both halves and made NaN-proof.
    ///
    /// Two things are asserted, and they pull in opposite directions:
    ///
    /// 1. The ORDER must not move. Score descending, `(id_l, id_r)` ascending on ties — that is
    ///    HF `SpmConverter`'s ranking, and every llama/gemma3 GGUF (no `tokenizer.ggml.merges`)
    ///    tokenizes through it. A flip here is a silent re-tokenization of every prompt.
    /// 2. A NaN score must not panic the sort. `partial_cmp().unwrap_or(Equal)` is non-transitive
    ///    in the presence of NaN, and `sort_by` detects that and panics with "user-provided
    ///    comparison function does not correctly implement a total order" — so one corrupt entry
    ///    in `tokenizer.ggml.scores` used to abort model load. The full sort below is the actual
    ///    regression test: it is exactly the call site's `sort_by`, and it panics under the old
    ///    comparator.
    #[test]
    fn spm_merge_order_is_a_total_order_and_keeps_its_direction() {
        use std::cmp::Ordering::{Greater, Less};
        // Higher score sorts FIRST (earlier merge), regardless of ids.
        assert_eq!(spm_merge_order((-1.0, 9, 9), (-2.0, 0, 0)), Less);
        assert_eq!(spm_merge_order((-2.0, 0, 0), (-1.0, 9, 9)), Greater);
        // Equal scores: lower (id_l, id_r) sorts first, id_l dominating id_r.
        assert_eq!(spm_merge_order((-1.0, 3, 7), (-1.0, 4, 0)), Less);
        assert_eq!(spm_merge_order((-1.0, 3, 7), (-1.0, 3, 8)), Less);
        assert_eq!(
            spm_merge_order((-1.0, 3, 7), (-1.0, 3, 7)),
            std::cmp::Ordering::Equal
        );

        // A candidate list with NaN scores sprinkled through it. Two properties of this data are
        // deliberate and BOTH are needed to reproduce the panic:
        //
        //   * scores are not monotone in `(id_l, id_r)` — a real candidate list isn't either, since
        //     a candidate carries the MERGED piece's score alongside the two SPLIT pieces' ids. On
        //     monotone data the old comparator's id tiebreak happens to agree with the scores and
        //     no cycle forms, so the sort completes and the bug hides.
        //   * enough elements that the sort actually runs its merge (n >= ~32); the insertion sort
        //     used for tiny slices does not detect the violation.
        //
        // Under `partial_cmp().unwrap_or(Equal)` a NaN ties with every finite score, the id
        // tiebreak then decides those pairs, and `NaN > y`, `y > z`, `z > NaN` becomes reachable —
        // a cycle. `sort_by` detects it and panics. Verified: this exact shape panics with the old
        // comparator at every n >= 32.
        let n = 128u32;
        let mut cand: Vec<(f64, u32, u32)> = (0..n)
            .map(|i| {
                let score = if i % 16 == 7 {
                    f64::NAN
                } else {
                    -(((i as f64) * 7.0) % 31.0)
                };
                (score, i, n - i)
            })
            .collect();
        cand.sort_by(|a, b| spm_merge_order(*a, *b));
        // Nothing dropped, and the finite entries keep their descending-score order among
        // themselves (NaN sorts to one end and does not interleave).
        assert_eq!(cand.len(), n as usize, "no candidate may be dropped");
        let finite: Vec<f64> = cand.iter().map(|c| c.0).filter(|s| !s.is_nan()).collect();
        assert!(
            finite.windows(2).all(|w| w[0] >= w[1]),
            "finite scores must stay descending: {finite:?}"
        );
    }

    // Validate the GGUF-derived tokenizer against the HF tokenizer.json sidecar (same model).
    // Skips if the test model isn't present.
    #[test]
    fn embedded_tokenizer_matches_sidecar() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        // The sidecar tokenizer.json must sit beside the GGUF (HF cache blobs are content-addressed
        // with no sidecar, so this runs only where a snapshot ships tokenizer.json).
        let side = gguf.with_file_name("tokenizer.json");
        if !side.exists() {
            eprintln!("skip: no tokenizer.json sidecar beside the GGUF");
            return;
        }
        let g = Gguf::open(&gguf).unwrap();
        let derived = build_tokenizer(&g).unwrap();
        let sidecar = Tokenizer::from_file(&side).unwrap();
        for s in [
            "Hello world",
            "The quick brown fox.",
            "<|im_start|>user\nWhat is two plus two?<|im_end|>\n<|im_start|>assistant\n",
            "café déjà vu — 123 + 456 = 579",
            "def f(x):\n    return x * 2\n",
        ] {
            let a = derived.encode(s, false).unwrap();
            let b = sidecar.encode(s, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "token id mismatch on {s:?}");
        }
        // <think>/</think> are user-defined (non-special): skip_special must KEEP them, while real
        // special tokens (<|im_end|>) are dropped — matching the sidecar.
        let think = "<think>\nreasoning\n</think>\n\nanswer<|im_end|>";
        let ids = derived.encode(think, false).unwrap();
        let d = derived.decode(ids.get_ids(), true).unwrap();
        assert!(
            d.contains("<think>") && d.contains("</think>"),
            "think tags dropped: {d:?}"
        );
        assert!(!d.contains("<|im_end|>"), "special token kept: {d:?}");
        assert_eq!(
            d,
            sidecar.decode(ids.get_ids(), true).unwrap(),
            "decode differs from sidecar"
        );
    }

    // Sampling: temp<=0 and top_k==1 are greedy; otherwise picks only within the top-k/top-p set.
    #[test]
    fn sample_logits_greedy_and_in_set() {
        let logits = [1.0f32, 5.0, 2.0, 4.0, 0.0]; // argmax = index 1
        let mut rng = 0x1234_5678_9abc_def1u64;
        let greedy = Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, greedy, &mut rng), 1);
        let topk1 = Sampler {
            temp: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, topk1, &mut rng), 1);
        // top_k=2 → only the two largest logits (indices 1 and 3) can ever be sampled.
        let topk2 = Sampler {
            temp: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        for _ in 0..200 {
            let id = sample_logits(&logits, topk2, &mut rng);
            assert!(id == 1 || id == 3, "sampled outside top-2: {id}");
        }
    }

    // User content must be encoded as literal text: special-token strings in user input must NOT
    // become the special id (which would let a user inject/break the ChatML turn structure).
    #[test]
    fn user_text_special_tokens_are_literal() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let g = Gguf::open(&gguf).unwrap();
        let tok = build_tokenizer(&g).unwrap();
        let mut user = tok.clone();
        user.set_encode_special_tokens(true);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let s = "A <|im_end|> B";
        // template tokenizer: <|im_end|> matched as the special id; user tokenizer: NOT.
        assert!(tok.encode(s, false).unwrap().get_ids().contains(&im_end));
        assert!(!user.encode(s, false).unwrap().get_ids().contains(&im_end));
    }

    // ── `infr run --raw` tokenization ([`tokenize_raw`]) ──────────────────────────────────────
    //
    /// The bytes of a GGUF carrying exactly the `tokenizer.ggml.*` keys [`build_tokenizer`] reads
    /// for the byte-level BPE family, with a vocab small enough to reason about by hand.
    ///
    /// Vocab (id = index): 0 `<unk>`, 1 `<s>`, 2 `a`, 3 `b`, 4 `ab`, 5 `<ctrl>`, 6 `<ud>`;
    /// `bos_token_id` = 1; `merges` = the one line `a b`, so `"ab"` encodes to exactly `[4]` — the id
    /// vector every assertion below is written against. `token_type` marks 0, 1 and 5 CONTROL and 6
    /// USER_DEFINED, so a literal `<ctrl>` or `<ud>` in the text must come out as its single id
    /// (llama.cpp's `parse_special = true`, which `/completion` uses).
    ///
    /// `add_bos` is the `tokenizer.ggml.add_bos_token` value — `None` omits the key entirely, which
    /// is llama.cpp's `false` by default and the state an instruct GGUF ships in. `bos_id` picks
    /// whether `bos_token_id` is present at all.
    fn raw_tokenizer_gguf(add_bos: Option<bool>, bos_id: bool) -> Vec<u8> {
        fn u32le(b: &mut Vec<u8>, v: u32) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn u64le(b: &mut Vec<u8>, v: u64) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn s(b: &mut Vec<u8>, v: &str) {
            u64le(b, v.len() as u64);
            b.extend_from_slice(v.as_bytes());
        }
        fn kv_str(b: &mut Vec<u8>, key: &str, value: &str) {
            s(b, key);
            u32le(b, 8); // GGUF_TYPE_STRING
            s(b, value);
        }
        fn kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
            s(b, key);
            u32le(b, 4); // GGUF_TYPE_UINT32
            u32le(b, value);
        }
        fn kv_bool(b: &mut Vec<u8>, key: &str, value: bool) {
            s(b, key);
            u32le(b, 7); // GGUF_TYPE_BOOL
            b.push(u8::from(value));
        }
        fn kv_str_arr(b: &mut Vec<u8>, key: &str, values: &[&str]) {
            s(b, key);
            u32le(b, 9); // GGUF_TYPE_ARRAY
            u32le(b, 8); // element type: STRING
            u64le(b, values.len() as u64);
            for v in values {
                s(b, v);
            }
        }
        fn kv_u32_arr(b: &mut Vec<u8>, key: &str, values: &[u32]) {
            s(b, key);
            u32le(b, 9); // GGUF_TYPE_ARRAY
            u32le(b, 4); // element type: UINT32
            u64le(b, values.len() as u64);
            for v in values {
                u32le(b, *v);
            }
        }

        // One entry per KV below. The count must match exactly or the tensor-table parse starts in
        // the middle of the metadata.
        let mut kv: Vec<Vec<u8>> = Vec::new();
        let mut b = Vec::new();
        kv_str(&mut b, "tokenizer.ggml.model", "gpt2");
        kv.push(b);
        let mut b = Vec::new();
        kv_str_arr(
            &mut b,
            "tokenizer.ggml.tokens",
            &["<unk>", "<s>", "a", "b", "ab", "<ctrl>", "<ud>"],
        );
        kv.push(b);
        let mut b = Vec::new();
        kv_str_arr(&mut b, "tokenizer.ggml.merges", &["a b"]);
        kv.push(b);
        let mut b = Vec::new();
        kv_u32_arr(&mut b, "tokenizer.ggml.token_type", &[3, 3, 1, 1, 1, 3, 4]);
        kv.push(b);
        if bos_id {
            let mut b = Vec::new();
            kv_u32(&mut b, "tokenizer.ggml.bos_token_id", 1);
            kv.push(b);
        }
        if let Some(v) = add_bos {
            let mut b = Vec::new();
            kv_bool(&mut b, "tokenizer.ggml.add_bos_token", v);
            kv.push(b);
        }

        let mut bytes = Vec::new();
        u32le(&mut bytes, 0x4655_4747); // GGUF magic
        u32le(&mut bytes, 3); // version
        u64le(&mut bytes, 0); // tensor count: a tokenizer-only fixture never reaches the data region
        u64le(&mut bytes, kv.len() as u64); // metadata count
        for entry in &kv {
            bytes.extend_from_slice(entry);
        }
        bytes
    }

    /// Write a fixture, open it, build the GGUF-derived tokenizer, run `f`, then remove the file —
    /// the byte-level precedent of `crate::config`'s `config_from_fixture` (`tempfile` is not a
    /// dev-dependency here, and pid + counter keeps parallel tests off one path).
    fn with_raw_tokenizer<T>(
        name: &str,
        bytes: &[u8],
        f: impl FnOnce(&Gguf, &Tokenizer) -> T,
    ) -> T {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "infr-llama-raw-tok-{name}-{}-{n}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).expect("write GGUF fixture");
        let g = Gguf::open(&path).expect("open GGUF fixture");
        let tok = build_tokenizer(&g).expect("build tokenizer from GGUF fixture");
        let out = f(&g, &tok);
        std::fs::remove_file(&path).expect("remove GGUF fixture");
        out
    }

    /// The bare-text ids of `text` on this tokenizer, specials matched as the vocab's flag says.
    fn bare_ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
        tok.encode(text, false).unwrap().get_ids().to_vec()
    }

    #[test]
    fn raw_tokenize_adds_bos_only_when_the_gguf_asks_for_it() {
        with_raw_tokenizer("bos", &raw_tokenizer_gguf(Some(true), true), |g, tok| {
            // The shared tokenizer on bare text: `"ab"` is the one merge, so `[4]`, no BOS. Nothing
            // a chat template would have added is in here — that is `--raw`'s whole point.
            assert_eq!(bare_ids(tok, "ab"), vec![4], "bare ids");
            // `add_bos_token=true` → llama.cpp prepends `bos_token_id` and changes nothing else.
            assert_eq!(tokenize_raw(g, tok, "ab").unwrap(), vec![1, 4]);
            assert_eq!(tokenize_raw(g, tok, "a").unwrap(), vec![1, 2]);
            // An empty prompt is BOS alone — the case llama.cpp's own comment at
            // src/llama-vocab.cpp:3406 spells out for `add_special = true`.
            assert_eq!(tokenize_raw(g, tok, "").unwrap(), vec![1]);
        });

        // The two ways a GGUF says "no BOS": the key set false, and the key absent, which leaves
        // llama.cpp's `add_bos` at its `false` initial value (src/llama-vocab.cpp:1824, :2616).
        for (case, add_bos) in [("false", Some(false)), ("absent", None)] {
            with_raw_tokenizer(case, &raw_tokenizer_gguf(add_bos, true), |g, tok| {
                assert_eq!(tokenize_raw(g, tok, "ab").unwrap(), vec![4]);
            });
        }

        // `add_bos_token=true` with no `bos_token_id` is a broken GGUF — llama.cpp GGML_ASSERTs on
        // it (src/llama-vocab.cpp:3413). A named error beats answering a shifted prompt.
        with_raw_tokenizer(
            "no-bos-id",
            &raw_tokenizer_gguf(Some(true), false),
            |g, tok| {
                let err = tokenize_raw(g, tok, "ab").unwrap_err().to_string();
                assert!(
                    err.contains("bos_token_id"),
                    "expected the missing-id error, got: {err}"
                );
            },
        );
    }

    #[test]
    fn raw_tokenize_parses_specials_out_of_the_prompt() {
        with_raw_tokenizer(
            "parse-special",
            &raw_tokenizer_gguf(Some(true), true),
            |g, tok| {
                let control = tok.token_to_id("<ctrl>").unwrap();
                // `/completion` runs with `parse_special = true`: a CONTROL marker written in the
                // prompt is its one id, with the text around it tokenized on either side.
                assert_eq!(tokenize_raw(g, tok, "<ctrl>").unwrap(), vec![1, control]);
                assert_eq!(
                    tokenize_raw(g, tok, "a<ctrl>b").unwrap(),
                    vec![1, 2, control, 3]
                );
                assert_eq!(
                    tokenize_raw(g, tok, "<ctrl>ab<ctrl>").unwrap(),
                    vec![1, control, 4, control]
                );
            },
        );
    }

    /// USER_DEFINED markers are cut out too (`tokenizer_st_partition`, `src/llama-vocab.cpp:3225`):
    /// `<ud>` is one id with the text around it tokenized on either side of it.
    #[test]
    fn raw_tokenize_cuts_user_defined_tokens_out_of_the_prompt() {
        with_raw_tokenizer(
            "user-defined",
            &raw_tokenizer_gguf(Some(true), true),
            |g, tok| {
                let ud = tok.token_to_id("<ud>").unwrap();
                assert_eq!(tokenize_raw(g, tok, "a<ud>b").unwrap(), vec![1, 2, ud, 3]);
                assert_eq!(tokenize_raw(g, tok, "<ud>").unwrap(), vec![1, ud]);
                assert_eq!(
                    tokenize_raw(g, tok, "<ud>ab<ud>").unwrap(),
                    vec![1, ud, 4, ud]
                );
            },
        );
    }

    #[test]
    fn raw_tokenize_follows_the_gguf_flags_on_a_real_model_too() {
        // The same two rules on a real vocab, asserted against the GGUF's own metadata rather than a
        // hand-written id vector. Self-skips without the model, like
        // `embedded_tokenizer_matches_sidecar`.
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let g = Gguf::open(&gguf).unwrap();
        let tok = build_tokenizer(&g).unwrap();
        let md = g.metadata();
        let add_bos = matches!(
            md.get("tokenizer.ggml.add_bos_token"),
            Some(MetaValue::Bool(true))
        );
        // Ordinary prose: no special string in it, so the two flags are the only difference from
        // what the chat path tokenizes.
        let bare = bare_ids(&tok, "The capital of France is");
        let raw = tokenize_raw(&g, &tok, "The capital of France is").unwrap();
        if add_bos {
            let bos =
                md.u64("tokenizer.ggml.bos_token_id")
                    .expect("add_bos_token=true must carry a bos_token_id") as u32;
            assert_eq!(raw, [vec![bos], bare].concat());
        } else {
            assert_eq!(raw, bare, "no add_bos_token means no BOS, ever");
        }
    }

    // ── `tokenize_raw` against llama.cpp itself, on the real model ─────────────────────────────
    //
    /// The check the fixture tests cannot make: they assert what the GGUF keys SAY, this one asserts
    /// what llama.cpp DOES with them. It runs `llama-tokenize` over the same prompt files a parity
    /// run feeds the engine and compares the id lists with [`tokenize_raw`], on the real model.
    ///
    /// `#[ignore]`d because it needs a llama.cpp build, the ~17 GB GGUF and the Hipfire prompt tree,
    /// none of which belongs in a CI run. All four come from the environment so no host path enters
    /// the repo, and anything missing prints a `skip:` line and passes — a test that fails for want
    /// of a 17 GB file teaches nobody anything:
    ///
    /// ```text
    /// LI_RAW_PARITY_GGUF=<model.gguf> LI_LLAMA_TOKENIZE=<llama-tokenize> \
    /// LI_RAW_PARITY_DIR=<prompt dir> LI_RAW_PARITY_FILES=<a.txt,b.txt> \
    ///   cargo test -p infr-llama raw_tokenize_matches_llama_cpp -- --ignored --nocapture
    /// ```
    ///
    /// `LI_RAW_PARITY_FILES` splits on `,` and whitespace (paths containing either are not
    /// supported) and resolves each entry against `LI_RAW_PARITY_DIR` unless it is absolute.
    /// `LD_LIBRARY_PATH` is inherited by the child, which is how a reference binary that needs the
    /// libs it was linked against gets them.
    ///
    /// The reference side is `llama-tokenize -m <gguf> -f <file> --ids --log-disable --no-escape`,
    /// and it is `/completion`'s tokenization of the same bytes, by its own source at `3612199ed`
    /// (`tools/tokenize/tokenize.cpp`):
    ///
    /// * `-f` reads the file VERBATIM — the tool's comment says it skips common's trailing-newline
    ///   strip because "a trailing newline is itself a token". The same bytes `--prompt-file` takes.
    /// * `--no-escape` is not optional: `common_params::escape` defaults to `true`
    ///   (`common/common.h:565`) and the tool runs `string_process_escapes` over the file, so without
    ///   it a literal backslash-n in a code prompt becomes a newline on the reference side ONLY.
    /// * `parse_special` is TRUE on both sides: `/completion` calls
    ///   `tokenize_input_prompts(.., add_special = true, parse_special = true)`
    ///   (`tools/server/server-context.cpp:4327`), and the tokenize tool defaults it to true
    ///   "like the old tokenize tool" (`common/arg.cpp:1407`). So `<|im_start|>` written in a
    ///   prompt file is the special id on the reference side. A `tokenize_raw` that keeps specials
    ///   as plain text FAILS this test on any prompt containing one, by design of the oracle.
    /// * BOS: the tool passes `llama_vocab_get_add_bos(vocab) && !--tokenize-no-bos` (flag left off)
    ///   as `add_special` — the model's own `add_bos_token`, i.e. [`tokenize_raw`]'s rule.
    ///   `/completion` passes `add_special = true` unconditionally, so the two references differ only
    ///   for a GGUF that sets `add_eos_token` / `add_sep_token`, which llama.cpp would append and
    ///   this tool would not. All three keys print below so a failure can be read against them, and
    ///   so does `tokenizer.ggml.pre`: the reference build on this host predates the pinned commit,
    ///   and between the two the only tokenizer changes are two new pre-tokenizer families, so a
    ///   prompt tokenized under any other `pre` is tokenized identically by both.
    ///
    /// [`Gguf::open`] maps the file and reads the KV metadata and the tensor table; the weights are
    /// never read, and the tokenizer path has no device to read them into.
    #[test]
    #[ignore = "needs a llama.cpp build, the parity GGUF and the prompt files (env vars)"]
    fn raw_tokenize_matches_llama_cpp() {
        fn env(name: &str) -> Option<String> {
            std::env::var(name).ok().filter(|v| !v.trim().is_empty())
        }
        const NEEDED: [&str; 4] = [
            "LI_RAW_PARITY_GGUF",
            "LI_LLAMA_TOKENIZE",
            "LI_RAW_PARITY_DIR",
            "LI_RAW_PARITY_FILES",
        ];
        for name in NEEDED {
            if env(name).is_none() {
                eprintln!("skip: {name} is unset — see the doc comment on this test");
                return;
            }
        }

        /// The five ids either side of `at`, which is what a mismatch has to show: the index alone
        /// says nothing about whether one side lost a token or merged two.
        fn around(ids: &[u32], at: usize) -> Vec<u32> {
            let lo = at.saturating_sub(5);
            ids[lo..(at + 6).min(ids.len())].to_vec()
        }

        /// `llama-tokenize --ids` prints the whole list as one bracketed, comma-separated line.
        fn llama_ids(
            tool: &std::path::Path,
            model: &std::path::Path,
            prompt: &std::path::Path,
        ) -> Vec<u32> {
            let out = std::process::Command::new(tool)
                .arg("-m")
                .arg(model)
                .arg("-f")
                .arg(prompt)
                .args(["--ids", "--log-disable", "--no-escape"])
                .output()
                .unwrap_or_else(|e| panic!("run {}: {e}", tool.display()));
            if !out.status.success() {
                panic!(
                    "{} failed on {} ({}): {}",
                    tool.display(),
                    prompt.display(),
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let body = text
                .strip_prefix('[')
                .and_then(|t| t.strip_suffix(']'))
                .unwrap_or_else(|| {
                    panic!("unexpected --ids output from {}: {text:?}", tool.display())
                });
            body.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<u32>()
                        .unwrap_or_else(|e| panic!("bad id {s:?} from {}: {e}", tool.display()))
                })
                .collect()
        }

        let gguf_path = std::path::PathBuf::from(env("LI_RAW_PARITY_GGUF").unwrap());
        let tool = std::path::PathBuf::from(env("LI_LLAMA_TOKENIZE").unwrap());
        let dir = std::path::PathBuf::from(env("LI_RAW_PARITY_DIR").unwrap());
        let prompts: Vec<std::path::PathBuf> = env("LI_RAW_PARITY_FILES")
            .unwrap()
            .split([',', ' ', '\t', '\r', '\n'])
            .filter(|s| !s.is_empty())
            .map(std::path::Path::new)
            .map(|p| {
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    dir.join(p)
                }
            })
            .collect();
        assert!(!prompts.is_empty(), "LI_RAW_PARITY_FILES named no file");

        let g = Gguf::open(&gguf_path).expect("open the parity GGUF");
        let tok = build_tokenizer(&g).expect("build the tokenizer from the parity GGUF");
        let md = g.metadata();
        println!("parity model: {}", gguf_path.display());
        println!(
            "  tokenizer.ggml.pre         = {:?}",
            md.get("tokenizer.ggml.pre").and_then(MetaValue::as_str)
        );
        for key in ["add_bos_token", "add_eos_token", "add_sep_token"] {
            let full = format!("tokenizer.ggml.{key}");
            println!("  {full:30} = {:?}", md.get(&full));
        }
        println!(
            "  reference = {} (LD_LIBRARY_PATH = {:?})",
            tool.display(),
            env("LD_LIBRARY_PATH").unwrap_or_default()
        );

        let mut total = 0usize;
        for prompt in &prompts {
            let bytes = std::fs::read(prompt)
                .unwrap_or_else(|e| panic!("read prompt {}: {e}", prompt.display()));
            let text = String::from_utf8(bytes).unwrap_or_else(|e| {
                panic!(
                    "{} is not valid UTF-8 ({} bytes), and tokenize_raw takes &str: {e}",
                    prompt.display(),
                    e.utf8_error().valid_up_to()
                )
            });
            let want = llama_ids(&tool, &gguf_path, prompt);
            let got = tokenize_raw(&g, &tok, &text).expect("tokenize_raw");
            let first_diff = got
                .iter()
                .zip(want.iter())
                .position(|(mine, theirs)| mine != theirs)
                .unwrap_or(got.len().min(want.len()));
            if first_diff != got.len() || got.len() != want.len() {
                panic!(
                    "{}: the two id lists differ at index {first_diff} (infr {} ids, llama.cpp {})\n  \
                     infr      [{first_diff}]: {:?}\n  llama.cpp [{first_diff}]: {:?}",
                    prompt.display(),
                    got.len(),
                    want.len(),
                    around(&got, first_diff),
                    around(&want, first_diff),
                );
            }
            let name = prompt.file_name().unwrap_or_default().to_string_lossy();
            println!("  {name}: {} ids, identical", got.len());
            total += got.len();
        }
        println!(
            "{} prompts, {total} ids: tokenize_raw is llama.cpp's, byte for byte",
            prompts.len()
        );
    }
}
