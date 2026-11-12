# Stage 1: CPU logits parity against llama.cpp (2026-11-07)

Engine under test: LucinaInference (infr fork, base f2f89cb), CPU reference backend
(`SeamModel::verify_logits_and_h_cpu`, all positions). Original: llama.cpp 3612199ed (production build on
the build host), CPU backend only, through `oracle/llama-logits` (no device offered, op offload off).

## Method

Three kinds of logits for the same tokens (the oracle's own tokenization of the prompt):

1. **llama.cpp** in several configurations it treats as equivalent: one batch of 512 (`b512`), 4 threads
   (`b512t4`), token by token (`seq`), f32 KV (`b512f32`).
2. **lucina-ref**, an exact-arithmetic reference (`crates/lucina-ref`): own GGUF reader, dequant ported
   from `ggml-quants.c`, every product, norm and exponential in f64, the gated delta rule run token by
   token. It follows llama.cpp's graph (qwen35.cpp, delta-net-base.cpp, ggml-cpu ops) and shares no code
   with the engine.
3. **The engine** (infr CPU).

Metrics per position (`crates/lucina-parity`): KL(p_ref || p_test) over the full 248,320-token vocabulary,
argmax agreement, rms(diff)/rms(ref spread), max |diff|. Argmax flips are counted separately where the
reference's top-2 margin is under 0.5 logits (near-ties: llama.cpp flips those against itself).

**Tolerance, and why:** the engine must stay within **2x the distance from exact arithmetic that llama.cpp
itself shows in its standard configurations**, on mean KL, max KL and relative RMS. It must also agree on
every decisive argmax. The f64 reference is the yardstick, so the bound is set by the original's own
precision rather than a guess, and k = 2 allows a different but equally legitimate rounding order (for
example f32 accumulation without activation quantization) without letting a real bug through. A wrong head
mapping, rope pairing or norm moves mean KL by orders of magnitude (see "sensitivity" below), not by 2x.
The same gate with llama.cpp `b512` as the reference (`Tolerance::from_self_spread`) is also run; it
passes too.

## Qwen3.5-9B Q4_K_M (`qwen35`, 32 layers), 149-token chat prompt

Prompt: `oracle/prompts/chat.txt` (chat template, code, Danish, Italian, `<think>` block).
Model: `~/models/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf` (types Q4_K, Q5_K, Q6_K, Q8_0, F32).

Distance from the f64 reference:

| Logits | mean KL | max KL | max rel RMS | max abs | top-1 decisive | top-1 all |
|---|---|---|---|---|---|---|
| llama.cpp b512 | 2.35e-3 | 0.041 | 0.559 | 6.98 | 128/128 | 146/149 |
| llama.cpp seq | 2.78e-3 | 0.070 | 0.539 | 7.01 | 128/128 | 147/149 |
| **infr CPU** | **4.16e-3** | **0.077** | **0.767** | 7.21 | **128/128** | 142/149 |
| tolerance (2x) | 5.56e-3 | 0.139 | 1.119 | - | all | - |

Gate result: **PASS** (`lucina-parity gate --ref work/f64/q35-9b-chat --self <llama.cpp runs> --test work/infr/q35-9b-chat --k 2`). Integration tests: `crates/lucina-parity/tests/oracle_parity.rs` (both gates; the llama.cpp-referenced one written by local Qwen through the coding-agent queue and reviewed).

Against llama.cpp b512 directly: infr mean KL 4.77e-3, max 0.055, decisive top-1 126/126; llama.cpp seq
vs b512: mean KL 2.68e-3, max 0.109. infr's worst positions (112, 104, 10-12, 140) are the same positions
where llama.cpp disagrees with itself most: they are numerically sensitive, not mapped wrongly.

What this also shows:
- **The f64 reference reproduces llama.cpp** within llama.cpp's own run-to-run spread (2.4e-3 to 2.8e-3
  mean KL, every decisive argmax equal), so its port of the qwen35 graph (interleaved per-head q/gate,
  IMROPE on 64 of 256 dims, `j % n_k_heads` head mapping, conv state, L2 norm, gated RMS norm) is right.
- infr's CPU path is about 1.8x further from exact math than llama.cpp. That is within the bound, but it
  is worth knowing: its CPU forward is not "more exact" than ggml's, even though it does not quantize
  activations the same way. Follow-up: find which op contributes it (per-layer hidden-state dumps).
- `b512t4` and `b512f32` were bit-identical to `b512`. The thread count is not expected to matter; f32 KV
  had no visible effect on this prompt. The cause was not investigated, so those two runs add nothing to
  the spread.

Timings (the build host, Tctl 88-96 °C, other sessions busy, so not a speed benchmark): llama.cpp CPU 92-109 s
per batch run, 349 s token by token; infr CPU 55 s; lucina-ref f64 185 s.

## Qwen3.8-27B UD-Q4_K_XL (the production file), CPU

Prompt `oracle/prompts/short.txt` (35 tokens: chat template, Danish, an empty `<think>` block, the start of
an answer).

| Comparison | mean KL | max KL | max rel RMS | top-1 decisive | top-1 all |
|---|---|---|---|---|---|
| llama.cpp seq vs llama.cpp b512 (self-spread) | 7.1e-4 | 9.5e-3 | 0.162 | 32/32 | 35/35 |
| **infr CPU vs llama.cpp b512** | **4.6e-4** | **5.2e-3** | 0.242 | **32/32** | **35/35** |
| llama.cpp b512 vs f64 | 2.43e-2 | 0.828 | 0.763 | 32/33 | 34/35 |
| llama.cpp seq vs f64 | 2.19e-2 | 0.752 | 0.828 | 32/33 | 34/35 |
| infr CPU vs f64 | 2.31e-2 | 0.794 | 0.883 | 32/33 | 34/35 |

Both gates **PASS**: the llama.cpp-referenced gate (infr is closer to llama.cpp b512 than llama.cpp's own
token-by-token run is), and the f64-referenced gate.

**Why llama.cpp and infr both sit about 0.02 mean KL from exact math on the 27B:** the distance is one
position. Position 25 (the `assistant` token after `<|im_start|>`) has KL 2.05 and relative RMS 0.96 against
f64, while every other position stays around 1e-5 to 1e-2. That token carries a massive activation. ggml-cpu
quantizes every matmul input to 8-bit blocks (Q8_K per 256 values for K/IQ weights, Q8_0 per 32 for
Q8_0/IQ4_NL), and one huge value in a block rounds its neighbours to zero. infr's CPU path does the same, so
it agrees with llama.cpp (KL 4.6e-4) and not with f64. Checked directly: with ggml's activation rounding
emulated in lucina-ref (`LUCINA_REF_ACT_QUANT=ggml`), position 25 drops to KL 0.075 and relative RMS 0.16
against llama.cpp, and mean KL to 2.7e-3.

Consequences:
- The gate's argmax rule is now "no more decisive flips than the original shows against the same
  reference" (`Tolerance::decisive_flips`). llama.cpp flips 1 of 33 against f64 on the 27B and 0 against
  itself everywhere.
- On the 27B the f64-referenced bounds are loose (max KL 1.66), because the original itself is far off at
  position 25. There the llama.cpp-referenced gate is the tight one.
- A GPU path that keeps activations in f16 will be **closer** to exact math than llama.cpp CPU at such
  tokens. Stage 2 therefore gates the GPU against both references and llama.cpp Vulkan's own spread.

Timings (other sessions loading the build host; not a benchmark): llama.cpp CPU 34 s batch, 231 s token by token; infr
CPU 21 s; lucina-ref f64 101 s.

## Sensitivity: does the gate catch real bugs?

Two deliberately wrong builds of lucina-ref (`LUCINA_REF_MUTANT`), 9B, first 48 tokens of the chat prompt, gate
against f64 with llama.cpp b512 and seq as the spread:

| Build | mean KL | max KL | top-1 decisive | Gate |
|---|---|---|---|---|
| `head_div`: DeltaNet v-head j reads k-head j/(n_v/n_k) instead of j % n_k | 2.47 | 11.5 | 19/37 | **FAIL** on every metric |
| `rope_adjacent`: rope on (2i, 2i+1) instead of (i, i+32) | 3.8e-3 | 0.076 | 37/37 | **FAIL, barely** (max KL 0.076 > 0.065) |
| infr (control) | 7.0e-3 | 0.050 | 37/37 | PASS |
| bound | 7.8e-3 | 0.065 | 0 flips | |

The head mapping (and anything of that size: norms, gates, conv, state layout) is caught by orders of
magnitude. **A wrong rope pairing is caught only just**, and its mean KL is lower than infr's. On a short
prompt, positions are small, only 1 layer in 4 is attention, and rope covers 64 of 256 dims, so positional
bugs hardly reach the logits. Logits on short prompts are not enough to vouch for positional code. Stage 2
adds a long-prompt gate (4k and 32k tokens, where rope and attention errors grow) and per-layer hidden-state
comparisons against llama.cpp (`llama-eval-callback` dumps).

## Tokenizer: a real bug, fixed

Corpus: `oracle/prompts/tokenizer-corpus.txt`, 144 lines (English, contractions, numbers, whitespace runs, code,
Danish, Italian, German, French, Spanish, Greek, Cyrillic, Arabic, Hebrew, Hindi, Thai, CJK, Korean, emoji ZWJ
and flags, combining marks, chat special tokens and near-misses, URLs, LaTeX). llama.cpp's own tokenization
(`llama-logits --tokenize-lines`, special tokens parsed) against the engine's (`lucina-parity tokenize`),
compared with `lucina-parity tokens-diff`.

- **Before: 110 of 144 lines matched.** infr had no arm for `tokenizer.ggml.pre = "qwen35"`, so the Qwen3.5/3.8
  vocabulary fell through to the GPT-2 ByteLevel default regex. That splits `snake_case`, `well-known`,
  `example.com`, URLs, `I'M`, tabs and newline runs, and Devanagari/Thai combining marks differently. The
  logits parity above was unaffected, because it fed the oracle's tokens. But every real prompt through infr's
  own tokenizer would have been off.
- **Fix** (commit 4f60614): `QWEN35_PRE_RE`, the regex from the model's tokenizer.json as quoted in llama.cpp
  `LLAMA_VOCAB_PRE_TYPE_QWEN35` (letters with `\p{M}`, case-insensitive contractions).
- **After: 144 of 144** on the 27B and on the 9B vocabulary (both produce the same llama.cpp tokens). Golden
  file `oracle/golden/qwen35-tokenizer-corpus.tokens.txt`, test `crates/lucina-parity/tests/tokenizer_parity.rs`
  (self-skips without a model). infr's own tokenizer unit tests still pass.
- This is an upstream bug too. It is listed as a contribution candidate; nothing has been sent.

## Dequantization

`lucina-ref` dequantizes every type in the production file (F32, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_NL,
IQ4_XS). `crates/lucina-ref/tests/ggml_dequant.rs` checks 8 rows of a real tensor per type **bit for bit**
against ggml's own `to_float` (`oracle/ggml-dequant`): 8/8 pass. Q3_K and the IQ formats were ported by
Mistral (mistral-large) and reviewed.

## How to reproduce

```
bash oracle/build.sh                                    # oracle against ~/llama.cpp (pin in docs/upstreams.toml)
M=~/models/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf
work/bin/llama-logits --model $M --prompt-file oracle/prompts/chat.txt --out work/ref/q35-9b-chat-b512 --ubatch 512 --threads 8
work/bin/llama-logits --model $M --prompt-file oracle/prompts/chat.txt --out work/ref/q35-9b-chat-seq --one-by-one --threads 8
cargo build --release -p lucina-parity -p lucina-ref
target/release/lucina-ref-logits --model $M --tokens work/ref/q35-9b-chat-b512.tokens.txt --out work/f64/q35-9b-chat
target/release/lucina-parity infr-logits --model $M --tokens work/ref/q35-9b-chat-b512.tokens.txt --out work/infr/q35-9b-chat
target/release/lucina-parity gate --n-vocab 248320 --ref work/f64/q35-9b-chat.logits.f32 \
  --self work/ref/q35-9b-chat-b512.logits.f32,work/ref/q35-9b-chat-seq.logits.f32 --test work/infr/q35-9b-chat.logits.f32 --k 2
cargo test --release -p lucina-parity --test oracle_parity -- --nocapture   # self-skips without the files
```
Run heavy steps under `systemd-run --user --scope -p CPUQuota=...` and pause at Tctl >= 90 °C.
