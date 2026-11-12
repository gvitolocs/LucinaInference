# LucinaInference stage 0: survey of Rust llama.cpp replacements (2026-11-07)

Goal: a full-Rust engine that replaces llama.cpp on the build host for **Qwen3.8-27B UD-Q4_K_XL** on the
**RX 7900 XTX through Vulkan**, at llama.cpp speed or better, then runs on Juno (AArch64 NEON).

## The target, read from the GGUF on the build host

`~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.56 GB):

- `general.architecture = qwen35`. "Dense" means no MoE, but the model is a **hybrid**: 48 Gated
  DeltaNet (linear attention) layers and 16 gated full-attention layers (every 4th), 64 blocks plus one
  MTP/nextn block (`block_count = 65`, `nextn_predict_layers = 1`).
- n_embd 5120, FFN 17408, 24 q heads / 4 kv heads of 256, MRoPE (`imrope`) on 64 of 256 dims,
  sections [11, 11, 10, 0], freq_base 1e7, rms eps 1e-6.
- DeltaNet: conv kernel 4, state 128, 16 k-groups, 48 v-heads, inner 6144.
- Vocab 248,320, `tokenizer.ggml.pre = qwen35`.
- Tensor types: F32 360, Q5_K 191, Q8_0 110, IQ4_XS 70, Q4_K 69, Q6_K 56, IQ4_NL 6, Q3_K 3, IQ3_S 1.
  An engine needs **all** of these, not only Q4_K.
- Production llama-server now runs `-c 122880` (drop-in `30-ctx.conf`, 2026-11-04), not 81,920.

## Strata: why it was removed on 2026-10-24 (do not reuse)

Sources: `~/Projects/AI/benchmarks/inference-2026-10-23/PROGRESS.md` ("Result and decision"),
`~/Projects/AI/Orchestra/QWEN_DO_ISSUES.md`, Orchestra memory `strata-default-provider.md`.

- Strata on the build host was a **HIP/ROCm** engine (gfx1100 code objects, hipBLASLt), not Vulkan, running the
  MoE **Qwen3.8-Flash-Next** at IQ2_XS / IQ1_M (68 GB on disk vs 63 GB RAM). Experts streamed from disk.
- Problems in production: memory.high cgroup throttling (5.8 tok/s with a 12 GB reserve), swap full,
  freezes at 16-20k-token cold reads, `switch_failed` when the GPU was busy.
- The deciding numbers were model quality and throughput on agent tasks: on the 11 hardest tasks, dense
  Qwen3.8-27B on llama.cpp passed 7 at 0.27 solved/min (median 81 s); the best Strata Coder config
  passed 5 at 0.046 solved/min. Full 20-task set: dense 16/20.
- Owner decision: dense 27B on llama.cpp is the only local offload model; Strata and its models were
  deleted. Nothing of the engine is left on disk.
- RustingStrata (GitHub) is a different, newer project; see below. Nothing in it is reused.

## Candidates

| Project | Licence | qwen35 hybrid | Quants we need | Vulkan | Evidence vs llama.cpp | Verdict |
|---|---|---|---|---|---|---|
| **infr** (kryptic-sh/infr, f2f89cb, 2026-10-29) | MIT | yes, CPU + Vulkan (`docs/qwen35.md`, `docs/qwen38.md` names Qwen3.8-27B) | all: Q3_K..Q6_K, Q8_0, IQ3_S, IQ4_NL, IQ4_XS | ash + glslc GLSL, 141 shaders incl. DeltaNet, conv1d, flash attention, coopmat GEMM, MMQ | own results on a **7900 XTX** RADV: Qwen3.6-27B Q4_K_M pp512 1.26x, tg128 1.03x, tg64@4k 0.99x; Qwen3.5-9B pp512 1.41x, tg128 1.01x (`docs/perf/results.md`, not reproduced yet) | **adopt (fork)** |
| RustingStrata | custom licence: no selling, credit required | lists qwen35 | wide | GLSL to SPIR-V, tested on RTX 3060 only | 1.7x decode only for MoE with CPU experts on a 3060; AMD issues #3, #10 (prefill 122 vs 827 t/s), NaN logits noted | reference |
| Lexmata llama-gguf 0.14 / llama-rs 0.23 | MIT/Apache | no qwen35 | K-quants | basic ops only, GPU path is CUDA | 0.3 t/s Mistral-7B CPU | reject |
| Ferrox | Apache-2.0 | not confirmed | K, IQ | "one kernel" | Metal only | reject |
| mistral.rs | MIT | not confirmed | GGUF | none | NVIDIA only | reject |
| candle | MIT/Apache | no | GGUF | none (issue #344) | none | reject |
| burn / cubecl | MIT/Apache | no LLM | none | SPIR-V via wgpu; no RDNA3 coopmat evidence | none | reference |
| Crane (on candle) | MIT | Qwen3.5-3.8 | GGUF | none | none | reject |
| rlx-qwen35 | unclear (sibling crate GPL-3) | yes, some host paths | K | flag only | none | reference |
| wgpu compute | MIT/Apache | n/a | n/a | coopmat experimental (wgpu 30), int-dot/subgroup status unverified | none | reject for kernels |
| ash + GLSL (glslc) | n/a | n/a | n/a | full: coopmat, subgroup size control, int dot | ggml-vulkan's own route | **the kernel route** |

infr facts checked in the clone (`~/scratch/lucinainference-survey/infr`): 1,573 commits (1,434 by one
author, mxaddict; 115 shinobu), ~180k lines of Rust, 15 crates (gguf, cpu, vulkan, metal, engine,
server, chat, llama, cli, testkit...), OpenAI server with streaming and tools, a CPU reference
forward that they state matches llama.cpp, `infr compare` against llama-bench. Its shaders cite
llama.cpp's kernels as the model they follow (flash_attn_cm1, d_split), so llama.cpp remains the
oracle for both behaviour and speed.

## Decision

**Adopt infr as the base: LucinaInference is a fork of infr** (`upstream` remote, pinned at
f2f89cb, merges only, push to upstream disabled; contributions upstream stay draft-only without the
owner's OK). Reasons: it is the only Rust engine that already runs qwen35 on Vulkan on our exact GPU,
with every quant type in our file, under MIT. Writing our own core would repeat months of its work
before reaching the same point.

What LucinaInference adds on top, and what we do not trust until measured:

1. An independent oracle harness against the pinned llama.cpp 3612199ed (`oracle/`), and logits
   parity tests on our own files (Qwen3.5-9B Q4_K_M, the 27B UD-Q4_K_XL). infr's own parity claims are
   not taken on faith.
2. Measured prefill and decode on the build host against our production llama-server (same model, same context,
   VRAM under 23.5 GB, `RADV_PERFTEST=nogttspill`).
3. llama-server API compatibility that our gateway and clients rely on (slots, prompt cache,
   checkpoints, `/slots` save and restore, tool-call format), so it can drop in behind
   `llama-server-qwen-proxy`.
4. Juno: the AArch64 NEON CPU path (infr-cpu has no NEON code today).

Fallback if infr fails parity or speed on our file: keep its GGUF/CPU crates, and port ggml-vulkan's
GLSL shaders (MIT) through ash ourselves.
