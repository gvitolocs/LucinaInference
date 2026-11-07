# Stage 2 plan: Qwen3.8-27B on the RX 7900 XTX through Vulkan (2026-11-07)

Goal: LucinaInference (infr fork) runs the production model `Qwen3.8-27B-UD-Q4_K_XL` on the 7900 XTX at
llama.cpp's speed or better, at production context, under 23.5 GB VRAM, with logits within the stage 1
tolerance. Nothing in this stage changes production; every GPU run happens in a window the coordinator
grants (llama-server-qwen stopped through its unit, the GPU taken through `the GPU lease tool`).

## The bar we have to meet (production today)

`~/bin/llama-server-vulkan.sh` with drop-ins (`llama-server-qwen.service.d/*`), llama.cpp 3612199ed:

| Setting | Value |
|---|---|
| Context | `-c 122880` (30-ctx.conf, since 2026-11-04), `-np 1` |
| KV cache | `--cache-type-k q4_0 --cache-type-v q4_0`, `--flash-attn on` |
| Batching | `-b 512 -ub 256` |
| Speculation | `--spec-type draft-mtp --spec-draft-n-max 2` (the built-in MTP head) |
| Cache | `--cache-ram 2048`, `--ctx-checkpoints 24`, `--checkpoint-min-step 256` |
| Env | `MESA_VK_DEVICE_SELECT=1002:744c`, `RADV_PERFTEST=nogttspill`, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` |
| Lease | 21,500 MB (35-lease-120k.conf) |

Measured production numbers so far (benchmarks/inference-2026-10-23/comparison.md, 24 h of real traffic at
ctx 81,920, so not a controlled benchmark): prefill median 559 tok/s (p95 597, prompts >= 2048 tokens),
decode median 67.8 tok/s with MTP (draft acceptance median 0.95), VRAM 18,540 MiB idle.

infr's published 7900 XTX ratios (Qwen3.6-27B Q4_K_M: pp512 1.26x, tg128 1.03x, tg64 at 4k depth
0.99x) were measured against llama.cpp **b9833** with f16 KV and no MTP. Our production build is far newer and
uses q4_0 KV plus MTP, so those ratios do not carry over. Step 3 measures them again.

## Steps

### 1. GPU window and harness (no production change)

- Ask the coordinator for a window. Stop `llama-server-qwen` (the proxy then answers "unavailable", the
  other lanes fall through). Record the start and end in `ops_record`.
- Every run: `the GPU lease tool run --name lucina-inference --mb <measured> --`, the production env above,
  `systemd-run --user --scope -p MemoryMax=...`. Stop everything at the end and restart the unit; check
  `/health` on :<gateway-port>.

### 2. Correctness on the GPU

- Add `lucina-parity vulkan-logits` (infr `SeamModel::prefill_logits_vulkan` / all-row variant).
- The 9B: Vulkan logits against the llama.cpp CPU oracle and the f64 reference (`work/ref`, `work/f64`).
  Self-spread for the GPU: llama.cpp Vulkan (production build, `-ngl 99`) against the llama.cpp CPU oracle.
  Gate: the same `Tolerance::from_self_spread` rule as stage 1, with llama.cpp Vulkan in the spread.
- The 27B: same, with the q4_0 KV cache on both sides, prompts of 149 tokens and of about 8k tokens (so
  flash attention's split-K path and the DeltaNet chunked path both run).
- Greedy 256-token continuations on five agent-style prompts: count the first divergence position against
  llama.cpp Vulkan.
- **Long-prompt and per-layer gates** (stage 1 showed a wrong rope pairing only just fails a 48-token
  logits gate): 4k- and 32k-token prompts on CPU first (oracle + lucina-ref + infr), then the GPU. Per-layer
  hidden states against llama.cpp's `llama-eval-callback` tensor dumps (`attn_residual`, `post_ffn`,
  `linear_attn_out` per layer), so a mismatch names its layer and op.
- Gate the GPU against both the llama.cpp CPU oracle and the f64 reference. At massive-activation tokens an
  f16-activation GPU path should be closer to exact math than llama.cpp CPU, whose 8-bit activation rounding
  dominates there (stage 1, position 25 of the 27B prompt).

### 3. Speed matrix (same flags as production on both sides)

Shapes: pp512, pp4096, pp(32k cold read), tg128 at depth 0, 4k, 32k, 64k and 120k, each r=3, Tctl logged
(pause at >= 90 °C), the GPU otherwise idle.

| Side | Command |
|---|---|
| llama.cpp | `llama-bench -m <27B> -fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -b 512 -p ... -n ... -d ...` (build 3612199ed) |
| infr | `infr bench` / `infr compare --llama-bench <that binary>` with `kv.type_k=q4_0 kv.type_v=q4_0` |
| MTP | llama-server with draft-mtp n_max 2 against `INFR_MTP=1`, on a recorded agent session replay (tokens/s at the client, acceptance rate) |

Numbers go to `docs/lucina/stage2-numbers.md` with the commit, the Mesa version and the Tctl range.

### 4. VRAM at production context

- Measure `drm-memory-vram` from `/proc/<pid>/fdinfo` at load, after a 120k-token prefill, and at peak during
  decode, for ctx 81,920 and 122,880, q4_0 KV.
- Hard limit 23.5 GB (above it RADV spills to GTT and decode drops to about 3.7 tok/s). Target: no more than
  production's 21.5 GB lease, so the GPU lease tool ledger does not change.

### 5. Gaps expected from reading the fork (to confirm in step 2/3)

- **q4_0 KV with head_dim 256 flash attention on the GPU**: infr has `quant_kv_q4_0` and a q4_0 path that
  skips the f16 dequant prepass; whether the hd=256 fast path covers q4_0 is unverified.
- **IQ4_NL (6 tensors) and IQ3_S (1 tensor)**: infr has `native_gemm_mmq_iq4_nl` / `_iq3_s`; check the mmv
  (decode) side too.
- **Recurrent-state checkpoints and the host prompt cache**: llama-server keeps up to 24 checkpoints of the
  DeltaNet state per slot and restores them on a mid-history edit. infr's server has KV reuse, but no
  checkpoint ring has been found yet. Without it, every edited agent turn re-prefills the whole context. This is a
  stage 3 blocker for the drop-in, not for the speed numbers.
- **MTP + q4_0 KV + Vulkan** together: infr's MTP landed for CPU and Vulkan with f16 KV.

If a kernel is slower than llama.cpp's, port that ggml-vulkan shader (MIT, notice kept) into
`crates/infr-vulkan/shaders` rather than tuning blind; the llama.cpp kernel is the oracle for speed as well.

### 6. Acceptance for stage 2

- Logits within tolerance on the 9B and 27B (CPU-oracle and f64-reference gates), q4_0 KV.
- Prefill and decode at every depth up to 120k equal to or faster than llama.cpp 3612199ed with the same
  flags; MTP-on decode equal or faster on the replay.
- VRAM at ctx 122,880 under 21.5 GB (hard limit 23.5 GB).
- Production untouched: the unit restarted after each window, `/health` green, `ops_record` entries made.

## Stage 3 preview: drop-in behind the gateway

`~/bin/llama-server-qwen-proxy.py` (:<gateway-port> → :<origin-port>) calls `/completion`, `/v1/chat/completions` (and
`/chat/completions`), `/slots` (save, restore, erase), `/health`, `/v1/models` (`/models`), `/metrics`.
infr's server has `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`, `/status`. To add: `/completion`,
`/slots` with save and restore, checkpoints, llama-server's sampler fields (`presence_penalty`, `top_k`, `min_p`,
`reasoning` off, the reasoning budget message), the `--alias` list, the API-key file, and `--sleep-idle-seconds`.
The acceptance is a replay of a recorded proxy session with byte-identical request handling, then a
shadow run beside production before any switch. The switch itself is the owner's decision.
