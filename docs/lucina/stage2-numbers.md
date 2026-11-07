# Stage 2 numbers

## Window 1: 5-minute probe (2026-11-07 21:36 UTC, owner-approved)

Procedure (`work/window.sh`, kept in the gitignored work dir; logs in `work/logs/window/`):
1. Held the coding-agent queue lock, after waiting for a running job to finish.
2. Queued a GPU lease for 21,500 MiB, then stopped `llama-server-qwen` (the gateway answers "busy" while a lease waits).
3. Ran the probe.
4. Restarted the unit, waited for `/health`, and sent one request through the gateway (:<gateway-port>, answer "42").

Downtime was 21:36:05 to 21:37:23, 78 s, of which 50 s was GPU work. Model: Qwen3.8-27B-UD-Q4_K_XL. infr fork at 5a44c0a
plus the `vulkan-window` probe. RADV, `RADV_PERFTEST=nogttspill`, Mesa 25.2.8. Tctl about 92 °C at the start.

| Measurement | Result | Caveat |
|---|---|---|
| Loads and runs on the XTX through Vulkan | yes | |
| pp512 | **1023.5 t/s** | r=1, KV printed as **f16** (the `--set kv.type_k=q4_0` was not honoured by `infr bench`), ubatch 1024 |
| tg128 | **34.4 t/s** | r=1, depth 0, f16 KV, **no MTP** |
| Peak VRAM, infr only (card minus 1,461 MiB of other users) | **~20.0 GB** during pp512, 19.0 GB during tg128 and logits | KV sized for about 530 tokens by the bench, not for 122,880 |
| 122,880-context session | not measured | `vulkan_session(122880)` returned in 0.1 s with 1.5 GB used: the session allocates lazily |

No llama-bench ran in this window, so there is **no like-for-like speed ratio yet**. Production's numbers come from
traffic, not a benchmark (prefill median 559 t/s on prompts of 2048+ tokens at ub 256; decode median 67.8 t/s with
MTP at about 0.95 acceptance).

**VRAM projection for production** (not measured): weights 17.5 GB. The q4_0 KV for 122,880 tokens is
16 attention layers × 2 × 4 heads × 256 dims × 0.5625 B × 122,880 ≈ **2.26 GB**; f16 KV would be about 8 GB and
does not fit. The ~2.5 GB of scratch seen at ub 1024 comes on top. That lands at about 22 GB for infr, plus about
1.5 GB used by other processes on the card, which is right at the 23.5 GB spill line. infr must run at ub 256 (as production
does) and its KV must really be q4_0. Measure both before any longer window.

### Vulkan logits (last token of a prefix) against the CPU references

| Row (prefix) | vs llama.cpp CPU b512 | vs f64 exact | vs infr CPU | llama.cpp seq vs b512 |
|---|---|---|---|---|
| 25 (`assistant` after `<|im_start|>`, massive activation) | KL 2.25, top-1 differs | **KL 5.4e-4**, top-1 equal | KL 2.18 | KL 9.5e-3 |
| 34 (end of prompt) | KL 8.6e-3, top-1 equal | KL 4.5e-3, top-1 equal | KL 6.1e-3 | KL 3.0e-3 |

So infr Vulkan keeps activations in higher precision and lands on exact math at row 25, where both 8-bit-activation CPU
paths (llama.cpp and infr) are far off. This was predicted in stage 1. It also means a GPU gate needs the f64 reference;
the llama.cpp CPU oracle alone would flag the GPU as wrong there.

## Before the full window (no GPU needed)

1. Make `infr bench` honour `kv.type_k/v = q4_0`, or find the right flag; the bench must match production.
2. Measure VRAM at 122,880 context: prefill to depth 120k in the window, or make the session allocate eagerly.
3. An all-rows Vulkan logits path, so one upload serves a whole prompt (each prefix now re-uploads 17.5 GB, about 9-10 s).
4. A llama-bench script with production flags (`-fa 1 -ctk q4_0 -ctv q4_0 -ub 256 -b 512`, depths 0/4k/32k/64k/120k).

## Time a full stage 2 window needs

| Part | Estimate |
|---|---|
| Correctness: 9B and 27B, 149-token, 4k and 32k prompts, against the oracle and f64 (with an all-rows path) | 15 min |
| llama-bench, production flags: pp512, pp4096, tg128 at depths 0/4k/32k/64k/120k, r=3 (deep depths dominate: a 120k prefill takes minutes) | 40 min |
| infr, the same matrix | 40 min |
| MTP: llama-server draft-mtp against `INFR_MTP=1` on a recorded agent replay | 15 min |
| VRAM at 122,880 context, q4_0 | 5 min |
| Restores and margins | 10 min |
| **Total** | **about 2 h**, at night when build-host is cool; it can be split into three windows of about 40 minutes (correctness+VRAM, llama.cpp matrix, infr matrix+MTP) |
