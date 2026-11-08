# GPU windows

`gpuwin.sh NAME INNER MAX_S HARD_END_EPOCH` runs one owner-approved GPU window on build-host. It waits for headroom
(memory, load, swap, the GPU lease tool budget), holds the coding-agent queue lock, stops llama-server-qwen through its unit, runs INNER
under a GPU lease with a VRAM watchdog, then restores production and confirms the gateway before it releases the flock.

The protocol is proved in `crates/lucina-rules/src/gate_rules.rs` (Verus, `verification/lucina-gate`). The shell
script itself is not proved: replay every window log with `target/release/lucina-gate-check <log>`.

`stage2-night.sh` is the 2026-11-08 night (three windows). The `w*-inner.sh` scripts are its measurements.
