# GPU windows

`gpuwin.sh NAME INNER MAX_S HARD_END_EPOCH` runs one owner-approved GPU window on the build host. It waits for headroom
(memory, load, swap, lease-tool budget), holds the coding-agent queue lock, stops the production unit, runs INNER
under a GPU lease with a VRAM watchdog, then restores production and confirms the gateway before it releases the lock.

Every machine-specific value (paths, ports, unit and tool names) comes from `scripts/local.env`; copy
`scripts/local.env.example` and fill it in. The scripts fail with a clear message if a value they need is missing.

The protocol is proved in `crates/lucina-rules/src/gate_rules.rs` (Verus, `verification/lucina-gate`). The shell
script itself is not proved: replay every window log with `target/release/lucina-gate-check <log>`.

`stage2-night.sh` is the 2026-11-08 night (three windows). The `w*-inner.sh` scripts are its measurements.
