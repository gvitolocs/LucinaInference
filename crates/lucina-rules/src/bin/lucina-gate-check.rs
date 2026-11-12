//! lucina-gate-check `<gpuwin log>`...: replays GPU-window logs through the Verus-proved protocol model
//! (`lucina_rules::gate_rules::check_trace`). Exit 0 when every window obeyed it, 1 otherwise.
use lucina_rules::gate_rules::{check_trace, parse_log};

fn main() {
    let mut bad = false;
    for path in std::env::args().skip(1) {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                // print-ok: the per-log verdict line — a log that cannot be read is reported, not
                // swallowed.
                eprintln!("{path}: {e}");
                bad = true;
                continue;
            }
        };
        let evs = parse_log(&text);
        match check_trace(&evs) {
            // print-ok: the per-log verdict line — one per replayed window log, which is the entire
            // product of this tool (`scripts/gpuwin/stage2-night*.sh` runs it after every window).
            Ok(()) => println!("{path}: OK ({} events)", evs.len()),
            Err(i) if i < evs.len() => {
                // print-ok: the per-log verdict line, naming the event that broke the proved protocol.
                println!(
                    "{path}: VIOLATION at event {i} ({:?}); events {:?}",
                    evs[i], evs
                );
                bad = true;
            }
            Err(_) => {
                // print-ok: the per-log verdict line for a window that ended mid-protocol.
                println!("{path}: VIOLATION: ended with the flock held or production down; events {evs:?}");
                bad = true;
            }
        }
    }
    std::process::exit(if bad { 1 } else { 0 });
}
