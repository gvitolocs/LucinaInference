//! lucina-gate-check <gpuwin log>...: replays GPU-window logs through the Verus-proved protocol model
//! (`lucina_rules::gate_rules::check_trace`). Exit 0 when every window obeyed it, 1 otherwise.
use lucina_rules::gate_rules::{check_trace, parse_log};

fn main() {
    let mut bad = false;
    for path in std::env::args().skip(1) {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{path}: {e}");
                bad = true;
                continue;
            }
        };
        let evs = parse_log(&text);
        match check_trace(&evs) {
            Ok(()) => println!("{path}: OK ({} events)", evs.len()),
            Err(i) if i < evs.len() => {
                println!("{path}: VIOLATION at event {i} ({:?}); events {:?}", evs[i], evs);
                bad = true;
            }
            Err(_) => {
                println!("{path}: VIOLATION: ended with the flock held or production down; events {evs:?}");
                bad = true;
            }
        }
    }
    std::process::exit(if bad { 1 } else { 0 });
}
