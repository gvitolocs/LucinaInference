//! The GPU-window protocol (`work/gpuwin.sh`) as a state machine, proved by Verus in this file
//! (`verification/lucina-gate/rules.rs`). The shell script is not verified; instead every window's log is replayed
//! through [`check_trace`] (`lucina-gate-check <log>`), so a run that broke the protocol is caught after the fact.
//!
//! The rules (coordinator, 2026-11-07/08): production (llama-server-qwen) is stopped only while this window holds the
//! coding-agent queue lock and has seen enough headroom since taking it; the flock is released only once production is back
//! and the gateway has answered; a window ends with the flock released and production up.
#![allow(unexpected_cfgs)]

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// What a window's log can say, in order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ev {
    /// `holding the coding-agent queue lock`
    Lock,
    /// a headroom check that passed (memory, load, swap, lease budget)
    HeadroomOk,
    /// a headroom check that failed
    HeadroomLow,
    /// `headroom gone while waiting; releasing the flock` (before any stop)
    Unlock,
    /// `stopping llama-server-qwen`
    Stop,
    /// `RESTORE: started llama-server-qwen`
    ProdStart,
    /// the gateway answered a request
    GatewayOk,
    /// `releasing the coding-agent queue lock`
    Release,
}

#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gate {
    pub locked: bool,
    /// headroom passed since the flock was taken
    pub checked: bool,
    /// production stopped and not yet confirmed back
    pub down: bool,
    pub prod_started: bool,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Production is only ever down while the flock is held.
pub open spec fn inv(g: Gate) -> bool {
    g.down ==> g.locked
}

pub open spec fn finished(g: Gate) -> bool {
    !g.locked && !g.down
}

pub open spec fn spec_step(g: Gate, e: Ev) -> Option<Gate> {
    match e {
        Ev::Lock => if g.locked { None } else { Some(Gate { locked: true, checked: false, ..g }) },
        Ev::HeadroomOk => Some(Gate { checked: g.locked, ..g }),
        Ev::HeadroomLow => Some(Gate { checked: false, ..g }),
        Ev::Unlock => if g.locked && !g.down { Some(Gate { locked: false, checked: false, ..g }) } else { None },
        Ev::Stop => if g.locked && g.checked && !g.down {
            Some(Gate { down: true, prod_started: false, ..g })
        } else { None },
        Ev::ProdStart => if g.down { Some(Gate { prod_started: true, ..g }) } else { None },
        Ev::GatewayOk => if g.down { if g.prod_started { Some(Gate { down: false, ..g }) } else { None } } else { Some(g) },
        Ev::Release => if g.locked && !g.down { Some(Gate { locked: false, checked: false, ..g }) } else { None },
    }
}

pub open spec fn run(s: Seq<Ev>) -> Option<Gate>
    decreases s.len(),
{
    if s.len() == 0 {
        Some(Gate { locked: false, checked: false, down: false, prod_started: false })
    } else {
        match run(s.drop_last()) {
            Some(g) => spec_step(g, s.last()),
            None => None,
        }
    }
}

/// Every legal step keeps the invariant.
pub proof fn step_keeps_inv(g: Gate, e: Ev)
    requires
        inv(g),
    ensures
        spec_step(g, e) matches Some(n) ==> inv(n),
{
}

/// Every state a legal trace reaches satisfies the invariant.
pub proof fn run_keeps_inv(s: Seq<Ev>)
    ensures
        run(s) matches Some(g) ==> inv(g),
    decreases s.len(),
{
    if s.len() > 0 {
        run_keeps_inv(s.drop_last());
        if let Some(g) = run(s.drop_last()) {
            step_keeps_inv(g, s.last());
        }
    }
}

} // verus!

/// One step; `None` is a protocol violation.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r == spec_step(g, e),
        // The two rules that matter, stated outright:
        e == Ev::Stop && r is Some ==> g.locked && g.checked,
        e == Ev::Release && r is Some ==> !g.down,
))]
pub fn step(g: Gate, e: Ev) -> Option<Gate> {
    match e {
        Ev::Lock => {
            if g.locked {
                None
            } else {
                Some(Gate { locked: true, checked: false, ..g })
            }
        }
        Ev::HeadroomOk => Some(Gate { checked: g.locked, ..g }),
        Ev::HeadroomLow => Some(Gate { checked: false, ..g }),
        Ev::Unlock => {
            if g.locked && !g.down {
                Some(Gate { locked: false, checked: false, ..g })
            } else {
                None
            }
        }
        Ev::Stop => {
            if g.locked && g.checked && !g.down {
                Some(Gate { down: true, prod_started: false, ..g })
            } else {
                None
            }
        }
        Ev::ProdStart => {
            if g.down {
                Some(Gate { prod_started: true, ..g })
            } else {
                None
            }
        }
        Ev::GatewayOk => {
            if g.down {
                if g.prod_started {
                    Some(Gate { down: false, ..g })
                } else {
                    None
                }
            } else {
                Some(g)
            }
        }
        Ev::Release => {
            if g.locked && !g.down {
                Some(Gate { locked: false, checked: false, ..g })
            } else {
                None
            }
        }
    }
}

/// Replays a window's events: `Ok` when every step is legal and the window ended with the flock released and
/// production up; otherwise the index of the first bad event (`events.len()` when the end state is wrong).
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r is Ok <==> (run(events@) matches Some(g) && finished(g)),
))]
pub fn check_trace(events: &[Ev]) -> Result<(), usize> {
    let mut g = Gate { locked: false, checked: false, down: false, prod_started: false };
    let mut i: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            i <= events.len(),
            run(events@.subrange(0, i as int)) == Some(g),
        decreases events.len() - i,
    ))]
    while i < events.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(events@.subrange(0, i as int + 1).drop_last() =~= events@.subrange(0, i as int));
            assert(events@.subrange(0, i as int + 1).last() == events@[i as int]);
        }
        match step(g, events[i]) {
            Some(n) => g = n,
            None => {
                #[cfg(verus_keep_ghost)]
                proof! { lemma_none_stays(events@, i as int + 1); }
                return Err(i);
            }
        }
        i += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! { assert(events@.subrange(0, events.len() as int) =~= events@); }
    if g.locked || g.down {
        Err(events.len())
    } else {
        Ok(())
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Once a prefix is illegal, the whole trace is.
pub proof fn lemma_none_stays(s: Seq<Ev>, k: int)
    requires
        0 <= k <= s.len(),
        run(s.subrange(0, k)) is None,
    ensures
        run(s) is None,
    decreases s.len() - k,
{
    if k < s.len() {
        assert(s.subrange(0, k + 1).drop_last() =~= s.subrange(0, k));
        lemma_none_stays(s, k + 1);
    } else {
        assert(s.subrange(0, k) =~= s);
    }
}

} // verus!

/// Map one line of a gpuwin log to an event (the parser is not proved; it only reads the fixed phrases the script
/// prints). Headroom lines need the next line to tell a pass from a fail in logs older than the explicit verdicts.
pub fn parse_log(text: &str) -> Vec<Ev> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l.contains("headroom ok") {
            out.push(Ev::HeadroomOk);
        } else if l.contains("headroom low") {
            out.push(Ev::HeadroomLow);
        } else if l.contains(": headroom avail") {
            // older logs: a check passed when the script went on to wait for the flock or to stop
            let next = lines.get(i + 1).copied().unwrap_or("");
            if next.contains("waiting for the coding-agent queue lock") || next.contains("stopping llama-server-qwen") {
                out.push(Ev::HeadroomOk);
            } else {
                out.push(Ev::HeadroomLow);
            }
        } else if l.contains("headroom gone while waiting") {
            out.push(Ev::Unlock);
        } else if l.contains("holding the coding-agent queue lock") {
            out.push(Ev::Lock);
        } else if l.contains("stopping llama-server-qwen") {
            out.push(Ev::Stop);
        } else if l.contains("RESTORE: started llama-server-qwen") {
            out.push(Ev::ProdStart);
        } else if l.contains("RESTORE: gateway answer: \"content\":\"") || l.contains("RESTORE: gateway confirmed") {
            out.push(Ev::GatewayOk);
        } else if l.contains("releasing the coding-agent queue lock") {
            out.push(Ev::Release);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use Ev::*;

    #[test]
    fn a_good_window_passes() {
        assert_eq!(check_trace(&[HeadroomOk, Lock, HeadroomOk, Stop, ProdStart, GatewayOk, Release]), Ok(()));
        assert_eq!(check_trace(&[Lock, HeadroomLow, Unlock, Lock, HeadroomOk, Stop, ProdStart, GatewayOk, Release]), Ok(()));
        assert_eq!(check_trace(&[]), Ok(()));
    }

    #[test]
    fn violations_are_caught() {
        assert_eq!(check_trace(&[HeadroomOk, Lock, Stop]), Err(2), "stop without a check after locking");
        assert_eq!(check_trace(&[Lock, HeadroomOk, Stop, ProdStart, Release]), Err(4), "released before the gateway");
        assert_eq!(check_trace(&[Lock, HeadroomOk, Stop, ProdStart]), Err(4), "ended down and locked");
        assert_eq!(check_trace(&[Stop]), Err(0), "stop without the flock");
    }

    #[test]
    fn parses_the_scripts_lines() {
        let log = "01:00 w: headroom avail 21 GB\n01:00 w: waiting for the coding-agent queue lock\n01:00 w: holding the coding-agent queue lock\n\
                   01:00 w: headroom avail 21 GB\n01:00 w: stopping llama-server-qwen (clock starts)\n\
                   01:10 RESTORE: started llama-server-qwen\n01:11 RESTORE: gateway answer: \"content\":\"42\"\n\
                   01:11 w: releasing the coding-agent queue lock\n";
        assert_eq!(parse_log(log), vec![HeadroomOk, Lock, HeadroomOk, Stop, ProdStart, GatewayOk, Release]);
    }
}
