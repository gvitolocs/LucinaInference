#!/usr/bin/env bash
# Shows that verification/lucina-gate/rules.rs can fail. Run from the repo root on the Verus host (see scripts/local.env.example):
#   verus-remote <repo> bash verification/lucina-gate/mutants.sh
ENTRY=lucina-gate
source verification/mutant-lib.sh
F=gate_rules.rs
# Stop without headroom after locking, or without the flock (in the model, then in the shipped step).
try $F 'Ev::Stop => if g.locked && g.checked && !g.down {' 'Ev::Stop => if g.locked && !g.down {'
try $F '            if g.locked && g.checked && !g.down {
                Some(Gate { down: true, prod_started: false, ..g })' '            if g.checked && !g.down {
                Some(Gate { down: true, prod_started: false, ..g })'
# Release while production is still down (model, then step).
try $F 'Ev::Release => if g.locked && !g.down {' 'Ev::Release => if g.locked {'
try $F 'Ev::Unlock => if g.locked && !g.down {' 'Ev::Unlock => if g.locked {'
# Production counted back without being started.
try $F 'Ev::GatewayOk => if g.down { if g.prod_started {' 'Ev::GatewayOk => if g.down { if true {'
# A headroom check before locking counted as one after it.
try $F 'Ev::HeadroomOk => Some(Gate { checked: g.locked, ..g }),' 'Ev::HeadroomOk => Some(Gate { checked: true, ..g }),'
# The trace check accepting a window that ended locked.
try $F '    if g.locked || g.down {' '    if g.down {'
finish
