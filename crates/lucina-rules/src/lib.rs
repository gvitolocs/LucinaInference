//! Pure rules proved by Verus in the shipped files (see verification/lucina-*/rules.rs). No dependencies.
#![allow(unexpected_cfgs)]
pub mod gguf_bounds;
pub mod kv_rules;
pub mod split_rules;
pub mod gate_rules;
pub mod spec_rules;
pub mod dequant_index;
