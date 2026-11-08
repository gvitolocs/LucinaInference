#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
// Verus proof of the SHIPPED code, not a model: this file only names the source files.
#[path = "../../crates/lucina-rules/src/gguf_bounds.rs"]
pub mod gguf_bounds;
#[path = "../../crates/lucina-rules/src/kv_rules.rs"]
pub mod kv_rules;

fn main() {}
