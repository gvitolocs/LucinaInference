//! The checked-in `INFR_*` key table: env name → config path → grammar → `migrated`.
//!
//! This is the authority on what the campaign had to move, because the compiler and the tests
//! read it. It was seeded from this command (the plan doc it was drafted against was deleted at
//! `3010e45`, once the migration landed):
//!
//! ```bash
//! grep -rhoE '"INFR_[A-Z_0-9]*"' crates/*/src crates/*/build.rs | tr -d '"' | sort -u \
//!   | grep -vE '_TEST_|^INFR_(TP|EP|CPU|METAL)$'
//! ```
//!
//! which yields **177** keys at `2dd0c5a`: [`KEYS`]'s 174 plus [`NOT_MIGRATED`]'s 3 — the sizes of
//! THOSE tables at THAT commit, not today's: `config::tests::manifest_matches_the_tree` re-runs
//! that scan, so every `INFR_*` literal added under `crates/*/src` has to join one of the two lists
//! to pass. (A naive
//! `grep 'env::var("INFR_'` found only 153 of them — 205 literal call sites — because ~27 reads
//! went through a helper that took the key name as a `&str`: `budget::env_flag`/`env_mib`/
//! `overflow_vram_reserve`, `pager::ring_bytes_policy`, `tier::EnvRows::get`, `fusion`'s
//! `disable_env`, `seam::parse_device_list`, `recorder::GemvKnobs::resolve` and
//! `recorder::cap_from_env`. Any inventory built from that one grep is wrong by ~15%. All of
//! those wrapper functions are gone now, folded into typed accessors as each crate moved onto
//! `Config`.)
//!
//! Three things read this table:
//! - `config::tests::env_layer_reads_every_key` — set each key through the injected reader
//!   and assert the layer noticed. This is what stops a knob being silently dropped mid-campaign.
//! - `config::tests::bool_knobs_are_value_reads` — the truth table of every `Bool`, `BoolInv`,
//!   `BoolOpt` and `BoolOptInv` entry, which is every switch in the table.
//! - `config::tests::manifest_matches_the_tree` — a new `INFR_*` literal in `crates/*/src`
//!   with no entry here fails the build's tests, so a feature branch cannot re-introduce an
//!   ungoverned knob.
//!
//! `config::tests::bool_knobs_match_the_manifest` is the fourth: it holds those tables against this
//! one in BOTH directions and fails on any row that carries a presence-style SWITCH grammar, so a
//! knob cannot come back reading presence without deleting the check first.
//!
//! A slice flips its knobs' `pending` → `migrated` as it moves their read sites.

/// How an env VALUE maps onto its config field. This is the per-KEY grammar; the per-TYPE
/// spelling used by the TOML file and `--set` lives in [`super::ConfigValue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    /// The `Bool` grammar (`env::boolean`, the ONE read for a plain on/off switch): unset ⇒
    /// the knob keeps its default; `0`/`false`/`off`/`no` ⇒ `false`; `1`/`true`/`on`/`yes` ⇒ `true`
    /// (ASCII case-insensitive); ANYTHING ELSE — the empty value included — is a startup error
    /// naming the variable. Every switch used to be [`Presence`][Grammar::Presence], which read
    /// `INFR_X=0` as ON.
    Bool,
    /// The `BoolInv` grammar (`env::boolean_inv`): a `Bool` switch whose KEY names the DISABLE, so
    /// the value is inverted on the way in. `1`/`true`/`on`/`yes` ⇒ the positive field is `false`;
    /// `0`/`false`/`off`/`no` ⇒ `true`; unset ⇒ the default; anything else ⇒ the same startup error.
    /// `=1` answers what presence answered, so a script that switched something off keeps doing it;
    /// `=0` is the half that used to mean the same thing as `=1`.
    BoolInv,
    /// The `BoolOpt` grammar (`env::boolean_opt`): `Bool` into an `Option<bool>` field, where the
    /// field's own `None` is the third state — "this layer has no opinion; decide from the
    /// hardware". Unset ⇒ `None`; ON spellings ⇒ `Some(true)`; OFF spellings ⇒ `Some(false)`;
    /// anything else ⇒ the same startup error. Replaces `TriState`, which forced ON for any value
    /// that was not exactly `"0"` — including the empty one.
    BoolOpt,
    /// `BoolOptInv` (`env::boolean_opt_inv`): [`BoolOpt`][Grammar::BoolOpt] for a key that names the
    /// DISABLE, so ON spellings give `Some(false)`.
    BoolOptInv,
    /// Set to anything (including empty) ⇒ the field is `true`. Kept ONLY for the `*_specified`
    /// MARKERS of value knobs (`INFR_UBATCH`, `INFR_KV_TYPE_K`/`_V`), which record that a layer
    /// NAMED the key rather than which way it points. A new on/off switch must be
    /// [`Bool`][Grammar::Bool], and `config::tests::bool_knobs_match_the_manifest` fails one that
    /// isn't.
    Presence,
    /// Set ⇒ the `Option<_>` field is cleared to `None` (`INFR_NO_GEMV_REG`). A MARKER on a value
    /// knob — the key carries no boolean; its presence means "drop the variant" — so it is the one
    /// presence-style grammar that survives, listed in `SWITCH_GRAMMARS_ARE_VALUE_READS`'s
    /// companion table below.
    PresenceClears,
    /// An integer.
    Int,
    /// A float.
    Float,
    /// A free string.
    Text,
    /// A filesystem path.
    Path,
    /// The shared size/count grammar (`crate::parse_size`).
    Size,
    /// A MiB count.
    Mib,
    /// A KV format name (`budget::parse_kv_dtype`).
    Dtype,
    /// A multi-GPU device list (`Vulkan0,Vulkan1,…`).
    DeviceList,
    /// Compared against an exact string literal, NOT parsed.
    Literal,
}

impl Grammar {
    /// The four grammars whose VALUE says which way the knob points — every on/off switch in the
    /// table is one of them, and `config::tests::bool_knobs_match_the_manifest` fails a boolean
    /// field reached by anything else.
    pub fn is_switch(self) -> bool {
        matches!(
            self,
            Grammar::Bool | Grammar::BoolInv | Grammar::BoolOpt | Grammar::BoolOptInv
        )
    }
}

/// What a BAD value does in the ENVIRONMENT layer today — which R1 freezes.
///
/// Most keys swallow it (`.and_then(parse).ok().unwrap_or(default)`); five reject it loudly and
/// must keep doing so. Every switch — [`Bool`][Grammar::Bool], [`BoolInv`][Grammar::BoolInv],
/// [`BoolOpt`][Grammar::BoolOpt], [`BoolOptInv`][Grammar::BoolOptInv] — carries `Error` too, by the
/// deliberate exception to R1 that a boolean is never swallowed: reading `INFR_KV_INLINE=0` as ON
/// cost a measurement window, so an unrecognized value stops the run instead. The FILE layer is
/// stricter for everyone: a value that does not parse into a known key's type is always a hard
/// error there, which is why `ctx = "banana"` fails to load from a file while `INFR_CTX=banana` is
/// still silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadValue {
    /// Falls back to the default. The layer reports "not specified".
    Ignored,
    /// Errors out of `ConfigLayer::env()`.
    Error,
}

/// One `INFR_*` knob.
#[derive(Debug, Clone, Copy)]
pub struct KnobKey {
    /// The environment variable. FROZEN — this campaign changes where the value goes, not what
    /// users type (R2).
    pub env: &'static str,
    /// The dotted config path it lands on. Checked against `Config::all_paths()` by a test, so it
    /// cannot drift from the TOML schema or from `--set`'s path set.
    pub path: &'static str,
    /// How the env VALUE maps onto the field.
    pub grammar: Grammar,
    /// What a bad value does in the env layer today.
    pub bad_value: BadValue,
    /// A value that provably makes the env layer specify this knob — used by
    /// `env_layer_reads_every_key` to prove it does.
    pub sample: &'static str,
    /// Has the read site moved onto `Config` yet? Flipped by the slice that migrates it.
    pub migrated: bool,
}

macro_rules! knobs {
    (@mig pending) => { false };
    (@mig migrated) => { true };
    ($($env:literal => $path:literal, $g:ident, $bv:ident, $sample:literal, $mig:ident;)*) => {
        /// Every `INFR_*` knob that becomes configuration. See [`NOT_MIGRATED`] for the rest.
        pub const KEYS: &[KnobKey] = &[$(
            KnobKey {
                env: $env,
                path: $path,
                grammar: Grammar::$g,
                bad_value: BadValue::$bv,
                sample: $sample,
                migrated: knobs!(@mig $mig),
            }
        ),*];
    };
}

knobs! {
    // ── device ───────────────────────────────────────────────────────────────
    "INFR_DEV"                  => "device.dev",               Text, Ignored, "vulkan0", migrated;
    "INFR_CTX"                  => "device.ctx",               Size, Ignored, "32k",     migrated;
    "INFR_UBATCH"               => "device.ubatch",            Int,  Ignored, "512",     migrated;
    "INFR_UBATCH_PARALLEL"      => "device.ubatch_parallel",   Int,  Ignored, "128",     migrated;
    "INFR_SUBMIT_DISPATCHES"    => "device.submit_dispatches", Int,  Error,   "64",      migrated;
    "INFR_SG"                   => "device.subgroup_pref",     Literal, Error, "16",     migrated;

    // ── sampling ─────────────────────────────────────────────────────────────
    "INFR_TEMP"                 => "sampling.temp",       Float,      Ignored, "0.6",  migrated;
    "INFR_TOP_K"                => "sampling.top_k",      Int,        Ignored, "40",   migrated;
    "INFR_TOP_P"                => "sampling.top_p",      Float,      Ignored, "0.9",  migrated;
    "INFR_SEED"                 => "sampling.seed",       Int,        Ignored, "42",   migrated;
    "INFR_MAX_NEW"              => "sampling.max_new",    Int,        Ignored, "256",  migrated;
    "INFR_IGNORE_EOS"           => "sampling.ignore_eos", Bool,       Error,   "1",    migrated;
    "INFR_NO_THINK"             => "sampling.no_think",   Bool,       Error,   "1",    migrated;

    // ── kv ───────────────────────────────────────────────────────────────────
    "INFR_KV_TYPE_K"            => "kv.type_k",              Dtype,       Ignored, "q8_0", migrated;
    "INFR_KV_TYPE_V"            => "kv.type_v",              Dtype,       Ignored, "q8_0", migrated;
    "INFR_KV_Q8"                => "kv.force_q8",            Bool,        Error,   "1",    migrated;
    "INFR_KV_SLOTS"             => "kv.slots",               Int,         Ignored, "8",    migrated;
    "INFR_NO_KV_RING"           => "kv.ring",                BoolInv,     Error,   "1",    migrated;
    "INFR_KV_INLINE"            => "kv.inline_decode",       Bool,        Error,   "1",    migrated;
    "INFR_KV_COOPMAT_BDA"       => "kv.coopmat_bda",         Bool,        Error,   "1",    migrated;
    "INFR_KV_OVERFLOW"          => "kv.overflow",            Bool,        Error,   "1",    migrated;
    "INFR_KV_OVERFLOW_VRAM_MB"  => "kv.overflow_vram_mb",    Mib,         Ignored, "512",  migrated;
    "INFR_KV_OVERFLOW_RESERVE_MB" => "kv.overflow_reserve_mb", Mib,       Ignored, "128",  migrated;

    // ── paging ───────────────────────────────────────────────────────────────
    "INFR_CACHE"                => "paging.cache",                     Size,     Ignored, "8g",  migrated;
    "INFR_PAGER_RING"           => "paging.ring",                      Size,     Ignored, "1g",  migrated;
    "INFR_PAGER_STATS"          => "paging.stats",                     Bool,     Error,   "1",   migrated;
    "INFR_DRAM_CACHE"           => "paging.dram",                      Size,     Ignored, "8g",  migrated;
    "INFR_DRAM_BYPASS"          => "paging.dram_bypass",               Bool,     Error,   "1",   migrated;
    "INFR_LAYER_MAJOR"          => "paging.layer_major",               BoolOpt,  Error,   "1",   migrated;

    // ── kernels.vulkan — coopmat / capability masking ────────────────────────
    "INFR_NO_COOPMAT"   => "kernels.vulkan.coopmat",      BoolInv,     Error,   "1", migrated;
    "INFR_CM_8X8"       => "kernels.vulkan.coopmat_8x8",  Bool,        Error,   "1", migrated;
    "INFR_BF16_COOPMAT" => "kernels.vulkan.bf16_coopmat", Bool,        Error,   "1", migrated;
    "INFR_F8_COOPMAT"   => "kernels.vulkan.f8_coopmat",   Bool,        Error,   "1", migrated;
    "INFR_F8_PREPACK"   => "kernels.vulkan.f8_prepack",   Bool,        Error,   "1", migrated;
    "INFR_I8_COOPMAT"   => "kernels.vulkan.i8_coopmat",   Bool,        Error,   "1", migrated;
    "INFR_I8_ROW_SCALE" => "kernels.vulkan.i8_row_scale", Bool,        Error,   "1", migrated;
    "INFR_NO_F16"       => "kernels.vulkan.f16",          BoolInv,     Error,   "1", migrated;
    "INFR_NO_I8DOT"     => "kernels.vulkan.i8_dot",       BoolInv,     Error,   "1", migrated;

    // ── kernels.vulkan — GEMM / GEMV tiers ───────────────────────────────────
    "INFR_NO_GEMM_WARP"     => "kernels.vulkan.gemm_warp",      BoolInv,     Error,   "1",  migrated;
    "INFR_GEMM_WIDE_TILE"   => "kernels.vulkan.gemm_wide_tile", Bool,        Error,   "1",  migrated;
    "INFR_GEMM_DIRECT_A"    => "kernels.vulkan.gemm_direct_a",  Bool,        Error,   "1",  migrated;
    "INFR_NO_SMALL_BM"      => "kernels.vulkan.small_bm",       BoolInv,     Error,   "1",  migrated;
    "INFR_NO_BM16"          => "kernels.vulkan.bm16",           BoolInv,     Error,   "1",  migrated;
    "INFR_NO_MMQ"           => "kernels.vulkan.mmq",            BoolInv,     Error,   "1",  migrated;
    "INFR_NO_MMQ_FALLBACK"  => "kernels.vulkan.mmq_fallback",   BoolInv,     Error,   "1",  migrated;
    "INFR_NO_MMV"           => "kernels.vulkan.mmv",            BoolInv,     Error,   "1",  migrated;
    "INFR_MMV_DECODE"       => "kernels.vulkan.mmv_decode",     Bool,        Error,   "1",  migrated;
    "INFR_NO_MMV_M4"        => "kernels.vulkan.mmv_m4",         BoolInv,     Error,   "1",  migrated;
    "INFR_NO_MMV_O4"        => "kernels.vulkan.mmv_o4",         BoolInv,     Error,   "1",  migrated;
    "INFR_MMV_MW"           => "kernels.vulkan.mmv_mw",         BoolOpt,     Error,   "0",  migrated;
    "INFR_MMV_MW_WARPS"     => "kernels.vulkan.mmv_mw_warps",   Int,         Ignored, "4",  migrated;
    "INFR_MMV_FUSE_QUANT"   => "kernels.vulkan.mmv_fuse_quant", Bool,        Error,   "1",  migrated;
    "INFR_NO_MROW"          => "kernels.vulkan.mrow",           BoolInv,     Error,   "1",  migrated;
    "INFR_NO_MROW16"        => "kernels.vulkan.mrow16",         BoolInv,     Error,   "1",  migrated;
    "INFR_NO_F32_MROW"      => "kernels.vulkan.f32_mrow",       BoolInv,     Error,   "1",  migrated;
    "INFR_NO_F32_V4"        => "kernels.vulkan.f32_v4",         BoolInv,     Error,   "1",  migrated;
    "INFR_MOE_SMALL_M"      => "kernels.vulkan.moe_small_m",    Int,         Ignored, "16", migrated;
    "INFR_CANVAS_CHUNK_N"   => "kernels.vulkan.canvas_chunk_n", Int,         Ignored, "5",  migrated;

    // ── kernels.vulkan — attention ───────────────────────────────────────────
    "INFR_FLASH_SPLITS"   => "kernels.vulkan.flash_splits",     Int,              Ignored, "2",  migrated;
    "INFR_FLASH_BM"       => "kernels.vulkan.flash_bm32",       Literal,          Ignored, "32", migrated;
    "INFR_FLASH_MIN_ROWS" => "kernels.vulkan.flash_min_rows",   Int,              Ignored, "8",  migrated;
    "INFR_FLASH_STAGE"    => "kernels.vulkan.flash_stage",      Bool,             Error,   "1",  migrated;
    "INFR_FLASH_DEQUANT"  => "kernels.vulkan.flash_dequant",    Bool,             Error,   "1",  migrated;
    "INFR_NO_FLASH_WARP"  => "kernels.vulkan.flash_warp",       BoolInv,          Error,   "1",  migrated;
    "INFR_NO_NC_FA"       => "kernels.vulkan.nc_fa",            BoolInv,          Error,   "1",  migrated;
    "INFR_NC_FA_WIDE"     => "kernels.vulkan.nc_fa_wide",       Bool,             Error,   "1",  migrated;
    "INFR_FLASH_CM1"      => "kernels.vulkan.flash_cm1",        Bool,             Error,   "1",  migrated;
    "INFR_NO_QK_WARP"     => "kernels.vulkan.qk_warp",          BoolInv,          Error,   "1",  migrated;
    "INFR_NO_PV_WARP"     => "kernels.vulkan.pv_warp",          BoolInv,          Error,   "1",  migrated;
    "INFR_PV_SPLITS"      => "kernels.vulkan.pv_splits",        Int,              Ignored, "2",  migrated;
    "INFR_NO_ATTN_HD"     => "kernels.vulkan.no_attn_hd_spec",  Bool,             Error,   "1",  migrated;
    "INFR_NO_ATTN_DECODE" => "kernels.vulkan.attn_decode",      BoolInv,          Error,   "1",  migrated;
    "INFR_NO_MROWS_ATTN"  => "kernels.vulkan.mrows_attn",       BoolOptInv,       Error,   "1",  migrated;
    "INFR_MROWS_ATTN"     => "kernels.vulkan.mrows_attn",       BoolOpt,          Error,   "1",  migrated;
    "INFR_NO_MLA_SG"      => "kernels.vulkan.mla_sg",           BoolInv,          Error,   "1",  migrated;

    // ── kernels.vulkan — DeltaNet + misc ─────────────────────────────────────
    "INFR_DN_CHUNK_SCAN"      => "kernels.vulkan.dn_chunk_scan",      BoolInv,     Error,   "1", migrated;
    "INFR_NO_DN_CHUNK"        => "kernels.vulkan.dn_chunk",           BoolInv,     Error,   "1", migrated;
    "INFR_NO_DN_SPLIT"        => "kernels.vulkan.dn_split",           BoolInv,     Error,   "1", migrated;
    "INFR_DELTA_STRIDED"      => "kernels.vulkan.delta_strided",      Bool,        Error,   "1", migrated;
    "INFR_NO_PUSH_DESC"       => "kernels.vulkan.push_desc",          BoolInv,     Error,   "1", migrated;
    "INFR_NO_PIPELINE_CACHE"  => "kernels.vulkan.pipeline_cache_disk", BoolInv,     Error,   "1", migrated;
    "INFR_NO_VRAM_GUARD"      => "kernels.vulkan.no_vram_guard",      Bool,        Error,   "1", migrated;
    "INFR_NO_MOE_SM_POOL"     => "kernels.vulkan.no_moe_sm_pool",     Bool,        Error,   "1", migrated;
    "INFR_SEAM_NO_REPLAY"     => "kernels.vulkan.no_replay",          Bool,        Error,   "1", migrated;
    "INFR_NO_GPU_POS"         => "kernels.vulkan.gpu_pos",            BoolInv,     Error,   "1", migrated;
    "INFR_NO_FUSE_ADD"        => "kernels.vulkan.fuse_add",           BoolInv,     Error,   "1", migrated;

    // ── kernels.vulkan — BDA chunk caps ──────────────────────────────────────
    "INFR_BDA_CHUNK_ELEMS" => "kernels.vulkan.bda_chunk_elems", Int, Ignored, "1024", migrated;
    "INFR_BDA_CHUNK_BYTES" => "kernels.vulkan.bda_chunk_bytes", Int, Ignored, "4096", migrated;

    // ── kernels.vulkan — resident-BDA weight arena ───────────────────────────
    // The MiB floor of one arena block; the stranded-tail cost of every value of it is measured on
    // the 27B in `infr-vulkan`'s `bda_gguf_matches_the_measured_27b_arena`. `0` is refused (minimum 1).
    "INFR_BDA_BLOCK_MIN_MIB" => "kernels.vulkan.bda_block_min_mib", Int, Ignored, "4", migrated;

    // ── kernels.vulkan.gemv ──────────────────────────────────────────────────
    "INFR_NO_GEMV_RM"     => "kernels.vulkan.gemv.no_rm",     Bool,           Error,   "1",     migrated;
    "INFR_GEMV_RM"        => "kernels.vulkan.gemv.rm",        Int,            Ignored, "4",     migrated;
    "INFR_GEMV_RM_MAXOUT" => "kernels.vulkan.gemv.rm_maxout", Int,            Ignored, "16384", migrated;
    "INFR_GEMV_RM_MINOUT" => "kernels.vulkan.gemv.rm_minout", Int,            Ignored, "1024",  migrated;
    "INFR_NO_GEMV_SG"     => "kernels.vulkan.gemv.no_sg",     Bool,           Error,   "1",     migrated;
    "INFR_NO_GEMV_ID_SG"  => "kernels.vulkan.gemv.no_id_sg",  Bool,           Error,   "1",     migrated;
    "INFR_GEMV_SG_MINOUT" => "kernels.vulkan.gemv.sg_minout", Int,            Ignored, "1024",  migrated;
    "INFR_GEMV_SG_MAXOUT" => "kernels.vulkan.gemv.sg_maxout", Int,            Ignored, "4096",  migrated;
    "INFR_GEMV_SG_NR"     => "kernels.vulkan.gemv.sg_nr",     Int,            Ignored, "4",     migrated;
    "INFR_NO_GEMV_REG"    => "kernels.vulkan.gemv.variant",   PresenceClears, Ignored, "1",     migrated;
    "INFR_GEMV_VARIANT"   => "kernels.vulkan.gemv.variant",   Text,           Ignored, "rm",    migrated;

    // ── kernels.metal ────────────────────────────────────────────────────────
    "INFR_METAL_NO_F16_NATIVE"    => "kernels.metal.f16_native",    BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_F32_NATIVE"    => "kernels.metal.f32_native",    BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_BF16_NATIVE"   => "kernels.metal.bf16_native",   BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_F16_CMM"       => "kernels.metal.f16_cmm",       BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_BF16_CMM"      => "kernels.metal.bf16_cmm",      BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_F32_CMM"       => "kernels.metal.f32_cmm",       BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_F16_RT"        => "kernels.metal.f16_rt",        BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_BF16_RT"       => "kernels.metal.bf16_rt",       BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_F32_RT"        => "kernels.metal.f32_rt",        BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_KQUANT_NATIVE" => "kernels.metal.kquant_native", BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_Q5K_RT"        => "kernels.metal.q5k_rt",        BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_RMSNORM_VEC4"  => "kernels.metal.rmsnorm_vec4",  BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_CONV1D_PAR"    => "kernels.metal.conv1d_par",    BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_DN_GATE_PREP"  => "kernels.metal.dn_gate_prep",  BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NO_DN_NORM_PREP"  => "kernels.metal.dn_norm_prep",  BoolInv,     Error,   "1", migrated;
    "INFR_METAL_LMHEAD_MRV"       => "kernels.metal.lmhead_mrv_uncapped", Bool,     Error,   "1", migrated;
    "INFR_METAL_NODELTA"          => "kernels.metal.deltanet",      BoolInv,     Error,   "1", migrated;
    "INFR_METAL_NOMOE"            => "kernels.metal.moe",           BoolInv,     Error,   "1", migrated;

    // ── kernels.cpu ──────────────────────────────────────────────────────────
    "INFR_CPU_SPIN"        => "kernels.cpu.spin",      Int,           Ignored, "4096", migrated;
    "INFR_CPU_NO_SPINPOOL" => "kernels.cpu.spinpool",  BoolInv,       Error,   "1",    migrated;
    "INFR_CPU_REPACK_MB"   => "kernels.cpu.repack_mb", Int,           Ignored, "1024", migrated;

    // ── kernels — graph shape, `infr-llama` ──────────────────────────────────
    "INFR_NO_QKV_FUSE"      => "kernels.qkv_fuse",      BoolInv,     Error,   "1", migrated;
    "INFR_NO_GATED_RMSNORM" => "kernels.gated_rmsnorm", BoolInv,     Error,   "1", migrated;

    // ── spec ─────────────────────────────────────────────────────────────────
    "INFR_MTP"                => "spec.mtp",             Literal,     Ignored, "1",         migrated;
    "INFR_MTP_MEASURE"        => "spec.mtp_measure",     Literal,     Ignored, "1",         migrated;
    "INFR_NO_MTP_CKPT"        => "spec.mtp_ckpt",        BoolInv,     Error,   "1",         migrated;
    "INFR_NO_MTP_REPRIME"     => "spec.mtp_reprime",     BoolInv,     Error,   "1",         migrated;
    "INFR_NO_MTP_DRAFT_CHAIN" => "spec.mtp_draft_chain", BoolInv,     Error,   "1",         migrated;
    "INFR_SPEC_DRAFT"         => "spec.draft",           Path,        Ignored, "/tmp/d.gguf", migrated;
    "INFR_SPEC_K"             => "spec.k",               Int,         Ignored, "4",         migrated;
    "INFR_SPEC_DEBUG"         => "spec.debug",           Bool,        Error,   "1",         migrated;
    "INFR_DECODE_CHAIN"       => "spec.decode_chain",    Int,         Ignored, "4",         migrated;
    "INFR_NO_GPU_DRAFT_PROB"  => "spec.gpu_draft_prob",  BoolInv,     Error,   "1",         migrated;
    "INFR_NO_GPU_MTP_ACCEPT"  => "spec.gpu_mtp_accept",  BoolInv,     Error,   "1",         migrated;
    "INFR_NO_GPU_ARGMAX"      => "spec.gpu_argmax",      BoolInv,     Error,   "1",         migrated;
    "INFR_NO_GPU_SAMPLE"      => "spec.gpu_sample",      BoolInv,     Error,   "1",         migrated;
    "INFR_NO_GPU_EMBED"       => "spec.gpu_embed",       BoolInv,     Error,   "1",         migrated;

    // ── multi ────────────────────────────────────────────────────────────────
    "INFR_PIPELINE"        => "multi.pipeline",        DeviceList,  Error,   "0,1", migrated;
    "INFR_TENSOR_PARALLEL" => "multi.tensor_parallel", DeviceList,  Error,   "0,1", migrated;
    "INFR_EXPERT_PARALLEL" => "multi.expert_parallel", DeviceList,  Error,   "0",   migrated;
    "INFR_PIPELINE_HOST"   => "multi.pipeline_p2p",    BoolInv,     Error,   "1",   migrated;
    "INFR_TP_HOST"         => "multi.tp_p2p",          BoolInv,     Error,   "1",   migrated;
    "INFR_EP_HOST"         => "multi.ep_p2p",          BoolInv,     Error,   "1",   migrated;

    // ── prof ─────────────────────────────────────────────────────────────────
    // One `INFR_PROF_*` prefix, and each name says what it does. Thirteen keys became eight: the
    // per-op profile had four (INFR_PROF2 / INFR_PROF_OPS / INFR_PROF2_SHAPES / INFR_METAL_PROFILE)
    // and host-side stage timing had five, one per pipeline (INFR_PROF, INFR_PROF_DEC,
    // INFR_PROF_PF, INFR_MTP_TIME, INFR_DIFFUSION_TIME). Old spellings are dropped CLEANLY.
    "INFR_PROF_OPS"                => "prof.ops",               Bool,     Error,   "1",           migrated;
    "INFR_PROF_OP_SHAPES"          => "prof.op_shapes",         Bool,     Error,   "1",           migrated;
    "INFR_PROF_STAGES"             => "prof.stages",            Bool,     Error,   "1",           migrated;
    "INFR_PROF_VRAM"               => "prof.vram",              Bool,     Error,   "1",           migrated;
    "INFR_PROF_DIFFUSION_TRACE"    => "prof.diffusion_trace",   Bool,     Error,   "1",           migrated;
    "INFR_PROF_OUT"                => "prof.out",               Path,     Ignored, "/tmp/p.json", migrated;
    "INFR_PROF_METAL_DEVICE_TIME"  => "prof.metal_device_time", Text,     Ignored, "counters",    migrated;
    "INFR_PROF_METAL_DEBUG"        => "prof.metal_debug",       Bool,     Error,   "1",           migrated;

    // ── debug ────────────────────────────────────────────────────────────────
    "INFR_DEBUG_BDA_CHUNK"     => "debug.bda_chunk",       Bool,     Error,   "1", migrated;
    "INFR_DEBUG_COOPMAT"       => "debug.coopmat",         Bool,     Error,   "1", migrated;
    "INFR_DEBUG_WIDE_DISPATCH" => "debug.wide_dispatch",   Bool,     Error,   "1", migrated;
    "INFR_DEBUG_CHAT"          => "debug.chat",            Bool,     Error,   "1", migrated;
    "INFR_MOE_COUNTS_DEBUG"    => "debug.moe_counts",      Bool,     Error,   "1", migrated;
    "INFR_MOE_COUNTS_DUMP"     => "debug.moe_counts_dump", Bool,     Error,   "1", migrated;
    "INFR_POISON_UNINIT"       => "debug.poison_uninit",   Bool,     Error,   "1", migrated;
    "INFR_NOBARRIER"           => "debug.no_barrier",      Bool,     Error,   "1", migrated;
    "INFR_FULLBARRIER"         => "debug.full_barrier",    Bool,     Error,   "1", migrated;

    // ── serve ────────────────────────────────────────────────────────────────
    "INFR_API_KEY"        => "serve.api_key",        Text, Ignored, "hunter2", migrated;
    "INFR_MAX_TOKENS_CAP" => "serve.max_tokens_cap", Int,  Ignored, "4096",    migrated;
    // `0`/unset = no deadline, so the env layer does NOT filter non-positive values the way
    // `INFR_MAX_TOKENS_CAP` does — see `ServeCfg::request_timeout_secs`.
    "INFR_REQUEST_TIMEOUT_SECS" => "serve.request_timeout_secs", Int, Ignored, "300", migrated;
    // Same "`0` is a VALUE, not a bad one" grammar as the deadline above: `0` disables the periodic
    // throughput line, so the env layer must not filter it out.
    "INFR_SERVE_STATS_SECS" => "serve.stats_interval_secs", Int, Ignored, "10", migrated;

    // ── hub ──────────────────────────────────────────────────────────────────
    // Same "`0` is a VALUE" grammar as the two `serve` knobs above: `0` means "one connection",
    // so the env layer must not filter it out.
    "INFR_PULL_JOBS" => "hub.pull_jobs", Int, Ignored, "4", migrated;
}

/// `INFR_*` keys that exist in `crates/*/src` (or a `build.rs`) and deliberately do NOT become
/// configuration. Listed here so the drift test can tell "excluded on purpose" from
/// "someone added a knob and forgot the manifest".
pub const NOT_MIGRATED: &[(&str, &str)] = &[
    (
        "INFR_PROFILE",
        "build-time input: read by build.rs in core/cpu/gguf/llama/vulkan to set cfg(infr_profile). \
         A runtime Config cannot exist when it is read.",
    ),
    (
        "INFR_LLAMA_DIFFUSION_CLI",
        "test/dev fixture: points at a llama.cpp binary on disk (cli/main.rs).",
    ),
    (
        "INFR_DIFFUSION_VISUAL",
        "CLI presentation only, NOT a Config field: it is `infr run --diffusion-visual`, \
         a plain clap flag whose `env =` fallback keeps the old spelling working. The literal that \
         remains in `cli/main.rs` is that clap attribute, not a `std::env::var` read.",
    ),
    (
        "INFR_PARITY_DISCRETE_LEASE",
        "test harness only: the id of the GPU lock/lease a device test run holds on the discrete \
         card. Read by `infr-testkit`'s `LEASE_VAR` (`lease_or_skip` refuses to open a Vulkan \
         device without it, `kv_inline_q4_parity`'s gate refuses that one card with it absent) and \
         named nowhere else in the tree. The runtime has no read site — no process that owns a \
         `Config` is deciding whether it may touch a card — so it must not become a Config field.",
    ),
];

/// `INFR_*` spellings that the derivation command's filter (module docs, above) drops and that
/// are NOT knobs at all.
///
/// - `INFR_*_TEST_*` — `#[cfg(test)]` fixtures that exist only to prove a wrapper reads its own
///   variable; they vanish with the wrapper.
/// - `INFR_TP` / `INFR_EP` — `label` strings passed to `parse_device_spec` for error messages, not
///   variables. The real keys are `INFR_TENSOR_PARALLEL` / `INFR_EXPERT_PARALLEL`.
/// - `INFR_CPU` / `INFR_METAL` — DEAD. Removed as backend selectors; the only references left are
///   `cli/main.rs`'s `legacy_metal_cpu_flags_are_no_longer_read` regression test and doc comments.
///   Do not add them to `Config`; do not remove that test.
pub const NOT_KNOBS: &[&str] = &["INFR_TP", "INFR_EP", "INFR_CPU", "INFR_METAL"];
