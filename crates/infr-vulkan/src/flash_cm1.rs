//! Host side of the coopmat-1 flash-attention prefill at head_dim 256 (`shaders/attn_flash_cm1.comp`):
//! the shared-memory budget that decides whether the kernel fits the device, the gate the routing
//! asks about, and the split-K plan. Pure arithmetic, no device: the adapter gates on
//! [`admitted`], the recorder and the adapter's scratch sizing both read [`split_plan`].
//!
//! Translated from llama.cpp v1.0.0 (b1b770d, MIT licence, Copyright (c) 2023-2026 The ggml
//! authors), `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: `get_fa_tuning_params_coopmat1` (:1290-1318),
//! `ggml_vk_flash_attn_coopmat_shmem_support` (:8364-8410) and the split_k choice in
//! `ggml_vk_flash_attn` (:8698-8742).

/// Query rows per workgroup (`coopmat_block_rows`, ggml-vulkan.cpp:1300 at b1b770d).
pub const BR: u32 = 16;
/// Kv columns per block (`coopmat_block_cols * num_subgroups`, :1306).
pub const BC: u32 = 64;
/// Subgroups per workgroup (`num_subgroups`, :1303), each owning `BR / ROW_SPLIT` query rows.
pub const ROW_SPLIT: u32 = 4;
/// The one head dim the shader is built for (HSK == HSV).
pub const HD: u32 = 256;

/// Shared-memory bytes the cm1 flash tile needs at head dims `hsk`/`hsv`, with S/O accumulated in
/// f32 (`acc_f32`) or f16, mirroring `ggml_vk_flash_attn_coopmat_shmem_support` term by term at
/// `shmem_staging == 0`: :1316 turns staging on only for NVIDIA with hsk < 256 AND hsv < 256, so no
/// device gets it at this head dim.
///
/// Deviations from the upstream sum, each because infr's shader does not declare the array:
/// no `tmpsh` (upstream's mask-skip vote), no `iq_shmem` (the IQ4_NL table; infr builds one KV
/// format per pipeline), no `slope` (ALiBi). And `pvsh` is sized by the accumulator type — upstream
/// counts it as f16vec4 for every non-BF16 build; infr's O accumulator is f32, so its `pvsh` is
/// vec4 (16 B per element). [`upstream_shmem_bytes`] keeps the literal upstream sum for comparison.
pub fn shmem_bytes(hsk: u32, hsv: u32, acc_f32: bool) -> u32 {
    const MAT_BR: u32 = 16;
    const MAT_BC: u32 = 16;
    let row_split = BC / MAT_BC;
    let hsk_pad = hsk.next_multiple_of(16);
    let hsv_pad = hsv.next_multiple_of(16);
    let acctype = if acc_f32 { 4 } else { 2 };
    let f16vec4 = 8;

    let qstride = hsk_pad / 4 + 2;
    let qf = BR * qstride * f16vec4;
    let psh_stride = BR / 4 + 2;
    let psh = BC * psh_stride * f16vec4;
    // flash_attn_cm1.comp:50-52: no padding for hsk > 128 ("to make it fit in 48KB shmem").
    let sfshstride = if hsk <= 128 { BR + 8 } else { BR };
    let sfsh = BC * sfshstride * acctype;
    let kvshstride = MAT_BR / 4 + 2;
    let vsh_stride = MAT_BC / 4 * row_split;
    let ksh = BC * kvshstride.max(vsh_stride) * f16vec4;
    let osh_stride = ROW_SPLIT * MAT_BR / 4;
    let pvsh = MAT_BC * osh_stride * 4 * acctype;
    let _ = hsv_pad; // only feeds kvshstride under shmem_staging, which is 0 here
    qf + psh + sfsh + ksh + pvsh
}

/// The literal upstream `total_size` of `ggml_vk_flash_attn_coopmat_shmem_support` (b1b770d
/// :8364-8410) for a non-BF16 K type, `shmem_staging` as given. Reference only — the gate uses
/// [`shmem_bytes`], which is what infr's shader declares.
pub fn upstream_shmem_bytes(hsk: u32, hsv: u32, f32acc: bool, shmem_staging: bool) -> u32 {
    let (br, bc) = (BR, BC);
    let (mat_br, mat_bc) = (16u32, 16u32);
    let row_split = bc / mat_bc;
    let hsk_pad = hsk.next_multiple_of(16);
    let hsv_pad = hsv.next_multiple_of(16);
    let acctype = if f32acc { 4 } else { 2 };
    let f16vec4 = 8;
    let tmpsh = (bc / mat_bc) * 4;
    let iq_shmem = 16 * 2;
    let qf = br * (hsk_pad / 4 + 2) * f16vec4;
    let psh = bc * (br / 4 + 2) * f16vec4;
    let sfsh = bc * (if hsk <= 128 { br + 8 } else { br }) * acctype;
    let kvshstride = (if shmem_staging {
        hsk_pad.max(hsv_pad)
    } else {
        mat_br
    }) / 4
        + 2;
    let vsh_stride = mat_bc / 4 * row_split;
    let ksh = bc * kvshstride.max(vsh_stride) * f16vec4;
    let pvsh = mat_bc * (ROW_SPLIT * mat_br / 4) * f16vec4;
    let slope = br * acctype;
    tmpsh + iq_shmem + qf + psh + sfsh + ksh + pvsh + slope
}

/// Whether the cm1 tile fits a device whose `maxComputeSharedMemorySize` is `max_shared` — the
/// comparison at the end of `ggml_vk_flash_attn_coopmat_shmem_support`. infr's build is f32
/// accumulation at `HD`, so that is what is checked. A `false` here must route the shape away
/// from the kernel (pipeline creation would fail, or the dispatch would fault).
pub fn shmem_support(max_shared: u32) -> bool {
    shmem_bytes(HD, HD, true) <= max_shared
}

/// Whether an hd-`hd` prefill may route to this kernel: the shader exists only at [`HD`], the
/// `kernels.vulkan.flash_cm1` knob must be on (default off — slower prefill, see its doc), and the
/// tile must fit the device. The adapter's own hd term is
/// `adapter::flash_hd_admitted`, which ORs hd 128 (a different kernel) onto this; kept separate so
/// the routing wiring is testable with no device.
pub fn admitted(hd: usize, enabled: bool, max_shared: u32) -> bool {
    hd == HD as usize && enabled && shmem_support(max_shared)
}

/// How a cm1 prefill splits its kv range across workgroups (`gl_WorkGroupID.y`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitPlan {
    /// 1 = one pass writing the normalized output; >1 = partials merged by `attn_flash_combine`.
    pub n_splits: u32,
    /// Kv columns per split, a multiple of [`BC`] (the last split may be short).
    pub ksplit: u32,
}

impl SplitPlan {
    /// Bytes of the `(po, pm, pl)` partial scratch the plan needs for `mpad` rows, or 0 at one split.
    pub fn partial_bytes(&self, mpad: usize, nh: usize, hd: usize) -> (usize, usize, usize) {
        if self.n_splits <= 1 {
            return (0, 0, 0);
        }
        let s = self.n_splits as usize;
        (s * mpad * nh * hd * 4, s * mpad * nh * 4, s * mpad * nh * 4)
    }
}

/// The split-K choice of `ggml_vk_flash_attn` (b1b770d :8698-8742) for a prefill: no GQA folding
/// (upstream folds only when `N <= 8` rows), so `total = ceil(rows / BR) * nh` workgroups; split
/// only when that is below twice the shader-core count, `split_k = 2 * cores / total`, then the
/// split length is rounded up to a whole `BC` block and the count recomputed from it.
/// `shader_cores == 0` (unknown) uses upstream's placeholder 16. `forced` is `INFR_FLASH_SPLITS`
/// (`kernels.vulkan.flash_splits`), clamped to the number of kv blocks.
pub fn split_plan(
    rows: usize,
    kv_len: usize,
    nh: usize,
    shader_cores: u32,
    forced: Option<u32>,
) -> SplitPlan {
    let kv = kv_len.max(1) as u32;
    let cores = if shader_cores == 0 { 16 } else { shader_cores };
    let total = (rows.div_ceil(BR as usize) * nh).max(1) as u32;
    let mut split_k = match forced {
        Some(f) => f.max(1),
        None if total < cores * 2 => cores * 2 / total,
        None => 1,
    };
    split_k = split_k.min(kv.div_ceil(BC));
    if split_k > 1 {
        let ksplit = (kv / split_k).max(1).next_multiple_of(BC);
        split_k = kv.div_ceil(ksplit);
        if split_k > 1 {
            return SplitPlan {
                n_splits: split_k,
                ksplit,
            };
        }
    }
    SplitPlan {
        n_splits: 1,
        ksplit: kv.next_multiple_of(BC),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RDNA3 (RX 7900 XTX, RADV) reports 64 KiB `maxComputeSharedMemorySize` and 96 compute units.
    const RDNA3_SHARED: u32 = 65536;
    const RDNA3_CUS: u32 = 96;

    #[test]
    fn shmem_budget_at_head_dim_256() {
        // Qf 16*(64+2)*8 = 8448, Psh 64*6*8 = 3072, sfsh 64*16*4 = 4096 (unpadded, hsk > 128),
        // kvsh 64*max(6,16)*8 = 8192, pvsh 16*16*16 = 4096 (f32 O).
        assert_eq!(
            shmem_bytes(256, 256, true),
            8448 + 3072 + 4096 + 8192 + 4096
        );
        assert_eq!(shmem_bytes(256, 256, true), 27904);
        // f16 accumulation halves sfsh and pvsh.
        assert_eq!(
            shmem_bytes(256, 256, false),
            8448 + 3072 + 2048 + 8192 + 2048
        );
        // hsk <= 128 keeps the +2-vec4 padding on sfsh (Br + 8 scalars per column).
        assert_eq!(
            shmem_bytes(128, 128, true),
            16 * 34 * 8 + 3072 + 64 * 24 * 4 + 8192 + 4096
        );

        // The upstream sum at hd 256: 16 + 32 + 8448 + 3072 + 4096 + 8192 + 2048 + 64 (f32acc).
        assert_eq!(upstream_shmem_bytes(256, 256, true, false), 25968);
        // The staging tile (kvsh = Bc x max(hsk, hsv)) is what :1316 keeps off this head dim: it is
        // enabled only for NVIDIA with hsk < 256 AND hsv < 256, so 0 everywhere at 256 — which is
        // also the configuration infr ports. It would not fit a 48 KiB NVIDIA limit either.
        assert_eq!(
            upstream_shmem_bytes(256, 256, true, true),
            25968 - 8192 + 64 * 66 * 8
        );
        assert!(upstream_shmem_bytes(256, 256, true, true) > 48 * 1024);
    }

    #[test]
    fn shmem_support_against_device_limits() {
        assert!(shmem_support(RDNA3_SHARED), "RDNA3 64 KiB fits");
        assert!(shmem_support(48 * 1024), "NVIDIA 48 KiB fits");
        assert!(shmem_support(32 * 1024), "MoltenVK 32 KiB fits");
        assert!(shmem_support(27904));
        assert!(!shmem_support(27903), "one byte short is refused");
        assert!(!shmem_support(16 * 1024), "a 16 KiB device is refused");
    }

    /// The gate the routing asks about, on the three things it reads. `adapter::flash_hd_admitted`
    /// turns a `false` here into the hd-256 old route, and is tested with the tier it produces.
    #[test]
    fn cm1_is_admitted_only_by_dim_knob_and_fit() {
        // hd 256, knob on, device that fits: the only admitting combination.
        assert!(admitted(256, true, RDNA3_SHARED));
        assert!(admitted(256, true, 27904), "exactly the tile size fits");
        // Knob off — the SHIPPED DEFAULT — refuses however big the device is.
        assert!(!admitted(256, false, RDNA3_SHARED));
        assert!(!admitted(256, false, u32::MAX));
        // One byte short of the tile refuses.
        assert!(!admitted(256, true, 27903));
        // No other head dim: the shader is built at HD only, and hd 128 in particular must never be
        // routed here (it has its own flash kernel; see the adapter's hd term).
        for hd in [4, 64, 128, 192, 255, 257, 512] {
            assert!(!admitted(hd, true, RDNA3_SHARED), "hd {hd} is not cm1's");
            assert!(!admitted(hd, false, RDNA3_SHARED), "hd {hd} is not cm1's");
        }
    }

    #[test]
    fn split_plan_matches_upstream_rule() {
        // Qwen3.8-27B prefill chunk: rows 256, nh 24 → 16 * 24 = 384 workgroups >= 2 * 96 → no split.
        let p = split_plan(256, 32768, 24, RDNA3_CUS, None);
        assert_eq!(
            p,
            SplitPlan {
                n_splits: 1,
                ksplit: 32768
            }
        );
        assert_eq!(p.partial_bytes(256, 24, 256), (0, 0, 0));
        let p = split_plan(256, 4096, 24, RDNA3_CUS, None);
        assert_eq!(p.n_splits, 1);

        // rows 64: 4 * 24 = 96 < 192 → split_k = 2, split_kv = roundup(16384, 64), 2 splits.
        let p = split_plan(64, 32768, 24, RDNA3_CUS, None);
        assert_eq!(
            p,
            SplitPlan {
                n_splits: 2,
                ksplit: 16384
            }
        );
        assert_eq!(
            p.partial_bytes(64, 24, 256),
            (2 * 64 * 24 * 256 * 4, 2 * 64 * 24 * 4, 2 * 64 * 24 * 4)
        );

        // rows 24 at kv 8200: 2 * 24 = 48 → split_k = 4, split_kv = roundup(2050, 64) = 2112 →
        // ceil(8200 / 2112) = 4.
        let p = split_plan(24, 8200, 24, RDNA3_CUS, None);
        assert_eq!(
            p,
            SplitPlan {
                n_splits: 4,
                ksplit: 2112
            }
        );

        // Splits never outnumber kv blocks: 100 keys are 2 blocks.
        let p = split_plan(16, 100, 1, RDNA3_CUS, None);
        assert_eq!(
            p,
            SplitPlan {
                n_splits: 2,
                ksplit: 64
            }
        );
        let p = split_plan(16, 64, 1, RDNA3_CUS, None);
        assert_eq!(
            p,
            SplitPlan {
                n_splits: 1,
                ksplit: 64
            }
        );

        // Unknown core count → upstream's placeholder 16: 384 >= 32 → no split.
        assert_eq!(split_plan(256, 32768, 24, 0, None).n_splits, 1);
        // INFR_FLASH_SPLITS overrides the heuristic and is kept to a whole-block split length.
        let p = split_plan(256, 32768, 24, RDNA3_CUS, Some(4));
        assert_eq!(
            p,
            SplitPlan {
                n_splits: 4,
                ksplit: 8192
            }
        );
        assert_eq!(split_plan(256, 32768, 24, RDNA3_CUS, Some(1)).n_splits, 1);
        // The plan always covers the kv range.
        for (rows, kv, nh) in [
            (64, 32769, 24),
            (24, 8191, 24),
            (16, 1, 8),
            (100, 70_144, 24),
        ] {
            let p = split_plan(rows, kv, nh, RDNA3_CUS, None);
            assert!(p.ksplit.is_multiple_of(BC));
            assert!(
                (p.n_splits * p.ksplit) as usize >= kv,
                "{rows} {kv} {nh}: {p:?}"
            );
            assert!(
                ((p.n_splits - 1) * p.ksplit) < kv as u32,
                "no empty split: {p:?}"
            );
        }
    }
}
