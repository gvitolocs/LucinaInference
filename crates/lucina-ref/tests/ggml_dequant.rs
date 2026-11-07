//! Bit-exactness check of `lucina_ref::quant::dequant_row` against ggml itself: `work/bin/ggml-dequant`
//! (built from the pinned llama.cpp) writes ggml's own dequantized rows as f32 LE, and every value must
//! match `to_bits()`. Each test self-skips when the model file or the oracle binary is missing.
use std::path::{Path, PathBuf};
use std::process::Command;

const ROWS: usize = 8;

const M27: &str = "~/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf";
const M9: &str = "~/models/qwen3.5-9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf";

fn oracle_bin() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../work/bin/ggml-dequant")
}

/// Dequantize `ROWS` rows of `tensor` in `model` with both ggml and the reference and require
/// bit-identical f32 output.
fn check(model: &str, tensor: &str) {
    let bin = oracle_bin();
    let model_path = Path::new(model);
    if !model_path.is_file() || !bin.is_file() {
        eprintln!("skip: {} or {} is missing", model_path.display(), bin.display());
        return;
    }

    let g = lucina_ref::gguf::Gguf::open(model_path).expect("open gguf");
    let info = g.info(tensor).expect("tensor in gguf");
    let ne0 = info.dims[0] as usize;
    let (blk, sz) = lucina_ref::quant::block_layout(info.ggml_type).expect("block layout");
    let row_bytes = ne0 / blk * sz;
    let bytes = g.bytes(info).expect("tensor bytes");

    // ggml's own rows, 8 rows x ne0 f32 LE
    let out = std::env::temp_dir().join(format!(
        "lucina-ggml-dequant-{}-{}.f32",
        std::process::id(),
        tensor.replace(['.', '/'], "_")
    ));
    let status = Command::new(&bin)
        .arg(model_path)
        .arg(tensor)
        .arg(ROWS.to_string())
        .arg(&out)
        .status()
        .expect("run ggml-dequant");
    assert!(status.success(), "ggml-dequant exited with {status}");
    let raw = std::fs::read(&out).expect("read ggml-dequant output");
    let _ = std::fs::remove_file(&out);
    assert_eq!(raw.len(), ROWS * ne0 * 4, "ggml-dequant output size for {tensor}");
    let mut expected = vec![0f32; ROWS * ne0];
    for (i, c) in raw.chunks_exact(4).enumerate() {
        expected[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }

    // the same rows through the reference
    let mut actual = vec![0f32; ROWS * ne0];
    for r in 0..ROWS {
        lucina_ref::quant::dequant_row(
            info.ggml_type,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &mut actual[r * ne0..(r + 1) * ne0],
        )
        .expect("dequant_row");
    }

    for i in 0..expected.len() {
        if expected[i].to_bits() != actual[i].to_bits() {
            panic!(
                "{tensor} row {} element {}: ggml {} ({:#010x}) != ref {} ({:#010x})",
                i / ne0,
                i % ne0,
                expected[i],
                expected[i].to_bits(),
                actual[i],
                actual[i].to_bits()
            );
        }
    }
    eprintln!("ok: {tensor} ({} rows x {ne0} values bit-identical)", ROWS);
}

#[test]
fn q3_k_blk_0_ffn_up() {
    check(M27, "blk.0.ffn_up.weight");
}

#[test]
fn iq4_nl_blk_1_ffn_down() {
    check(M27, "blk.1.ffn_down.weight");
}

#[test]
fn iq3_s_blk_14_ffn_down() {
    check(M27, "blk.14.ffn_down.weight");
}

#[test]
fn iq4_xs_blk_0_ffn_down() {
    check(M27, "blk.0.ffn_down.weight");
}

#[test]
fn q4_k_token_embd() {
    check(M27, "token_embd.weight");
}

#[test]
fn q6_k_output() {
    check(M27, "output.weight");
}

#[test]
fn q5_k_blk_3_attn_k() {
    check(M9, "blk.3.attn_k.weight");
}

#[test]
fn q8_0_blk_0_ssm_out() {
    check(M9, "blk.0.ssm_out.weight");
}
