//! Hostile GGUF files against lucina-ref's reader (bounds proved in lucina-rules::gguf_bounds): every damaged file
//! must come back as an error, never a panic, an out-of-bounds slice or a huge allocation.
use lucina_ref::gguf::Gguf;
use std::path::PathBuf;

fn s(b: &mut Vec<u8>, x: &str) {
    b.extend_from_slice(&(x.len() as u64).to_le_bytes());
    b.extend_from_slice(x.as_bytes());
}

/// A valid file: one string pair (general.architecture = "qwen35"), one 256-element Q4_K tensor at offset 0.
/// Returns (bytes, position of n_kv, position of the tensor's n_dims, position of its offset, position of the string length).
fn valid() -> (Vec<u8>, usize, usize, usize, usize) {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
    let kv_pos = b.len();
    b.extend_from_slice(&1u64.to_le_bytes()); // n_kv
    s(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    let str_pos = b.len();
    s(&mut b, "qwen35");
    s(&mut b, "t");
    let nd_pos = b.len();
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&256u64.to_le_bytes());
    b.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
    let off_pos = b.len();
    b.extend_from_slice(&0u64.to_le_bytes());
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend_from_slice(&[0x11u8; 144]);
    (b, kv_pos, nd_pos, off_pos, str_pos)
}

fn tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lucina-hostile-{}-{name}.gguf", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

/// Open and read every tensor; Ok(()) only if both succeed. Panics if the reader panics.
fn open_and_read(name: &str, bytes: &[u8]) -> Result<(), String> {
    let p = tmp(name, bytes);
    let r = std::panic::catch_unwind(|| -> Result<(), String> {
        let g = Gguf::open(&p).map_err(|e| e.to_string())?;
        for t in g.tensors.values() {
            g.bytes(t).map_err(|e| e.to_string())?;
        }
        Ok(())
    });
    let _ = std::fs::remove_file(&p);
    match r {
        Ok(v) => v,
        Err(_) => panic!("{name}: the reader PANICKED on a hostile file"),
    }
}

fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

#[test]
fn the_valid_file_opens() {
    let (b, ..) = valid();
    open_and_read("valid", &b).unwrap();
}

#[test]
fn lying_counts_are_refused_before_allocating() {
    let (b, kv_pos, ..) = valid();
    let mut x = b.clone();
    put_u64(&mut x, kv_pos, u64::MAX);
    assert!(open_and_read("nkv", &x).is_err());
    let mut x = b.clone();
    put_u64(&mut x, kv_pos - 8, 1 << 60);
    assert!(open_and_read("ntensors", &x).is_err());
}

#[test]
fn a_huge_string_length_is_refused() {
    let (b, _, _, _, str_pos) = valid();
    let mut x = b.clone();
    put_u64(&mut x, str_pos, u64::MAX - 3);
    assert!(open_and_read("strlen", &x).is_err());
}

#[test]
fn too_many_dimensions_out_of_file_and_misaligned_offsets_are_refused() {
    let (b, _, nd_pos, off_pos, _) = valid();
    let mut x = b.clone();
    x[nd_pos..nd_pos + 4].copy_from_slice(&5u32.to_le_bytes());
    assert!(open_and_read("ndims", &x).is_err());
    let mut x = b.clone();
    put_u64(&mut x, off_pos, 1 << 40);
    assert!(open_and_read("offset-far", &x).is_err());
    let mut x = b.clone();
    put_u64(&mut x, off_pos, 32);
    assert!(open_and_read("offset-past-end", &x).is_err(), "the 144 bytes would end past the file");
    let mut x = b.clone();
    put_u64(&mut x, off_pos, 1);
    assert!(open_and_read("offset-misaligned", &x).is_err());
}

#[test]
fn every_truncation_is_an_error() {
    let (b, ..) = valid();
    for n in 0..b.len() {
        assert!(open_and_read(&format!("trunc{n}"), &b[..n]).is_err(), "truncated at {n} was accepted");
    }
}

#[test]
fn random_damage_never_panics() {
    let (b, ..) = valid();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..3000 {
        let mut x = b.clone();
        for _ in 0..(1 + next() % 4) {
            let at = (next() % x.len() as u64) as usize;
            x[at] = next() as u8;
        }
        let _ = open_and_read(&format!("rand{i}"), &x); // Ok or Err, but never a panic
    }
}
