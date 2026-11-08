//! GGUF bounds rules: every count, offset and size read from a (possibly hostile) GGUF file goes through these before
//! it indexes memory or sizes an allocation.
//!
//! Verus proves these in this very file (`verification/lucina-gguf/rules.rs`; the attributes are only seen by Verus).
//! What each promises is in its `ensures`. Format facts follow llama.cpp 3612199ed `ggml/src/gguf.cpp`: at most
//! `GGML_MAX_DIMS` (4) dimensions, `general.alignment` (default 32) for the data section and every tensor offset.
#![allow(unexpected_cfgs)]

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// `GGML_MAX_DIMS`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const MAX_DIMS: usize = 4;

/// The smallest encoding of one metadata pair: key length (8) + key (at least 1) + value type (4) + smallest value (1).
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const MIN_KV_BYTES: u64 = 14;
/// The smallest encoding of one tensor-info record: name length (8) + name (1) + n_dims (4) + one dim (8) + type (4) +
/// offset (8).
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const MIN_TENSOR_INFO_BYTES: u64 = 33;

#[cfg(verus_keep_ghost)]
verus! {

global size_of usize == 8;

/// The product of a sequence of dimensions.
pub open spec fn prod(s: Seq<u64>) -> int
    decreases s.len(),
{
    if s.len() == 0 { 1 } else { prod(s.drop_last()) * s.last() as int }
}

} // verus!

/// `a * b`, or `None` when it does not fit in a u64.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r matches Some(v) ==> v as int == a as int * b as int,
        r is None ==> a as int * b as int > u64::MAX as int,
))]
pub fn mul_u64(a: u64, b: u64) -> Option<u64> {
    if a == 0 {
        #[cfg(verus_keep_ghost)]
        proof! { assert(a as int * b as int == 0) by (nonlinear_arith) requires a == 0; }
        return Some(0);
    }
    if b > u64::MAX / a {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(a as int * b as int > u64::MAX as int) by (nonlinear_arith)
                requires a > 0, b > u64::MAX as int / a as int;
        }
        None
    } else {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(a as int * b as int <= u64::MAX as int) by (nonlinear_arith)
                requires a > 0, b <= u64::MAX as int / a as int;
        }
        Some(a * b)
    }
}

/// The element count of a tensor: the product of its dimensions, refusing more than `MAX_DIMS` of them, a zero
/// dimension count, and any product that overflows.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        // 4 and not the constant, so that changing the constant is refused.
        r matches Some(n) ==> 1 <= dims.len() <= 4 && n as int == prod(dims@),
        dims.len() > MAX_DIMS || dims.len() == 0 ==> r is None,
))]
pub fn numel(dims: &[u64]) -> Option<u64> {
    if dims.len() == 0 || dims.len() > MAX_DIMS {
        return None;
    }
    let mut acc: u64 = 1;
    let mut i: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            i <= dims.len(),
            1 <= dims.len() <= MAX_DIMS,
            acc as int == prod(dims@.subrange(0, i as int)),
        decreases dims.len() - i,
    ))]
    while i < dims.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(dims@.subrange(0, i as int + 1).drop_last() =~= dims@.subrange(0, i as int));
        }
        match mul_u64(acc, dims[i]) {
            Some(v) => acc = v,
            None => return None,
        }
        i += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! { assert(dims@.subrange(0, dims.len() as int) =~= dims@); }
    Some(acc)
}

/// Round `x` up to a multiple of `align`, or `None` past u64.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        align > 0,
    ensures
        r matches Some(v) ==> v >= x && v % align == 0 && v - x < align,
))]
pub fn align_up(x: u64, align: u64) -> Option<u64> {
    let rem = x % align;
    if rem == 0 {
        return Some(x);
    }
    let add = align - rem;
    if x > u64::MAX - add {
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        let k = x as int / align as int;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(x as int, align as int);
        assert(x as int + add as int == (k + 1) * align as int) by (nonlinear_arith)
            requires x as int == align as int * k + rem as int, add as int == align as int - rem as int;
        vstd::arithmetic::div_mod::lemma_mod_multiples_basic(k + 1, align as int);
    }
    Some(x + add)
}

/// The end of `n` bytes read at `pos` of a buffer of `len` bytes, or `None` when they are not all inside it.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    ensures
        r matches Some(end) ==> pos <= end <= len && end as int == pos as int + n as int,
        r is None ==> pos as int + n as int > len as int,
))]
pub fn take_end(pos: usize, n: u64, len: usize) -> Option<usize> {
    if pos > len {
        return None;
    }
    let room = (len - pos) as u64;
    if n > room {
        return None;
    }
    Some(pos + n as usize)
}

/// Whether `count` items of at least `min_bytes` each can still be in the `remaining` bytes of the file. A count that
/// fails this is a lie, and must be refused before anything is allocated for it.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        min_bytes > 0,
    ensures
        r ==> count as int * min_bytes as int <= remaining as int,
        r ==> count as int <= remaining as int,
))]
pub fn count_fits(count: u64, min_bytes: u64, remaining: u64) -> bool {
    let ok = count <= remaining / min_bytes;
    #[cfg(verus_keep_ghost)]
    proof! {
        if ok {
            assert(count as int * min_bytes as int <= remaining as int) by (nonlinear_arith)
                requires min_bytes > 0, count as int <= remaining as int / min_bytes as int;
            assert(count as int <= count as int * min_bytes as int) by (nonlinear_arith)
                requires min_bytes >= 1, count >= 0;
        }
    }
    ok
}

/// Where a tensor's bytes are in the file: `(start, length)`. Refused unless the element count is a whole number of
/// blocks, the offset is aligned, the byte length fits in u64, and the whole range lies inside the file.
#[cfg_attr(verus_keep_ghost, verus_spec(r =>
    requires
        blk_elems > 0,
        align > 0,
    ensures
        r matches Some(sl) ==> numel % blk_elems == 0,
        r matches Some(sl) ==> offset % align == 0,
        r matches Some(sl) ==> sl.0 as int == data_start as int + offset as int,
        r matches Some(sl) ==> sl.1 as int == (numel / blk_elems) as int * blk_bytes as int,
        r matches Some(sl) ==> sl.0 as int + sl.1 as int <= file_len as int,
))]
pub fn tensor_span(
    file_len: u64,
    data_start: u64,
    offset: u64,
    numel: u64,
    blk_elems: u64,
    blk_bytes: u64,
    align: u64,
) -> Option<(u64, u64)> {
    if numel % blk_elems != 0 || offset % align != 0 {
        return None;
    }
    let len = match mul_u64(numel / blk_elems, blk_bytes) {
        Some(v) => v,
        None => return None,
    };
    if data_start > u64::MAX - offset {
        return None;
    }
    let start = data_start + offset;
    if start > file_len || len > file_len - start {
        return None;
    }
    Some((start, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_refuses_overflow_and_too_many_dims() {
        assert_eq!(numel(&[5120, 248320]), Some(5120 * 248320));
        assert_eq!(numel(&[u64::MAX, 2]), None);
        assert_eq!(numel(&[1, 1, 1, 1, 1]), None);
        assert_eq!(numel(&[]), None);
        assert_eq!(numel(&[0, u64::MAX]), Some(0));
    }

    #[test]
    fn tensor_span_keeps_inside_the_file() {
        // a Q4_K tensor of 256 elements = one 144-byte block
        assert_eq!(tensor_span(1000, 64, 32, 256, 256, 144, 32), Some((96, 144)));
        assert_eq!(tensor_span(200, 64, 32, 256, 256, 144, 32), None, "runs past the end");
        assert_eq!(tensor_span(1000, 64, 33, 256, 256, 144, 32), None, "misaligned");
        assert_eq!(tensor_span(1000, 64, 32, 255, 256, 144, 32), None, "not whole blocks");
        assert_eq!(tensor_span(u64::MAX, u64::MAX - 10, 32, 256, 256, 144, 32), None, "start overflows");
        assert_eq!(tensor_span(u64::MAX, 0, 0, u64::MAX - 255, 256, 1 << 20, 32), None, "length overflows");
    }

    #[test]
    fn counts_and_reads() {
        assert!(count_fits(10, 14, 140));
        assert!(!count_fits(11, 14, 140));
        assert!(!count_fits(u64::MAX, 1, 1 << 40));
        assert_eq!(take_end(10, 5, 20), Some(15));
        assert_eq!(take_end(10, 11, 20), None);
        assert_eq!(take_end(21, 0, 20), None);
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(64, 32), Some(64));
        assert_eq!(align_up(u64::MAX, 32), None);
    }
}
