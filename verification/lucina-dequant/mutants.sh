#!/usr/bin/env bash
# Shows that verification/lucina-dequant/rules.rs can fail. Run from the repo root on verus-host:
#   verus-remote <repo> bash verification/lucina-dequant/mutants.sh
ENTRY=lucina-dequant
source verification/mutant-lib.sh
F=dequant_index.rs
# Layout: a wrong block size.
try $F "        12 => Some((QK_K, Q4_K_BYTES))," "        12 => Some((QK_K, Q5_K_BYTES)),"
# get_scale_min_k4: a missing 6-bit mask, the wrong neighbour byte.
try $F "        (x & 63, y & 63)" "        (x & 127, y & 63)"
try $F "        let (x, y) = (b[off + j], b[off + j + 4]);" "        let (x, y) = (b[off + j], b[off + j + 3]);"
# Q8_0: reading the scale bytes as quants.
try $F "    b[2 + i] as i8 as i32" "    b[1 + i] as i8 as i32"
# Q4_K: quants start one byte late, the high nibble not shifted down.
try $F "    let qs_off = 16 + 32 * t + r % 32;" "    let qs_off = 17 + 32 * t + r % 32;"
try $F "    let quant = if r < 32 { q & 0xF } else { q >> 4 };
    K4Parts { quant, scale, min, qs_off, sub }" "    let quant = if r < 32 { q & 0xF } else { q >> 3 };
    K4Parts { quant, scale, min, qs_off, sub }"
# Q5_K: high bits read from the quant bytes.
try $F "    let qh_off = 16 + l;" "    let qh_off = 48 + l;"
# Q6_K: the second group's quants not moved past the first 32 bytes, scales one group off.
try $F "    let ql_off = 64 * h + l + if g % 2 == 1 { 32 } else { 0 };" "    let ql_off = 64 * h + l;"
try $F "    let sc_off = 192 + 8 * h + l / 16 + 2 * g;" "    let sc_off = 192 + 8 * h + l / 16 + g;"
try $F "    let v: u8 = low | (((hb >> sh) & 3) << 4);" "    let v: u8 = low | (((hb >> sh) & 7) << 4);"
# Q3_K: the high-mask bit not advancing per half, the hmask correction too big.
try $F "    let hm_bit = (4 * n + j) as u8;" "    let hm_bit = j as u8;"
try $F "    let adj: i32 = if b[l] & m != 0 { 0 } else { 4 };" "    let adj: i32 = if b[l] & m != 0 { 0 } else { 5 };"
# IQ4_NL: ggml's nibble order replaced by even/odd interleaving (the bug the first draft had).
try $F "    let off = 2 + i % 16;" "    let off = 2 + i / 2;"
# IQ4_XS: the sub-block offset dropped (the other first-draft bug), the 2 high scale bits not masked.
try $F "    let qs_off = 8 + 16 * ib + j % 16;" "    let qs_off = 8 + j % 16;"
try $F "    let ls = ((sl >> s1) & 0xF) | (((sh >> s2) & 3) << 4);" "    let ls = ((sl >> s1) & 0xF) | ((sh >> s2) << 4);"
# IQ3_S: the wrong qh bit, signs of the wrong half, the grid's high bit at 512.
try $F "    let qh_bit = (2 * l + g2) as u8;" "    let qh_bit = (2 * l) as u8;"
try $F "    let sign_off = 74 + 4 * k + l;" "    let sign_off = 74 + 4 * k;"
try $F "    let high: u32 = if (qh >> qh_bit) & 1 != 0 { 256 } else { 0 };" "    let high: u32 = if (qh >> qh_bit) & 1 != 0 { 512 } else { 0 };"
finish
