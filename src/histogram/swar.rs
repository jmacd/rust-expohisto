// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! SWAR (SIMD Within A Register) — per-word parallel pairwise summation.
//!
//! [`widen_into`] widens SWAR lanes in a single u64 word from one
//! [`Width`] to another by chaining single-step pair-sum operations.
//! Each step sums adjacent lane pairs, doubling the lane width.

use super::width::Width;

// ── Single-step SWAR primitives ──────────────────────────────────────

/// B1 → B2: sum pairs of 1-bit lanes.
#[inline(always)]
fn step_b1_b2(w: u64) -> u64 {
    (w & 0x5555_5555_5555_5555) + ((w >> 1) & 0x5555_5555_5555_5555)
}

/// B2 → B4: sum pairs of 2-bit lanes.
#[inline(always)]
fn step_b2_b4(w: u64) -> u64 {
    (w & 0x3333_3333_3333_3333) + ((w >> 2) & 0x3333_3333_3333_3333)
}

/// B4 → U8: sum pairs of 4-bit lanes.
#[inline(always)]
fn step_b4_u8(w: u64) -> u64 {
    (w & 0x0F0F_0F0F_0F0F_0F0F) + ((w >> 4) & 0x0F0F_0F0F_0F0F_0F0F)
}

/// U8 → U16: sum pairs of 8-bit lanes.
#[inline(always)]
fn step_u8_u16(w: u64) -> u64 {
    (w & 0x00FF_00FF_00FF_00FF) + ((w >> 8) & 0x00FF_00FF_00FF_00FF)
}

/// U16 → U32: sum pairs of 16-bit lanes.
#[inline(always)]
fn step_u16_u32(w: u64) -> u64 {
    (w & 0x0000_FFFF_0000_FFFF) + ((w >> 16) & 0x0000_FFFF_0000_FFFF)
}

/// U32 → U64: sum pair of 32-bit lanes.
#[inline(always)]
fn step_u32_u64(w: u64) -> u64 {
    (w & 0x0000_0000_FFFF_FFFF) + (w >> 32)
}

// ── Public API ───────────────────────────────────────────────────────

/// Widen a single u64 word in place from `before` lane width to
/// `after` lane width by chaining SWAR pair-sum steps.
/// Uses count_ones() shortcuts for B1→U32 and B1→U64.
#[inline]
pub(crate) fn widen_into(before: Width, after: Width, word: &mut u64) {
    use Width::*;
    debug_assert!(before < after);
    let w = *word;
    *word = match (before, after) {
        // B1 → *
        (B1, B2)  => step_b1_b2(w),
        (B1, B4)  => step_b2_b4(step_b1_b2(w)),
        (B1, U8)  => step_b4_u8(step_b2_b4(step_b1_b2(w))),
        (B1, U16) => step_u8_u16(step_b4_u8(step_b2_b4(step_b1_b2(w)))),
        (B1, U32) => {
            let lo = (w as u32).count_ones() as u64;
            let hi = ((w >> 32) as u32).count_ones() as u64;
            lo | (hi << 32)
        }
        (B1, U64) => w.count_ones() as u64,

        // B2 → *
        (B2, B4)  => step_b2_b4(w),
        (B2, U8)  => step_b4_u8(step_b2_b4(w)),
        (B2, U16) => step_u8_u16(step_b4_u8(step_b2_b4(w))),
        (B2, U32) => step_u16_u32(step_u8_u16(step_b4_u8(step_b2_b4(w)))),
        (B2, U64) => step_u32_u64(step_u16_u32(step_u8_u16(step_b4_u8(step_b2_b4(w))))),

        // B4 → *
        (B4, U8)  => step_b4_u8(w),
        (B4, U16) => step_u8_u16(step_b4_u8(w)),
        (B4, U32) => step_u16_u32(step_u8_u16(step_b4_u8(w))),
        (B4, U64) => step_u32_u64(step_u16_u32(step_u8_u16(step_b4_u8(w)))),

        // U8 → *
        (U8, U16) => step_u8_u16(w),
        (U8, U32) => step_u16_u32(step_u8_u16(w)),
        (U8, U64) => step_u32_u64(step_u16_u32(step_u8_u16(w))),

        // U16 → *
        (U16, U32) => step_u16_u32(w),
        (U16, U64) => step_u32_u64(step_u16_u32(w)),

        // U32 → U64
        (U32, U64) => step_u32_u64(w),

        _ => {
            debug_assert!(false, "widen_into: invalid pair ({before:?}, {after:?})");
            w
        }
    };
}

/// OR-fold all SWAR lanes within a word into a single representative
/// value.  The result has the same highest-set-bit as the true
/// maximum lane, so `Width::from_max_value(or_fold_lanes(w, word))`
/// gives the exact minimum width needed to hold any lane.
#[inline]
pub(crate) fn or_fold_lanes(width: Width, w: u64) -> u64 {
    use Width::*;
    match width {
        B1 => (w != 0) as u64,
        B2 => {
            let w = w | (w >> 2);
            let w = w | (w >> 4);
            let w = w | (w >> 8);
            let w = w | (w >> 16);
            (w | (w >> 32)) & 0x3
        }
        B4 => {
            let w = w | (w >> 4);
            let w = w | (w >> 8);
            let w = w | (w >> 16);
            (w | (w >> 32)) & 0xF
        }
        U8 => {
            let w = w | (w >> 8);
            let w = w | (w >> 16);
            (w | (w >> 32)) & 0xFF
        }
        U16 => {
            let w = w | (w >> 16);
            (w | (w >> 32)) & 0xFFFF
        }
        U32 => (w | (w >> 32)) & 0xFFFF_FFFF,
        U64 => w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Width::*;

    fn widen(before: Width, after: Width, w: u64) -> u64 {
        let mut word = w;
        widen_into(before, after, &mut word);
        word
    }

    #[test]
    fn b1_to_b2() {
        assert_eq!(widen(B1, B2, 0), 0);
        // All ones: each pair (1,1) sums to 2 = 0b10, repeated 32×.
        assert_eq!(widen(B1, B2, u64::MAX), 0xAAAA_AAAA_AAAA_AAAA);
        // Alternating 0b01 in each pair → sum = 1.
        assert_eq!(widen(B1, B2, 0x5555_5555_5555_5555), 0x5555_5555_5555_5555);
    }

    #[test]
    fn b1_to_u64_matches_popcount() {
        for w in [0u64, 1, u64::MAX, 0xDEAD_BEEF_CAFE_BABE, 0x8000_0000_0000_0001] {
            assert_eq!(widen(B1, U64, w), w.count_ones() as u64);
        }
    }

    #[test]
    fn b1_to_u8_popcount_per_byte() {
        let w = 0xFF_00_0F_F0_AA_55_01_80u64;
        let result = widen(B1, U8, w);
        for i in 0..8 {
            let orig_byte = (w >> (i * 8)) & 0xFF;
            let result_byte = (result >> (i * 8)) & 0xFF;
            assert_eq!(result_byte, orig_byte.count_ones() as u64, "byte {i}");
        }
    }

    #[test]
    fn b1_to_u16_popcount_per_u16() {
        let w = u64::MAX;
        let result = widen(B1, U16, w);
        for i in 0..4 {
            assert_eq!((result >> (i * 16)) & 0xFFFF, 16, "lane {i}");
        }
    }

    #[test]
    fn b1_to_u32_popcount_per_u32() {
        let w = u64::MAX;
        let result = widen(B1, U32, w);
        assert_eq!(result & 0xFFFF_FFFF, 32);
        assert_eq!(result >> 32, 32);
    }

    #[test]
    fn u32_to_u64() {
        let w = 3u64 | (5u64 << 32);
        assert_eq!(widen(U32, U64, w), 8);
    }

    #[test]
    fn u16_to_u32() {
        let w = 1u64 | (2u64 << 16) | (3u64 << 32) | (4u64 << 48);
        let result = widen(U16, U32, w);
        assert_eq!(result & 0xFFFF_FFFF, 3);
        assert_eq!(result >> 32, 7);
    }

    #[test]
    fn u8_to_u16() {
        let w = 0x01_02_03_04_05_06_07_08u64;
        let result = widen(U8, U16, w);
        assert_eq!(result & 0xFFFF, 0x08 + 0x07);
        assert_eq!((result >> 16) & 0xFFFF, 0x06 + 0x05);
        assert_eq!((result >> 32) & 0xFFFF, 0x04 + 0x03);
        assert_eq!((result >> 48) & 0xFFFF, 0x02 + 0x01);
    }

    #[test]
    fn b4_to_u8() {
        // 16 B4 lanes, each = 0xF (max). Pairs sum to 30 = 0x1E.
        let w = u64::MAX;
        let result = widen(B4, U8, w);
        for i in 0..8 {
            assert_eq!((result >> (i * 8)) & 0xFF, 30, "byte {i}");
        }
    }

    #[test]
    fn b2_to_b4() {
        // 32 B2 lanes, each = 3 (max). Pairs sum to 6 = 0b0110.
        let w = u64::MAX;
        let result = widen(B2, B4, w);
        for i in 0..16 {
            assert_eq!((result >> (i * 4)) & 0xF, 6, "nibble {i}");
        }
    }

    /// Verify multi-step direct widening matches stepwise chaining.
    #[test]
    fn chained_consistency() {
        let all = [B1, B2, B4, U8, U16, U32, U64];
        let words = [0u64, 1, u64::MAX, 0xDEAD_BEEF_CAFE_BABE, 0x5555_5555_5555_5555];
        for &before in &all {
            for &after in &all {
                if before >= after {
                    continue;
                }
                for &w in &words {
                    // Mask to valid lane values for `before` width.
                    let mask = before.counter_max();
                    let spw = before.slots_per_u64();
                    let bps = before.bits_per_slot();
                    let mut masked = 0u64;
                    for s in 0..spw {
                        masked |= ((w >> (s * bps)) & mask) << (s * bps);
                    }
                    // Widen directly.
                    let direct = widen(before, after, masked);
                    // Widen step-by-step through each intermediate width.
                    let mut stepped = masked;
                    let b = before as usize;
                    let a = after as usize;
                    for level in b..a {
                        let from = crate::histogram::width::ALL_WIDTHS[level];
                        let to = crate::histogram::width::ALL_WIDTHS[level + 1];
                        widen_into(from, to, &mut stepped);
                    }
                    assert_eq!(
                        direct, stepped,
                        "mismatch ({before:?},{after:?}) w={masked:#018X}: direct={direct:#018X} stepped={stepped:#018X}"
                    );
                }
            }
        }
    }

    /// Verify or_fold_lanes gives the same from_max_value as true max extraction.
    #[test]
    fn or_fold_matches_true_max() {
        use crate::histogram::width::Width;

        let widths = [B2, B4, U8, U16, U32, U64];
        let test_words: &[u64] = &[
            0,
            1,
            u64::MAX,
            0xDEAD_BEEF_CAFE_BABE,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0100_0000_0000_0000,
            0x8000_0000_0000_0001,
        ];

        for &width in &widths {
            let bps = width.bits_per_slot();
            let spw = width.slots_per_u64();
            let mask = width.counter_max();

            for &raw in test_words {
                // Mask to valid lane values.
                let mut word = 0u64;
                for s in 0..spw {
                    word |= ((raw >> (s * bps)) & mask) << (s * bps);
                }

                // True max: extract each lane, take max.
                let mut true_max = 0u64;
                for s in 0..spw {
                    let lane = (word >> (s * bps)) & mask;
                    true_max = true_max.max(lane);
                }

                let folded = or_fold_lanes(width, word);

                // from_max_value should agree.
                assert_eq!(
                    Width::from_max_value(folded),
                    Width::from_max_value(true_max),
                    "width={width:?} word={word:#018X}: folded={folded}, true_max={true_max}"
                );
            }
        }
    }

    #[test]
    fn or_fold_b1() {
        assert_eq!(or_fold_lanes(B1, 0), 0);
        assert_eq!(or_fold_lanes(B1, 1), 1);
        assert_eq!(or_fold_lanes(B1, u64::MAX), 1);
    }

    #[test]
    fn or_fold_u64() {
        assert_eq!(or_fold_lanes(U64, 42), 42);
        assert_eq!(or_fold_lanes(U64, 0), 0);
    }
}
