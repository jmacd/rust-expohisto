// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! SWAR (SIMD Within A Register) — per-word parallel pairwise summation.
//!
//! These primitives are no longer used by the production downscale path
//! (which uses a clone + scatter-add approach), but are retained for
//! unit tests that verify SWAR correctness in isolation.

#![allow(dead_code)]

use super::bucket_width::BucketWidth;

/// Per-width SWAR parameters: `(shift, lane_mask)`.
///
/// - `shift`: number of bits to shift the upper half-slots down.
/// - `lane_mask`: keeps only the lower half-slot in each pair.
/// - `!lane_mask`: overflow mask — bits set here after a step mean the
///   pair-sum overflowed the original width.
///
/// Indexed by [`BucketWidth::level()`] (0=B1 … 5=U32).
pub(crate) const SWAR_TABLE: [(u32, u64); 6] = [
    (1, 0x5555_5555_5555_5555),  // B1
    (2, 0x3333_3333_3333_3333),  // B2
    (4, 0x0F0F_0F0F_0F0F_0F0F), // B4
    (8, 0x00FF_00FF_00FF_00FF),  // U8
    (16, 0x0000_FFFF_0000_FFFF), // U16
    (32, 0x0000_0000_FFFF_FFFF), // U32
];

/// Single SWAR step: sum adjacent counters at the current width into
/// the next wider width, in place.
#[inline]
pub(crate) fn swar_step(data: &mut [u64], width: BucketWidth) {
    debug_assert_ne!(width, BucketWidth::U64, "cannot widen past U64");
    let (shift, mask) = SWAR_TABLE[width.level()];
    for w in data.iter_mut() {
        let x = *w;
        *w = ((x >> shift) & mask) + (x & mask);
    }
}

/// Checks whether any widened pair-sum overflows the original width.
/// Called after `swar_step` has already written the wider sums.
#[inline]
pub(crate) fn swar_has_overflow(data: &[u64], original_width: BucketWidth) -> bool {
    if original_width == BucketWidth::U64 {
        return false;
    }
    let (_, mask) = SWAR_TABLE[original_width.level()];
    data.iter().any(|&w| w & !mask != 0)
}

/// Compacts narrowed half-words into full words, pairing two source
/// words into one destination word and zeroing the freed tail.
#[inline]
fn compact_with<F: Fn(u64) -> u64>(data: &mut [u64], narrow: F) {
    let n = data.len();
    for i in (0..n).step_by(2) {
        let lo = narrow(data[i]);
        let hi = if i + 1 < n { narrow(data[i + 1]) } else { 0 };
        data[i / 2] = lo | (hi << 32);
    }
    for w in &mut data[n.div_ceil(2)..n] {
        *w = 0;
    }
}

/// Shifts all slot values up by one position, inserting a zero at slot 0.
///
/// This effectively decrements the logical `index_base` by one, turning
/// an odd base into an even one so that a normal SWAR step pairs the
/// correct indices.
///
/// Precondition: the top slot of the last word must be zero (the live
/// range must not fill the entire capacity).
#[inline]
pub(crate) fn swar_shift_up_one(data: &mut [u64], width: BucketWidth) {
    debug_assert_ne!(width, BucketWidth::U64);
    let bits = width.bits();
    let n = data.len();
    if n == 0 {
        return;
    }
    debug_assert!(
        data[n - 1] >> (64 - bits) == 0,
        "top slot must be zero before shift",
    );
    // Process high-to-low so each word reads from the (unmodified) word below.
    for i in (1..n).rev() {
        data[i] = (data[i] << bits) | (data[i - 1] >> (64 - bits));
    }
    data[0] <<= bits;
}

/// Progressive bit-compaction: at each stage, merge adjacent groups by
/// OR-shifting, then mask to keep only the compacted result.
#[inline]
fn compact_lanes(mut x: u64, stages: &[(u32, u64)]) -> u64 {
    for &(shift, mask) in stages {
        x = (x | (x >> shift)) & mask;
    }
    x
}

/// Narrows a single word from `wider(original_width)` format back to
/// `original_width`, compacting the result into the low 32 bits.
///
/// Stage parameters are derived from [`SWAR_TABLE`]: stage K uses
/// `(SWAR_TABLE[K].0, SWAR_TABLE[K+1].1)` — the shift of level K and
/// the lane mask of level K+1.
#[inline]
pub(crate) fn narrow_word(w: u64, original_width: BucketWidth) -> u64 {
    const COMPACT_STAGES: [(u32, u64); 5] = [
        (SWAR_TABLE[0].0, SWAR_TABLE[1].1),
        (SWAR_TABLE[1].0, SWAR_TABLE[2].1),
        (SWAR_TABLE[2].0, SWAR_TABLE[3].1),
        (SWAR_TABLE[3].0, SWAR_TABLE[4].1),
        (SWAR_TABLE[4].0, SWAR_TABLE[5].1),
    ];

    let level = original_width.level();
    debug_assert!(level <= 5, "can only narrow sub-U64 widths");

    if level == 5 {
        // U32: one sum per word, just mask the carry bit.
        w & SWAR_TABLE[5].1
    } else {
        compact_lanes(w & SWAR_TABLE[level].1, &COMPACT_STAGES[level..])
    }
}

/// After a SWAR step that produced no overflow, narrow the widened sums
/// back to the original width and compact words 2:1.
///
/// The data is currently in `wider(original_width)` format with all values
/// fitting in `original_width`. This function bit-compresses each word
/// (via [`narrow_word`]) and packs pairs of words into one, freeing the
/// upper half of the array.
#[inline]
pub(crate) fn swar_narrow_compact(data: &mut [u64], original_width: BucketWidth) {
    compact_with(data, |w| narrow_word(w, original_width));
}
