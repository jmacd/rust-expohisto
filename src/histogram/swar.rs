// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! SWAR (SIMD Within A Register) — per-word parallel pairwise summation.
//!
//! [`swar_step`] sums adjacent packed counters within each u64 word,
//! doubling the lane width.  [`swar_shift_up`] rotates slot data to
//! align group boundaries before reduction.
//!
//! Additional primitives ([`swar_has_overflow`], [`narrow_word`],
//! [`swar_narrow_compact`]) are available for testing but are not
//! used by the current downscale implementation.

use super::width::Width;

/// Per-width SWAR parameters: `(shift, lane_mask)`.
///
/// - `shift`: number of bits to shift the upper half-slots down.
/// - `lane_mask`: keeps only the lower half-slot in each pair.
/// - `!lane_mask`: overflow mask — bits set here after a step mean the
///   pair-sum overflowed the original width.
///
/// Indexed by [`Width::level()`] (0=B1 … 5=U32).
pub(crate) const SWAR_TABLE: [(u32, u64); 6] = [
    (1, 0x5555_5555_5555_5555),  // B1
    (2, 0x3333_3333_3333_3333),  // B2
    (4, 0x0F0F_0F0F_0F0F_0F0F),  // B4
    (8, 0x00FF_00FF_00FF_00FF),  // U8
    (16, 0x0000_FFFF_0000_FFFF), // U16
    (32, 0x0000_0000_FFFF_FFFF), // U32
];

/// Single SWAR step: sum adjacent counters at the current width into
/// the next wider width, in place.
#[inline]
pub(crate) fn swar_step(data: &mut [u64], width: Width) {
    debug_assert_ne!(width, Width::U64, "cannot widen past U64");
    let (shift, mask) = SWAR_TABLE[width.level()];
    for w in data.iter_mut() {
        let x = *w;
        *w = ((x >> shift) & mask) + (x & mask);
    }
}

/// Checks whether any widened pair-sum overflows the original width.
/// Called after `swar_step` has already written the wider sums.
#[cfg(test)]
#[inline]
pub(crate) fn swar_has_overflow(data: &[u64], original_width: Width) -> bool {
    if original_width == Width::U64 {
        return false;
    }
    let (_, mask) = SWAR_TABLE[original_width.level()];
    data.iter().any(|&w| w & !mask != 0)
}

/// Compacts narrowed half-words into full words, pairing two source
/// words into one destination word and zeroing the freed tail.
#[cfg(test)]
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

/// Shifts all slot values up (toward higher physical positions) by
/// `count` slots, inserting zeros at the bottom.
///
/// This effectively decrements the logical `index_base` by `count`,
/// aligning it to a group boundary so that SWAR steps pair the
/// correct indices.
///
/// Precondition: the top `count` slots of the data must be zero
/// (the live range must not fill the entire capacity minus `count`).
#[cfg(test)]
#[inline]
pub(crate) fn swar_shift_up(data: &mut [u64], width: Width, count: usize) {
    if count == 0 {
        return;
    }
    let bits = width.bits();
    let spw = width.slots_per_word();
    let n = data.len();
    if n == 0 {
        return;
    }

    // Decompose the shift into whole words + remaining slots.
    let word_shift = count / spw;
    let slot_shift = count % spw;
    let bit_shift = slot_shift * bits;

    // Shift whole words first (high to low), then sub-word bits.
    if word_shift >= n {
        // Entire data is shifted out — zero everything.
        for w in data.iter_mut() {
            *w = 0;
        }
        return;
    }

    if bit_shift == 0 {
        // Pure word-granularity shift.
        for i in (word_shift..n).rev() {
            data[i] = data[i - word_shift];
        }
    } else {
        let complement = 64 - bit_shift;
        for i in (word_shift + 1..n).rev() {
            data[i] =
                (data[i - word_shift] << bit_shift) | (data[i - word_shift - 1] >> complement);
        }
        data[word_shift] = data[0] << bit_shift;
    }

    // Zero the vacated low words.
    for w in &mut data[..word_shift] {
        *w = 0;
    }
}

/// Progressive bit-compaction: at each stage, merge adjacent groups by
/// OR-shifting, then mask to keep only the compacted result.
#[cfg(test)]
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
#[cfg(test)]
#[inline]
pub(crate) fn narrow_word(w: u64, original_width: Width) -> u64 {
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
#[cfg(test)]
#[inline]
pub(crate) fn swar_narrow_compact(data: &mut [u64], original_width: Width) {
    compact_with(data, |w| narrow_word(w, original_width));
}
