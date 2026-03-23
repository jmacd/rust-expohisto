// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Counter width for bucket data.

/// The current width of bucket counters, in bits.
///
/// Counters start at 1-bit (maximizing initial bucket count) and widen
/// in place through the chain: 1→2→4→8→16→32→64 bits.
///
/// The special `B0` variant represents **literal mode**, where the data
/// pool stores raw `f64` bit patterns instead of bucket counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BucketWidth {
    /// Literal mode: data pool stores raw f64 bit patterns, not counters.
    B0 = 0,
    /// 1-bit counters (max 1 per bucket — presence bitmap).
    B1 = 1,
    /// 2-bit counters (max 3 per bucket).
    B2 = 2,
    /// 4-bit counters (max 15 per bucket).
    B4 = 4,
    /// 1-byte counters (max 255 per bucket).
    U8 = 8,
    /// 2-byte counters (max 65,535 per bucket).
    U16 = 16,
    /// 4-byte counters (max ~4 billion per bucket).
    U32 = 32,
    /// 8-byte counters.
    U64 = 64,
}

/// All counter widths in level order (excludes B0), for computed lookups.
pub(crate) const ALL_WIDTHS: [BucketWidth; 7] = [
    BucketWidth::B1,
    BucketWidth::B2,
    BucketWidth::B4,
    BucketWidth::U8,
    BucketWidth::U16,
    BucketWidth::U32,
    BucketWidth::U64,
];

impl BucketWidth {
    /// Returns true if this is `B0` (literal mode).
    #[inline]
    pub const fn is_literal(self) -> bool {
        self as u8 == 0
    }

    /// Returns the bit width of one counter.
    ///
    /// Panics on `B0` in debug mode (literal mode has no counter width).
    #[inline]
    pub(crate) const fn bits(self) -> usize {
        debug_assert!(!self.is_literal(), "B0 has no counter width");
        self as usize
    }

    /// Returns the ordinal level (0=B1 … 6=U64), used to index
    /// [`ALL_WIDTHS`] and the SWAR table.
    #[inline]
    pub(crate) const fn level(self) -> usize {
        self.bits().trailing_zeros() as usize
    }

    /// Returns the number of buckets that fit in `word_count` u64 words.
    #[inline]
    pub(crate) const fn capacity(self, word_count: usize) -> usize {
        (word_count * 64) / self.bits()
    }

    /// Returns the number of counter slots per u64 word.
    #[inline]
    pub(crate) const fn slots_per_word(self) -> usize {
        64 / self.bits()
    }

    /// Rounds a bucket index down to the first slot in its u64 word.
    #[inline]
    pub(crate) const fn word_start(self, index: i32) -> i32 {
        index & !(self.slots_per_word() as i32 - 1)
    }

    /// Returns the next wider counter width, or `None` if already at U64.
    ///
    /// For `B0`, returns `B1` (promotion from literal to bucket mode).
    #[inline]
    pub(crate) const fn wider(self) -> Option<BucketWidth> {
        if self.is_literal() {
            return Some(BucketWidth::B1);
        }
        let l = self.level();
        if l < 6 {
            Some(ALL_WIDTHS[l + 1])
        } else {
            None
        }
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    pub(crate) const fn counter_max(self) -> u64 {
        u64::MAX >> (64 - self.bits())
    }

    /// Returns the narrowest width whose `counter_max()` ≥ `value`,
    /// or `None` if `value` is 0 (no width needed).
    #[inline]
    pub(crate) const fn from_max_value(value: u64) -> Option<Self> {
        if value == 0 {
            return None;
        }
        // Bits needed to represent `value`: 64 - leading_zeros.
        // Round up to the next valid width (power-of-two bit count).
        let raw_bits = 64 - value.leading_zeros(); // u32, 1..=64
        let width_bits = raw_bits.next_power_of_two(); // 1,2,4,8,16,32,64
                                                       // width_bits is already a valid BucketWidth discriminant.
        Some(ALL_WIDTHS[width_bits.trailing_zeros() as usize])
    }
}
