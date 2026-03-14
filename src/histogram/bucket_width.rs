// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Counter width for bucket data.

/// The current width of bucket counters, in bits.
///
/// Counters start at 1-bit (maximizing initial bucket count) and widen
/// in place through the chain: 1→2→4→8→16→32→64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BucketWidth {
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

/// All widths in level order, for computed lookups.
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
    /// Returns the bit width of one counter.
    #[inline]
    pub(crate) const fn bits(self) -> usize {
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

    /// Returns the next wider counter width, or `None` if already at u64.
    #[inline]
    pub(crate) const fn wider(self) -> Option<BucketWidth> {
        let l = self.level();
        if l < 6 {
            Some(ALL_WIDTHS[l + 1])
        } else {
            None
        }
    }

    /// Returns the width `steps` levels wider, or `None` if it would
    /// exceed U64.
    #[inline]
    pub(crate) const fn widen_by(self, steps: i32) -> Option<BucketWidth> {
        let target = self.level() + steps as usize;
        if target > 6 {
            None
        } else {
            Some(ALL_WIDTHS[target])
        }
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    pub(crate) const fn counter_max(self) -> u64 {
        u64::MAX >> (64 - self.bits())
    }
}
