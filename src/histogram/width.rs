// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Counter width for bucket data.

/// The current width of bucket counters, in bits.
///
/// Counters start at 1-bit (maximizing initial bucket count) and widen
/// in place through the chain: 1→2→4→8→16→32→64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Width {
    /// 1-bit counters (max 1 per bucket — presence bitmap).
    B1 = 0,
    /// 2-bit counters (max 3 per bucket).
    B2 = 1,
    /// 4-bit counters (max 15 per bucket).
    B4 = 2,
    /// 1-byte counters (max 255 per bucket).
    U8 = 3,
    /// 2-byte counters (max 65,535 per bucket).
    U16 = 4,
    /// 4-byte counters (max ~4 billion per bucket).
    U32 = 5,
    /// 8-byte counters.
    U64 = 6,
}

/// All counter widths in level order (excludes B0), for computed lookups.
pub(crate) const ALL_WIDTHS: [Width; 7] = [
    Width::B1,
    Width::B2,
    Width::B4,
    Width::U8,
    Width::U16,
    Width::U32,
    Width::U64,
];

impl Width {
    /// Returns the log2 of bits.
    #[inline]
    pub(crate) const fn log2(self) -> usize {
        self as usize
    }

    // Number of in-word widening change steps possible.
    #[inline]
    pub(crate) const fn to_u64_widen_by(self) -> u32 {
        Self::U64 as u32 - self as u32
    }

    /// Returns number of bits in one slot.
    #[inline]
    pub(crate) const fn bits_per_slot(self) -> usize {
        1 << self.log2()
    }

    /// Number of slots per u64.
    #[inline]
    pub(crate) const fn slots_per_u64(self) -> usize {
        64 >> self.log2()
    }

    /// Mask for the sub-u64 index values at this width.
    #[inline]
    pub(crate) const fn slot_mask_u64(self) -> i32 {
        // B1 -> 0x3f
        // B2 -> 0x1f
        // B4 -> 0xf
        // U8 -> 0x7
        // U16 -> 0x3
        // U32 -> 0x1
        // U64 -> 0
        self.slots_per_u64() as i32 - 1
    }

    /// Rounds a bucket index down to the first slot in its u64 word.
    #[inline]
    pub(crate) const fn slot_start_u64(self, index: i32) -> i32 {
        index & !self.slot_mask_u64()
    }

    /// Rounds a bucket index up to the last slot in its u64 word.
    #[inline]
    pub(crate) const fn slot_end_u64(self, index: i32) -> i32 {
        index | self.slot_mask_u64()
    }

    /// Returns the next-wider counter width or None.
    #[inline]
    pub(crate) const fn wider_by(self, change: u32) -> Option<Width> {
        let value = self as usize + change as usize;
        if value > Self::U64 as usize {
            None
        } else {
            Some(ALL_WIDTHS[value])
        }
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    pub(crate) const fn counter_max(self) -> u64 {
        u64::MAX >> (64 - self.bits_per_slot())
    }

    /// Returns the narrowest viable width.
    #[inline]
    pub(crate) const fn from_max_value(value: u64) -> Self {
        if value == 0 {
            return Self::B1;
        }
        let leading = 64 - value.leading_zeros();
        let width = leading.next_power_of_two();

        ALL_WIDTHS[width.trailing_zeros() as usize]
    }
}
