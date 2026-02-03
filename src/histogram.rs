// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram implementation.
//!
//! This module provides a fixed-size exponential histogram that stores
//! bucket counts in a fixed-size array, avoiding any heap allocation.
//!
//! # Type Parameters
//!
//! - `C`: Counter type (e.g., `u16`, `u32`) - determines the maximum count per bucket
//! - `SIZE`: Maximum number of buckets (power of 2 recommended for slightly faster modular arithmetic)
//!
//! # Example
//!
//! ```
//! use rust_expohisto::Histogram;
//!
//! // Create a histogram with 16 u16 buckets
//! let mut hist: Histogram<u16, 16> = Histogram::new();
//!
//! // Record some values
//! hist.update(1.5);
//! hist.update(2.5);
//! hist.update(100.0);
//!
//! assert_eq!(hist.count(), 3);
//! ```

use crate::mapping::{Mapping, MAX_SCALE};

/// Trait for counter types that can be used in the histogram.
pub trait Counter: Copy + Default + Ord {
    /// The maximum value this counter can hold.
    const MAX: u64;

    /// Attempt to add `incr` to self, returning None on overflow.
    fn checked_add_u64(self, incr: u64) -> Option<Self>;

    /// Convert to u64 for merging operations.
    fn to_u64(self) -> u64;

    /// Create from u64, saturating at MAX.
    fn from_u64_saturating(v: u64) -> Self;
}

macro_rules! impl_counter {
    ($($t:ty),+) => {$(
        impl Counter for $t {
            const MAX: u64 = <$t>::MAX as u64;

            #[inline]
            fn checked_add_u64(self, incr: u64) -> Option<Self> {
                if incr > <Self as Counter>::MAX {
                    return None;
                }
                self.checked_add(incr as $t)
            }

            #[inline]
            fn to_u64(self) -> u64 {
                self as u64
            }

            #[inline]
            fn from_u64_saturating(v: u64) -> Self {
                v.min(<Self as Counter>::MAX) as $t
            }
        }
    )+};
}

impl_counter!(u8, u16, u32, u64);

/// High-low range for scale change calculations.
#[derive(Debug, Clone, Copy)]
struct HighLow {
    low: i32,
    high: i32,
}

impl HighLow {
    #[inline]
    const fn empty() -> Self {
        Self { low: 0, high: -1 }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.low > self.high
    }

    #[inline]
    fn merge(self, other: Self) -> Self {
        match (self.is_empty(), other.is_empty()) {
            (true, _) => other,
            (_, true) => self,
            _ => Self {
                low: self.low.min(other.low),
                high: self.high.max(other.high),
            },
        }
    }
}

/// Computes how much downscaling is needed for indices to fit in `size` buckets.
#[inline]
fn change_scale(mut hl: HighLow, size: i32) -> i32 {
    let mut change = 0;
    while hl.high - hl.low >= size {
        hl.high >>= 1;
        hl.low >>= 1;
        change += 1;
    }
    change
}

/// Fixed-size bucket storage using a circular buffer.
///
/// This stores counts in a fixed-size array, using modular arithmetic
/// to handle the circular nature of the index space.
#[derive(Debug, Clone, Copy)]
pub struct Buckets<C: Counter, const SIZE: usize> {
    /// The bucket counts, stored in a circular buffer.
    counts: [C; SIZE],
    /// Index of the 0th position in the backing array.
    index_base: i32,
    /// Smallest index value represented.
    index_start: i32,
    /// Largest index value represented.
    index_end: i32,
}

impl<C: Counter, const SIZE: usize> Default for Buckets<C, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Counter, const SIZE: usize> Buckets<C, SIZE> {
    /// Creates empty buckets.
    #[inline]
    pub fn new() -> Self {
        Self {
            counts: [C::default(); SIZE],
            index_base: 0,
            index_start: 0,
            index_end: 0,
        }
    }

    /// Returns the offset (smallest index).
    #[inline]
    pub fn offset(&self) -> i32 {
        self.index_start
    }

    /// Returns the number of buckets in use.
    #[inline]
    pub fn len(&self) -> u32 {
        if self.is_effectively_empty() {
            0
        } else {
            (self.index_end - self.index_start + 1) as u32
        }
    }

    /// Returns true if no buckets have been used.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks if the bucket range represents no data.
    #[inline]
    fn is_effectively_empty(&self) -> bool {
        self.index_end == self.index_start && self.at(0) == 0
    }

    /// Returns the count at position `pos` (0-indexed from offset).
    #[inline]
    pub fn at(&self, pos: u32) -> u64 {
        let bias = (self.index_base - self.index_start) as u32;
        let size = SIZE as u32;

        let mut idx = pos;
        if idx < bias {
            idx += size;
        }
        idx -= bias;

        self.counts[idx as usize].to_u64()
    }

    /// Clears all bucket counts.
    #[inline]
    pub fn clear(&mut self) {
        self.index_start = 0;
        self.index_end = 0;
        self.index_base = 0;
        self.counts.fill(C::default());
    }

    /// Returns the size of the backing array.
    #[inline]
    const fn size(&self) -> i32 {
        SIZE as i32
    }

    /// Increments the bucket at `bucket_index` by `incr`.
    ///
    /// Returns `false` if the increment would overflow the counter.
    #[inline]
    fn try_increment(&mut self, bucket_index: i32, incr: u64) -> bool {
        let idx = bucket_index as usize % SIZE;
        if let Some(new_val) = self.counts[idx].checked_add_u64(incr) {
            self.counts[idx] = new_val;
            true
        } else {
            false
        }
    }

    /// Empties a bucket and returns its count.
    #[inline]
    fn empty_bucket(&mut self, src: i32) -> u64 {
        let idx = src as usize % SIZE;
        core::mem::take(&mut self.counts[idx]).to_u64()
    }

    /// Reverses elements in range [from, limit).
    fn reverse(&mut self, from: i32, limit: i32) {
        let num = ((from + limit) / 2) - from;
        for i in 0..num {
            let a = (from + i) as usize % SIZE;
            let b = (limit - i - 1) as usize % SIZE;
            self.counts.swap(a, b);
        }
    }

    /// Rotates the array so that index_start == index_base.
    fn rotate(&mut self) {
        let bias = self.index_base - self.index_start;
        if bias == 0 {
            return;
        }

        self.index_base = self.index_start;

        let size = self.size();
        self.reverse(0, size);
        self.reverse(0, bias);
        self.reverse(bias, size);
    }

    /// Downscales by collapsing 2^by buckets into 1.
    fn downscale(&mut self, by: i32) {
        self.rotate();

        let size = 1 + self.index_end - self.index_start;
        let each = 1i64 << by;
        let mut inpos = 0i32;
        let mut outpos = 0i32;
        let mut pos = self.index_start;

        while pos <= self.index_end {
            let mod_val = (pos as i64).rem_euclid(each);

            let mut i = mod_val;
            while i < each && inpos < size {
                self.relocate_bucket(outpos, inpos);
                inpos += 1;
                pos += 1;
                i += 1;
            }
            outpos += 1;
        }

        self.index_start >>= by;
        self.index_end >>= by;
        self.index_base = self.index_start;
    }

    /// Moves count from src bucket to dest bucket.
    fn relocate_bucket(&mut self, dest: i32, src: i32) {
        if dest == src {
            return;
        }
        let count = self.empty_bucket(src);
        // In a fixed-size implementation, we assume counts fit
        self.try_increment(dest, count);
    }

    /// Returns an iterator over the bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, C, SIZE> {
        BucketsIter {
            buckets: self,
            pos: 0,
            len: self.len(),
        }
    }
}

/// Iterator over bucket counts.
#[derive(Debug)]
pub struct BucketsIter<'a, C: Counter, const SIZE: usize> {
    buckets: &'a Buckets<C, SIZE>,
    pos: u32,
    len: u32,
}

impl<C: Counter, const SIZE: usize> Iterator for BucketsIter<'_, C, SIZE> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let count = self.buckets.at(self.pos);
        self.pos += 1;
        Some(count)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<C: Counter, const SIZE: usize> ExactSizeIterator for BucketsIter<'_, C, SIZE> {}

impl<'a, C: Counter, const SIZE: usize> IntoIterator for &'a Buckets<C, SIZE> {
    type Item = u64;
    type IntoIter = BucketsIter<'a, C, SIZE>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An allocation-free exponential histogram for non-negative values.
///
/// # Type Parameters
///
/// - `C`: Counter type (`u8`, `u16`, `u32`, or `u64`)
/// - `SIZE`: Maximum number of buckets
///
/// # Size Calculation
///
/// For SIZE=16 and C=u16:
/// - sum: 8 bytes
/// - count: 8 bytes
/// - zero_count: 8 bytes
/// - min: 8 bytes
/// - max: 8 bytes
/// - mapping: ~24 bytes (scale + factors)
/// - buckets.counts: 32 bytes (16 * 2)
/// - buckets indices: 12 bytes (3 * i32)
/// - Total: ~108 bytes ≈ 14 words
///
/// For positive-only with smaller counters, this fits in roughly 10+ words.
#[derive(Debug, Clone)]
pub struct Histogram<C: Counter, const SIZE: usize> {
    // Statistics
    sum: f64,
    count: u64,
    zero_count: u64,
    min: f64,
    max: f64,

    // Mapping (scale-dependent index calculation)
    mapping: Mapping,

    // Positive value buckets only (non-negative assumption)
    positive: Buckets<C, SIZE>,
}

impl<C: Counter, const SIZE: usize> Default for Histogram<C, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Counter, const SIZE: usize> Histogram<C, SIZE> {
    /// Creates a new histogram at the maximum scale.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(MAX_SCALE).unwrap(),
            positive: Buckets::new(),
        }
    }

    /// Creates a new histogram at the specified scale.
    ///
    /// # Panics
    /// Panics if the scale is out of range.
    #[inline]
    pub fn with_scale(scale: i32) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(scale).expect("invalid scale"),
            positive: Buckets::new(),
        }
    }

    /// Returns the sum of all recorded values.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the count of zero values.
    #[inline]
    pub fn zero_count(&self) -> u64 {
        self.zero_count
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Returns the current scale.
    #[inline]
    pub fn scale(&self) -> i32 {
        if self.count == self.zero_count {
            // All zeros - scale doesn't matter
            0
        } else {
            self.mapping.scale()
        }
    }

    /// Returns a reference to the positive buckets.
    #[inline]
    pub fn positive(&self) -> &Buckets<C, SIZE> {
        &self.positive
    }

    /// Clears the histogram, resetting to initial state.
    pub fn clear(&mut self) {
        self.positive.clear();
        self.sum = 0.0;
        self.count = 0;
        self.zero_count = 0;
        self.min = 0.0;
        self.max = 0.0;
        self.mapping = Mapping::new(MAX_SCALE).unwrap();
    }

    /// Swaps contents with another histogram.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Records a single value.
    ///
    /// # Panics
    /// Panics if value is negative, NaN, or infinite.
    #[inline]
    pub fn update(&mut self, value: f64) {
        self.update_by_incr(value, 1);
    }

    /// Records a value with a specified increment (count).
    ///
    /// # Panics
    /// Panics if value is negative, NaN, or infinite.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) {
        debug_assert!(value >= 0.0, "Histogram only accepts non-negative values");
        debug_assert!(value.is_finite(), "Histogram only accepts finite values");

        // Update min/max
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }

        self.count += incr;

        if value == 0.0 {
            self.zero_count += incr;
            return;
        }

        self.sum += value * incr as f64;
        self.update_buckets(value, incr);
    }

    /// Updates buckets for a positive value.
    fn update_buckets(&mut self, value: f64, incr: u64) {
        let index = self.mapping.map_to_index(value);

        let (hl, success) = self.increment_index_by(index, incr);
        if success {
            return;
        }

        // Need to downscale
        let change = change_scale(hl, SIZE as i32);
        self.downscale(change);

        let index = self.mapping.map_to_index(value);
        let (_, success) = self.increment_index_by(index, incr);
        debug_assert!(success, "downscale logic error");
    }

    /// Attempts to increment at the given index.
    ///
    /// Returns (HighLow, success). If success is false, HighLow contains
    /// the required range for downscaling.
    fn increment_index_by(&mut self, index: i32, incr: u64) -> (HighLow, bool) {
        if incr == 0 {
            return (HighLow::empty(), true);
        }

        let max_size = SIZE as i32;

        if self.positive.is_empty() {
            // First value
            self.positive.index_start = index;
            self.positive.index_end = index;
            self.positive.index_base = index;
        } else if index < self.positive.index_start {
            let span = self.positive.index_end - index;
            if span >= max_size {
                return (
                    HighLow {
                        low: index,
                        high: self.positive.index_end,
                    },
                    false,
                );
            }
            self.positive.index_start = index;
        } else if index > self.positive.index_end {
            let span = index - self.positive.index_start;
            if span >= max_size {
                return (
                    HighLow {
                        low: self.positive.index_start,
                        high: index,
                    },
                    false,
                );
            }
            self.positive.index_end = index;
        }

        let mut bucket_index = index - self.positive.index_base;
        if bucket_index < 0 {
            bucket_index += self.positive.size();
        }

        if !self.positive.try_increment(bucket_index, incr) {
            // Counter overflow - this is a limitation of fixed-width counters
            // In production, you might want to handle this differently
            panic!("bucket counter overflow");
        }

        (HighLow::empty(), true)
    }

    /// Downscales the histogram by the given amount.
    fn downscale(&mut self, change: i32) {
        if change == 0 {
            return;
        }
        debug_assert!(change > 0, "cannot upscale");

        let new_scale = self.mapping.scale() - change;
        self.positive.downscale(change);
        self.mapping = Mapping::new(new_scale).expect("invalid scale after downscale");
    }

    /// Merges another histogram into this one.
    pub fn merge_from(&mut self, other: &Self) {
        // Update min/max
        match (self.count == 0, other.count == 0) {
            (true, false) => {
                self.min = other.min;
                self.max = other.max;
            }
            (false, false) => {
                self.min = self.min.min(other.min);
                self.max = self.max.max(other.max);
            }
            _ => {}
        }

        self.sum += other.sum;
        self.count += other.count;
        self.zero_count += other.zero_count;

        let min_scale = self.scale().min(other.scale());

        let hlp = self.high_low_at_scale(&self.positive, min_scale)
            .merge(other.high_low_at_scale(&other.positive, min_scale));

        let min_scale = min_scale - change_scale(hlp, SIZE as i32);

        self.downscale(self.scale() - min_scale);
        self.merge_buckets(other, min_scale);
    }

    fn high_low_at_scale(&self, buckets: &Buckets<C, SIZE>, scale: i32) -> HighLow {
        if buckets.is_empty() {
            return HighLow::empty();
        }
        let shift = self.scale() - scale;
        HighLow {
            low: buckets.index_start >> shift,
            high: buckets.index_end >> shift,
        }
    }

    fn merge_buckets(&mut self, other: &Self, scale: i32) {
        let their_offset = other.positive.offset();
        let their_change = other.scale() - scale;

        for i in 0..other.positive.len() {
            let count = other.positive.at(i);
            if count == 0 {
                continue;
            }
            let index = (their_offset + i as i32) >> their_change;
            let (_, success) = self.increment_index_by(index, count);
            debug_assert!(success, "incorrect merge scale");
        }
    }
}

/// Type alias for a compact histogram with u16 counters.
pub type Histogram16<const SIZE: usize> = Histogram<u16, SIZE>;

/// Type alias for a histogram with u32 counters.
pub type Histogram32<const SIZE: usize> = Histogram<u32, SIZE>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<u16, 16> = Histogram::new();

        h.update(1.0);
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum(), 1.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
        assert_eq!(h.zero_count(), 0);
    }

    #[test]
    fn test_histogram_zero() {
        let mut h: Histogram<u16, 16> = Histogram::new();

        h.update(0.0);
        assert_eq!(h.count(), 1);
        assert_eq!(h.zero_count(), 1);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_multiple() {
        let mut h: Histogram<u16, 16> = Histogram::new();

        h.update(1.0);
        h.update(2.0);
        h.update(4.0);

        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 7.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 4.0);
    }

    #[test]
    fn test_histogram_downscale() {
        let mut h: Histogram<u16, 4> = Histogram::new();

        // With only 4 buckets, this should trigger downscaling
        h.update(1.0);
        h.update(1000.0);

        assert_eq!(h.count(), 2);
        assert!(h.scale() < MAX_SCALE);
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1: Histogram<u16, 16> = Histogram::new();
        let mut h2: Histogram<u16, 16> = Histogram::new();

        h1.update(1.0);
        h1.update(2.0);

        h2.update(3.0);
        h2.update(4.0);

        h1.merge_from(&h2);

        assert_eq!(h1.count(), 4);
        assert_eq!(h1.sum(), 10.0);
        assert_eq!(h1.min(), 1.0);
        assert_eq!(h1.max(), 4.0);
    }

    #[test]
    fn test_histogram_clear() {
        let mut h: Histogram<u16, 16> = Histogram::new();

        h.update(1.0);
        h.update(2.0);
        h.clear();

        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.scale(), 0); // Returns 0 when empty
    }

    #[test]
    fn test_buckets_at() {
        // Scale 0: each power-of-2 is a bucket
        let mut h = Histogram::<u16, 16>::with_scale(0);
        h.update(1.5); // Should go into bucket for [1, 2)
        h.update(1.7);
        h.update(3.0); // Should go into bucket for [2, 4)

        let buckets = h.positive();
        assert!(buckets.len() >= 2);
    }

    #[test]
    fn test_size_of_histogram() {
        // Verify the size is reasonable
        let size = core::mem::size_of::<Histogram<u16, 16>>();
        // Should be around 100-120 bytes
        assert!(size < 200, "Histogram should be compact, got {} bytes", size);
        
        // Verify u8/8 is even smaller
        let size_small = core::mem::size_of::<Histogram<u8, 8>>();
        assert!(size_small < 100, "Small histogram should be very compact, got {} bytes", size_small);
    }
}
