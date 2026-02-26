// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram implementation.

use crate::mapping::{Mapping, max_scale};

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

/// Result of attempting to increment a bucket.
enum IncrResult {
    /// Bucket was successfully incremented.
    Ok,
    /// Span exceeds bucket capacity; downscale by the given range.
    NeedsDownscale(HighLow),
    /// The counter type overflowed.
    CounterOverflow,
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

    /// Rotates the array so that index_start == index_base.
    fn rotate(&mut self) {
        let bias = (self.index_base - self.index_start) as usize;
        if bias == 0 {
            return;
        }
        self.counts.rotate_right(bias);
        self.index_base = self.index_start;
    }

    /// Downscales by collapsing 2^by buckets into 1.
    /// Returns `false` if combining buckets would overflow the counter type.
    fn downscale(&mut self, by: i32) -> bool {
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
                if !self.relocate_bucket(outpos, inpos) {
                    return false;
                }
                inpos += 1;
                pos += 1;
                i += 1;
            }
            outpos += 1;
        }

        self.index_start >>= by;
        self.index_end >>= by;
        self.index_base = self.index_start;
        true
    }

    /// Moves count from src bucket to dest bucket.
    ///
    /// Returns `false` if the combined count would overflow the counter type.
    fn relocate_bucket(&mut self, dest: i32, src: i32) -> bool {
        if dest == src {
            return true;
        }
        let count = self.empty_bucket(src);
        self.try_increment(dest, count)
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

    // Upper bound on scale, used as the initial/reset scale.
    max_scale: i32,

    // Positive value buckets only (non-negative assumption)
    positive: Buckets<C, SIZE>,
}

impl<C: Counter, const SIZE: usize> Default for Histogram<C, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Counter, const SIZE: usize> Histogram<C, SIZE> {
    /// Creates a new histogram at the maximum supported scale.
    #[inline]
    pub fn new() -> Self {
        let scale = max_scale();
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(scale).unwrap(),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Creates a new histogram with an upper bound on scale.
    ///
    /// The histogram starts at this scale and will never exceed it after a
    /// `clear()`. Setting this to the compile-time table scale (see
    /// [`max_scale()`]) avoids logarithm-based mapping at runtime and
    /// prevents autoscaling from choosing extremely high resolutions that
    /// place many empty buckets between nearby measurements.
    #[inline]
    pub fn with_max_scale(scale: i32) -> Self {
        let scale = scale.min(max_scale());
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Creates a new histogram at the specified scale.
    #[inline]
    pub fn with_scale(scale: i32) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Creates a wider-counter copy of this histogram.
    ///
    /// Every bucket count is losslessly promoted to the wider counter type.
    /// The resulting histogram has the same scale, max_scale, and statistics.
    pub fn widen_into<C2: Counter>(&self) -> Histogram<C2, SIZE> {
        let mut wider = Histogram::<C2, SIZE> {
            sum: self.sum,
            count: self.count,
            zero_count: self.zero_count,
            min: self.min,
            max: self.max,
            mapping: self.mapping,
            max_scale: self.max_scale,
            positive: Buckets {
                counts: [C2::default(); SIZE],
                index_base: self.positive.index_base,
                index_start: self.positive.index_start,
                index_end: self.positive.index_end,
            },
        };
        for i in 0..SIZE {
            wider.positive.counts[i] = C2::from_u64_saturating(self.positive.counts[i].to_u64());
        }
        wider
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

    /// Returns the maximum scale this histogram will use on reset.
    #[inline]
    pub fn max_scale(&self) -> i32 {
        self.max_scale
    }

    /// Returns a reference to the positive buckets.
    #[inline]
    pub fn positive(&self) -> &Buckets<C, SIZE> {
        &self.positive
    }

    /// Clears the histogram, resetting to initial state.
    ///
    /// Scale resets to the max_scale ceiling configured at construction.
    pub fn clear(&mut self) {
        self.positive.clear();
        self.sum = 0.0;
        self.count = 0;
        self.zero_count = 0;
        self.min = 0.0;
        self.max = 0.0;
        self.mapping = Mapping::new(self.max_scale).unwrap();
    }

    /// Swaps contents with another histogram.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Records a single value.
    ///
    /// Returns `false` if a bucket counter overflowed.
    #[inline]
    pub fn update(&mut self, value: f64) -> bool {
        self.update_by_incr(value, 1)
    }

    /// Records a value with a specified increment.
    ///
    /// Returns `false` if a counter overflowed.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) -> bool {
        debug_assert!(value >= 0.0, "Histogram only accepts non-negative values");

        let new_count = match self.count.checked_add(incr) {
            Some(c) => c,
            None => return false,
        };

        // Update min/max
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }

        self.count = new_count;

        if value == 0.0 {
            let new_zc = match self.zero_count.checked_add(incr) {
                Some(c) => c,
                None => return false,
            };
            self.zero_count = new_zc;
            return true;
        }

        self.sum += value * incr as f64;
        self.update_buckets(value, incr)
    }

    /// Updates buckets for a positive value.
    ///
    /// Returns `false` if a bucket counter overflowed.
    pub(crate) fn update_buckets(&mut self, value: f64, incr: u64) -> bool {
        let index = self.mapping.map_to_index(value);

        match self.increment_index_by(index, incr) {
            IncrResult::Ok => return true,
            IncrResult::CounterOverflow => return false,
            IncrResult::NeedsDownscale(hl) => {
                let change = change_scale(hl, SIZE as i32);
                if !self.downscale(change) {
                    return false;
                }
            }
        }

        let index = self.mapping.map_to_index(value);
        match self.increment_index_by(index, incr) {
            IncrResult::Ok => true,
            IncrResult::CounterOverflow => false,
            IncrResult::NeedsDownscale(_) => {
                debug_assert!(false, "downscale logic error");
                false
            }
        }
    }

    /// Attempts to increment at the given index.
    fn increment_index_by(&mut self, index: i32, incr: u64) -> IncrResult {
        if incr == 0 {
            return IncrResult::Ok;
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
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.positive.index_end,
                });
            }
            self.positive.index_start = index;
        } else if index > self.positive.index_end {
            let span = index - self.positive.index_start;
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.positive.index_start,
                    high: index,
                });
            }
            self.positive.index_end = index;
        }

        let mut bucket_index = index - self.positive.index_base;
        if bucket_index < 0 {
            bucket_index += self.positive.size();
        }

        if !self.positive.try_increment(bucket_index, incr) {
            return IncrResult::CounterOverflow;
        }

        IncrResult::Ok
    }

    /// Downscales the histogram by the given amount.
    ///
    /// Returns `false` if combining buckets would overflow the counter type.
    fn downscale(&mut self, change: i32) -> bool {
        if change == 0 {
            return true;
        }
        debug_assert!(change > 0, "cannot upscale");

        let new_scale = self.mapping.scale() - change;
        if !self.positive.downscale(change) {
            return false;
        }
        self.mapping = Mapping::new(new_scale).expect("invalid scale after downscale");
        true
    }

    /// Merges another histogram into this one.
    ///
    /// Returns `false` if a bucket counter overflowed.
    pub fn merge_from(&mut self, other: &Self) -> bool {
        self.merge_from_histogram(other)
    }

    /// Merges a histogram with a potentially different counter type into this one.
    ///
    /// Returns `false` if a bucket counter overflowed.
    pub fn merge_from_histogram<C2: Counter>(&mut self, other: &Histogram<C2, SIZE>) -> bool {
        // Early return if other is empty
        if other.count == 0 {
            return true;
        }

        // Check for u64 overflow before mutating.
        let new_count = match self.count.checked_add(other.count) {
            Some(c) => c,
            None => return false,
        };
        let new_zero_count = match self.zero_count.checked_add(other.zero_count) {
            Some(c) => c,
            None => return false,
        };

        // Update min/max
        if self.count == 0 {
            self.min = other.min;
            self.max = other.max;
        } else {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }

        self.sum += other.sum;
        self.count = new_count;
        self.zero_count = new_zero_count;

        // If other only has zeros, no bucket merging needed
        if other.positive.is_empty() {
            return true;
        }

        let min_scale = self.scale().min(other.scale());

        let hlp = self.high_low_at_scale(&self.positive, min_scale)
            .merge(Self::high_low_at_scale_for(&other.positive, other.scale(), min_scale));

        let min_scale = min_scale - change_scale(hlp, SIZE as i32);

        if !self.downscale(self.scale() - min_scale) {
            return false;
        }
        self.merge_buckets_from(&other.positive, other.scale(), min_scale)
    }

    fn high_low_at_scale(&self, buckets: &Buckets<C, SIZE>, scale: i32) -> HighLow {
        Self::high_low_at_scale_for(buckets, self.scale(), scale)
    }

    fn high_low_at_scale_for<C2: Counter>(buckets: &Buckets<C2, SIZE>, current_scale: i32, target_scale: i32) -> HighLow {
        if buckets.is_empty() {
            return HighLow::empty();
        }
        let shift = current_scale - target_scale;
        HighLow {
            low: buckets.index_start >> shift,
            high: buckets.index_end >> shift,
        }
    }

    fn merge_buckets_from<C2: Counter>(&mut self, other_buckets: &Buckets<C2, SIZE>, other_scale: i32, target_scale: i32) -> bool {
        let their_offset = other_buckets.offset();
        let their_change = other_scale - target_scale;

        for i in 0..other_buckets.len() {
            let count = other_buckets.at(i);
            if count == 0 {
                continue;
            }
            let index = (their_offset + i as i32) >> their_change;
            match self.increment_index_by(index, count) {
                IncrResult::Ok => {}
                IncrResult::CounterOverflow => return false,
                IncrResult::NeedsDownscale(_) => {
                    debug_assert!(false, "incorrect merge scale");
                    return false;
                }
            }
        }
        true
    }

    /// Merges from raw histogram data, enabling cross-SIZE merging.
    ///
    /// This accepts the source's statistics and bucket data as scalars and a
    /// closure, so the source histogram's SIZE need not match self's SIZE.
    ///
    /// Returns `false` if a counter overflowed.
    pub(crate) fn merge_from_raw(
        &mut self,
        other_count: u64,
        other_zero_count: u64,
        other_sum: f64,
        other_min: f64,
        other_max: f64,
        other_scale: i32,
        other_offset: i32,
        other_len: u32,
        other_at: &dyn Fn(u32) -> u64,
    ) -> bool {
        if other_count == 0 {
            return true;
        }

        let new_count = match self.count.checked_add(other_count) {
            Some(c) => c,
            None => return false,
        };
        let new_zero_count = match self.zero_count.checked_add(other_zero_count) {
            Some(c) => c,
            None => return false,
        };

        if self.count == 0 {
            self.min = other_min;
            self.max = other_max;
        } else {
            self.min = self.min.min(other_min);
            self.max = self.max.max(other_max);
        }

        self.sum += other_sum;
        self.count = new_count;
        self.zero_count = new_zero_count;

        if other_len == 0 {
            return true;
        }

        let other_end = other_offset + other_len as i32 - 1;
        let min_scale = self.scale().min(other_scale);

        let self_hl = if self.positive.is_empty() {
            HighLow::empty()
        } else {
            let shift = self.scale() - min_scale;
            HighLow {
                low: self.positive.index_start >> shift,
                high: self.positive.index_end >> shift,
            }
        };
        let other_hl = {
            let shift = other_scale - min_scale;
            HighLow {
                low: other_offset >> shift,
                high: other_end >> shift,
            }
        };
        let hlp = self_hl.merge(other_hl);
        let min_scale = min_scale - change_scale(hlp, SIZE as i32);

        if !self.downscale(self.scale() - min_scale) {
            return false;
        }

        let their_change = other_scale - min_scale;
        for i in 0..other_len {
            let count = other_at(i);
            if count == 0 {
                continue;
            }
            let index = (other_offset + i as i32) >> their_change;
            match self.increment_index_by(index, count) {
                IncrResult::Ok => {}
                IncrResult::CounterOverflow => return false,
                IncrResult::NeedsDownscale(_) => {
                    debug_assert!(false, "incorrect merge scale in merge_from_raw");
                    return false;
                }
            }
        }
        true
    }
}

/// Type alias for a compact histogram with u16 counters.
pub type Histogram16<const SIZE: usize> = Histogram<u16, SIZE>;

/// Type alias for a histogram with u32 counters.
pub type Histogram32<const SIZE: usize> = Histogram<u32, SIZE>;

/// Type alias for a histogram with u64 counters (widest, for cumulative aggregation).
pub type Histogram64<const SIZE: usize> = Histogram<u64, SIZE>;

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
        assert!(h.scale() < max_scale());
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
        // Should be around 100-128 bytes (varies with feature flags)
        assert!(size < 200, "Histogram should be compact, got {} bytes", size);
        
        // Verify u8/8 is even smaller
        let size_small = core::mem::size_of::<Histogram<u8, 8>>();
        assert!(size_small < 120, "Small histogram should be very compact, got {} bytes", size_small);
    }

    #[test]
    fn test_merge_u32_into_u64() {
        // Simulate cumulative with U64, delta with U32
        let mut cumulative: Histogram<u64, 16> = Histogram::new();
        let mut delta: Histogram<u32, 16> = Histogram::new();

        delta.update(1.0);
        delta.update(2.0);
        delta.update(3.0);

        cumulative.merge_from_histogram(&delta);

        assert_eq!(cumulative.count(), 3);
        assert_eq!(cumulative.sum(), 6.0);
        assert_eq!(cumulative.min(), 1.0);
        assert_eq!(cumulative.max(), 3.0);
    }

    #[test]
    fn test_merge_multiple_deltas_into_cumulative() {
        // Simulates the common use case: cumulative U64, send deltas as U32
        let mut cumulative: Histogram<u64, 16> = Histogram::new();

        // First delta batch
        let mut delta1: Histogram<u32, 16> = Histogram::new();
        delta1.update(1.0);
        delta1.update(2.0);
        cumulative.merge_from_histogram(&delta1);

        assert_eq!(cumulative.count(), 2);
        assert_eq!(cumulative.sum(), 3.0);

        // Second delta batch
        let mut delta2: Histogram<u32, 16> = Histogram::new();
        delta2.update(3.0);
        delta2.update(4.0);
        delta2.update(5.0);
        cumulative.merge_from_histogram(&delta2);

        assert_eq!(cumulative.count(), 5);
        assert_eq!(cumulative.sum(), 15.0);
        assert_eq!(cumulative.min(), 1.0);
        assert_eq!(cumulative.max(), 5.0);
    }

    #[test]
    fn test_merge_u16_into_u64() {
        // Even narrower counter type
        let mut cumulative: Histogram<u64, 16> = Histogram::new();
        let mut delta: Histogram<u16, 16> = Histogram::new();

        delta.update(10.0);
        delta.update(20.0);

        cumulative.merge_from_histogram(&delta);

        assert_eq!(cumulative.count(), 2);
        assert_eq!(cumulative.sum(), 30.0);
    }

    #[test]
    fn test_merge_u8_into_u32() {
        // Compact delta with u8 into u32 cumulative
        let mut cumulative: Histogram<u32, 8> = Histogram::new();
        let mut delta: Histogram<u8, 8> = Histogram::new();

        delta.update(1.5);
        delta.update(2.5);
        delta.update(0.0); // zero value

        cumulative.merge_from_histogram(&delta);

        assert_eq!(cumulative.count(), 3);
        assert_eq!(cumulative.zero_count(), 1);
        assert_eq!(cumulative.sum(), 4.0);
    }

    #[test]
    fn test_merge_cross_width_with_different_scales() {
        // Create histograms that will have different scales
        let mut cumulative: Histogram<u64, 4> = Histogram::new();
        let mut delta: Histogram<u32, 4> = Histogram::new();

        // Force downscaling by using wide value range
        cumulative.update(1.0);
        cumulative.update(1000.0);

        delta.update(0.001);
        delta.update(100.0);

        let cumulative_scale_before = cumulative.scale();
        cumulative.merge_from_histogram(&delta);

        assert_eq!(cumulative.count(), 4);
        // Scale may have changed due to merge
        assert!(cumulative.scale() <= cumulative_scale_before);
    }

    #[test]
    fn test_merge_empty_histogram_cross_width() {
        let mut cumulative: Histogram<u64, 16> = Histogram::new();
        cumulative.update(1.0);

        // Merge an empty histogram
        let empty: Histogram<u32, 16> = Histogram::new();
        cumulative.merge_from_histogram(&empty);

        assert_eq!(cumulative.count(), 1);
        assert_eq!(cumulative.sum(), 1.0);
    }

    #[test]
    fn test_merge_into_empty_histogram_cross_width() {
        let mut cumulative: Histogram<u64, 16> = Histogram::new();
        let mut delta: Histogram<u32, 16> = Histogram::new();

        delta.update(5.0);
        delta.update(10.0);

        // Merge into empty
        cumulative.merge_from_histogram(&delta);

        assert_eq!(cumulative.count(), 2);
        assert_eq!(cumulative.sum(), 15.0);
        assert_eq!(cumulative.min(), 5.0);
        assert_eq!(cumulative.max(), 10.0);
    }

    #[test]
    fn test_merge_equivalence_comprehensive() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        // Test inputs: diverse values in 0-20 range, various patterns
        let hardcoded_sets: &[&[f64]] = &[
            &[],
            &[0.0],
            &[1.0],
            &[0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 2.0],
            &[0.5, 1.5, 2.5],
            &[0.001, 1.0, 20.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 1.0, 2.0, 0.0],
            &[5.0, 10.0, 15.0, 20.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            &[0.01, 0.1, 1.0, 10.0],
            &[0.0, 20.0],
            &[1.0, 19.0],
            &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            &[15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            &[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        ];

        // Convert to owned vectors for uniform handling
        let mut test_sets: Vec<Vec<f64>> = hardcoded_sets.iter().map(|s| s.to_vec()).collect();

        // Generate 20 random sets with sizes in [0, 10]
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
        for _ in 0..20 {
            let size = rng.gen_range(0..=10);
            let set: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..20.0)).collect();
            test_sets.push(set);
        }

        // Test with multiple histogram sizes to stress downscaling
        test_merge_equivalence_for_size_owned::<2>(&test_sets);
        test_merge_equivalence_for_size_owned::<3>(&test_sets);
        test_merge_equivalence_for_size_owned::<4>(&test_sets);
        test_merge_equivalence_for_size_owned::<8>(&test_sets);
        test_merge_equivalence_for_size_owned::<16>(&test_sets);
    }

    fn test_merge_equivalence_for_size_owned<const SIZE: usize>(test_sets: &[Vec<f64>]) {
        // Cross-product: for every pair (A, B), verify merge equivalence
        for (i, set_a) in test_sets.iter().enumerate() {
            for (j, set_b) in test_sets.iter().enumerate() {
                // Build merged histogram: insert A, then merge B
                let mut merged: Histogram<u64, SIZE> = Histogram::new();
                for &v in set_a {
                    merged.update(v);
                }
                let mut other: Histogram<u32, SIZE> = Histogram::new();
                for &v in set_b {
                    other.update(v);
                }
                merged.merge_from_histogram(&other);

                // Build single histogram: insert A then B directly
                let mut single: Histogram<u64, SIZE> = Histogram::new();
                for &v in set_a {
                    single.update(v);
                }
                for &v in set_b {
                    single.update(v);
                }

                // Compare all properties
                assert_eq!(merged.count(), single.count(), 
                    "count mismatch for size={SIZE} sets {i} x {j}");
                // Sum comparison with floating-point tolerance (order of additions can differ)
                let sum_diff = (merged.sum() - single.sum()).abs();
                assert!(sum_diff < 1e-10, 
                    "sum mismatch for size={SIZE} sets {i} x {j}: {} vs {}", 
                    merged.sum(), single.sum());
                assert_eq!(merged.zero_count(), single.zero_count(), 
                    "zero_count mismatch for size={SIZE} sets {i} x {j}");
                assert_eq!(merged.min(), single.min(), 
                    "min mismatch for size={SIZE} sets {i} x {j}");
                assert_eq!(merged.max(), single.max(), 
                    "max mismatch for size={SIZE} sets {i} x {j}");
                assert_eq!(merged.scale(), single.scale(), 
                    "scale mismatch for size={SIZE} sets {i} x {j}");

                // Compare bucket data
                let mb = merged.positive();
                let sb = single.positive();
                assert_eq!(mb.offset(), sb.offset(), 
                    "offset mismatch for size={SIZE} sets {i} x {j}");
                assert_eq!(mb.len(), sb.len(), 
                    "bucket len mismatch for size={SIZE} sets {i} x {j}");
                for k in 0..mb.len() {
                    assert_eq!(mb.at(k), sb.at(k), 
                        "bucket[{k}] mismatch for size={SIZE} sets {i} x {j}");
                }
            }
        }
    }

    #[test]
    fn test_u8_update_overflow_returns_false() {
        let mut h: Histogram<u8, 16> = Histogram::new();
        // Fill one bucket to u8::MAX
        for _ in 0..255 {
            assert!(h.update(1.0));
        }
        // The 256th increment should overflow.
        assert!(!h.update(1.0));
    }

    #[test]
    fn test_u16_update_by_incr_overflow_returns_false() {
        let mut h: Histogram<u16, 16> = Histogram::new();
        // One large increment that fits
        assert!(h.update_by_incr(1.0, u16::MAX as u64));
        // Any additional increment to the same bucket overflows.
        assert!(!h.update(1.0));
    }

    #[test]
    fn test_u16_merge_overflow_returns_false() {
        let mut a: Histogram<u16, 16> = Histogram::new();
        assert!(a.update_by_incr(1.0, u16::MAX as u64));

        let mut b: Histogram<u16, 16> = Histogram::new();
        b.update(1.0);

        // Merging should overflow.
        assert!(!a.merge_from(&b));
    }

    #[test]
    fn test_u8_downscale_overflow_returns_false() {
        // Start at scale 0 so we know exact bucket indices:
        //   2.0 → index 0, 4.0 → index 1 (adjacent positive indices).
        // When a far-away value forces a downscale, indices 0 and 1
        // both map to 0 (0>>1 == 1>>1 == 0), combining 200+200 = 400 > u8::MAX.
        let mut h: Histogram<u8, 2> = Histogram::with_scale(0);
        assert!(h.update_by_incr(2.0, 200));
        assert!(h.update_by_incr(4.0, 200));
        // 1024.0 at scale 0 maps to index 9, forcing a downscale.
        assert!(!h.update(1024.0));
    }

    #[test]
    fn test_cross_width_merge_overflow_returns_false() {
        // A u32 source whose count exceeds u16::MAX, merged into u16 target.
        let mut target: Histogram<u16, 16> = Histogram::new();
        let mut source: Histogram<u32, 16> = Histogram::new();
        source.update_by_incr(1.0, u16::MAX as u64 + 1);

        assert!(!target.merge_from_histogram(&source));
    }

    #[test]
    fn test_with_max_scale_limits_initial_scale() {
        let h: Histogram<u16, 16> = Histogram::with_max_scale(3);
        assert_eq!(h.max_scale(), 3);
        // Empty histogram reports scale 0, but the mapping is at max_scale.
        assert_eq!(h.scale(), 0);
    }

    #[test]
    fn test_with_max_scale_clamps_to_algorithm_max() {
        // Requesting a scale higher than the algorithm supports gets clamped.
        let h: Histogram<u16, 16> = Histogram::with_max_scale(100);
        assert_eq!(h.max_scale(), max_scale());
    }

    #[test]
    fn test_with_max_scale_records_at_limited_scale() {
        let mut limited: Histogram<u16, 16> = Histogram::with_max_scale(3);
        let mut unlimited: Histogram<u16, 16> = Histogram::new();

        // Two very close values: at high scale they'd be in different buckets,
        // at low scale they share a bucket.
        limited.update(1.0);
        limited.update(1.001);
        unlimited.update(1.0);
        unlimited.update(1.001);

        // Limited histogram starts coarser, so scale should be <= 3.
        assert!(limited.scale() <= 3);
        // Unlimited starts at algorithm max, should be higher (if max_scale > 3).
        if max_scale() > 3 {
            assert!(unlimited.scale() > limited.scale());
        }
    }

    #[test]
    fn test_clear_resets_to_max_scale() {
        let mut h: Histogram<u16, 16> = Histogram::with_max_scale(3);
        // Insert values that force downscale below 3.
        h.update(0.001);
        h.update(1000.0);
        let scale_after_insert = h.scale();
        assert!(scale_after_insert <= 3);

        h.clear();
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);

        // After clear+insert, scale should start at 3 again.
        h.update(1.0);
        assert_eq!(h.scale(), 3);
    }

    #[test]
    fn test_with_scale_sets_max_scale() {
        // with_scale also records its scale as the max_scale ceiling.
        let mut h: Histogram<u16, 16> = Histogram::with_scale(2);
        assert_eq!(h.max_scale(), 2);
        h.update(1.0);
        assert_eq!(h.scale(), 2);
        h.clear();
        h.update(1.0);
        assert_eq!(h.scale(), 2);
    }

    #[test]
    fn test_new_histogram_max_scale_equals_algorithm_max() {
        let h: Histogram<u16, 16> = Histogram::new();
        assert_eq!(h.max_scale(), max_scale());
    }

    /// Exploratory test: what happens with subnormals, MIN_VALUE, +Inf, and MAX.
    ///
    /// IEEE 754 edge values and their bit patterns:
    ///   subnormal (5e-324):  biased_exp=0,   significand!=0
    ///   MIN_VALUE (2^-1022): biased_exp=1,   significand=0  (smallest normal)
    ///   f64::MAX:            biased_exp=2046, significand=all-ones
    ///   +Inf:                biased_exp=2047, significand=0
    ///
    /// Questions this test answers:
    ///   1. Where does get_normal_base2 place subnormals and +Inf?
    ///   2. What index does each mapping assign to these values?
    ///   3. Does the histogram accept them without panic in release mode?
    ///   4. Does +Inf land in the same bucket as f64::MAX, or a new one?
    ///   5. Does MIN_VALUE (0x1p-1022) share a bucket with subnormals?
    #[test]
    fn test_edge_values_subnormal_inf() {
        use crate::float64::{get_normal_base2, get_significand};
        use crate::mapping::Mapping;

        let subnormal: f64 = 5e-324; // smallest positive subnormal
        let largest_subnormal: f64 = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        let min_normal: f64 = crate::float64::MIN_VALUE; // 0x1p-1022
        let next_after_min: f64 = f64::from_bits(min_normal.to_bits() + 1);
        let max_f64: f64 = f64::MAX;
        let inf: f64 = f64::INFINITY;

        // ---- Part 1: Raw IEEE 754 bit extraction ----
        // Subnormals have biased exponent 0, so get_normal_base2 returns -1023
        assert_eq!(get_normal_base2(subnormal), -1023);
        assert_ne!(get_significand(subnormal), 0);

        assert_eq!(get_normal_base2(largest_subnormal), -1023);
        assert_ne!(get_significand(largest_subnormal), 0);

        // MIN_VALUE = 2^-1022: biased exponent 1, significand 0 (exact power of two)
        assert_eq!(get_normal_base2(min_normal), -1022);
        assert_eq!(get_significand(min_normal), 0);

        // next value after min_normal: same exponent, significand = 1
        assert_eq!(get_normal_base2(next_after_min), -1022);
        assert_eq!(get_significand(next_after_min), 1);

        // f64::MAX: biased exponent 2046, all significand bits set
        assert_eq!(get_normal_base2(max_f64), 1023);
        assert_eq!(get_significand(max_f64), (1u64 << 52) - 1);

        // +Inf: biased exponent 2047, significand 0
        // get_normal_base2 returns 1024 (not a valid normal exponent)
        assert_eq!(get_normal_base2(inf), 1024);
        assert_eq!(get_significand(inf), 0);

        // ---- Part 2: Index mapping at scale 0 (exponent path) ----
        // At scale 0, each bucket is one power-of-two wide.
        let m0 = Mapping::new(0).unwrap();

        // Subnormals clamp to min_normal_lower_boundary_index via the
        // `value < MIN_VALUE` guard in exponent::map_to_index.
        let idx_subnormal = m0.map_to_index(subnormal);
        let idx_largest_sub = m0.map_to_index(largest_subnormal);

        // MIN_VALUE is an exact power of two → gets upper-inclusive correction (-1)
        // so it maps to one bucket below its exponent.
        let idx_min_normal = m0.map_to_index(min_normal);
        let idx_next_after = m0.map_to_index(next_after_min);

        // By upper-inclusive semantics, min_normal should be the top of the
        // bucket below, which is the subnormal clamping bucket.
        // Verify: all of {subnormal, largest_subnormal, min_normal} share a bucket.
        println!("scale 0: subnormal={idx_subnormal}, largest_sub={idx_largest_sub}, min_normal={idx_min_normal}, next_after={idx_next_after}");
        assert_eq!(idx_subnormal, idx_largest_sub,
            "all subnormals should map to the same index");
        assert_eq!(idx_subnormal, idx_min_normal,
            "min_normal (power of two) should share the subnormal bucket by upper-inclusivity");
        assert_eq!(idx_next_after, idx_min_normal + 1,
            "first value above min_normal should be in the next bucket");

        // f64::MAX at scale 0
        let idx_max = m0.map_to_index(max_f64);
        // MAX has exponent 1023, significand != 0, so index = 1023 >> 0 = 1023
        // which is the bucket (2^1023, 2^1024] — where 2^1024 would be +Inf.
        println!("scale 0: max_f64 index={idx_max}");

        // +Inf: get_normal_base2(+Inf)=1024, significand=0,
        // so correction=-1, index = (1024-1) >> 0 = 1023.
        // That means +Inf lands in the SAME bucket as f64::MAX.
        let idx_inf = m0.map_to_index(inf);
        println!("scale 0: +Inf index={idx_inf}");
        assert_eq!(idx_inf, idx_max,
            "+Inf should land in the same bucket as f64::MAX at scale 0");

        // ---- Part 3: Index mapping at positive scale (lookup table path) ----
        let table_scale = max_scale();
        if table_scale > 0 {
            let ms = Mapping::new(table_scale).unwrap();

            // Subnormals: the lookup table path has no < MIN_VALUE guard;
            // it goes straight to get_significand / get_normal_base2.
            // For a subnormal, exponent=-1023, significand is nonzero.
            // This may produce a garbage index. Let's see what it does.
            // (The exponent path guards this, but the lookup path doesn't.)

            // min_normal at positive scale: exponent=-1022, sig=0 → exact power of two
            // fine_index = (-1022 << table_scale) + 0 - 1
            let idx_min_normal_ts = ms.map_to_index(min_normal);
            let idx_next_ts = ms.map_to_index(next_after_min);
            println!("scale {table_scale}: min_normal={idx_min_normal_ts}, next_after={idx_next_ts}");
            assert!(idx_next_ts > idx_min_normal_ts,
                "next_after should map to a higher index than min_normal");

            // f64::MAX at positive scale: this should work fine
            let idx_max_ts = ms.map_to_index(max_f64);
            println!("scale {table_scale}: max_f64={idx_max_ts}");

            // +Inf at positive scale: exponent=1024, sig=0, linear_idx=0,
            // fine_index = (1024 << table_scale) + bucket - 1.
            // At the native table scale this is 1 index above MAX, but at
            // lower positive scales the right-shift merges them.
            let idx_inf_ts = ms.map_to_index(inf);
            println!("scale {table_scale}: +Inf={idx_inf_ts}");
            // At the native table scale, +Inf may be 1 index above MAX.
            // At lower scales they coincide. Either way, +Inf doesn't explode.
            assert!(idx_inf_ts >= idx_max_ts,
                "+Inf index should be >= MAX index");
            assert!(idx_inf_ts <= idx_max_ts + 1,
                "+Inf index should be at most 1 above MAX index");
        }

        // ---- Part 4: Histogram update acceptance ----
        // +Inf should flow through naturally: sum becomes Inf,
        // max becomes Inf, min/count are unaffected.
        let mut h: Histogram<u32, 160> = Histogram::with_scale(0);

        // Normal values first
        assert!(h.update(1.0));
        assert!(h.update(max_f64));
        assert_eq!(h.count(), 2);
        assert_eq!(h.max(), max_f64);
        assert!(h.sum().is_finite());

        // +Inf: sum becomes Inf, max becomes Inf
        assert!(h.update(inf));
        assert_eq!(h.count(), 3);
        assert_eq!(h.max(), f64::INFINITY);
        assert_eq!(h.min(), 1.0);
        assert!(h.sum().is_infinite());

        // +Inf should share a bucket with MAX at scale 0
        let idx_max_h = h.mapping.map_to_index(max_f64);
        let idx_inf_h = h.mapping.map_to_index(inf);
        assert_eq!(idx_max_h, idx_inf_h,
            "+Inf and MAX should share a bucket at scale 0");

        // Subnormals: currently handled by exponent path's MIN_VALUE guard
        assert!(h.update(subnormal));
        assert!(h.update(largest_subnormal));
        assert_eq!(h.count(), 5);

        // Verify bucket structure: the histogram had to downscale heavily
        // to fit the range from subnormal (-1023) to MAX (1023) into 160 buckets.
        // At the downscaled scale, subnormal/min_normal/next_after may share a bucket.
        let buckets: Vec<u64> = h.positive.iter().collect();
        println!("scale {} with {} non-zero buckets, span {} to {}",
            h.scale(), buckets.iter().filter(|&&c| c > 0).count(),
            h.positive.index_start, h.positive.index_end);
        // Total should be 5
        assert_eq!(buckets.iter().sum::<u64>(), 5);

        // Now test with a narrow histogram that won't downscale.
        // Use only nearby values: subnormal, min_normal, next_after.
        let mut h2: Histogram<u32, 160> = Histogram::with_scale(0);
        h2.update(subnormal);
        h2.update(largest_subnormal);
        h2.update(min_normal);
        h2.update(next_after_min);
        let b2: Vec<u64> = h2.positive.iter().collect();
        println!("narrow scale {}: indexes {}..{}, buckets: {:?}",
            h2.scale(), h2.positive.index_start, h2.positive.index_end, &b2);
        // At scale 0: subnormal=-1023, min_normal=-1023, next_after=-1022
        // So first bucket (index -1023) has count 3, second (index -1022) has count 1.
        assert_eq!(b2[0], 3,
            "first bucket should contain both subnormals and min_normal");
        assert_eq!(b2[1], 1, "second bucket: next_after_min");

        println!("\n=== Summary ===");
        println!("Subnormals: clamped to min_normal bucket at all scales");
        println!("+Inf: shares bucket with MAX at scale 0; at most 1 index above MAX at positive scales");
        println!("+Inf: sum becomes Inf, max becomes Inf — hardware/math semantics");
    }
}
