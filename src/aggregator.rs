// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Runtime-configurable exponential histogram with three resolution levels
//! and automatic counter widening.
//!
//! [`ExpoHistogram`] wraps the fixed-size [`Histogram`] in an enum, letting
//! users choose bucket resolution at construction time rather than via const
//! generics. Counters start at u16 and automatically widen to u32 then u64
//! when a bucket would overflow.

use crate::histogram::{Counter, Histogram};

/// Number of buckets for the [`Resolution::Small`] variant.
pub const SMALL_SIZE: usize = 16;

/// Number of buckets for the [`Resolution::Large`] variant.
pub const LARGE_SIZE: usize = 160;

/// Minimum statistics without bucket resolution (min, max, sum, count).
#[derive(Debug, Clone)]
pub struct MMSC {
    sum: f64,
    count: u64,
    zero_count: u64,
    min: f64,
    max: f64,
}

impl MMSC {
    /// Creates an empty MMSC.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
        }
    }

    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    #[inline]
    pub fn zero_count(&self) -> u64 {
        self.zero_count
    }

    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    #[inline]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Records a value with a specified increment.
    ///
    /// Returns `false` if a counter overflowed.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) -> bool {
        debug_assert!(value >= 0.0, "MMSC only accepts non-negative values");

        let new_count = match self.count.checked_add(incr) {
            Some(c) => c,
            None => return false,
        };

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
        } else {
            self.sum += value * incr as f64;
        }

        true
    }

    /// Clears all statistics.
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Merges another source's statistics into this MMSC.
    fn merge_stats(
        &mut self,
        other_count: u64,
        other_zero_count: u64,
        other_sum: f64,
        other_min: f64,
        other_max: f64,
    ) -> bool {
        if other_count == 0 {
            return true;
        }
        let new_count = match self.count.checked_add(other_count) {
            Some(c) => c,
            None => return false,
        };
        let new_zc = match self.zero_count.checked_add(other_zero_count) {
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
        self.count = new_count;
        self.zero_count = new_zc;
        self.sum += other_sum;
        true
    }
}

impl Default for MMSC {
    fn default() -> Self {
        Self::new()
    }
}

/// Desired bucket resolution for an [`ExpoHistogram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// No buckets — only min, max, sum, count, zero_count.
    Empty,
    /// 16 buckets.
    Small,
    /// 160 buckets.
    Large,
}

/// The current counter width of a bucketed variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterWidth {
    /// 16-bit counters (max 65,535 per bucket).
    U16,
    /// 32-bit counters (max ~4 billion per bucket).
    U32,
    /// 64-bit counters.
    U64,
}

/// Runtime-configurable exponential histogram.
///
/// Wraps [`Histogram`] for two fixed bucket sizes (16, 160) with automatic
/// counter widening. Counters start at u16 and widen to u32 then u64 when a
/// bucket overflow occurs during `update` or `merge`.
///
/// The [`Resolution::Empty`] variant tracks only min/max/sum/count with no
/// buckets at all.
#[derive(Debug, Clone)]
pub enum ExpoHistogram {
    /// Statistics only, no bucket resolution.
    Empty(MMSC),
    /// 16 buckets, u16 counters.
    Small16(Box<Histogram<u16, SMALL_SIZE>>),
    /// 16 buckets, u32 counters.
    Small32(Box<Histogram<u32, SMALL_SIZE>>),
    /// 16 buckets, u64 counters.
    Small64(Box<Histogram<u64, SMALL_SIZE>>),
    /// 160 buckets, u16 counters.
    Large16(Box<Histogram<u16, LARGE_SIZE>>),
    /// 160 buckets, u32 counters.
    Large32(Box<Histogram<u32, LARGE_SIZE>>),
    /// 160 buckets, u64 counters.
    Large64(Box<Histogram<u64, LARGE_SIZE>>),
}

impl ExpoHistogram {
    /// Creates a new histogram with the given resolution at the maximum
    /// supported scale. Counters start at u16.
    pub fn new(resolution: Resolution) -> Self {
        match resolution {
            Resolution::Empty => Self::Empty(MMSC::new()),
            Resolution::Small => Self::Small16(Box::new(Histogram::new())),
            Resolution::Large => Self::Large16(Box::new(Histogram::new())),
        }
    }

    /// Creates a new histogram with the given resolution and an upper bound
    /// on scale. Counters start at u16.
    pub fn with_max_scale(resolution: Resolution, max_scale: i32) -> Self {
        match resolution {
            Resolution::Empty => Self::Empty(MMSC::new()),
            Resolution::Small => Self::Small16(Box::new(Histogram::with_max_scale(max_scale))),
            Resolution::Large => Self::Large16(Box::new(Histogram::with_max_scale(max_scale))),
        }
    }

    /// Returns which resolution this histogram was created with.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        match self {
            Self::Empty(_) => Resolution::Empty,
            Self::Small16(_) | Self::Small32(_) | Self::Small64(_) => Resolution::Small,
            Self::Large16(_) | Self::Large32(_) | Self::Large64(_) => Resolution::Large,
        }
    }

    /// Returns the current counter width of the bucketed variants.
    ///
    /// Returns `None` for the [`Resolution::Empty`] variant.
    #[inline]
    pub fn counter_width(&self) -> Option<CounterWidth> {
        match self {
            Self::Empty(_) => None,
            Self::Small16(_) | Self::Large16(_) => Some(CounterWidth::U16),
            Self::Small32(_) | Self::Large32(_) => Some(CounterWidth::U32),
            Self::Small64(_) | Self::Large64(_) => Some(CounterWidth::U64),
        }
    }

    /// Returns the sum of all recorded values.
    #[inline]
    pub fn sum(&self) -> f64 {
        dispatch!(self, sum())
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        dispatch!(self, count())
    }

    /// Returns the count of zero values.
    #[inline]
    pub fn zero_count(&self) -> u64 {
        dispatch!(self, zero_count())
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        dispatch!(self, min())
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
        dispatch!(self, max())
    }

    /// Returns the current scale.
    ///
    /// Always returns 0 for the [`Resolution::Empty`] variant.
    #[inline]
    pub fn scale(&self) -> i32 {
        match self {
            Self::Empty(_) => 0,
            Self::Small16(h) => h.scale(),
            Self::Small32(h) => h.scale(),
            Self::Small64(h) => h.scale(),
            Self::Large16(h) => h.scale(),
            Self::Large32(h) => h.scale(),
            Self::Large64(h) => h.scale(),
        }
    }

    /// Returns the maximum scale this histogram will use on reset.
    ///
    /// Always returns 0 for the [`Resolution::Empty`] variant.
    #[inline]
    pub fn max_scale(&self) -> i32 {
        match self {
            Self::Empty(_) => 0,
            Self::Small16(h) => h.max_scale(),
            Self::Small32(h) => h.max_scale(),
            Self::Small64(h) => h.max_scale(),
            Self::Large16(h) => h.max_scale(),
            Self::Large32(h) => h.max_scale(),
            Self::Large64(h) => h.max_scale(),
        }
    }

    /// Returns the bucket offset (smallest bucket index).
    #[inline]
    pub fn bucket_offset(&self) -> i32 {
        match self {
            Self::Empty(_) => 0,
            Self::Small16(h) => h.positive().offset(),
            Self::Small32(h) => h.positive().offset(),
            Self::Small64(h) => h.positive().offset(),
            Self::Large16(h) => h.positive().offset(),
            Self::Large32(h) => h.positive().offset(),
            Self::Large64(h) => h.positive().offset(),
        }
    }

    /// Returns the number of buckets in use.
    #[inline]
    pub fn bucket_len(&self) -> u32 {
        match self {
            Self::Empty(_) => 0,
            Self::Small16(h) => h.positive().len(),
            Self::Small32(h) => h.positive().len(),
            Self::Small64(h) => h.positive().len(),
            Self::Large16(h) => h.positive().len(),
            Self::Large32(h) => h.positive().len(),
            Self::Large64(h) => h.positive().len(),
        }
    }

    /// Returns the count in bucket at position `pos` (0-indexed from offset).
    #[inline]
    pub fn bucket_at(&self, pos: u32) -> u64 {
        match self {
            Self::Empty(_) => 0,
            Self::Small16(h) => h.positive().at(pos),
            Self::Small32(h) => h.positive().at(pos),
            Self::Small64(h) => h.positive().at(pos),
            Self::Large16(h) => h.positive().at(pos),
            Self::Large32(h) => h.positive().at(pos),
            Self::Large64(h) => h.positive().at(pos),
        }
    }

    /// Records a single value.
    ///
    /// Widens the counter type automatically on overflow.
    #[inline]
    pub fn update(&mut self, value: f64) -> bool {
        self.update_by_incr(value, 1)
    }

    /// Records a value with a specified increment.
    ///
    /// Widens the counter type automatically on overflow. Returns `false`
    /// only if the u64 count/zero_count overflows or u64 counters overflow.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) -> bool {
        match self {
            Self::Empty(m) => return m.update_by_incr(value, incr),
            Self::Small16(h) => {
                let snapshot = h.clone();
                if h.update_by_incr(value, incr) { return true; }
                let mut wider: Box<Histogram<u32, SMALL_SIZE>> = Box::new(snapshot.widen_into());
                if !wider.update_by_incr(value, incr) { return false; }
                *self = Self::Small32(wider);
            }
            Self::Small32(h) => {
                let snapshot = h.clone();
                if h.update_by_incr(value, incr) { return true; }
                let mut wider: Box<Histogram<u64, SMALL_SIZE>> = Box::new(snapshot.widen_into());
                if !wider.update_by_incr(value, incr) { return false; }
                *self = Self::Small64(wider);
            }
            Self::Small64(h) => return h.update_by_incr(value, incr),
            Self::Large16(h) => {
                let snapshot = h.clone();
                if h.update_by_incr(value, incr) { return true; }
                let mut wider: Box<Histogram<u32, LARGE_SIZE>> = Box::new(snapshot.widen_into());
                if !wider.update_by_incr(value, incr) { return false; }
                *self = Self::Large32(wider);
            }
            Self::Large32(h) => {
                let snapshot = h.clone();
                if h.update_by_incr(value, incr) { return true; }
                let mut wider: Box<Histogram<u64, LARGE_SIZE>> = Box::new(snapshot.widen_into());
                if !wider.update_by_incr(value, incr) { return false; }
                *self = Self::Large64(wider);
            }
            Self::Large64(h) => return h.update_by_incr(value, incr),
        }
        true
    }

    /// Clears the histogram, resetting to initial state.
    ///
    /// Preserves resolution but resets counter width back to u16.
    pub fn clear(&mut self) {
        match self {
            Self::Empty(m) => m.clear(),
            Self::Small16(h) => h.clear(),
            Self::Small32(h) => {
                *self = Self::Small16(Box::new(Histogram::with_max_scale(h.max_scale())));
            }
            Self::Small64(h) => {
                *self = Self::Small16(Box::new(Histogram::with_max_scale(h.max_scale())));
            }
            Self::Large16(h) => h.clear(),
            Self::Large32(h) => {
                *self = Self::Large16(Box::new(Histogram::with_max_scale(h.max_scale())));
            }
            Self::Large64(h) => {
                *self = Self::Large16(Box::new(Histogram::with_max_scale(h.max_scale())));
            }
        }
    }

    /// Swaps contents with another histogram (works across resolutions).
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Merges another histogram into this one. The source may be any
    /// resolution or counter width.
    ///
    /// Widens the counter type automatically on overflow. Returns `false`
    /// only if the u64 count/zero_count overflows or u64 counters overflow.
    pub fn merge_from(&mut self, other: &Self) -> bool {
        match self {
            Self::Empty(m) => {
                return m.merge_stats(
                    other.count(),
                    other.zero_count(),
                    other.sum(),
                    other.min(),
                    other.max(),
                );
            }
            // For widenable variants: snapshot, try, widen snapshot on failure.
            Self::Small16(h) => {
                let snapshot = h.clone();
                if merge_raw_into(h.as_mut(), other) { return true; }
                let mut wider: Box<Histogram<u32, SMALL_SIZE>> = Box::new(snapshot.widen_into());
                if !merge_raw_into(wider.as_mut(), other) { return false; }
                *self = Self::Small32(wider);
            }
            Self::Small32(h) => {
                let snapshot = h.clone();
                if merge_raw_into(h.as_mut(), other) { return true; }
                let mut wider: Box<Histogram<u64, SMALL_SIZE>> = Box::new(snapshot.widen_into());
                if !merge_raw_into(wider.as_mut(), other) { return false; }
                *self = Self::Small64(wider);
            }
            Self::Small64(h) => return merge_raw_into(h.as_mut(), other),
            Self::Large16(h) => {
                let snapshot = h.clone();
                if merge_raw_into(h.as_mut(), other) { return true; }
                let mut wider: Box<Histogram<u32, LARGE_SIZE>> = Box::new(snapshot.widen_into());
                if !merge_raw_into(wider.as_mut(), other) { return false; }
                *self = Self::Large32(wider);
            }
            Self::Large32(h) => {
                let snapshot = h.clone();
                if merge_raw_into(h.as_mut(), other) { return true; }
                let mut wider: Box<Histogram<u64, LARGE_SIZE>> = Box::new(snapshot.widen_into());
                if !merge_raw_into(wider.as_mut(), other) { return false; }
                *self = Self::Large64(wider);
            }
            Self::Large64(h) => return merge_raw_into(h.as_mut(), other),
        }
        true
    }

}

/// Dispatch a no-arg stats getter across all variants.
macro_rules! dispatch {
    ($self:expr, $method:ident()) => {
        match $self {
            ExpoHistogram::Empty(m) => m.$method(),
            ExpoHistogram::Small16(h) => h.$method(),
            ExpoHistogram::Small32(h) => h.$method(),
            ExpoHistogram::Small64(h) => h.$method(),
            ExpoHistogram::Large16(h) => h.$method(),
            ExpoHistogram::Large32(h) => h.$method(),
            ExpoHistogram::Large64(h) => h.$method(),
        }
    };
}
use dispatch;

/// Helper: extract raw bucket data from any `ExpoHistogram` source and merge
/// into a `Histogram<C, SIZE>`.
fn merge_raw_into<C: Counter, const SIZE: usize>(
    target: &mut Histogram<C, SIZE>,
    source: &ExpoHistogram,
) -> bool {
    match source {
        ExpoHistogram::Empty(m) => target.merge_from_raw(
            m.count, m.zero_count, m.sum, m.min, m.max,
            0, 0, 0, &|_| 0,
        ),
        ExpoHistogram::Small16(h) => merge_hist_raw_into(target, h),
        ExpoHistogram::Small32(h) => merge_hist_raw_into(target, h),
        ExpoHistogram::Small64(h) => merge_hist_raw_into(target, h),
        ExpoHistogram::Large16(h) => merge_hist_raw_into(target, h),
        ExpoHistogram::Large32(h) => merge_hist_raw_into(target, h),
        ExpoHistogram::Large64(h) => merge_hist_raw_into(target, h),
    }
}

fn merge_hist_raw_into<C: Counter, C2: Counter, const SIZE: usize, const SIZE2: usize>(
    target: &mut Histogram<C, SIZE>,
    source: &Histogram<C2, SIZE2>,
) -> bool {
    let b = source.positive();
    target.merge_from_raw(
        source.count(), source.zero_count(), source.sum(),
        source.min(), source.max(),
        source.scale(), b.offset(), b.len(), &|i| b.at(i),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_resolution() {
        let mut h = ExpoHistogram::new(Resolution::Empty);
        assert_eq!(h.resolution(), Resolution::Empty);
        assert!(h.update(1.0));
        assert!(h.update(2.0));
        assert!(h.update(0.0));

        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 3.0);
        assert_eq!(h.min(), 0.0);
        assert_eq!(h.max(), 2.0);
        assert_eq!(h.zero_count(), 1);
        assert_eq!(h.bucket_len(), 0);
        assert_eq!(h.scale(), 0);
    }

    #[test]
    fn test_small_resolution() {
        let mut h = ExpoHistogram::new(Resolution::Small);
        assert_eq!(h.resolution(), Resolution::Small);
        assert_eq!(h.counter_width(), Some(CounterWidth::U16));
        h.update(1.0);
        h.update(2.0);
        assert_eq!(h.count(), 2);
        assert_eq!(h.sum(), 3.0);
        assert!(h.bucket_len() > 0);
    }

    #[test]
    fn test_large_resolution() {
        let mut h = ExpoHistogram::new(Resolution::Large);
        assert_eq!(h.resolution(), Resolution::Large);
        assert_eq!(h.counter_width(), Some(CounterWidth::U16));
        h.update(1.0);
        h.update(2.0);
        assert_eq!(h.count(), 2);
        assert_eq!(h.sum(), 3.0);
        assert!(h.bucket_len() > 0);
    }

    #[test]
    fn test_with_max_scale() {
        let mut h = ExpoHistogram::with_max_scale(Resolution::Small, 3);
        assert_eq!(h.max_scale(), 3);
        h.update(1.0);
        assert!(h.scale() <= 3);
    }

    #[test]
    fn test_merge_same_resolution() {
        let mut a = ExpoHistogram::new(Resolution::Small);
        let mut b = ExpoHistogram::new(Resolution::Small);
        a.update(1.0);
        a.update(2.0);
        b.update(3.0);
        b.update(4.0);
        assert!(a.merge_from(&b));
        assert_eq!(a.count(), 4);
        assert_eq!(a.sum(), 10.0);
        assert_eq!(a.min(), 1.0);
        assert_eq!(a.max(), 4.0);
    }

    #[test]
    fn test_merge_small_into_large() {
        let mut large = ExpoHistogram::new(Resolution::Large);
        let mut small = ExpoHistogram::new(Resolution::Small);
        large.update(1.0);
        small.update(2.0);
        small.update(3.0);
        assert!(large.merge_from(&small));
        assert_eq!(large.count(), 3);
        assert_eq!(large.sum(), 6.0);
    }

    #[test]
    fn test_merge_large_into_small() {
        let mut small = ExpoHistogram::new(Resolution::Small);
        let mut large = ExpoHistogram::new(Resolution::Large);
        small.update(1.0);
        large.update(2.0);
        large.update(3.0);
        assert!(small.merge_from(&large));
        assert_eq!(small.count(), 3);
        assert_eq!(small.sum(), 6.0);
    }

    #[test]
    fn test_merge_into_empty() {
        let mut empty = ExpoHistogram::new(Resolution::Empty);
        let mut large = ExpoHistogram::new(Resolution::Large);
        large.update(5.0);
        large.update(10.0);
        assert!(empty.merge_from(&large));
        assert_eq!(empty.count(), 2);
        assert_eq!(empty.sum(), 15.0);
        assert_eq!(empty.min(), 5.0);
        assert_eq!(empty.max(), 10.0);
        assert_eq!(empty.bucket_len(), 0);
    }

    #[test]
    fn test_merge_empty_into_large() {
        let mut large = ExpoHistogram::new(Resolution::Large);
        let mut empty = ExpoHistogram::new(Resolution::Empty);
        large.update(1.0);
        empty.update_by_incr(2.0, 3);
        assert!(large.merge_from(&empty));
        assert_eq!(large.count(), 4);
        assert_eq!(large.sum(), 7.0);
    }

    #[test]
    fn test_clear_preserves_resolution() {
        let mut h = ExpoHistogram::with_max_scale(Resolution::Large, 4);
        h.update(1.0);
        h.update(100.0);
        h.clear();
        assert_eq!(h.count(), 0);
        assert_eq!(h.resolution(), Resolution::Large);
        assert_eq!(h.max_scale(), 4);
    }

    #[test]
    fn test_swap_cross_resolution() {
        let mut a = ExpoHistogram::new(Resolution::Small);
        let mut b = ExpoHistogram::new(Resolution::Large);
        a.update(1.0);
        b.update(2.0);

        a.swap(&mut b);

        assert_eq!(a.resolution(), Resolution::Large);
        assert_eq!(a.sum(), 2.0);
        assert_eq!(b.resolution(), Resolution::Small);
        assert_eq!(b.sum(), 1.0);
    }

    #[test]
    fn test_merge_equivalence_across_resolutions() {
        let values = &[0.5, 1.0, 2.0, 5.0, 10.0, 0.0];

        let mut target_a = ExpoHistogram::new(Resolution::Large);
        let mut source_small = ExpoHistogram::new(Resolution::Small);
        for &v in values {
            source_small.update(v);
        }
        target_a.merge_from(&source_small);

        let mut target_b = ExpoHistogram::new(Resolution::Large);
        let mut source_large = ExpoHistogram::new(Resolution::Large);
        for &v in values {
            source_large.update(v);
        }
        target_b.merge_from(&source_large);

        assert_eq!(target_a.count(), target_b.count());
        assert!((target_a.sum() - target_b.sum()).abs() < 1e-10);
        assert_eq!(target_a.zero_count(), target_b.zero_count());
        assert_eq!(target_a.min(), target_b.min());
        assert_eq!(target_a.max(), target_b.max());
    }

    #[test]
    fn test_large_has_finer_resolution_than_small() {
        let mut small = ExpoHistogram::new(Resolution::Small);
        let mut large = ExpoHistogram::new(Resolution::Large);

        for v in [1.0, 2.0, 4.0, 8.0, 16.0] {
            small.update(v);
            large.update(v);
        }

        assert!(large.scale() >= small.scale());
    }

    // --- Counter widening tests ---

    #[test]
    fn test_auto_widen_u16_to_u32_on_update() {
        let mut h = ExpoHistogram::new(Resolution::Small);
        assert_eq!(h.counter_width(), Some(CounterWidth::U16));

        // Fill one bucket to u16::MAX
        assert!(h.update_by_incr(1.0, u16::MAX as u64));
        assert_eq!(h.counter_width(), Some(CounterWidth::U16));

        // One more overflows u16 — should auto-widen to u32.
        assert!(h.update(1.0));
        assert_eq!(h.counter_width(), Some(CounterWidth::U32));
        assert_eq!(h.count(), u16::MAX as u64 + 1);
        assert_eq!(h.resolution(), Resolution::Small);
    }

    #[test]
    fn test_auto_widen_u32_to_u64_on_update() {
        let mut h = ExpoHistogram::new(Resolution::Small);

        // Force to u32 first.
        assert!(h.update_by_incr(1.0, u16::MAX as u64));
        assert!(h.update(1.0));
        assert_eq!(h.counter_width(), Some(CounterWidth::U32));

        // Now fill to u32::MAX.
        assert!(h.update_by_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1)));
        assert_eq!(h.counter_width(), Some(CounterWidth::U32));

        // One more overflows u32 — should auto-widen to u64.
        assert!(h.update(1.0));
        assert_eq!(h.counter_width(), Some(CounterWidth::U64));
        assert_eq!(h.count(), u32::MAX as u64 + 1);
    }

    #[test]
    fn test_auto_widen_on_merge() {
        let mut target = ExpoHistogram::new(Resolution::Small);
        assert!(target.update_by_incr(1.0, u16::MAX as u64));
        assert_eq!(target.counter_width(), Some(CounterWidth::U16));

        let mut source = ExpoHistogram::new(Resolution::Small);
        source.update(1.0);

        // Merge would overflow u16, triggers widening.
        assert!(target.merge_from(&source));
        assert_eq!(target.counter_width(), Some(CounterWidth::U32));
        assert_eq!(target.count(), u16::MAX as u64 + 1);
    }

    #[test]
    fn test_auto_widen_large_variant() {
        let mut h = ExpoHistogram::new(Resolution::Large);
        assert_eq!(h.counter_width(), Some(CounterWidth::U16));

        assert!(h.update_by_incr(1.0, u16::MAX as u64));
        assert!(h.update(1.0));
        assert_eq!(h.counter_width(), Some(CounterWidth::U32));
        assert_eq!(h.resolution(), Resolution::Large);
        assert_eq!(h.count(), u16::MAX as u64 + 1);
    }

    #[test]
    fn test_clear_resets_counter_width() {
        let mut h = ExpoHistogram::with_max_scale(Resolution::Small, 4);

        // Force widening.
        h.update_by_incr(1.0, u16::MAX as u64);
        h.update(1.0);
        assert_eq!(h.counter_width(), Some(CounterWidth::U32));

        h.clear();
        assert_eq!(h.counter_width(), Some(CounterWidth::U16));
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 4);
    }

    #[test]
    fn test_auto_widen_preserves_bucket_data() {
        let mut h = ExpoHistogram::new(Resolution::Small);

        // Put different values in different buckets, then overflow one.
        h.update_by_incr(1.0, 100);
        h.update_by_incr(2.0, 200);
        h.update_by_incr(4.0, u16::MAX as u64);

        let count_before = h.count();
        let sum_before = h.sum();
        let scale_before = h.scale();

        // This overflows u16 on the 4.0 bucket → triggers widening.
        assert!(h.update(4.0));
        assert_eq!(h.counter_width(), Some(CounterWidth::U32));
        assert_eq!(h.count(), count_before + 1);
        assert!((h.sum() - (sum_before + 4.0)).abs() < 1e-10);
        assert_eq!(h.scale(), scale_before);
    }

    #[test]
    fn test_auto_widen_downscale_overflow() {
        // A downscale can also overflow counters when two u16 buckets are
        // combined. Verify the widening path handles this.
        let mut h = ExpoHistogram::with_max_scale(Resolution::Small, 0);
        // At scale 0: 2.0→index 0, 4.0→index 1. Fill both close to max.
        h.update_by_incr(2.0, u16::MAX as u64 - 100);
        h.update_by_incr(4.0, u16::MAX as u64 - 100);

        // 131072 = 2^17, index 16 at scale 0. With SIZE=16 the span (16)
        // forces a downscale that merges indices 0 and 1, overflowing u16.
        assert!(h.update(131072.0));
        assert!(h.counter_width() == Some(CounterWidth::U32)
             || h.counter_width() == Some(CounterWidth::U64));
    }
}
