// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram with a unified flat memory layout.
//!
//! `Histogram<N>` stores everything in fixed struct fields plus a `[u64; N]`
//! data pool used for bucket counters.

use core::fmt;

use crate::float64::{NAN_INF_BIASED, get_biased_exponent, get_significand, unbias_exponent};
use crate::mapping::{Scale, ScaleError, table_scale};

mod bucket_ops;
mod merge;
mod swar;
pub mod width;

mod bucket_view;
pub use bucket_view::{BucketView, BucketsIter};

#[cfg(feature = "quantile")]
mod quantile;
#[cfg(feature = "quantile")]
pub use quantile::{QuantileIter, QuantileValue};

mod view;
pub use view::HistogramView;
pub use width::{SlotAddr, Width};

/// Compact histogram configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Settings {
    scale: Scale,
    width: Width,
}

impl Settings {
    /// Creates settings from a scale and width.
    #[inline]
    pub const fn new(scale: Scale, width: Width) -> Self {
        Self { scale, width }
    }

    /// Returns the scale.
    #[inline]
    pub const fn scale(&self) -> Scale {
        self.scale
    }

    /// Returns the width.
    #[inline]
    pub const fn width(&self) -> Width {
        self.width
    }
}

/// Error returned when the total count would exceed `u64::MAX`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    /// Overflow of a u64 counter.
    Overflow,
    /// Extreme values like Inf, NaN, and zero values.
    Extreme,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "histogram total count overflow",
            Self::Extreme => "extreme value received",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Aggregate statistics of a histogram: count, sum, min, max.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    /// Total number of observations.
    pub count: u64,
    /// Sum of all observed values.
    pub sum: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
}

impl Stats {
    /// Empty stats.
    pub const EMPTY: Self = Self {
        count: 0,
        sum: 0.0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
}

/// Describes the bucket layout of an exponential histogram.
///
/// Used by [`Histogram::merge_from_raw`] to pass the source histogram's
/// bucket metadata without requiring a full `Histogram` instance.
#[derive(Debug, Clone, Copy)]
pub struct BucketDescriptor {
    /// Exponential histogram scale.
    pub scale: i32,
    /// Index of the first bucket.
    pub offset: i32,
    /// Number of contiguous buckets. Note that some buckets may be
    /// zero, including at the extremes. Callers are expected to skip
    /// and adjust for leading/trailing buckets.
    pub len: u32,
}

/// High-low range helper.
#[derive(Debug, Clone, Copy)]
struct HighLow {
    low: i32,
    high: i32,
}

impl HighLow {
    /// Computes how much downscaling is needed.
    #[inline]
    const fn change_steps(mut self, size: usize) -> u32 {
        let mut change = 0;
        while (self.high - self.low) as usize >= size {
            self.high >>= 1;
            self.low >>= 1;
            change += 1;
        }
        change
    }
}

/// Result of attempting to increment a bucket.
enum IncrResult {
    Ok,
    NeedsDownscale(HighLow),
    CounterOverflow(u64),
}

/// An allocation-free exponential histogram for non-negative values.
pub struct Histogram<const N: usize> {
    initial: Settings,
    current: Settings,

    word_base: i32,
    word_start: i32,
    word_end: i32,

    stats: Stats,

    data: [u64; N],
}

impl<const N: usize> Clone for Histogram<N> {
    fn clone(&self) -> Self {
        Self {
            initial: self.initial,
            current: self.current,
            word_base: self.word_base,
            word_start: self.word_start,
            word_end: self.word_end,
            stats: self.stats,
            data: self.data,
        }
    }
}

impl<const N: usize> fmt::Debug for Histogram<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Histogram");
        let stats = self.stats();
        s.field("width", &self.current.width)
            .field("count", &stats.count)
            .field("sum", &stats.sum)
            .field("min", &stats.min)
            .field("max", &stats.max)
            .field("scale", &self.current.scale.scale())
            .field("slot_count", &self.current_slot_count())
            .finish()
    }
}

impl<const N: usize> Default for Histogram<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Histogram<N> {
    /// Returns the aggregate statistics (count, sum, min, max).
    #[inline]
    pub(crate) const fn stats(&self) -> Stats {
        self.stats
    }

    /// Checked increment of count by `incr`. Returns `None` on overflow.
    #[inline]
    const fn checked_add_count(&self, incr: u64) -> Option<u64> {
        self.stats.count.checked_add(incr)
    }

    /// Commits merged statistics. The incoming `stats` carry the
    /// already-computed `sum` and `count` (self + other) and the
    /// other side's `min`/`max` which are merged via `f64::min`/`max`.
    fn commit_stats(&mut self, stats: &Stats) {
        self.stats.sum = stats.sum;
        self.stats.min = self.stats.min.min(stats.min);
        self.stats.max = self.stats.max.max(stats.max);
        self.stats.count = stats.count;
    }

    /// Returns true if no non-zero values have been recorded.
    #[inline]
    pub(crate) const fn buckets_empty(&self) -> bool {
        if self.word_end != self.word_start {
            return false;
        }
        self.data[0] == 0
    }

    /// Number of u64 data words in use at the current width.
    #[inline]
    const fn current_word_count(&self) -> i32 {
        if self.buckets_empty() {
            0
        } else {
            self.word_end - self.word_start + 1
        }
    }

    /// Number of buckets defined at the current width.
    #[inline]
    const fn current_slot_count(&self) -> i32 {
        self.current
            .width
            .word_to_slot_index(self.current_word_count())
    }

    /// Returns the slot address of a bucket indx.
    #[inline]
    const fn slot_addr(&self, slot: i32) -> SlotAddr<'_> {
        self.current.width.slot_addr(slot)
    }

    /// Returns the for a physical slot.
    #[inline]
    const fn start_addr(&self) -> SlotAddr<'_> {
        self.slot_addr(self.current.width.word_to_slot_index(self.word_start))
    }

    #[inline]
    pub(crate) const fn size_hint(&self, addr: &SlotAddr) -> usize {
        addr.size_hint(self.word_end)
    }

    /// Shifts all three index fields right by `by` positions.
    #[inline]
    fn shift_indices(&mut self, by: u32) {
        self.word_start >>= by;
        self.word_end >>= by;
        self.word_base >>= by;
    }

    // /// Returns the bucket data as a slice.
    // #[inline]
    // const fn bucket_data(&self) -> &[u64] {
    //     &self.data
    // }

    // /// Returns the bucket data as a mutable slice.
    // #[inline]
    // fn bucket_data_mut(&mut self) -> &mut [u64] {
    //     &mut self.data
    // }

    /// Gets the value at a slot address.
    #[inline]
    pub(super) const fn bucket_get(&self, addr: &SlotAddr) -> u64 {
        let idx = addr.data_index(N);
        let word = self.data[idx];
        addr.retrieve_counter(word)
    }

    /// Attempts to add `incr` to a physical slot. Returns false on overflow.
    #[inline]
    fn bucket_try_increment(&mut self, addr: &SlotAddr, incr: u64) -> Result<(), u64> {
        let idx = addr.data_index(N);
        let word = self.data[idx];
        let count = addr.retrieve_counter(word);

        let new_count = match count.checked_add(incr) {
            None => {
                // Safety: the total count would overflow before the
                // try_increment of an individual bucket would.
                unreachable!()
            }
            Some(c) => {
                if c > self.current.width.counter_max() {
                    return Err(c);
                }
                c
            }
        };

        self.data[idx] = addr.update_counter_in_word(word, new_count);
        Ok(())
    }

    /// Creates a new histogram at the maximum supported scale.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        // The limit at 2 ensures MIN_SCALE is sufficient to cover the
        // entire range.
        const { assert!(N >= 2, "requires >= 2 u64 buckets") };

        // The limit at 250 allows up to 16k single-bit buckets and
        // limits the histogram struct to 2048 bytes, noting that the
        // structure itself uses 6 u64.
        //
        // Note that nothing breaks when we allow N to grow above this
        // limit, just performance. The algorithms here are designed
        // for cache-line sized data.
        const { assert!(N <= 250, "requires <= 250 u64 buckets") };

        let settings = Settings::new(
            Scale::new(table_scale()).expect("table scale is valid"),
            Width::B1,
        );
        Self {
            initial: settings,
            current: settings,
            word_base: 0,
            word_start: 0,
            word_end: 0,
            stats: Stats::EMPTY,
            data: [0u64; N],
        }
    }

    /// Sets the maximum scale.
    #[inline]
    pub fn with_scale(mut self, scale: i32) -> Result<Self, ScaleError> {
        let s = Scale::new(scale)?;
        self.initial.scale = s;
        self.current.scale = s;
        Ok(self)
    }

    /// Sets the minimum bucket width.
    #[inline]
    #[must_use]
    pub fn with_min_width(mut self, width: Width) -> Self {
        self.initial.width = width;
        self.current.width = width;
        self
    }

    /// Returns a read-only view of the histogram.
    ///
    #[inline]
    pub fn view(&self) -> HistogramView<'_, N> {
        HistogramView { hist: self }
    }

    /// Returns the current counter width.
    #[inline]
    pub const fn width(&self) -> Width {
        self.current.width
    }

    /// Swaps contents with another histogram.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Resets the histogram to its initial state.
    pub fn clear(&mut self) {
        self.current = self.initial;
        self.word_base = 0;
        self.word_start = 0;
        self.word_end = 0;
        self.stats = Stats::EMPTY;
        self.data.fill(0);
    }

    /// Records a single value.
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<(), Error> {
        self.record_incr(value, 1)
    }

    /// Records a value with a specified increment.
    pub fn record_incr(&mut self, value: f64, incr: u64) -> Result<(), Error> {
        // Extract the raw exponent and significand (sign bit is ignored).
        let mut biased_exp = get_biased_exponent(value);
        let mut significand = get_significand(value);

        let new_count = self.checked_add_count(incr).ok_or(Error::Overflow)?;

        // Handle the extreme cases.
        match biased_exp {
            0 => {
                if significand == 0 {
                    // Zero case: no bucket, no min/max update.
                    self.stats.count = new_count;
                    return Ok(());
                } else {
                    // Round up to MIN_VALUE.
                    biased_exp = 1;
                    significand = 0;
                }
            }
            NAN_INF_BIASED => {
                // Inf and NaN cases.
                return Err(Error::Extreme);
            }
            _ => {
                // Normal exponents, only positive.
                if value.is_sign_negative() {
                    return Err(Error::Extreme);
                }
            }
        }

        let base2_exp = unbias_exponent(biased_exp);

        self.stats.min = self.stats.min.min(value);
        self.stats.max = self.stats.max.max(value);
        self.update_decomposed(significand, base2_exp, incr)?;
        self.stats.sum += value * incr as f64;
        self.stats.count = new_count;
        Ok(())
    }

    /// Updates buckets for a decomposed value.
    fn update_decomposed(
        &mut self,
        significand: u64,
        base2_exp: i32,
        incr: u64,
    ) -> Result<(), Error> {
        self.retry_increment(incr, |h| {
            h.current.scale.map_decomposed(significand, base2_exp)
        })
    }

    /// Retries an increment until it succeeds, performing downscale or
    /// widen as needed between attempts.  `index_fn` is called each
    /// iteration because the scale may have changed.
    fn retry_increment(
        &mut self,
        incr: u64,
        mut index_fn: impl FnMut(&Self) -> i32,
    ) -> Result<(), Error> {
        loop {
            let index = index_fn(self);
            let result = self.try_increment(index, incr);
            if self.resolve_increment(result)? {
                return Ok(());
            }
        }
    }

    /// Decreases the scale by `decrease` steps.
    fn change_scale(&mut self, decrease: u32) {
        let new_scale = self.current.scale.scale() - decrease as i32;
        self.current.scale =
            Scale::new(new_scale).expect("two buckets fit entire range at min_scale");
    }

    fn downscale_by(&mut self, change: u32) -> Result<(), Error> {
        if change == 0 {
            return Ok(());
        }

        self.do_downscale(change)?;
        self.change_scale(change);
        Ok(())
    }

    /// Attempts to add `incr` into the bucket at `index`.
    fn try_increment(&mut self, slot_index: i32, incr: u64) -> IncrResult {
        if incr == 0 {
            return IncrResult::Ok;
        }

        let width = self.current.width;
        let addr = width.slot_addr(slot_index);
        let word_index = addr.word_index();

        if self.buckets_empty() {
            self.word_start = word_index;
            self.word_end = self.word_start;
            self.word_base = self.word_start;
        } else if word_index < self.word_start {
            let diff = (self.word_end - word_index) as usize;
            if diff >= N {
                return IncrResult::NeedsDownscale(HighLow {
                    low: word_index,
                    high: self.word_end,
                });
            }
            self.word_start = word_index;
        } else if word_index > self.word_end {
            let diff = (word_index - self.word_start) as usize;
            if diff >= N {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.word_start,
                    high: word_index,
                });
            }
            self.word_end = word_index;
        }

        if let Err(oflow) = self.bucket_try_increment(&addr, incr) {
            return IncrResult::CounterOverflow(oflow);
        }

        IncrResult::Ok
    }

    /// Handles the result of `try_increment`, performing downscale
    /// or widen as needed.  Returns `Ok(true)` when the increment
    /// succeeded, `Ok(false)` when the caller should retry.
    fn resolve_increment(&mut self, result: IncrResult) -> Result<bool, Error> {
        match result {
            IncrResult::Ok => Ok(true),
            IncrResult::CounterOverflow(total) => {
                let new_width = Width::from_max_value(total);
                let change = new_width.subtract(self.current.width);
                self.downscale_by(change)?;
                Ok(false)
            }
            IncrResult::NeedsDownscale(hl) => {
                let change = hl.change_steps(N);
                self.downscale_by(change)?;
                Ok(false)
            }
        }
    }
}

// Compile-time test that Histogram is Send + Sync
const fn _assert_send_sync<T: Send + Sync>() {}
const _: () = _assert_send_sync::<Histogram<2>>();

#[cfg(test)]
mod tests;
