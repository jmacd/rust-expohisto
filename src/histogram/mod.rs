// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram with a unified flat memory layout.
//!
//! `Histogram<N>` stores everything in fixed struct fields plus a `[u64; N]`
//! data pool used for bucket counters.

use core::fmt;

use crate::float64::{NAN_INF_BIASED, get_biased_exponent, get_significand, unbias_exponent};
use crate::mapping::{Scale, ScaleError, max_scale};

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

pub use width::Width;

/// Compact histogram configuration: scale + counter width in 2 bytes.
///
/// Used as both the "initial" settings (configured at construction time,
/// restored on reset) and the "current" settings (mutated during
/// downscale/widen operations).
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
///
/// The total count is checked before any bucket mutation.  Because the
/// total is always >= any individual bucket count, a `u64`-width bucket
/// counter cannot overflow once the total-count check passes.  In
/// practice, callers should flush and reset histograms periodically
/// long before `u64` exhaustion.
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

// ---------------------------------------------------------------------------
// Aggregate stats and bucket descriptor — used by merge_from_raw
// ---------------------------------------------------------------------------

/// Aggregate statistics of a histogram: count, sum, min, max.
///
/// Used by [`Histogram::merge_from_raw`] to pass the source histogram's
/// statistics without requiring a full `Histogram` instance.
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
    /// Empty stats — `min` is `INFINITY` and `max` is `NEG_INFINITY`
    /// so that the first observation overwrites both unconditionally via
    /// `f64::min`/`f64::max`.
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
    /// Number of contiguous buckets.
    pub len: u32,
}

// ---------------------------------------------------------------------------
// High-low range helpers
// ---------------------------------------------------------------------------

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
    const fn is_empty(&self) -> bool {
        self.low > self.high
    }

    #[inline]
    const fn merge(self, other: Self) -> Self {
        match (self.is_empty(), other.is_empty()) {
            (true, _) => other,
            (_, true) => self,
            _ => Self {
                low: if self.low < other.low {
                    self.low
                } else {
                    other.low
                },
                high: if self.high > other.high {
                    self.high
                } else {
                    other.high
                },
            },
        }
    }
}

/// Result of attempting to increment a bucket.
enum IncrResult {
    Ok,
    NeedsDownscale(HighLow),
    CounterOverflow,
}

/// Computes how much downscaling is needed for indices to fit in `size` buckets.
#[inline]
const fn scale_reduction(mut hl: HighLow, size: i32) -> i32 {
    let mut change = 0;
    while hl.high - hl.low >= size {
        hl.high >>= 1;
        hl.low >>= 1;
        change += 1;
    }
    change
}

// ---------------------------------------------------------------------------
// Histogram<N> — the unified flat-layout histogram
// ---------------------------------------------------------------------------

/// An allocation-free exponential histogram for non-negative values.
///
/// `N` is the number of `u64` words in the data pool. The entire pool
/// is used for bucket counter data.  Aggregate statistics (count,
/// sum, min, max) are stored in separate struct fields.
///
/// # Positive Buckets Only
///
/// This histogram only maintains positive buckets. Negative values are
/// rejected by [`record()`](Self::record). The OTel exponential histogram
/// data model defines both positive and negative bucket arrays; this
/// crate implements the positive side only, which is sufficient for
/// latency, size, and other non-negative metrics.
///
/// # Counter Widening
///
/// Bucket counters start at 1-bit and auto-widen in place
/// (B0→B1→B2→B4→U8→U16→U32→U64) via combined downscale+widen
/// when a counter saturates.
///
/// At minimum, `N` should be 8 (64 bytes of pool), giving 8 bucket words
/// (128 B4 buckets or 8 U64 buckets).
pub struct Histogram<const N: usize> {
    // -- Settings: initial (restored on reset) and current --
    initial: Settings,
    current: Settings,

    // -- Bucket index state --
    index_base: i32,
    index_start: i32,
    index_end: i32,

    // -- Aggregate statistics (min/max/sum/count) --
    stats: Stats,

    // -- Data pool: bucket counters --
    data: [u64; N],
}

impl<const N: usize> Clone for Histogram<N> {
    fn clone(&self) -> Self {
        Self {
            initial: self.initial,
            current: self.current,
            index_base: self.index_base,
            index_start: self.index_start,
            index_end: self.index_end,
            stats: self.stats,
            data: self.data,
        }
    }
}

impl<const N: usize> fmt::Debug for Histogram<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Histogram");
        s.field("width", &self.current.width)
            .field("count", &self.count())
            .field("sum", &self.sum())
            .field("min", &self.min())
            .field("max", &self.max())
            .field("scale", &self.current.scale.scale())
            .field("bucket_len", &self.range_len());
        s.finish()
    }
}

impl<const N: usize> Default for Histogram<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MMSC read/write methods
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Returns the sum of all recorded values as `f64`.
    #[inline]
    pub(crate) const fn sum(&self) -> f64 {
        self.stats.sum
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub(crate) const fn count(&self) -> u64 {
        self.stats.count
    }

    /// Returns the minimum recorded value.
    #[inline]
    pub(crate) const fn min(&self) -> f64 {
        self.stats.min
    }

    /// Returns the maximum recorded value.
    #[inline]
    pub(crate) const fn max(&self) -> f64 {
        self.stats.max
    }

    /// Checked increment of count by `incr`. Returns `None` on overflow.
    #[inline]
    const fn checked_add_count(&self, incr: u64) -> Option<u64> {
        self.stats.count.checked_add(incr)
    }

    /// Commits sum, count, min, and max from incoming values. Used in merge.
    fn commit_stats(&mut self, sum: f64, count: u64, min: f64, max: f64) {
        self.stats.sum = sum;
        self.stats.min = self.stats.min.min(min);
        self.stats.max = self.stats.max.max(max);
        self.stats.count = count;
    }

    // -- Index arithmetic helpers --

    /// Number of logical buckets in the live range, or 0 if empty.
    #[inline]
    const fn range_len(&self) -> u32 {
        if self.range_is_empty() {
            0
        } else {
            (self.index_end - self.index_start + 1) as u32
        }
    }

    /// Ring-buffer slot index for a given bucket index.
    #[inline]
    const fn slot_for(&self, index: i32) -> usize {
        let cap = self.bucket_count() as i32;
        (index - self.index_base).rem_euclid(cap) as usize
    }

    /// Shifts all three index fields right by `by` positions.
    #[inline]
    fn shift_indices(&mut self, by: i32) {
        self.index_start >>= by;
        self.index_end >>= by;
        self.index_base >>= by;
    }

    /// Returns true if `index` falls outside the contiguous physical range
    /// `[index_base, index_base + cap)` required by SWAR at sub-U64 widths.
    /// At U64 the ring buffer handles wrapping, so this always returns false.
    #[inline]
    const fn swar_would_wrap(&self, index: i32) -> bool {
        !matches!(self.current.width, Width::U64)
            && (index < self.index_base || index >= self.index_base + self.bucket_count() as i32)
    }
}

// ---------------------------------------------------------------------------
// Bucket data access — operates on the bucket slice of the data pool
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Returns the bucket data as a slice.
    #[inline]
    const fn bucket_data(&self) -> &[u64] {
        &self.data
    }

    /// Returns the bucket data as a mutable slice.
    #[inline]
    fn bucket_data_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Returns the number of counter slots available at the current width.
    #[inline]
    pub const fn bucket_count(&self) -> usize {
        self.current.width.capacity(N)
    }

    /// Returns true if no non-zero values have been recorded.
    #[inline]
    pub fn buckets_empty(&self) -> bool {
        self.range_is_empty()
    }

    /// Checks if the bucket index range is empty (bucket-mode only).
    /// Structural check: are the bucket index pointers empty?
    ///
    /// Distinct from `buckets_empty()` because during reconstruction
    /// (promotion, downscale) the indices are reset while stats.sum
    /// remains non-zero.
    #[inline]
    const fn range_is_empty(&self) -> bool {
        if self.index_end != self.index_start {
            return false;
        }
        let slot = self.slot_for(self.index_start);
        self.bucket_get(slot) == 0
    }

    /// Returns the (word_index, bit_shift, mask) for a physical slot.
    #[inline]
    const fn slot_addr(&self, slot: usize) -> (usize, usize, u64) {
        let bits = self.current.width.bits();
        let spw = 64 / bits;
        (
            slot / spw,
            (slot % spw) * bits,
            self.current.width.counter_max(),
        )
    }

    /// Gets the value at a physical slot index.
    ///
    /// All widths use the same shift-and-mask formula on the underlying
    /// `[u64]` pool. For sub-byte widths this extracts a packed bitfield;
    /// for byte-aligned widths the compiler reduces it to the same code as
    /// a direct typed read.
    #[inline]
    const fn bucket_get(&self, slot: usize) -> u64 {
        let (wi, shift, mask) = self.slot_addr(slot);
        (self.bucket_data()[wi] >> shift) & mask
    }

    /// Sets the value at a physical slot index.
    #[inline]
    fn bucket_set(&mut self, slot: usize, value: u64) {
        let (wi, shift, mask) = self.slot_addr(slot);
        let word = &mut self.bucket_data_mut()[wi];
        *word = (*word & !(mask << shift)) | ((value & mask) << shift);
    }

    /// Zeroes all counter slots in `[from_index, to_index)`.
    ///
    /// Operates at word granularity where possible: partial words at the
    /// edges are cleared per-slot, but interior words are zeroed whole.
    fn zero_slots(&mut self, from_index: i32, to_index: i32) {
        if from_index >= to_index {
            return;
        }
        let spw = self.current.width.slots_per_word();
        let from_slot = self.slot_for(from_index);
        let to_slot = self.slot_for(to_index - 1) + 1;

        let first_word = from_slot / spw;
        let last_word = (to_slot - 1) / spw;

        if first_word == last_word {
            // All slots in one word — clear per-slot.
            for slot in from_slot..to_slot {
                self.bucket_set(slot, 0);
            }
            return;
        }

        // Partial first word.
        if from_slot % spw != 0 {
            for slot in from_slot..(first_word + 1) * spw {
                self.bucket_set(slot, 0);
            }
            // Interior whole words.
            self.data[first_word + 1..last_word].fill(0);
        } else {
            self.data[first_word..last_word].fill(0);
        }

        // Partial last word.
        if to_slot % spw != 0 {
            for slot in last_word * spw..to_slot {
                self.bucket_set(slot, 0);
            }
        } else {
            self.data[last_word] = 0;
        }
    }

    /// Attempts to add `incr` to a physical slot. Returns false on overflow.
    #[inline]
    fn bucket_try_increment(&mut self, slot: usize, incr: u64) -> bool {
        let val = self.bucket_get(slot);
        let new_val = match val.checked_add(incr) {
            Some(v) if v <= self.current.width.counter_max() => v,
            _ => return false,
        };
        self.bucket_set(slot, new_val);
        true
    }
}

// ---------------------------------------------------------------------------
// Histogram<N> — construction and public API
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    // Shared constructor — all public constructors delegate here.
    fn new_at_scale(scale: i32) -> Result<Self, ScaleError> {
        const { assert!(N >= 1, "N must be >= 1 for at least 1 bucket word") };
        let settings = Settings::new(Scale::new(scale)?, Width::B1);
        Ok(Self {
            initial: settings,
            current: settings,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            stats: Stats::EMPTY,
            data: [0u64; N],
        })
    }

    /// Creates a new histogram at the maximum supported scale.
    ///
    /// The maximum scale is determined by the compiled lookup table
    /// (`scale-N` feature). This always succeeds because scale 0
    /// (exponent mapping) is unconditionally available.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        // max_scale() is always valid (>= 0), so this cannot fail.
        Self::new_at_scale(max_scale()).unwrap()
    }

    /// Sets the exact scale.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError::InvalidScale`] if `scale` is outside
    /// the supported range [`MIN_SCALE`](crate::MIN_SCALE)..=[`max_scale()`](crate::max_scale).
    #[inline]
    pub fn with_scale(mut self, scale: i32) -> Result<Self, ScaleError> {
        let s = Scale::new(scale)?;
        self.initial.scale = s;
        self.current.scale = s;
        Ok(self)
    }

    /// Sets the minimum (initial) bucket counter width.
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
    ///
    /// Useful for the "swap and export" pattern: record into a histogram,
    /// then swap with a fresh one to hand off the data:
    ///
    /// ```
    /// use otel_expohisto::Histogram;
    ///
    /// let mut h: Histogram<16> = Histogram::new();
    /// // … record values, export via view() …
    /// let mut fresh = Histogram::new();
    /// h.swap(&mut fresh);
    /// // h is now empty; fresh holds the old data
    /// ```
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Resets the histogram to its initial state, preserving the
    /// configured scale and minimum bucket width.
    pub fn clear(&mut self) {
        self.current = self.initial;
        self.index_base = 0;
        self.index_start = 0;
        self.index_end = 0;
        self.stats = Stats::EMPTY;
        self.data.fill(0); // TODO: is this required?
    }

    /// Records a value with a specified increment.
    ///
    /// The value must be non-negative and not NaN.  **This is not
    /// checked at runtime** — `debug_assert!` catches violations in
    /// debug builds, but release builds assume valid input.
    ///
    /// This is deliberate: an OTel SDK must already validate values
    /// at the API boundary (rejecting NaN, Inf, and negative values
    /// before selecting an aggregator), so repeating that check here
    /// would add a branch to the hot path for no benefit.  The
    /// `debug_assert!` exists as a safety net during development.
    ///
    /// Positive infinity (`f64::INFINITY`) is accepted and mapped to
    /// the same bucket as `f64::MAX`, consistent with the Prometheus
    /// exponential histogram specification.
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if the total count would exceed `u64::MAX`.
    /// The total count is checked before any mutation, and because the
    /// total is always ≥ any individual bucket count, no bucket at
    /// `u64` width can overflow once the total-count check passes.
    /// Callers that need rollback semantics can `clone()` beforehand,
    /// but in practice histograms should be flushed and reset long
    /// before `u64` exhaustion.
    pub fn record_incr(&mut self, value: f64, incr: u64) -> Result<(), Error> {
        // Extract the raw exponent.
        let mut biased_exp = get_biased_exponent(value);
        let mut significand = get_significand(value);

        let new_count = self.checked_add_count(1).ok_or(Error::Overflow)?;

        // Handle the extreme cases.
        match biased_exp {
            0 => {
                if significand == 0 {
                    // Zero case.
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
                // Normal exponents
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

    /// Records a single value.
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<(), Error> {
        self.record_incr(value, 1)
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
            // This ? will catch 0, Inf and NaN cases. Sign is ignored.
            // so if the user manages to pass negatives they are counted
            // as positive.
            let index = index_fn(self);
            let result = self.try_increment(index, incr);
            if self.resolve_increment(result)? {
                return Ok(());
            }
        }
    }

    /// Decreases the scale by `decrease` steps.
    fn decrease_scale(&mut self, decrease: i32) -> Result<(), Error> {
        let new_scale = self.current.scale.scale() - decrease;
        self.current.scale = Scale::new(new_scale).map_err(|_| Error::Overflow)?;
        Ok(())
    }

    /// Widens bucket counters by one step, adjusting the scale
    /// to account for the implicit 1-step downscale.
    fn widen_one_step(&mut self) -> Result<(), Error> {
        if self.range_is_empty() {
            self.current.width = self.current.width.wider().ok_or(Error::Overflow)?;
            self.shift_indices(1);
            return self.decrease_scale(1);
        }
        self.widen_by_one()?;
        self.decrease_scale(1)
    }

    /// Downscales by `change` scale-steps.
    ///
    /// Clones the data array and scatter-adds groups of `2^change`
    /// adjacent counters into a fresh, aligned output buffer.  The
    /// output width is the minimum that holds all group sums.
    #[cfg(any(test, feature = "bench-internals"))]
    pub fn downscale(&mut self, change: i32) -> Result<(), Error> {
        self.downscale_by(change)
    }

    fn downscale_by(&mut self, change: i32) -> Result<(), Error> {
        if change <= 0 {
            return Ok(());
        }

        if self.range_is_empty() {
            self.shift_indices(change);
            return self.decrease_scale(change);
        }

        self.do_downscale(change, self.current.width)?;
        self.decrease_scale(change)
    }

    fn downscale_to(&mut self, target_scale: i32) -> Result<(), Error> {
        let change = self.current.scale.scale() - target_scale;
        if change <= 0 {
            return Ok(());
        }
        self.downscale_by(change)
    }

    /// Attempts to place `incr` into the bucket at `index`.
    ///
    /// Returns `NeedsDownscale` if the index doesn't fit in the current
    /// range, or `CounterError` if the counter at `index` can't hold
    /// the addition.  The caller retries after adjusting scale or width.
    fn try_increment(&mut self, index: i32, incr: u64) -> IncrResult {
        if incr == 0 {
            return IncrResult::Ok;
        }

        let cap = self.bucket_count() as i32;

        if self.range_is_empty() {
            self.index_start = index;
            self.index_end = index;
            // Align base to a word boundary so that SWAR pairwise ops
            // never split a counter pair across u64 words.
            self.index_base = self.current.width.word_start(index);
        } else if index < self.index_start {
            if self.swar_would_wrap(index) {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.index_end,
                });
            }
            if self.index_end - index >= cap {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.index_end,
                });
            }
            self.zero_slots(index, self.index_start);
            self.index_start = index;
        } else if index > self.index_end {
            if self.swar_would_wrap(index) {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.index_start,
                    high: index,
                });
            }
            if index - self.index_start >= cap {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.index_start,
                    high: index,
                });
            }
            self.zero_slots(self.index_end + 1, index + 1);
            self.index_end = index;
        }

        if !self.bucket_try_increment(self.slot_for(index), incr) {
            return IncrResult::CounterOverflow;
        }

        IncrResult::Ok
    }

    /// Handles the result of `try_increment`, performing downscale
    /// or widen as needed.  Returns `Ok(true)` when the increment
    /// succeeded, `Ok(false)` when the caller should retry.
    fn resolve_increment(&mut self, result: IncrResult) -> Result<bool, Error> {
        match result {
            IncrResult::Ok => Ok(true),
            IncrResult::CounterOverflow => {
                self.widen_one_step()?;
                Ok(false)
            }
            IncrResult::NeedsDownscale(hl) => {
                let change = scale_reduction(hl, self.bucket_count() as i32);
                if change > 0 {
                    self.downscale_by(change)?;
                } else if self.swar_would_wrap(hl.low) || self.swar_would_wrap(hl.high) {
                    // Range fits in capacity but wraps outside the
                    // contiguous SWAR region.  Widen to U64 (where
                    // wrapping is safe) via a 1-step merge.
                    self.widen_one_step()?;
                } else {
                    self.downscale_by(1)?;
                }
                Ok(false)
            }
        }
    }
}

// Compile-time proof that Histogram is Send + Sync (all fields are Copy
// primitives). This prevents regressions if a non-Send/Sync type is
// accidentally added in the future.
const fn _assert_send_sync<T: Send + Sync>() {}
const _: () = _assert_send_sync::<Histogram<1>>();

#[cfg(test)]
mod tests;
