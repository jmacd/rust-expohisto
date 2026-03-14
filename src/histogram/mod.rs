// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram with a unified flat memory layout.
//!
//! `Histogram<N>` stores everything in fixed struct fields plus a `[u64; N]`
//! data pool used for bucket counters (or raw literal values before
//! promotion).
//!
//! Bucket counters start at 1-bit and widen through the chain
//! 1→2→4→8→16→32→64 bits via combined downscale+widen when a counter
//! saturates. Sub-byte transitions use parallel bit-sum (SWAR) — the
//! popcount algorithm's building blocks.

use core::fmt;

use crate::mapping::{max_scale, Mapping};

mod bucket_ops;
pub mod bucket_width;
mod swar;

mod bucket_view;
pub use bucket_view::{BucketView, BucketsIter};

mod quantile;
pub use quantile::{QuantileIter, QuantileValue};

mod view;
pub use view::HistogramView;

pub use bucket_width::BucketWidth;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned when a histogram operation would overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("histogram counter overflow")
    }
}

impl std::error::Error for Overflow {}

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
    /// Empty stats (all zeros).
    pub const EMPTY: Self = Self {
        count: 0,
        sum: 0.0,
        min: 0.0,
        max: 0.0,
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

// (BucketWidth is in bucket_width.rs)

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
    Ok,
    NeedsDownscale(HighLow),
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

// ---------------------------------------------------------------------------
// Histogram<N> — the unified flat-layout histogram
// ---------------------------------------------------------------------------

/// An allocation-free exponential histogram for non-negative values.
///
/// `N` is the number of `u64` words in the data pool. The entire pool is
/// used for bucket counter data (or literal values before promotion).
/// Aggregate statistics (count, sum, min, max) are stored in separate
/// struct fields.
///
/// # Literal Mode
///
/// New histograms start in **literal mode**, where the data pool stores
/// raw `f64` bit patterns instead of bucket counters. This holds up to
/// `N` values (e.g. 8 values for `Histogram<8>`).
/// When the next non-zero observation would exceed capacity, the
/// histogram **promotes** to bucket mode: all stored literals are
/// replayed through `update_buckets` at the configured max scale.
///
/// Read operations are accessed through [`view()`](Self::view), which
/// promotes from literal mode if needed and returns a [`HistogramView`]
/// with `&self` accessors. Literal mode can be disabled with
/// [`with_literal_mode(false)`](Self::with_literal_mode).
///
/// # Counter Widening
///
/// Bucket counters start at 1-bit and auto-widen in place
/// (1→2→4 bits → u8 → u16 → u32 → u64) via combined downscale+widen
/// when a counter saturates.
///
/// At minimum, `N` should be 8 (64 bytes of pool), giving 8 bucket words
/// (128 B4 buckets or 8 U64 buckets).
pub struct Histogram<const N: usize> {
    // -- Fixed metadata (never relocates) --
    mapping: Mapping,
    limit_scale: i8,
    min_bucket_width: BucketWidth,
    bucket_width: BucketWidth,
    /// When true, `data[..]` holds raw f64 bit patterns (literals)
    /// instead of bucket counters. `index_end` is repurposed as the literal
    /// count.
    literal: bool,
    /// Whether literal mode is enabled (persists across `clear()`).
    literal_enabled: bool,
    index_base: i32,
    index_start: i32,
    index_end: i32,

    // -- Aggregate statistics (min/max/sum/count) --
    stats: Stats,

    // -- Data pool: bucket counters or literal values --
    data: [u64; N],
}

impl<const N: usize> Clone for Histogram<N> {
    fn clone(&self) -> Self {
        Self {
            mapping: self.mapping,
            limit_scale: self.limit_scale,
            min_bucket_width: self.min_bucket_width,
            bucket_width: self.bucket_width,
            literal: self.literal,
            literal_enabled: self.literal_enabled,
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
        s.field("bucket_width", &self.bucket_width)
            .field("count", &self.count())
            .field("sum", &self.sum())
            .field("min", &self.min())
            .field("max", &self.max());
        if self.literal {
            s.field("mode", &"literal");
            s.field("literal_count", &self.literal_count());
        } else {
            s.field("mode", &"bucket");
            s.field("scale", &self.mapping.scale());
            s.field("bucket_len", &self.bucket_range_len());
        }
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
    pub(crate) fn sum(&self) -> f64 {
        self.stats.sum
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub(crate) fn count(&self) -> u64 {
        self.stats.count
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub(crate) fn min(&self) -> f64 {
        self.stats.min
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub(crate) fn max(&self) -> f64 {
        self.stats.max
    }

    /// Checked increment of count by `incr`. Returns `None` on overflow.
    #[inline]
    fn checked_add_count(&self, incr: u64) -> Option<u64> {
        self.stats.count.checked_add(incr)
    }

    /// Commits sum, count, min, and max from incoming values.
    ///
    /// If the histogram is currently empty (count == 0), min and max
    /// are set directly.  Otherwise min and max are merged via the
    /// respective comparison.
    fn commit_stats(&mut self, sum: f64, count: u64, min: f64, max: f64) {
        self.stats.sum = sum;
        if self.stats.count == 0 {
            self.stats.min = min;
            self.stats.max = max;
        } else {
            if min < self.stats.min {
                self.stats.min = min;
            }
            if max > self.stats.max {
                self.stats.max = max;
            }
        }
        self.stats.count = count;
    }

    /// Returns the total count stored across all positive buckets.
    fn non_zero_count(&self) -> u64 {
        if self.literal {
            return self.literal_count() as u64;
        }
        let len = self.bucket_range_len();
        let mut total = 0u64;
        for pos in 0..len {
            let index = self.index_start + pos as i32;
            total = total.saturating_add(self.bucket_get(self.slot_for(index)));
        }
        total
    }

    // -- Index arithmetic helpers --

    /// Number of logical buckets in the live range, or 0 if empty.
    #[inline]
    fn bucket_range_len(&self) -> u32 {
        if self.is_effectively_empty() {
            0
        } else {
            (self.index_end - self.index_start + 1) as u32
        }
    }

    /// Ring-buffer slot index for a given bucket index.
    #[inline]
    fn slot_for(&self, index: i32) -> usize {
        let cap = self.bucket_capacity() as i32;
        (index - self.index_base).rem_euclid(cap) as usize
    }

    /// Shifts all three index fields right by `by` positions.
    #[inline]
    fn shift_indices(&mut self, by: i32) {
        self.index_start >>= by;
        self.index_end >>= by;
        self.index_base >>= by;
    }

    /// Clamps `index_end` so the live range does not exceed capacity.
    #[inline]
    fn clamp_index_end(&mut self) {
        let cap = self.bucket_capacity() as i32;
        let max_end = self.index_start + cap - 1;
        if self.index_end > max_end {
            self.index_end = max_end;
        }
    }

    /// Returns true if the topmost slot in the last data word is occupied.
    #[inline]
    fn top_slot_occupied(&self) -> bool {
        let bits = self.bucket_width.bits();
        let n = self.bucket_word_count();
        n > 0 && self.bucket_data()[n - 1] >> (64 - bits) != 0
    }
}

// ---------------------------------------------------------------------------
// Bucket data access — operates on the bucket slice of the data pool
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Returns the number of u64 words available for bucket data.
    #[inline]
    fn bucket_word_count(&self) -> usize {
        N
    }

    /// Returns the bucket data as a slice.
    #[inline]
    fn bucket_data(&self) -> &[u64] {
        &self.data
    }

    /// Returns the bucket data as a mutable slice.
    #[inline]
    fn bucket_data_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Number of logical buckets available at the current width.
    #[inline]
    pub fn bucket_capacity(&self) -> usize {
        self.bucket_width.capacity(self.bucket_word_count())
    }

    /// Eagerly promotes from literal mode to bucket mode.
    /// No-op if already in bucket mode.
    #[inline]
    fn ensure_promoted(&mut self) {
        if self.literal {
            let _ = self.promote();
        }
    }

    /// Returns true if no buckets have been used.
    #[inline]
    pub fn buckets_empty(&self) -> bool {
        if self.literal {
            return self.literal_count() == 0;
        }
        self.is_effectively_empty()
    }

    /// Checks if the bucket range represents no data.
    #[inline]
    fn is_effectively_empty(&self) -> bool {
        if self.index_end != self.index_start {
            return false;
        }
        let slot = self.slot_for(self.index_start);
        self.bucket_get(slot) == 0
    }

    /// Trims leading and trailing zero buckets from the index range.
    fn trim_bucket_range(&mut self) {
        while self.index_end > self.index_start {
            if self.bucket_get(self.slot_for(self.index_end)) != 0 {
                break;
            }
            self.index_end -= 1;
        }
        while self.index_start < self.index_end {
            if self.bucket_get(self.slot_for(self.index_start)) != 0 {
                break;
            }
            self.index_start += 1;
        }
    }

    /// Gets the value at a physical slot index.
    ///
    /// All widths use the same shift-and-mask formula on the underlying
    /// `[u64]` pool. For sub-byte widths this extracts a packed bitfield;
    /// for byte-aligned widths the compiler reduces it to the same code as
    /// a direct typed read.
    #[inline]
    fn bucket_get(&self, slot: usize) -> u64 {
        let data = self.bucket_data();
        let bits = self.bucket_width.bits();
        let spw = 64 / bits;
        (data[slot / spw] >> ((slot % spw) * bits)) & self.bucket_width.counter_max()
    }

    /// Sets the value at a physical slot index.
    #[inline]
    fn bucket_set(&mut self, slot: usize, value: u64) {
        let bits = self.bucket_width.bits();
        let spw = 64 / bits;
        let mask = self.bucket_width.counter_max();
        let word = &mut self.bucket_data_mut()[slot / spw];
        let shift = (slot % spw) * bits;
        *word = (*word & !(mask << shift)) | ((value & mask) << shift);
    }

    /// Attempts to add `incr` to a physical slot. Returns false on overflow.
    #[inline]
    fn bucket_try_increment(&mut self, slot: usize, incr: u64) -> bool {
        let val = self.bucket_get(slot);
        let new_val = match val.checked_add(incr) {
            Some(v) if v <= self.bucket_width.counter_max() => v,
            _ => return false,
        };
        self.bucket_set(slot, new_val);
        true
    }
}

// (BucketView and BucketsIter are in bucket_view.rs)

// (Bucket operations — widen/downscale — are in bucket_ops.rs)

// (SWAR free functions are in swar.rs)

// ---------------------------------------------------------------------------
// Histogram<N> — construction and public API
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    // Shared constructor — all public constructors delegate here.
    fn new_at_scale(scale: i32) -> Self {
        const { assert!(N >= 1, "N must be >= 1 for at least 1 bucket word") };
        Self {
            mapping: Mapping::new(scale).expect("invalid scale"),
            limit_scale: scale as i8,
            min_bucket_width: BucketWidth::B1,
            bucket_width: BucketWidth::B1,
            literal: true,
            literal_enabled: true,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            stats: Stats::EMPTY,
            data: [0u64; N],
        }
    }

    /// Creates a new histogram at the maximum supported scale.
    ///
    /// # Panics
    ///
    /// Panics if no valid mapping algorithm feature is enabled, or if `N` is
    /// too small to hold stats and bucket data.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::new_at_scale(max_scale())
    }

    /// Creates a new histogram with an upper bound on scale.
    ///
    /// The scale is clamped to [`max_scale()`].
    ///
    /// # Panics
    ///
    /// Panics if the clamped scale is not supported by the mapping algorithm.
    #[inline]
    #[must_use]
    pub fn with_max_scale(scale: i32) -> Self {
        Self::new_at_scale(scale.min(max_scale()))
    }

    /// Creates a new histogram at the specified scale.
    ///
    /// Unlike [`with_max_scale`](Self::with_max_scale), this does not clamp
    /// to [`max_scale()`].
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not supported by the mapping algorithm.
    #[inline]
    #[must_use]
    pub fn with_scale(scale: i32) -> Self {
        Self::new_at_scale(scale)
    }

    /// Sets the minimum (initial) bucket counter width.
    ///
    /// By default, counters start at 1-bit (B1). Setting a higher floor
    /// (e.g. `BucketWidth::U8`) trades bucket capacity for avoiding the
    /// CPU cost of sub-byte bit-level indexing and SWAR widening.
    ///
    /// This also becomes the width used after `clear()`.
    #[inline]
    #[must_use]
    pub fn with_min_bucket_width(mut self, width: BucketWidth) -> Self {
        self.min_bucket_width = width;
        self.bucket_width = width;
        self
    }

    /// Enables or disables literal mode.
    ///
    /// When disabled, the histogram starts directly in bucket mode. This
    /// setting persists across `clear()`.
    ///
    /// Useful for benchmarks or when the caller knows the value range upfront.
    #[inline]
    #[must_use]
    pub fn with_literal_mode(mut self, enabled: bool) -> Self {
        self.literal = enabled;
        self.literal_enabled = enabled;
        self
    }

    /// Returns true if the histogram is in literal mode.
    #[inline]
    pub fn is_literal(&self) -> bool {
        self.literal
    }

    /// Returns a promoted read-only view of the histogram.
    ///
    /// Promotes from literal mode if needed. The returned
    /// [`HistogramView`] provides access to scale, stats, positive
    /// buckets, and quantile estimation — all via `&self`.
    ///
    /// ```
    /// use otel_expohisto::Histogram;
    ///
    /// let mut h: Histogram<16> = Histogram::new();
    /// h.update(1.5).unwrap();
    /// h.update(2.7).unwrap();
    ///
    /// let v = h.view();
    /// assert_eq!(v.count(), 2);
    /// println!("scale = {}", v.scale());
    /// ```
    #[inline]
    pub fn view(&mut self) -> HistogramView<'_, N> {
        self.ensure_promoted();
        HistogramView { hist: self }
    }

    /// Returns the number of literal values stored (0 if not in literal mode).
    #[inline]
    fn literal_count(&self) -> usize {
        if self.literal {
            self.index_end as usize
        } else {
            0
        }
    }

    /// Maximum number of literals that can be stored.
    #[inline]
    fn literal_capacity(&self) -> usize {
        self.bucket_word_count()
    }

    /// Returns the stored literal values as a slice.
    #[inline]
    fn literal_values(&self) -> &[u64] {
        &self.data[..self.literal_count()]
    }

    /// Returns the configured scale limit for this histogram.
    ///
    /// This is the scale the histogram resets to on [`clear()`](Self::clear)
    /// and the upper bound used by [`with_max_scale`](Self::with_max_scale).
    /// For the global maximum supported by the mapping algorithm, see
    /// [`max_scale()`](crate::max_scale).
    #[inline]
    pub fn limit_scale(&self) -> i32 {
        self.limit_scale as i32
    }

    /// Returns the current bucket counter width.
    #[inline]
    pub fn bucket_width(&self) -> BucketWidth {
        self.bucket_width
    }

    /// Clears the histogram, resetting to initial state.
    pub fn clear(&mut self) {
        self.reset_bucket_state();
        self.stats = Stats::EMPTY;
        self.literal = self.literal_enabled;
        self.mapping = Mapping::new(self.limit_scale as i32).unwrap();
    }

    /// Resets the bucket-related fields to empty state. Does not touch
    /// stats, literal flag, or mapping.
    fn reset_bucket_state(&mut self) {
        self.data.fill(0);
        self.bucket_width = self.min_bucket_width;
        self.index_start = 0;
        self.index_end = 0;
        self.index_base = 0;
    }

    /// Swaps contents with another histogram.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Records a single value.
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if a bucket counter or the total count would overflow.
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<(), Overflow> {
        self.update_by_incr(value, 1)
    }

    /// Records a value with a specified increment.
    ///
    /// Returns `Err(Overflow)` if the count or bucket counters would overflow.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        debug_assert!(value >= 0.0, "Histogram only accepts non-negative values");

        if incr == 0 {
            return Ok(());
        }

        let new_count = self.checked_add_count(incr).ok_or(Overflow)?;

        if value != 0.0 {
            if self.literal {
                self.update_literal(value, incr)?;
            } else {
                self.update_buckets(value, incr)?;
            }
        }

        let new_sum = self.sum() + value * incr as f64;
        self.commit_stats(new_sum, new_count, value, value);
        Ok(())
    }

    /// Stores a value in literal mode, promoting to bucket mode on overflow.
    fn update_literal(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        debug_assert!(self.literal);

        let count = self.literal_count();
        let cap = self.literal_capacity();
        let needed = incr as usize;

        if count + needed <= cap {
            // Fits — store `incr` copies of the value's bit pattern.
            let start = count;
            let bits = value.to_bits();
            for i in 0..needed {
                self.data[start + i] = bits;
            }
            self.index_end += incr as i32;
            Ok(())
        } else {
            // Overflow — promote with the trigger value included.
            self.promote_with(value, incr)
        }
    }

    /// Promotes from literal mode to bucket mode, optionally including
    /// a trigger value that caused overflow of literal capacity.
    fn promote_impl(&mut self, trigger: Option<(f64, u64)>) -> Result<(), Overflow> {
        debug_assert!(self.literal);

        let count = self.literal_count();

        if count == 0 && trigger.is_none() {
            self.literal = false;
            return Ok(());
        }

        // Collect stored literal values before we clobber the data pool.
        let mut literals = [0u64; N];
        literals[..count].copy_from_slice(self.literal_values());

        // Reset to empty bucket mode at limit_scale and replay values
        // through the normal update path, which handles widening and
        // downscaling incrementally.
        self.literal = false;
        self.reset_bucket_state();
        self.mapping = Mapping::new(self.limit_scale as i32).map_err(|_| Overflow)?;

        for &bits in literals[..count].iter() {
            let v = f64::from_bits(bits);
            self.update_buckets(v, 1)?;
        }

        if let Some((trigger_val, trigger_incr)) = trigger {
            self.update_buckets(trigger_val, trigger_incr)?;
        }

        Ok(())
    }

    /// Promotes from literal mode to bucket mode, including a trigger value
    /// that caused overflow of literal capacity.
    #[inline]
    fn promote_with(&mut self, trigger: f64, trigger_incr: u64) -> Result<(), Overflow> {
        self.promote_impl(Some((trigger, trigger_incr)))
    }

    /// Promotes from literal mode to bucket mode without a trigger value.
    /// Used when self is a merge destination and needs to accept bucket data.
    #[inline]
    fn promote(&mut self) -> Result<(), Overflow> {
        self.promote_impl(None)
    }

    /// Updates buckets for a positive value.
    fn update_buckets(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        self.retry_increment(incr, |h| h.mapping.map_to_index(value))
    }

    /// Retries an increment until it succeeds, performing downscale or
    /// widen as needed between attempts.  `index_fn` is called each
    /// iteration because the mapping scale may have changed.
    fn retry_increment(
        &mut self,
        incr: u64,
        mut index_fn: impl FnMut(&Self) -> i32,
    ) -> Result<(), Overflow> {
        loop {
            let index = index_fn(self);
            let result = self.increment_index_by(index, incr);
            if self.handle_incr_result(result)? {
                return Ok(());
            }
        }
    }

    /// Decreases the mapping scale by `decrease` steps.
    fn adjust_scale(&mut self, decrease: i32) -> Result<(), Overflow> {
        let new_scale = self.mapping.scale() - decrease;
        self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
        Ok(())
    }

    /// Widens bucket counters by one step, adjusting the mapping scale
    /// to account for any implicit downscale during widening.
    /// Re-inserts any deferred value displaced by an odd-base shift.
    fn widen_one_step(&mut self) -> Result<(), Overflow> {
        let (by, deferred) = self.bucket_widen().ok_or(Overflow)?;
        self.adjust_scale(by)?;
        if let Some((idx, val)) = deferred {
            let base_scale = self.mapping.scale();
            self.retry_increment(val, |h| {
                idx >> (base_scale - h.mapping.scale())
            })?;
        }
        Ok(())
    }

    /// Downscales by `change` scale-steps using adaptive SWAR merge.
    ///
    /// Processes one merge step at a time. Each step does a SWAR
    /// pairwise sum and checks for overflow:
    ///
    /// - **No overflow**: narrows back to the original width (preserving
    ///   bucket capacity) and continues to the next step.
    /// - **Overflow**: accepts the wider format and continues at the new
    ///   width. This simultaneously merges AND widens in one SWAR pass.
    ///
    /// If `index_base` is odd (possible after several no-overflow
    /// merges or after a widen that fills all post-widen capacity),
    /// shifts data up by one slot to restore even alignment. When
    /// the top slot is occupied, that slot's value is saved and
    /// re-inserted after all downscale steps complete.
    ///
    /// At U64 width, remaining steps use `bucket_downscale_u64`
    /// (scatter-write collapse).
    #[cfg(any(test, feature = "bench-internals"))]
    pub fn do_downscale(&mut self, change: i32) -> Result<(), Overflow> {
        self.do_downscale_impl(change)
    }

    fn do_downscale_impl(&mut self, change: i32) -> Result<(), Overflow> {
        if change <= 0 {
            return Ok(());
        }

        if self.is_effectively_empty() {
            self.shift_indices(change);
            return self.adjust_scale(change);
        }

        let mut remaining = change;

        // Deferred values: up to 6 (one per sub-U64 width level).
        let mut deferred: [(i32, u64); 6] = [(0, 0); 6];
        let mut n_deferred = 0usize;

        // Phase 1: Adaptive SWAR merge at sub-U64 widths.
        while remaining > 0 && self.bucket_width != BucketWidth::U64 {
            let (steps, overflow) =
                self.swar_merge_step(false).ok_or(Overflow)?;

            // Shift all previously deferred indices by this step.
            for d in &mut deferred[..n_deferred] {
                d.0 >>= steps;
            }

            // Collect new deferred (already at post-step scale).
            if let Some(d) = overflow {
                debug_assert!(n_deferred < deferred.len());
                deferred[n_deferred] = d;
                n_deferred += 1;
            }

            self.adjust_scale(steps)?;
            remaining -= steps;
        }

        // Phase 2: At U64, scatter-write for remaining steps.
        if remaining > 0 {
            debug_assert_eq!(self.bucket_width, BucketWidth::U64);

            // Shift deferred indices by U64 steps.
            for d in &mut deferred[..n_deferred] {
                d.0 >>= remaining;
            }

            self.bucket_downscale_u64(remaining);
            self.adjust_scale(remaining)?;
        }

        self.trim_bucket_range();

        // Phase 3: Re-insert deferred values.
        // All deferred indices are relative to the scale at this point.
        // Use a single reference scale so that if re-inserting one value
        // triggers a further downscale, subsequent indices adjust correctly.
        let deferred_scale = self.mapping.scale();
        for &(idx, val) in &deferred[..n_deferred] {
            self.retry_increment(val, |h| {
                idx >> (deferred_scale - h.mapping.scale())
            })?;
        }

        Ok(())
    }

    fn downscale_to(&mut self, target_scale: i32) -> Result<(), Overflow> {
        let change = self.mapping.scale() - target_scale;
        if change <= 0 {
            return Ok(());
        }
        self.do_downscale_impl(change)
    }

    /// Attempts to increment at the given index.
    fn increment_index_by(&mut self, index: i32, incr: u64) -> IncrResult {
        if incr == 0 {
            return IncrResult::Ok;
        }

        let max_size = self.bucket_capacity() as i32;

        if self.buckets_empty() {
            self.index_start = index;
            self.index_end = index;
            // Align base to one word of slots at the current width
            // so that counter pairs always share a word for SWAR widening.
            let spw = self.bucket_width.slots_per_word() as i32;
            self.index_base = index & !(spw - 1);
        } else if index < self.index_start {
            // At sub-U64, indices below index_base would wrap and break
            // SWAR. Force a downscale (which widens to U64 first, where
            // wrapping is safe).
            if self.bucket_width != BucketWidth::U64 && index < self.index_base {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.index_end,
                });
            }
            let span = self.index_end.saturating_sub(index);
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.index_end,
                });
            }
            for idx in index..self.index_start {
                self.bucket_set(self.slot_for(idx), 0);
            }
            self.index_start = index;
        } else if index > self.index_end {
            // At sub-U64, indices at or above index_base + cap would
            // wrap and break SWAR.
            if self.bucket_width != BucketWidth::U64 && index >= self.index_base + max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.index_start,
                    high: index,
                });
            }
            let span = index.saturating_sub(self.index_start);
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.index_start,
                    high: index,
                });
            }
            for idx in (self.index_end + 1)..=index {
                self.bucket_set(self.slot_for(idx), 0);
            }
            self.index_end = index;
        }

        if !self.bucket_try_increment(self.slot_for(index), incr) {
            return IncrResult::CounterOverflow;
        }

        IncrResult::Ok
    }

    /// Handles the result of [`increment_index_by`], performing downscale
    /// or widen as needed.  Returns `Ok(true)` when the increment
    /// succeeded, `Ok(false)` when the caller should retry.
    fn handle_incr_result(&mut self, result: IncrResult) -> Result<bool, Overflow> {
        match result {
            IncrResult::Ok => Ok(true),
            IncrResult::CounterOverflow => {
                self.widen_one_step()?;
                Ok(false)
            }
            IncrResult::NeedsDownscale(hl) => {
                let change = change_scale(hl, self.bucket_capacity() as i32);
                if change > 0 {
                    self.do_downscale_impl(change)?;
                } else if self.bucket_width != BucketWidth::U64 {
                    self.widen_one_step()?;
                } else {
                    self.do_downscale_impl(1)?;
                }
                Ok(false)
            }
        }
    }

    // -- Merge --

    /// Merges another histogram (same N) into this one.
    pub fn merge_from(&mut self, other: &Self) -> Result<(), Overflow> {
        self.merge_from_impl(other, true)
    }

    /// Merges a histogram of a different size into this one.
    pub fn merge_from_other<const M: usize>(
        &mut self,
        other: &Histogram<M>,
    ) -> Result<(), Overflow> {
        self.merge_from_impl(other, false)
    }

    /// Shared merge implementation.
    ///
    /// When `adopt_width` is true and self is empty, adopts the source's
    /// bucket width to avoid unnecessary widening steps (same-size only).
    fn merge_from_impl<const M: usize>(
        &mut self,
        other: &Histogram<M>,
        adopt_width: bool,
    ) -> Result<(), Overflow> {
        if other.literal {
            return self.merge_literal_from(other);
        }
        if adopt_width && !other.buckets_empty() && self.buckets_empty() {
            let saved_width = self.bucket_width;
            self.bucket_width = self.bucket_width.max(other.bucket_width);
            let result = self.merge_from_histogram(other);
            if result.is_err() {
                self.bucket_width = saved_width;
            }
            return result;
        }
        self.merge_from_histogram(other)
    }

    /// Merges from raw histogram data, enabling cross-size merging.
    ///
    /// # Arguments
    ///
    /// * `stats` — aggregate statistics (count, sum, min, max) of the source
    /// * `buckets` — bucket layout (scale, offset, len) of the source
    /// * `at` — returns the count at bucket position `i` (0-indexed from offset)
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if a bucket counter or the total count would overflow.
    pub fn merge_from_raw(
        &mut self,
        stats: &Stats,
        buckets: &BucketDescriptor,
        at: &dyn Fn(u32) -> u64,
    ) -> Result<(), Overflow> {
        if stats.count == 0 {
            return Ok(());
        }

        let snapshot = self.clone();
        match self.merge_from_raw_inner(stats, buckets, at) {
            Ok(()) => Ok(()),
            Err(e) => {
                *self = snapshot;
                Err(e)
            }
        }
    }

    fn merge_from_raw_inner(
        &mut self,
        stats: &Stats,
        buckets: &BucketDescriptor,
        at: &dyn Fn(u32) -> u64,
    ) -> Result<(), Overflow> {
        let new_count = self.checked_add_count(stats.count).ok_or(Overflow)?;
        let new_sum = self.sum() + stats.sum;

        if buckets.len > 0 {
            if self.literal {
                self.promote()?;
            }

            let other_end = buckets.offset + buckets.len as i32 - 1;
            let cap = self.bucket_capacity() as i32;
            let min_scale = self.mapping.scale().min(buckets.scale);

            let self_hl = self.high_low_at_scale(min_scale);
            let other_hl = {
                let shift = buckets.scale - min_scale;
                HighLow {
                    low: buckets.offset >> shift,
                    high: other_end >> shift,
                }
            };
            let hlp = self_hl.merge(other_hl);
            let min_scale = min_scale - change_scale(hlp, cap);

            self.downscale_to(min_scale)?;

            for i in 0..buckets.len {
                let count = at(i);
                if count == 0 {
                    continue;
                }
                self.retry_increment(count, |h| {
                    let shift = buckets.scale - h.mapping.scale();
                    (buckets.offset + i as i32) >> shift
                })?;
            }

            self.trim_bucket_range();
        }

        self.commit_stats(new_sum, new_count, stats.min, stats.max);
        Ok(())
    }

    /// Merges literal values from another histogram into this one.
    fn merge_literal_from<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Overflow> {
        debug_assert!(other.literal);
        if other.count() == 0 {
            return Ok(());
        }
        let snapshot = self.clone();
        match self.merge_literal_from_inner(other) {
            Ok(()) => Ok(()),
            Err(e) => {
                *self = snapshot;
                Err(e)
            }
        }
    }

    fn merge_literal_from_inner<const M: usize>(
        &mut self,
        other: &Histogram<M>,
    ) -> Result<(), Overflow> {
        let new_count = self.checked_add_count(other.count()).ok_or(Overflow)?;
        let new_sum = self.sum() + other.sum();
        if self.literal {
            self.promote()?;
        }
        for &bits in other.literal_values() {
            self.update_buckets(f64::from_bits(bits), 1)?;
        }
        self.commit_stats(new_sum, new_count, other.min(), other.max());
        Ok(())
    }

    /// Builds [`Stats`] + [`BucketDescriptor`] from a histogram and
    /// delegates to [`merge_from_raw`](Self::merge_from_raw).
    fn merge_from_histogram<const M: usize>(
        &mut self,
        other: &Histogram<M>,
    ) -> Result<(), Overflow> {
        self.merge_from_raw(
            &Stats {
                count: other.count(),
                sum: other.sum(),
                min: other.min(),
                max: other.max(),
            },
            &BucketDescriptor {
                scale: other.mapping.scale(),
                offset: other.index_start,
                len: other.bucket_range_len(),
            },
            &|i| {
                let index = other.index_start + i as i32;
                other.bucket_get(other.slot_for(index))
            },
        )
    }

    fn high_low_at_scale(&self, target_scale: i32) -> HighLow {
        if self.buckets_empty() {
            return HighLow::empty();
        }
        let shift = self.mapping.scale() - target_scale;
        HighLow {
            low: self.index_start >> shift,
            high: self.index_end >> shift,
        }
    }
}

// (Quantile estimation is in quantile.rs)

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;

#[cfg(test)]
mod regression;
