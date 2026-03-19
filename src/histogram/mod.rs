// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram with a unified flat memory layout.
//!
//! `Histogram<N>` stores everything in fixed struct fields plus a `[u64; N]`
//! data pool used for bucket counters (or raw literal values before
//! promotion).
//!
//! Bucket counters start at 1-bit and widen through the chain
//! 1→2→4→8→16→32→64 bits.  Both downscale and counter-overflow use SWAR
//! (SIMD-within-a-register) pairwise-merge steps at sub-U64 widths, then
//! scatter-write at U64.  Each merge step is self-contained: any value
//! displaced by an odd-base alignment shift is fixed up immediately.

use core::fmt;

use crate::mapping::{max_scale, Mapping, MappingError};

mod bucket_ops;
pub mod bucket_width;
mod literal;
mod merge;
mod swar;

mod bucket_view;
pub use bucket_view::{BucketView, BucketsIter};

#[cfg(feature = "boundary")]
mod quantile;
#[cfg(feature = "boundary")]
pub use quantile::{QuantileIter, QuantileValue};

mod view;
pub use view::HistogramView;

pub use bucket_width::BucketWidth;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned when the total count would exceed `u64::MAX`.
///
/// The total count is checked before any bucket mutation.  Because the
/// total is always ≥ any individual bucket count, a `u64`-width bucket
/// counter cannot overflow once the total-count check passes.  In
/// practice, callers should flush and reset histograms periodically
/// long before `u64` exhaustion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("histogram total count overflow")
    }
}

#[cfg(feature = "std")]
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
    const fn is_empty(&self) -> bool {
        self.low > self.high
    }

    #[inline]
    const fn merge(self, other: Self) -> Self {
        match (self.is_empty(), other.is_empty()) {
            (true, _) => other,
            (_, true) => self,
            _ => Self {
                low: if self.low < other.low { self.low } else { other.low },
                high: if self.high > other.high { self.high } else { other.high },
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
/// `N` is the number of `u64` words in the data pool. The entire pool is
/// used for bucket counter data (or literal values before promotion).
/// Aggregate statistics (count, sum, min, max) are stored in separate
/// struct fields.
///
/// # Positive Buckets Only
///
/// This histogram only maintains positive buckets. Negative values are
/// rejected by [`record()`](Self::record). The OTel exponential histogram
/// data model defines both positive and negative bucket arrays; this
/// crate implements the positive side only, which is sufficient for
/// latency, size, and other non-negative metrics.
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
/// Read operations are accessed through [`view()`](Self::view),
/// which promotes from literal mode if needed and returns a
/// [`HistogramView`] with `&self` accessors.
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
    min_bucket_width: BucketWidth,
    bucket_width: BucketWidth,
    /// When true, `data[..]` holds raw f64 bit patterns (literals)
    /// instead of bucket counters.  `index_end` is repurposed as the
    /// literal count — this saves a struct field in a fixed-size type
    /// where every byte matters.  Code that reads `index_end` must
    /// check `self.literal` first; `literal_count()` provides the
    /// safe accessor.
    literal: bool,
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
            min_bucket_width: self.min_bucket_width,
            bucket_width: self.bucket_width,
            literal: self.literal,
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
            s.field("bucket_len", &self.range_len());
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

    /// Commits sum, count, min, and max from incoming values.
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

    /// Returns true if `index` falls outside the contiguous physical range
    /// `[index_base, index_base + cap)` required by SWAR at sub-U64 widths.
    /// At U64 the ring buffer handles wrapping, so this always returns false.
    #[inline]
    const fn swar_would_wrap(&self, index: i32) -> bool {
        !matches!(self.bucket_width, BucketWidth::U64)
            && (index < self.index_base
                || index >= self.index_base + self.bucket_capacity() as i32)
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

    /// Number of logical buckets available at the current width.
    #[inline]
    pub const fn bucket_capacity(&self) -> usize {
        self.bucket_width.capacity(N)
    }

    /// Eagerly promotes from literal mode to bucket mode.
    /// No-op if already in bucket mode.
    ///
    /// The error from `promote()` is intentionally discarded: literal
    /// mode stores at most `N` values, each replayed with `incr = 1`.
    /// Even at the narrowest counter width (B1 = 1-bit), the data pool
    /// holds `64 * N` counters — far more than `N` — so replay cannot
    /// overflow.
    #[inline]
    fn ensure_promoted(&mut self) {
        if self.literal {
            let _ = self.promote();
        }
    }

    /// Returns true if no buckets have been used.
    #[inline]
    pub const fn buckets_empty(&self) -> bool {
        if self.literal {
            return self.literal_count() == 0;
        }
        self.range_is_empty()
    }

    /// Checks if the bucket range represents no data.
    #[inline]
    const fn range_is_empty(&self) -> bool {
        if self.index_end != self.index_start {
            return false;
        }
        let slot = self.slot_for(self.index_start);
        self.bucket_get(slot) == 0
    }

    /// Trims leading and trailing zero buckets from the index range.
    ///
    /// Uses word-level checks to skip `slots_per_word` counters at a
    /// time when the entire word is zero, falling back to per-counter
    /// checks only at partially-occupied word boundaries.
    fn trim_bucket_range(&mut self) {
        let spw = self.bucket_width.slots_per_word() as i32;

        // Trim trailing zeros.
        while self.index_end > self.index_start {
            let slot = self.slot_for(self.index_end);
            let wi = slot / spw as usize;

            // If this is the last slot in its word and the word is
            // all zero, skip the whole word.
            if slot % spw as usize == (spw as usize - 1) && self.data[wi] == 0 {
                // Don't go below index_start.
                let skip = spw.min(self.index_end - self.index_start);
                self.index_end -= skip;
                continue;
            }
            if self.bucket_get(slot) != 0 {
                break;
            }
            self.index_end -= 1;
        }

        // Trim leading zeros.
        while self.index_start < self.index_end {
            let slot = self.slot_for(self.index_start);
            let wi = slot / spw as usize;

            if slot % spw as usize == 0 && self.data[wi] == 0 {
                let skip = spw.min(self.index_end - self.index_start);
                self.index_start += skip;
                continue;
            }
            if self.bucket_get(slot) != 0 {
                break;
            }
            self.index_start += 1;
        }
    }

    /// Returns the (word_index, bit_shift, mask) for a physical slot.
    #[inline]
    const fn slot_addr(&self, slot: usize) -> (usize, usize, u64) {
        let bits = self.bucket_width.bits();
        let spw = 64 / bits;
        (slot / spw, (slot % spw) * bits, self.bucket_width.counter_max())
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
        let spw = self.bucket_width.slots_per_word();
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
    fn new_at_scale(scale: i32) -> Result<Self, MappingError> {
        const { assert!(N >= 1, "N must be >= 1 for at least 1 bucket word") };
        Ok(Self {
            mapping: Mapping::new(scale)?,
            min_bucket_width: BucketWidth::B1,
            bucket_width: BucketWidth::B1,
            literal: true,
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
    /// Returns [`MappingError::InvalidScale`] if `scale` is outside
    /// the supported range [`MIN_SCALE`](crate::MIN_SCALE)..=[`max_scale()`](crate::max_scale).
    #[inline]
    pub fn with_scale(mut self, scale: i32) -> Result<Self, MappingError> {
        self.mapping = Mapping::new(scale)?;
        Ok(self)
    }

    /// Sets the minimum (initial) bucket counter width.
    ///
    /// By default, counters start at 1-bit (B1). Setting a higher floor
    /// (e.g. `BucketWidth::U8`) trades bucket capacity for avoiding the
    /// CPU cost of sub-byte bit-level indexing and SWAR widening.
    #[inline]
    #[must_use]
    pub fn with_min_bucket_width(mut self, width: BucketWidth) -> Self {
        self.min_bucket_width = width;
        self.bucket_width = width;
        self
    }

    /// Disables literal mode — starts directly in bucket mode.
    /// Available only for benchmarks and tests.
    #[cfg(any(test, feature = "bench-internals"))]
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn with_literal_mode(mut self, enabled: bool) -> Self {
        self.literal = enabled;
        self
    }

    /// Returns true if the histogram is in literal mode.
    #[inline]
    pub const fn is_literal(&self) -> bool {
        self.literal
    }

    /// Returns a read-only view of the histogram.
    ///
    /// Takes `&mut self` because it may need to promote from literal
    /// mode to bucket mode internally.  The returned [`HistogramView`]
    /// provides access to scale, stats, positive buckets, and quantile
    /// estimation — all via `&self`.
    ///
    /// # Why `&mut self` and not `&self`?
    ///
    /// Shared-read access (via `&self`) is intentionally not supported.
    /// In the OTel aggregation pattern, each collector owns its
    /// histogram exclusively: record into it, then [`swap`](Self::swap)
    /// or [`merge_from`](Self::merge_from) to hand off data.  Shared
    /// access is unnecessary and the internal-mutability machinery
    /// required to support it (`Cell`/`OnceCell`) would add complexity
    /// and runtime cost to every read path for no practical benefit.
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
    const fn literal_count(&self) -> usize {
        if self.literal {
            self.index_end as usize
        } else {
            0
        }
    }

    /// Maximum number of literals that can be stored.
    #[inline]
    const fn literal_capacity(&self) -> usize {
        N
    }

    /// Returns the stored literal values as a slice.
    #[inline]
    const fn literal_values(&self) -> &[u64] {
        // Split the data slice to get [0..literal_count()]
        // Using split_at for const-compatible slicing.
        let (head, _) = self.data.split_at(self.literal_count());
        head
    }

    /// Returns the current bucket counter width.
    #[inline]
    pub const fn bucket_width(&self) -> BucketWidth {
        self.bucket_width
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
    ///
    /// There is no `clear()` method — the histogram's initial scale is
    /// not stored separately. To reset, swap with a freshly constructed
    /// histogram at your desired scale:
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

    /// Records a single value.
    ///
    /// The value must be non-negative and not NaN.  See
    /// [`record`](Self::record) for why this is not checked at runtime.
    ///
    /// Positive infinity (`f64::INFINITY`) is accepted and mapped to
    /// the same bucket as `f64::MAX`, consistent with the Prometheus
    /// exponential histogram specification.
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if the total count would exceed `u64::MAX`.
    /// This is the only fallible check — because the total count is
    /// always ≥ any individual bucket count, a bucket counter at `u64`
    /// width cannot overflow when the total count fits. In practice,
    /// callers should flush and reset histograms periodically long
    /// before `u64` exhaustion.
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<(), Overflow> {
        self.record(value, 1)
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
    pub fn record(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        debug_assert!(!value.is_nan(), "NaN is not a valid histogram value");
        debug_assert!(value >= 0.0, "negative values are not supported");

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
            let result = self.try_increment(index, incr);
            if self.resolve_increment(result)? {
                return Ok(());
            }
        }
    }

    /// Decreases the mapping scale by `decrease` steps.
    fn decrease_scale(&mut self, decrease: i32) -> Result<(), Overflow> {
        let new_scale = self.mapping.scale() - decrease;
        self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
        Ok(())
    }

    /// Widens bucket counters by one step, adjusting the mapping scale
    /// to account for the implicit 1-step downscale.
    fn widen_one_step(&mut self) -> Result<(), Overflow> {
        if self.range_is_empty() {
            self.bucket_width = self.bucket_width.wider().ok_or(Overflow)?;
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
    pub fn downscale(&mut self, change: i32) -> Result<(), Overflow> {
        self.downscale_by(change)
    }

    fn downscale_by(&mut self, change: i32) -> Result<(), Overflow> {
        if change <= 0 {
            return Ok(());
        }

        if self.range_is_empty() {
            self.shift_indices(change);
            return self.decrease_scale(change);
        }

        self.do_downscale(change, self.bucket_width)?;
        self.decrease_scale(change)
    }

    fn downscale_to(&mut self, target_scale: i32) -> Result<(), Overflow> {
        let change = self.mapping.scale() - target_scale;
        if change <= 0 {
            return Ok(());
        }
        self.downscale_by(change)
    }

    /// Attempts to place `incr` into the bucket at `index`.
    ///
    /// Returns `NeedsDownscale` if the index doesn't fit in the current
    /// range, or `CounterOverflow` if the counter at `index` can't hold
    /// the addition.  The caller retries after adjusting scale or width.
    fn try_increment(&mut self, index: i32, incr: u64) -> IncrResult {
        if incr == 0 {
            return IncrResult::Ok;
        }

        let cap = self.bucket_capacity() as i32;

        if self.buckets_empty() {
            self.index_start = index;
            self.index_end = index;
            // Align base to a word boundary so that SWAR pairwise ops
            // never split a counter pair across u64 words.
            let spw = self.bucket_width.slots_per_word() as i32;
            self.index_base = index & !(spw - 1);
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
    fn resolve_increment(&mut self, result: IncrResult) -> Result<bool, Overflow> {
        match result {
            IncrResult::Ok => Ok(true),
            IncrResult::CounterOverflow => {
                self.widen_one_step()?;
                Ok(false)
            }
            IncrResult::NeedsDownscale(hl) => {
                let change = scale_reduction(hl, self.bucket_capacity() as i32);
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

// (Quantile estimation is in quantile.rs)

// ---------------------------------------------------------------------------

// Compile-time proof that Histogram is Send + Sync (all fields are Copy
// primitives). This prevents regressions if a non-Send/Sync type is
// accidentally added in the future.
const fn _assert_send_sync<T: Send + Sync>() {}
const _: () = _assert_send_sync::<Histogram<1>>();

#[cfg(test)]
mod tests;

#[cfg(test)]
mod regression;
