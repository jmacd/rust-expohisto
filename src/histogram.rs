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

// ---------------------------------------------------------------------------
// BucketWidth — counter width for bucket data
// ---------------------------------------------------------------------------

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
const ALL_WIDTHS: [BucketWidth; 7] = [
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
    const fn bits(self) -> usize {
        self as usize
    }

    /// Returns the ordinal level (0=B1 … 6=U64), used to index
    /// [`ALL_WIDTHS`] and [`SWAR_TABLE`].
    #[inline]
    const fn level(self) -> usize {
        self.bits().trailing_zeros() as usize
    }

    /// Returns the number of buckets that fit in `word_count` u64 words.
    #[inline]
    const fn capacity(self, word_count: usize) -> usize {
        (word_count * 64) / self.bits()
    }

    /// Returns the number of counter slots per u64 word.
    #[inline]
    const fn slots_per_word(self) -> usize {
        64 / self.bits()
    }

    /// Returns the next wider counter width, or `None` if already at u64.
    #[inline]
    const fn wider(self) -> Option<BucketWidth> {
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
    const fn widen_by(self, steps: i32) -> Option<BucketWidth> {
        let target = self.level() + steps as usize;
        if target > 6 {
            None
        } else {
            Some(ALL_WIDTHS[target])
        }
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    const fn counter_max(self) -> u64 {
        u64::MAX >> (64 - self.bits())
    }
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
/// Read operations (`positive()`, `scale()`, `bucket_at()`) eagerly
/// promote before returning, so literal mode is transparent to callers.
/// Disable with [`with_literal_mode(false)`](Self::with_literal_mode).
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
    pub fn sum(&self) -> f64 {
        self.stats.sum
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        self.stats.count
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        self.stats.min
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
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

    /// Returns the number of buckets in use.
    ///
    /// Promotes from literal mode if needed.
    #[inline]
    pub fn bucket_len(&mut self) -> u32 {
        self.ensure_promoted();
        self.bucket_range_len()
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

    /// Returns the count at position `pos` (0-indexed from offset).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= bucket_len()`.
    #[inline]
    pub fn bucket_at(&mut self, pos: u32) -> u64 {
        self.ensure_promoted();
        let len = self.bucket_range_len();
        assert!(
            pos < len,
            "bucket_at: pos {} out of range (len {})",
            pos,
            len
        );
        let index = self.index_start + pos as i32;
        self.bucket_get(self.slot_for(index))
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

    /// Returns the bucket offset (smallest index).
    ///
    /// Promotes from literal mode if needed.
    #[inline]
    pub fn bucket_offset(&mut self) -> i32 {
        self.ensure_promoted();
        self.index_start
    }
}

// ---------------------------------------------------------------------------
// BucketView — read-only public view of bucket data
// ---------------------------------------------------------------------------

/// Read-only view of bucket data in a histogram.
///
/// Obtaining a `BucketView` via [`positive()`](Histogram::positive)
/// requires `&mut self` because literal-mode histograms are lazily
/// promoted to bucket mode on first read. After promotion, subsequent
/// reads are normal bucket lookups with no extra cost.
#[derive(Debug)]
pub struct BucketView<'a, const N: usize> {
    hist: &'a Histogram<N>,
}

impl<const N: usize> BucketView<'_, N> {
    /// Returns the offset (smallest index).
    #[inline]
    pub fn offset(&self) -> i32 {
        // BucketView is only created after promotion, so we're in bucket mode.
        self.hist.index_start
    }

    /// Number of logical buckets in use.
    #[inline]
    pub fn len(&self) -> u32 {
        self.hist.bucket_range_len()
    }

    /// Returns the current counter width.
    #[inline]
    pub fn width(&self) -> BucketWidth {
        self.hist.bucket_width
    }

    /// Returns true if no buckets are in use.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of logical buckets available at the current width.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.hist.bucket_capacity()
    }

    /// Returns the count at position `pos` (0-indexed from offset).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= len()`.
    #[inline]
    pub fn at(&self, pos: u32) -> u64 {
        let len = self.len();
        assert!(
            pos < len,
            "BucketView::at: pos {} out of range (len {})",
            pos,
            len
        );
        let index = self.hist.index_start + pos as i32;
        self.hist.bucket_get(self.hist.slot_for(index))
    }

    /// Returns an iterator over bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, N> {
        BucketsIter {
            hist: self.hist,
            pos: 0,
            len: self.len(),
        }
    }
}

impl<'a, const N: usize> IntoIterator for &'a BucketView<'a, N> {
    type Item = u64;
    type IntoIter = BucketsIter<'a, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over bucket counts.
#[derive(Debug)]
pub struct BucketsIter<'a, const N: usize> {
    hist: &'a Histogram<N>,
    pos: u32,
    len: u32,
}

impl<const N: usize> Iterator for BucketsIter<'_, N> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let index = self.hist.index_start + self.pos as i32;
        let count = self.hist.bucket_get(self.hist.slot_for(index));
        self.pos += 1;
        Some(count)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for BucketsIter<'_, N> {}

// ---------------------------------------------------------------------------
// Bucket operations — widen (SWAR) and downscale
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Widens bucket counters from the current width by `steps` scale-steps,
    /// using SWAR pairwise summation. Each step doubles the counter width
    /// and halves the index range (equivalent to a 1-step downscale).
    ///
    /// At sub-U64, data never wraps (indices are always in
    /// `[index_base, index_base + cap)`), so SWAR operates on a
    /// contiguous linear layout.
    ///
    /// Returns `None` if already at U64 or `steps` would exceed U64.
    /// Returns `Some(steps)` on success (= the scale decrease applied).
    fn bucket_widen(&mut self, steps: i32) -> Option<i32> {
        debug_assert!(steps >= 1);

        if self.is_effectively_empty() {
            let target = self.bucket_width.widen_by(steps)?;
            self.bucket_width = target;
            self.shift_indices(steps);
            return Some(steps);
        }

        debug_assert!(
            self.bucket_width != BucketWidth::U64,
            "cannot widen past U64",
        );
        debug_assert!(
            self.index_start >= self.index_base,
            "sub-U64 data must not wrap: start={} base={}",
            self.index_start,
            self.index_base,
        );

        let mut done = 0;
        while done < steps {
            done += self.swar_merge_step(true)?;
        }

        Some(done)
    }

    /// Performs one SWAR pairwise-merge step.
    ///
    /// If `index_base` is odd, shifts data up by one slot to restore
    /// even alignment. When the top slot is occupied, falls back to
    /// [`scalar_merge_step`](Self::scalar_merge_step).
    ///
    /// When `force_widen` is true, always accepts the wider format
    /// (used by `bucket_widen`). Otherwise checks for overflow and
    /// narrows back to the original width when possible (used by
    /// `do_downscale` to preserve bucket capacity).
    ///
    /// Returns the number of scale levels consumed, or `None` if
    /// already at U64.
    fn swar_merge_step(&mut self, force_widen: bool) -> Option<i32> {
        let width = self.bucket_width;
        if width == BucketWidth::U64 {
            return None;
        }

        let shifted = self.index_base & 1 != 0;

        if shifted {
            if self.top_slot_occupied() {
                return self.scalar_merge_step(force_widen);
            }
            swar_shift_up_one(self.bucket_data_mut(), width);
        }

        swar_step(self.bucket_data_mut(), width);

        if force_widen || swar_has_overflow(self.bucket_data(), width) {
            self.bucket_width = width.wider().unwrap();
        } else {
            swar_narrow_compact(self.bucket_data_mut(), width);
        }

        self.shift_indices(1);

        if shifted {
            self.clamp_index_end();
        }

        Some(1)
    }

    /// Scalar gather-scatter merge. Used when SWAR cannot operate
    /// (odd base with top slot occupied).
    ///
    /// If `force_widen` is true, always widens by one level (for the
    /// `CounterOverflow` path). Otherwise, only widens if any pair sum
    /// exceeds the current width's maximum.
    ///
    /// When the merged range exceeds the new capacity (which happens
    /// when base is odd and the range is fully packed), additional
    /// merge levels are applied until the range fits.
    ///
    /// Returns the number of scale steps performed (>= 1).
    #[allow(clippy::needless_range_loop)]
    fn scalar_merge_step(&mut self, force_widen: bool) -> Option<i32> {
        let width = self.bucket_width;
        let max_val = width.counter_max();
        let bucket_words = self.bucket_word_count();

        let mut cur_start = self.index_start >> 1;
        let mut cur_end = self.index_end >> 1;
        let mut cur_len = (cur_end - cur_start + 1) as usize;

        let mut sums = vec![0u64; cur_len];
        let mut needs_widen = force_widen;

        // Phase 1: gather from hardware layout, merging adjacent pairs.
        for old_idx in self.index_start..=self.index_end {
            let val = self.bucket_get(self.slot_for(old_idx));
            let out = ((old_idx >> 1) - cur_start) as usize;
            sums[out] = sums[out].saturating_add(val);
            if sums[out] > max_val {
                needs_widen = true;
            }
        }

        let mut target_width = if needs_widen { width.wider()? } else { width };

        let mut scale_steps = 1i32;

        // Phase 2: if the merged range doesn't fit in the new capacity,
        // keep merging pairs and widening until it does.
        loop {
            let target_cap = target_width.capacity(bucket_words);
            if cur_len <= target_cap {
                break;
            }
            // Re-pair the sums array for another merge level.
            let prev_start = cur_start;
            let prev_len = cur_len;
            cur_start >>= 1;
            cur_end >>= 1;
            cur_len = (cur_end - cur_start + 1) as usize;

            let mut next = vec![0u64; cur_len];
            for i in 0..prev_len {
                let old_idx = prev_start + i as i32;
                let out = ((old_idx >> 1) - cur_start) as usize;
                next[out] = next[out].saturating_add(sums[i]);
            }
            sums = next;

            if target_width != BucketWidth::U64 {
                target_width = target_width.wider()?;
            }
            scale_steps += 1;
        }

        // Phase 3: commit — clear data, write back.
        self.bucket_width = target_width;

        let spw = target_width.slots_per_word() as i32;
        self.rewrite_buckets(cur_start, cur_end, cur_start & !(spw - 1), &sums[..cur_len]);

        Some(scale_steps)
    }

    /// Clears bucket data and writes `sums` into the new index range.
    fn rewrite_buckets(&mut self, start: i32, end: i32, base: i32, sums: &[u64]) {
        self.bucket_data_mut().fill(0);
        self.index_start = start;
        self.index_end = end;
        self.index_base = base;
        for (i, &v) in sums.iter().enumerate() {
            let idx = start + i as i32;
            self.bucket_set(self.slot_for(idx), v);
        }
    }

    /// Downscales at U64 width by collapsing 2^by adjacent buckets.
    ///
    /// At U64 width, sums use saturating arithmetic and cannot
    /// meaningfully overflow.
    fn bucket_downscale_u64(&mut self, by: i32) {
        debug_assert_eq!(self.bucket_width, BucketWidth::U64);
        debug_assert!(by >= 1);

        if self.is_effectively_empty() {
            self.shift_indices(by);
            return;
        }

        let new_start = self.index_start >> by;
        let new_end = self.index_end >> by;
        let new_len = (new_end - new_start + 1) as usize;

        let mut sums = [0u64; 256];
        debug_assert!(new_len <= sums.len());

        for old_idx in self.index_start..=self.index_end {
            let val = self.bucket_get(self.slot_for(old_idx));
            let out = ((old_idx >> by) - new_start) as usize;
            sums[out] = sums[out].saturating_add(val);
        }

        self.rewrite_buckets(new_start, new_end, new_start, &sums[..new_len]);
    }
}

// ---------------------------------------------------------------------------
// SWAR — per-word parallel pairwise summation
// ---------------------------------------------------------------------------

/// Per-width SWAR parameters: `(shift, lane_mask)`.
///
/// - `shift`: number of bits to shift the upper half-slots down.
/// - `lane_mask`: keeps only the lower half-slot in each pair.
/// - `!lane_mask`: overflow mask — bits set here after a step mean the
///   pair-sum overflowed the original width.
///
/// Indexed by [`BucketWidth::level()`] (0=B1 … 5=U32).
const SWAR_TABLE: [(u32, u64); 6] = [
    (1, 0x5555_5555_5555_5555),  // B1
    (2, 0x3333_3333_3333_3333),  // B2
    (4, 0x0F0F_0F0F_0F0F_0F0F), // B4
    (8, 0x00FF_00FF_00FF_00FF),  // U8
    (16, 0x0000_FFFF_0000_FFFF), // U16
    (32, 0x0000_0000_FFFF_FFFF), // U32
];

/// Single SWAR step: sum adjacent counters at the current width into
/// the next wider width, in place.
#[inline]
fn swar_step(data: &mut [u64], width: BucketWidth) {
    debug_assert_ne!(width, BucketWidth::U64, "cannot widen past U64");
    let (shift, mask) = SWAR_TABLE[width.level()];
    for w in data.iter_mut() {
        let x = *w;
        *w = ((x >> shift) & mask) + (x & mask);
    }
}

/// Checks whether any widened pair-sum overflows the original width.
/// Called after `swar_step` has already written the wider sums.
#[inline]
fn swar_has_overflow(data: &[u64], original_width: BucketWidth) -> bool {
    if original_width == BucketWidth::U64 {
        return false;
    }
    let (_, mask) = SWAR_TABLE[original_width.level()];
    data.iter().any(|&w| w & !mask != 0)
}

/// Compacts narrowed half-words into full words, pairing two source
/// words into one destination word and zeroing the freed tail.
#[inline]
fn compact_with<F: Fn(u64) -> u64>(data: &mut [u64], narrow: F) {
    let n = data.len();
    for i in (0..n).step_by(2) {
        let lo = narrow(data[i]);
        let hi = if i + 1 < n { narrow(data[i + 1]) } else { 0 };
        data[i / 2] = lo | (hi << 32);
    }
    for w in &mut data[n.div_ceil(2)..n] {
        *w = 0;
    }
}

/// Shifts all slot values up by one position, inserting a zero at slot 0.
///
/// This effectively decrements the logical `index_base` by one, turning
/// an odd base into an even one so that a normal SWAR step pairs the
/// correct indices.
///
/// Precondition: the top slot of the last word must be zero (the live
/// range must not fill the entire capacity).
#[inline]
fn swar_shift_up_one(data: &mut [u64], width: BucketWidth) {
    debug_assert_ne!(width, BucketWidth::U64);
    let bits = width.bits();
    let n = data.len();
    if n == 0 {
        return;
    }
    debug_assert!(
        data[n - 1] >> (64 - bits) == 0,
        "top slot must be zero before shift",
    );
    // Process high-to-low so each word reads from the (unmodified) word below.
    for i in (1..n).rev() {
        data[i] = (data[i] << bits) | (data[i - 1] >> (64 - bits));
    }
    data[0] <<= bits;
}

/// Progressive bit-compaction: at each stage, merge adjacent groups by
/// OR-shifting, then mask to keep only the compacted result.
#[inline]
fn compact_lanes(mut x: u64, stages: &[(u32, u64)]) -> u64 {
    for &(shift, mask) in stages {
        x = (x | (x >> shift)) & mask;
    }
    x
}

/// Narrows a single word from `wider(original_width)` format back to
/// `original_width`, compacting the result into the low 32 bits.
///
/// Stage parameters are derived from [`SWAR_TABLE`]: stage K uses
/// `(SWAR_TABLE[K].0, SWAR_TABLE[K+1].1)` — the shift of level K and
/// the lane mask of level K+1.
#[inline]
fn narrow_word(w: u64, original_width: BucketWidth) -> u64 {
    const COMPACT_STAGES: [(u32, u64); 5] = [
        (SWAR_TABLE[0].0, SWAR_TABLE[1].1),
        (SWAR_TABLE[1].0, SWAR_TABLE[2].1),
        (SWAR_TABLE[2].0, SWAR_TABLE[3].1),
        (SWAR_TABLE[3].0, SWAR_TABLE[4].1),
        (SWAR_TABLE[4].0, SWAR_TABLE[5].1),
    ];

    let level = original_width.level();
    debug_assert!(level <= 5, "can only narrow sub-U64 widths");

    if level == 5 {
        // U32: one sum per word, just mask the carry bit.
        w & SWAR_TABLE[5].1
    } else {
        compact_lanes(w & SWAR_TABLE[level].1, &COMPACT_STAGES[level..])
    }
}

/// After a SWAR step that produced no overflow, narrow the widened sums
/// back to the original width and compact words 2:1.
///
/// The data is currently in `wider(original_width)` format with all values
/// fitting in `original_width`. This function bit-compresses each word
/// (via [`narrow_word`]) and packs pairs of words into one, freeing the
/// upper half of the array.
#[inline]
fn swar_narrow_compact(data: &mut [u64], original_width: BucketWidth) {
    compact_with(data, |w| narrow_word(w, original_width));
}

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

    /// Returns the current scale.
    ///
    /// Promotes from literal mode if needed. Returns 0 when no
    /// non-zero values have been recorded.
    #[inline]
    pub fn scale(&mut self) -> i32 {
        if self.literal {
            if self.literal_count() == 0 {
                return 0;
            }
            let _ = self.promote();
        }
        if self.non_zero_count() == 0 {
            0
        } else {
            self.mapping.scale()
        }
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

    /// Returns a read-only view of the positive buckets.
    ///
    /// Takes `&mut self` because literal-mode histograms are lazily
    /// promoted to bucket mode on first read. After promotion the
    /// histogram stays in bucket mode, so subsequent reads pay no
    /// extra cost.
    #[inline]
    pub fn positive(&mut self) -> BucketView<'_, N> {
        // promote() cannot fail here — if it could (which requires
        // an invalid limit_scale), we'd have failed at construction.
        self.ensure_promoted();
        BucketView { hist: self }
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

    // -- Rollback helper --

    /// Executes `f`, rolling back to the pre-call state on error.
    fn with_rollback<F>(&mut self, f: F) -> Result<(), Overflow>
    where
        F: FnOnce(&mut Self) -> Result<(), Overflow>,
    {
        let snapshot = self.clone();
        match f(self) {
            Ok(()) => Ok(()),
            Err(e) => {
                *self = snapshot;
                Err(e)
            }
        }
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
    /// On error, the histogram is unchanged.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        debug_assert!(value >= 0.0, "Histogram only accepts non-negative values");

        if incr == 0 {
            return Ok(());
        }

        let new_count = self.checked_add_count(incr).ok_or(Overflow)?;

        if value != 0.0 {
            self.with_rollback(|h| {
                if h.literal {
                    h.update_literal(value, incr)
                } else {
                    h.update_buckets(value, incr)
                }
            })?;
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

        for &bits in &literals[..count] {
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
    fn widen_one_step(&mut self) -> Result<(), Overflow> {
        let by = self.bucket_widen(1).ok_or(Overflow)?;
        self.adjust_scale(by)
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
    /// merges), shifts data up by one slot to restore even alignment
    /// before the SWAR step. Falls back to scalar gather-scatter for
    /// the rare case where the top slot is occupied.
    ///
    /// At U64 width, remaining steps use `bucket_downscale_u64`
    /// (scatter-write collapse).
    #[doc(hidden)]
    pub fn do_downscale(&mut self, change: i32) -> Result<(), Overflow> {
        if change <= 0 {
            return Ok(());
        }

        if self.is_effectively_empty() {
            self.shift_indices(change);
            return self.adjust_scale(change);
        }

        let mut remaining = change;

        // Phase 1: Adaptive SWAR merge at sub-U64 widths.
        while remaining > 0 && self.bucket_width != BucketWidth::U64 {
            let steps = self.swar_merge_step(false).ok_or(Overflow)?;
            self.adjust_scale(steps)?;
            remaining -= steps;
        }

        // Phase 2: At U64, scatter-write for remaining steps.
        if remaining > 0 {
            debug_assert_eq!(self.bucket_width, BucketWidth::U64);
            self.bucket_downscale_u64(remaining);
            self.adjust_scale(remaining)?;
        }

        self.trim_bucket_range();

        Ok(())
    }

    fn downscale_to(&mut self, target_scale: i32) -> Result<(), Overflow> {
        let change = self.mapping.scale() - target_scale;
        if change <= 0 {
            return Ok(());
        }
        self.do_downscale(change)
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
                    self.do_downscale(change)?;
                } else if self.bucket_width != BucketWidth::U64 {
                    self.widen_one_step()?;
                } else {
                    self.do_downscale(1)?;
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
    /// On error, the histogram is unchanged.
    pub fn merge_from_raw(
        &mut self,
        stats: &Stats,
        buckets: &BucketDescriptor,
        at: &dyn Fn(u32) -> u64,
    ) -> Result<(), Overflow> {
        if stats.count == 0 {
            return Ok(());
        }

        let new_count = self.checked_add_count(stats.count).ok_or(Overflow)?;
        let new_sum = self.sum() + stats.sum;

        self.with_rollback(|h| {
            if buckets.len > 0 {
                if h.literal {
                    h.promote()?;
                }

                let other_end = buckets.offset + buckets.len as i32 - 1;
                let cap = h.bucket_capacity() as i32;
                let min_scale = h.mapping.scale().min(buckets.scale);

                let self_hl = h.high_low_at_scale(min_scale);
                let other_hl = {
                    let shift = buckets.scale - min_scale;
                    HighLow {
                        low: buckets.offset >> shift,
                        high: other_end >> shift,
                    }
                };
                let hlp = self_hl.merge(other_hl);
                let min_scale = min_scale - change_scale(hlp, cap);

                h.downscale_to(min_scale)?;

                for i in 0..buckets.len {
                    let count = at(i);
                    if count == 0 {
                        continue;
                    }
                    h.retry_increment(count, |h| {
                        let shift = buckets.scale - h.mapping.scale();
                        (buckets.offset + i as i32) >> shift
                    })?;
                }
            }

            h.trim_bucket_range();
            h.commit_stats(new_sum, new_count, stats.min, stats.max);
            Ok(())
        })
    }

    /// Merges literal values from another histogram into this one.
    fn merge_literal_from<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Overflow> {
        debug_assert!(other.literal);
        if other.count() == 0 {
            return Ok(());
        }
        let new_count = self.checked_add_count(other.count()).ok_or(Overflow)?;
        let new_sum = self.sum() + other.sum();
        self.with_rollback(|h| {
            if h.literal {
                h.promote()?;
            }
            for &bits in other.literal_values() {
                h.update_buckets(f64::from_bits(bits), 1)?;
            }
            h.commit_stats(new_sum, new_count, other.min(), other.max());
            Ok(())
        })
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: count total across all positive buckets.
    fn bucket_total<const N: usize>(h: &mut Histogram<N>) -> u64 {
        h.positive().iter().sum()
    }

    fn derived_zero_count<const N: usize>(h: &mut Histogram<N>) -> u64 {
        h.count() - bucket_total(h)
    }

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(1.0).unwrap();
        assert_stats(&h, 1, 1.0, 1.0, 1.0);
        assert_eq!(derived_zero_count(&mut h), 0);
        assert_eq!(h.bucket_width(), BucketWidth::B1);
    }

    #[test]
    fn test_histogram_zero() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(0.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(derived_zero_count(&mut h), 1);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_multiple() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(2.0).unwrap();
        h.update(4.0).unwrap();
        assert_stats(&h, 3, 7.0, 1.0, 4.0);
    }

    #[test]
    fn test_histogram_downscale() {
        let mut h: Histogram<8> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(1000.0).unwrap();
        assert_eq!(h.count(), 2);
        assert!(h.scale() < max_scale());
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1: Histogram<16> = Histogram::new();
        let mut h2: Histogram<16> = Histogram::new();
        h1.update(1.0).unwrap();
        h1.update(2.0).unwrap();
        h2.update(3.0).unwrap();
        h2.update(4.0).unwrap();
        h1.merge_from(&h2).unwrap();
        assert_stats(&h1, 4, 10.0, 1.0, 4.0);
    }

    #[test]
    fn test_histogram_clear() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(2.0).unwrap();
        h.clear();
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.scale(), 0);
        assert_eq!(h.bucket_width(), BucketWidth::B1);
    }

    #[test]
    fn test_buckets_at() {
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update(1.5).unwrap();
        h.update(100.0).unwrap();
        h.update(1e10).unwrap();

        let buckets = h.positive();
        assert!(
            buckets.len() >= 2,
            "expected at least 2 buckets, got {} at scale {}",
            buckets.len(),
            h.scale()
        );
    }

    #[test]
    fn test_auto_widen_cascade() {
        let mut h: Histogram<16> = Histogram::new()
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);

        // B4 → U8 at threshold 15+1=16
        h.update_by_incr(1.0, 15).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 16);

        // U8 → U16 at threshold 255+1=256
        h.update_by_incr(1.0, 239).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), 256);

        // U16 → U32 at threshold 65535+1=65536
        h.update_by_incr(1.0, u16::MAX as u64 - 256).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        assert_eq!(h.count(), u16::MAX as u64 + 1);

        // U32 → U64 at threshold 4294967295+1
        h.update_by_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1))
            .unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U64);
    }

    #[test]
    fn test_auto_widen_b4_to_u8_from_b4_start() {
        let mut h: Histogram<16> = Histogram::new()
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        h.update_by_incr(1.0, 4).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update_by_incr(1.0, 11).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 16);
    }

    #[test]
    fn test_bucket_count_halves_on_widen() {
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        let initial_cap = h.bucket_capacity();
        assert_eq!(initial_cap, 16 * 16); // 256

        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.bucket_capacity(), 16 * 8); // 128
    }

    #[test]
    fn test_clear_resets_to_b4() {
        let mut h: Histogram<16> = Histogram::with_max_scale(3)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.clear();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_eq!(h.count(), 0);
        assert_eq!(h.limit_scale(), 3);
    }

    #[test]
    fn test_with_max_scale() {
        let h: Histogram<16> = Histogram::with_max_scale(3);
        assert_eq!(h.limit_scale(), 3);
    }

    #[test]
    fn test_with_max_scale_clamps() {
        let h: Histogram<16> = Histogram::with_max_scale(100);
        assert_eq!(h.limit_scale(), max_scale());
    }

    #[test]
    fn test_with_max_scale_records_at_limited_scale() {
        let mut limited: Histogram<16> = Histogram::with_max_scale(3);
        let mut unlimited: Histogram<16> = Histogram::new();
        limited.update(1.0).unwrap();
        limited.update(1.001).unwrap();
        unlimited.update(1.0).unwrap();
        unlimited.update(1.001).unwrap();
        assert!(limited.scale() <= 3);
        if max_scale() > 3 {
            assert!(unlimited.scale() > limited.scale());
        }
    }

    #[test]
    fn test_clear_resets_to_limit_scale() {
        let mut h: Histogram<16> = Histogram::with_max_scale(3);
        h.update(0.001).unwrap();
        h.update(1000.0).unwrap();
        assert!(h.scale() <= 3);
        h.clear();
        assert_eq!(h.count(), 0);
        assert_eq!(h.limit_scale(), 3);
        h.update(1.0).unwrap();
        assert_eq!(h.scale(), 3);
    }

    #[test]
    fn test_widen_preserves_data() {
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(1.0, 100).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        h.update_by_incr(256.0, 50).unwrap();
        h.update_by_incr(65536.0, 200).unwrap();

        let count_before = h.count();
        let sum_before = h.sum();

        h.update_by_incr(65536.0, 55).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.update(65536.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        assert_eq!(h.count(), count_before + 56);
        assert!((h.sum() - (sum_before + 56.0 * 65536.0)).abs() < 1.0);
    }

    #[test]
    fn test_merge_equivalence_comprehensive() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let hardcoded_sets: &[&[f64]] = &[
            &[],
            &[0.0],
            &[1.0],
            &[0.0, 0.0],
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
            &[
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            ],
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

        let mut test_sets: Vec<Vec<f64>> = hardcoded_sets.iter().map(|s| s.to_vec()).collect();

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let size = rng.gen_range(0..=10);
            let set: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..20.0)).collect();
            test_sets.push(set);
        }

        test_merge_equivalence_for_size::<8>(&test_sets);
        test_merge_equivalence_for_size::<12>(&test_sets);
        test_merge_equivalence_for_size::<16>(&test_sets);
        test_merge_equivalence_for_size::<20>(&test_sets);
    }

    fn test_merge_equivalence_for_size<const K: usize>(test_sets: &[Vec<f64>]) {
        for (i, set_a) in test_sets.iter().enumerate() {
            for (j, set_b) in test_sets.iter().enumerate() {
                let mut merged = build_from_values::<K>(set_a);
                let other = build_from_values::<K>(set_b);
                if let Err(e) = merged.merge_from(&other) {
                    panic!("merge_from failed for size={K} sets {i} x {j}: {e}\n  set_a: {set_a:?}\n  set_b: {set_b:?}\n  merged: {:?}\n  other: {:?}", merged, other);
                }

                let mut single = build_from_values::<K>(set_a);
                for &v in set_b.iter() {
                    single.update(v).unwrap();
                }

                let label = format!("size={K} sets {i} x {j}");
                assert_eq!(merged.count(), single.count(), "count mismatch for {label}");
                let ms = merged.sum();
                let ss = single.sum();
                let sum_diff = (ms - ss).abs();
                let denom = ms.abs().max(ss.abs()).max(1e-30);
                assert!(
                    sum_diff / denom < 1e-5,
                    "sum mismatch for {label}: {ms} vs {ss}"
                );
                assert_eq!(
                    derived_zero_count(&mut merged),
                    derived_zero_count(&mut single),
                    "zero_count mismatch for {label}"
                );
                assert_eq!(
                    bucket_total(&mut merged),
                    bucket_total(&mut single),
                    "bucket total mismatch for {label}"
                );
            }
        }
    }

    #[test]
    fn test_merge_regression_bucket_total() {
        // Regression: "bucket total mismatch for size=8 sets 2 x 35"
        let set_b: &[f64] = &[
            18.896147780359236,
            19.038540970281623,
            15.726266735088323,
            19.97053274796744,
            16.963914020801518,
        ];

        // Verify incremental bucket totals while building.
        let mut other: Histogram<8> = Histogram::new();
        for &v in set_b {
            other.update(v).unwrap();
            let bt = bucket_total(&mut other);
            let non_zero_count = other.count() - derived_zero_count(&mut other);
            assert_eq!(bt, non_zero_count, "bucket total mismatch after inserting {v}");
        }

        let set_a: &[f64] = &[1.0];
        let mut merged = build_from_values::<8>(set_a);
        merged.merge_from(&other).unwrap();

        let mut single = build_from_values::<8>(set_a);
        for &v in set_b {
            single.update(v).unwrap();
        }

        assert_eq!(
            bucket_total(&mut merged),
            bucket_total(&mut single),
            "bucket total mismatch: merged vs single"
        );
    }

    #[test]
    fn test_edge_values_inf() {
        use crate::mapping::Mapping;

        let max_f64: f64 = f64::MAX;
        let inf: f64 = f64::INFINITY;

        let m0 = Mapping::new(0).unwrap();
        let idx_max = m0.map_to_index(max_f64);
        let idx_inf = m0.map_to_index(inf);
        assert_eq!(idx_max, idx_inf);

        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update(1.0).unwrap();
        h.update(max_f64).unwrap();
        // f64::MAX fits in f64 sum, but after adding infinity the sum
        // is infinite.
        h.update(inf).unwrap();
        assert_eq!(h.count(), 3);
        assert_eq!(h.max(), f64::INFINITY);
        assert!(h.sum().is_infinite());
        assert_eq!(h.min(), 1.0);
    }

    #[test]
    fn test_edge_values_subnormals() {
        use crate::mapping::Mapping;

        let subnormal: f64 = 5e-324;
        let min_normal: f64 = crate::float64::MIN_VALUE;

        let m0 = Mapping::new(0).unwrap();
        assert_eq!(m0.map_to_index(subnormal), m0.map_to_index(min_normal));

        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update(subnormal).unwrap();
        h.update(min_normal).unwrap();
        assert_eq!(h.count(), 2);
        assert_eq!(h.positive().len(), 1);
    }

    #[test]
    fn test_exhaustive_u8_overflow() {
        // Insert 8 values spanning a wide index range at scale 0, each
        // with count 255. Starting at B1 with 320 slots (Histogram<8>),
        // counters widen B1→B2→B4→U8 (255 fits in U8), but the larger
        // initial capacity means the span still fits without reaching U64.
        let mut h: Histogram<8> = Histogram::with_scale(0);
        let num_buckets = 8;
        for i in 0..num_buckets {
            let val = 2.0_f64.powi(i * 8);
            h.update_by_incr(val, 255).unwrap();
        }
        // With B1 start, U8 has enough capacity for the span.
        assert!(
            h.bucket_width() >= BucketWidth::U8,
            "expected at least U8, got {:?}",
            h.bucket_width()
        );
        assert_eq!(h.count(), num_buckets as u64 * 255);
        // Adding one more should still be fine at U64 (no further widen needed).
        h.update(1.0).unwrap();
        assert_eq!(h.count(), num_buckets as u64 * 255 + 1);
    }

    #[test]
    fn test_successive_sub_byte_widening() {
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);

        h.update(1.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        for count in 2..=15u64 {
            h.update(1.0).unwrap();
            assert_eq!(h.count(), count);
            assert_eq!(
                h.bucket_width(),
                BucketWidth::B4,
                "expected B4 at count {count}"
            );
        }

        h.update(1.0).unwrap();
        assert_eq!(h.count(), 16);
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        assert!((h.sum() - 16.0).abs() < 1e-10);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
    }

    #[test]
    fn test_successive_sub_byte_widening_multi_bucket() {
        let mut h: Histogram<16> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        let num_buckets = 8;
        let values: Vec<f64> = (1..=num_buckets).map(|k| 2.0_f64.powi(k)).collect();

        for &v in &values {
            h.update(v).unwrap();
        }
        assert_eq!(h.count(), num_buckets as u64);
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        for &v in &values {
            h.update(v).unwrap();
        }
        assert_eq!(h.count(), 2 * num_buckets as u64);
        assert!(h.bucket_width() >= BucketWidth::B4);

        let target = 16 * num_buckets as u64;
        while h.count() < target {
            for &v in &values {
                h.update(v).unwrap();
            }
        }
        assert!(h.bucket_width() >= BucketWidth::U8);

        let expected_sum: f64 = values.iter().sum::<f64>() * 16.0;
        assert!(
            (h.sum() - expected_sum).abs() < 1e-6,
            "sum mismatch: got {} expected {}",
            h.sum(),
            expected_sum
        );
        assert_eq!(h.count(), target);
    }

    // -----------------------------------------------------------------------
    // Cross-size merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_different_sizes() {
        let mut collector: Histogram<16> = Histogram::new();
        let mut source: Histogram<8> = Histogram::new();

        source.update(1.0).unwrap();
        source.update(2.0).unwrap();
        source.update(4.0).unwrap();
        source.update(0.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        assert_eq!(collector.count(), 4);
        assert_eq!(derived_zero_count(&mut collector), 1);
        assert!((collector.sum() - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_merge_multiple_sources() {
        let mut collector: Histogram<20> = Histogram::new();

        for batch in 0..5 {
            let mut src: Histogram<16> = Histogram::new();
            for i in 0..10 {
                src.update((batch * 10 + i) as f64 * 0.1 + 0.1).unwrap();
            }
            collector.merge_from_other(&src).unwrap();
        }

        assert_eq!(collector.count(), 50);
        assert!(collector.sum() > 0.0);
    }

    #[test]
    fn test_merge_preserves_buckets() {
        let mut collector: Histogram<16> = Histogram::with_scale(0);
        let mut source: Histogram<16> = Histogram::with_scale(0);

        source.update(1.0).unwrap();
        source.update(2.0).unwrap();
        source.update(4.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        let mut direct: Histogram<16> = Histogram::with_scale(0);
        direct.update(1.0).unwrap();
        direct.update(2.0).unwrap();
        direct.update(4.0).unwrap();

        assert_eq!(collector.scale(), direct.scale());
        assert_eq!(collector.positive().offset(), direct.positive().offset());
        assert_eq!(collector.positive().len(), direct.positive().len());
        for i in 0..collector.positive().len() {
            assert_eq!(
                collector.positive().at(i),
                direct.positive().at(i),
                "bucket[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_merge_empty_into_populated() {
        let mut collector: Histogram<16> = Histogram::new();
        collector.update(1.0).unwrap();

        let empty: Histogram<8> = Histogram::new();
        collector.merge_from_other(&empty).unwrap();

        assert_eq!(collector.count(), 1);
        assert_eq!(collector.sum(), 1.0);
    }

    #[test]
    fn test_merge_into_empty() {
        let mut collector: Histogram<16> = Histogram::new();
        let mut source: Histogram<8> = Histogram::new();
        source.update(5.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        assert_eq!(collector.count(), 1);
        assert!((collector.sum() - 5.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Flat layout capacity tests
    // -----------------------------------------------------------------------

    mod flat_layout {
        use super::*;

        #[test]
        fn test_capacity() {
            let h: Histogram<16> = Histogram::new();
            assert_eq!(h.bucket_word_count(), 16);
            assert_eq!(h.bucket_capacity(), 1024); // 16 * 64 at B1
        }

        #[test]
        fn test_minimum_n() {
            let h: Histogram<8> = Histogram::new();
            assert_eq!(h.bucket_word_count(), 8);
            assert_eq!(h.bucket_capacity(), 512); // 8 * 64 at B1
        }

        #[test]
        fn test_struct_size() {
            use core::mem;
            let size = mem::size_of::<Histogram<16>>();
            // 16 u64 words (128 bytes) + Stats (32 bytes) + fixed metadata.
            // Data pool + stats dominate; struct should not exceed pool + 80 bytes overhead.
            assert!(
                size <= 128 + 80,
                "Histogram<16> unexpectedly large: {} bytes",
                size
            );
        }
    }

    // -----------------------------------------------------------------------
    // Narrow function unit tests
    // -----------------------------------------------------------------------

    /// Helper: pack 8 bytes into one u64, byte0 in the LSB.
    fn pack_u8x8(b: [u8; 8]) -> u64 {
        u64::from_le_bytes(b)
    }

    /// Helper: pack 16 nibbles into one u64, nibble0 in the low 4 bits.
    fn pack_b4x16(n: [u8; 16]) -> u64 {
        let mut w = 0u64;
        for (i, &nibble) in n.iter().enumerate() {
            w |= (nibble as u64 & 0xF) << (i * 4);
        }
        w
    }

    /// Helper: pack 4 u16s into one u64, short0 in the low 16 bits.
    fn pack_u16x4(s: [u16; 4]) -> u64 {
        (s[0] as u64) | ((s[1] as u64) << 16) | ((s[2] as u64) << 32) | ((s[3] as u64) << 48)
    }

    /// Helper: pack 2 u32s into one u64, int0 in the low 32 bits.
    fn pack_u32x2(lo: u32, hi: u32) -> u64 {
        (lo as u64) | ((hi as u64) << 32)
    }

    /// Asserts `swar_narrow_compact` produces `expected` prefix words
    /// and zeroes all freed tail words.
    fn assert_compact(width: BucketWidth, input: &[u64], expected: &[u64]) {
        let mut data = [0u64; 8];
        data[..input.len()].copy_from_slice(input);
        swar_narrow_compact(&mut data[..input.len()], width);
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                data[i], exp,
                "word {i}: got {:#018x}, expected {:#018x}",
                data[i], exp
            );
        }
        for (i, word) in data[expected.len()..input.len()].iter().enumerate() {
            assert_eq!(*word, 0, "word {} should be zeroed", expected.len() + i);
        }
    }

    /// Runs the full SWAR pipeline (step → overflow check → optional compact)
    /// and asserts the result.
    fn assert_swar_roundtrip(
        width: BucketWidth,
        input: &[u64],
        expect_overflow: bool,
        expected_after_step: &[u64],
        expected_after_compact: Option<&[u64]>,
    ) {
        let mut data = [0u64; 8];
        data[..input.len()].copy_from_slice(input);
        let slice = &mut data[..input.len()];

        swar_step(slice, width);
        for (i, &exp) in expected_after_step.iter().enumerate() {
            assert_eq!(
                slice[i], exp,
                "swar_step word {i}: got {:#018x}, expected {:#018x}",
                slice[i], exp
            );
        }

        assert_eq!(
            swar_has_overflow(slice, width),
            expect_overflow,
            "overflow mismatch"
        );

        if let Some(expected) = expected_after_compact {
            swar_narrow_compact(slice, width);
            for (i, &exp) in expected.iter().enumerate() {
                assert_eq!(
                    slice[i], exp,
                    "compact word {i}: got {:#018x}, expected {:#018x}",
                    slice[i], exp
                );
            }
            for (i, word) in slice[expected.len()..input.len()].iter().enumerate() {
                assert_eq!(*word, 0, "word {} should be zeroed after compact", expected.len() + i);
            }
        }
    }

    /// Helper for asserting `Stats` fields.
    fn assert_stats<const N: usize>(
        h: &Histogram<N>,
        count: u64,
        sum: f64,
        min: f64,
        max: f64,
    ) {
        assert_eq!(h.count(), count, "count");
        assert_eq!(h.sum(), sum, "sum");
        assert_eq!(h.min(), min, "min");
        assert_eq!(h.max(), max, "max");
    }

    /// Builds a literal-mode and a bucket-mode histogram from the same
    /// values and asserts they produce equivalent views.
    fn assert_literal_matches_bucket(values: &[f64]) {
        let mut lit: Histogram<8> = Histogram::new();
        let mut bkt: Histogram<8> = Histogram::new().with_literal_mode(false);
        for &v in values {
            lit.update(v).unwrap();
            bkt.update(v).unwrap();
        }
        assert_eq!(lit.scale(), bkt.scale(), "scale mismatch");
        assert_eq!(lit.count(), bkt.count(), "count mismatch");
        assert_eq!(lit.sum(), bkt.sum(), "sum mismatch");
        assert_eq!(lit.positive().offset(), bkt.positive().offset(), "offset");
        assert_eq!(lit.positive().len(), bkt.positive().len(), "len");
        for i in 0..lit.positive().len() {
            assert_eq!(
                lit.positive().at(i),
                bkt.positive().at(i),
                "bucket[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_narrow_u8_to_b4_zeroes() {
        assert_eq!(narrow_word(0, BucketWidth::B4), 0);
    }

    #[test]
    fn test_narrow_u8_to_b4() {
        let cases: &[([u8; 8], [u8; 16])] = &[
            (
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [15, 15, 15, 15, 15, 15, 15, 15],
                [15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [15, 0, 8, 0, 3, 0, 1, 0],
                [15, 0, 8, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];
        for (i, (input_bytes, expected_nibbles)) in cases.iter().enumerate() {
            let result = narrow_word(pack_u8x8(*input_bytes), BucketWidth::B4);
            let expected = pack_b4x16(*expected_nibbles);
            assert_eq!(
                result, expected,
                "case {i}: got {result:#018x}, expected {expected:#018x}"
            );
        }
    }

    #[test]
    fn test_narrow_u16_to_u8() {
        assert_eq!(narrow_word(0, BucketWidth::U8), 0);
        let cases: &[([u16; 4], [u8; 8])] = &[
            ([10, 20, 30, 40], [10, 20, 30, 40, 0, 0, 0, 0]),
            ([255, 255, 255, 255], [255, 255, 255, 255, 0, 0, 0, 0]),
        ];
        for (i, (input_shorts, expected_bytes)) in cases.iter().enumerate() {
            let result = narrow_word(pack_u16x4(*input_shorts), BucketWidth::U8);
            let expected = pack_u8x8(*expected_bytes) & 0xFFFF_FFFF;
            assert_eq!(
                result, expected,
                "case {i}: got {result:#018x}, expected {expected:#018x}"
            );
        }
    }

    #[test]
    fn test_narrow_u32_to_u16() {
        assert_eq!(narrow_word(0, BucketWidth::U16), 0);
        let cases: &[(u32, u32, u64)] = &[
            (1000, 2000, 1000 | (2000 << 16)),
            (65535, 65535, 65535 | (65535 << 16)),
        ];
        for (i, &(a, b, expected)) in cases.iter().enumerate() {
            let result = narrow_word(pack_u32x2(a, b), BucketWidth::U16);
            assert_eq!(
                result, expected,
                "case {i}: got {result:#018x}, expected {expected:#018x}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // swar_narrow_compact end-to-end tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_narrow_compact_two_words() {
        // B4: 2 words of U8 → 1 word of B4
        assert_compact(
            BucketWidth::B4,
            &[
                pack_u8x8([1, 2, 3, 4, 5, 6, 7, 8]),
                pack_u8x8([9, 10, 11, 12, 13, 14, 15, 0]),
            ],
            &[pack_b4x16([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
            ])],
        );
        // U8: 2 words of U16 → 1 word of U8
        assert_compact(
            BucketWidth::U8,
            &[pack_u16x4([10, 20, 30, 40]), pack_u16x4([50, 60, 70, 80])],
            &[pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80])],
        );
        // U16: 2 words of U32 → 1 word of U16
        assert_compact(
            BucketWidth::U16,
            &[pack_u32x2(100, 200), pack_u32x2(300, 400)],
            &[pack_u16x4([100, 200, 300, 400])],
        );
        // U32: 2 words of U64 → 1 word of U32
        assert_compact(
            BucketWidth::U32,
            &[1000u64, 2000u64],
            &[pack_u32x2(1000, 2000)],
        );
    }

    #[test]
    fn test_swar_narrow_compact_four_words() {
        // B4: 4 words of U8 → 2 words of B4
        assert_compact(
            BucketWidth::B4,
            &[
                pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]),
                pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]),
                pack_u8x8([3, 0, 0, 0, 0, 0, 0, 0]),
                pack_u8x8([0, 0, 0, 0, 0, 0, 0, 4]),
            ],
            &[
                pack_b4x16([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
                pack_b4x16([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]),
            ],
        );
    }

    #[test]
    fn test_swar_narrow_compact_odd_word_counts() {
        // B4: 1 word → in-place narrow
        let input = [pack_u8x8([3, 7, 0, 15, 0, 0, 5, 6])];
        let expected = narrow_word(input[0], BucketWidth::B4);
        assert_compact(BucketWidth::B4, &input, &[expected]);

        // B4: 3 words → 2 compacted words
        let w0 = pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]);
        let w1 = pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]);
        let w2 = pack_u8x8([3, 0, 0, 0, 0, 0, 0, 4]);
        let lo0 = narrow_word(w0, BucketWidth::B4);
        let hi0 = narrow_word(w1, BucketWidth::B4);
        let lo1 = narrow_word(w2, BucketWidth::B4);
        assert_compact(
            BucketWidth::B4,
            &[w0, w1, w2],
            &[lo0 | (hi0 << 32), lo1],
        );

        // U8: 3 words → 2 compacted words
        let expected1 = narrow_word(pack_u16x4([255, 0, 128, 1]), BucketWidth::U8);
        assert_compact(
            BucketWidth::U8,
            &[
                pack_u16x4([10, 20, 30, 40]),
                pack_u16x4([50, 60, 70, 80]),
                pack_u16x4([255, 0, 128, 1]),
            ],
            &[pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80]), expected1],
        );
    }

    // -----------------------------------------------------------------------
    // swar_has_overflow tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_has_overflow() {
        let cases: &[(&[u64], BucketWidth, bool)] = &[
            // B4: at-max → no overflow
            (
                &[pack_u8x8([15, 0, 8, 3, 1, 14, 7, 0])],
                BucketWidth::B4,
                false,
            ),
            // B4: one slot at 16 → overflow
            (
                &[pack_u8x8([15, 0, 16, 0, 0, 0, 0, 0])],
                BucketWidth::B4,
                true,
            ),
            // B4 boundary: all at max
            (
                &[pack_u8x8([15, 15, 15, 15, 15, 15, 15, 15])],
                BucketWidth::B4,
                false,
            ),
            // B4 boundary: one over
            (
                &[pack_u8x8([15, 15, 15, 16, 15, 15, 15, 15])],
                BucketWidth::B4,
                true,
            ),
            // U8: at-max
            (&[pack_u16x4([255, 0, 128, 1])], BucketWidth::U8, false),
            // U8: overflow
            (&[pack_u16x4([256, 0, 0, 0])], BucketWidth::U8, true),
            // U8 boundary: all at max
            (&[pack_u16x4([255, 255, 255, 255])], BucketWidth::U8, false),
            // U8 boundary: one over
            (&[pack_u16x4([255, 255, 256, 255])], BucketWidth::U8, true),
            // U16: at-max
            (&[pack_u32x2(65535, 0)], BucketWidth::U16, false),
            // U16: overflow
            (&[pack_u32x2(65536, 0)], BucketWidth::U16, true),
            // U16 boundary: all at max
            (&[pack_u32x2(65535, 65535)], BucketWidth::U16, false),
            // U32: at-max
            (&[u32::MAX as u64], BucketWidth::U32, false),
            // U32: overflow
            (&[u32::MAX as u64 + 1], BucketWidth::U32, true),
        ];
        for (i, &(data, width, expected)) in cases.iter().enumerate() {
            assert_eq!(
                swar_has_overflow(data, width),
                expected,
                "case {i}: width={width:?} expected={expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Full SWAR pipeline: swar_step → overflow check → narrow_compact
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_step_then_narrow_compact_roundtrip() {
        // B4: pair sums ≤ 15 → compact back to B4
        assert_swar_roundtrip(
            BucketWidth::B4,
            &[
                pack_b4x16([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                pack_b4x16([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 6, 0]),
            ],
            false,
            &[
                pack_u8x8([3, 7, 0, 0, 0, 0, 0, 0]),
                pack_u8x8([0, 0, 0, 0, 0, 0, 5, 6]),
            ],
            Some(&[pack_b4x16([
                3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6,
            ])]),
        );

        // U8: pair sums ≤ 255 → compact back to U8
        assert_swar_roundtrip(
            BucketWidth::U8,
            &[
                pack_u8x8([100, 50, 30, 20, 10, 5, 3, 1]),
                pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
            ],
            false,
            &[pack_u16x4([150, 50, 15, 4]), pack_u16x4([0, 0, 0, 0])],
            Some(&[pack_u8x8([150, 50, 15, 4, 0, 0, 0, 0])]),
        );

        // U16: pair sums ≤ 65535 → compact back to U16
        assert_swar_roundtrip(
            BucketWidth::U16,
            &[pack_u16x4([1000, 2000, 3000, 4000]), pack_u16x4([0; 4])],
            false,
            &[pack_u32x2(3000, 7000), pack_u32x2(0, 0)],
            Some(&[pack_u16x4([3000, 7000, 0, 0])]),
        );

        // U32: pair sum fits → compact back to U32
        assert_swar_roundtrip(
            BucketWidth::U32,
            &[pack_u32x2(100_000, 200_000), pack_u32x2(0, 0)],
            false,
            &[300_000u64, 0],
            Some(&[pack_u32x2(300_000, 0)]),
        );
    }

    #[test]
    fn test_swar_step_then_narrow_compact_overflow() {
        // B4 pair sums > 15 → overflow, keep widened result
        let mut data = [
            pack_b4x16([8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pack_b4x16([0; 16]),
        ];
        swar_step(&mut data, BucketWidth::B4);
        assert!(swar_has_overflow(&data, BucketWidth::B4));
        assert_eq!(data[0] & 0xFF, 17, "first byte sum should be 17");
    }

    // -----------------------------------------------------------------------
    // Adaptive merge (do_downscale) integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_do_downscale_width_behavior() {
        // Helper: insert ops into a B4 histogram at scale 0,
        // downscale(1), and verify the expected final width.
        let check = |ops: &[(f64, u64)], expected_width: BucketWidth, label: &str| {
            let mut h: Histogram<16> =
                Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
            for &(v, incr) in ops {
                h.update_by_incr(v, incr).unwrap();
            }
            assert_eq!(h.bucket_width(), BucketWidth::B4, "{label}: pre-check");
            assert_total_conserved(&mut h, 1);
            assert_eq!(h.bucket_width(), expected_width, "{label}");
        };

        check(&[(2.0, 5), (4.0, 7)], BucketWidth::B4, "small sums stay B4");
        check(&[(2.0, 10), (4.0, 10)], BucketWidth::U8, "overflow widens to U8");
        check(&[(2.0, 15), (4.0, 15)], BucketWidth::U8, "max B4 overflow widens to U8");
    }

    #[test]
    fn test_do_downscale_many_indices_preserves_width() {
        // Many small counts at spread-out indices → pair sums ≤ 2, stays B4.
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        for i in 0..8 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.bucket_width(), BucketWidth::B4);
    }

    // -----------------------------------------------------------------------
    // Reproducer for the sets 6 x 10 merge mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_sets_6_x_10_bucket_totals() {
        // Merged via merge_from must produce same bucket total as sequential inserts.
        let left: &[(f64, u64)] = &[(0.5, 1), (1.5, 1), (2.5, 1)];
        let right: &[(f64, u64)] = &[(5.0, 1), (10.0, 1), (15.0, 1), (20.0, 1)];
        merge_check::<8>(left, right, "sets_6_x_10");
    }

    // -----------------------------------------------------------------------
    // swar_step isolation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_step_single_word() {
        let cases: &[(BucketWidth, u64, u64, &str)] = &[
            // B4: 16 nibbles → 8 byte pair sums
            (
                BucketWidth::B4,
                pack_b4x16([1, 2, 3, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                pack_u8x8([3, 3, 0, 15, 0, 0, 0, 0]),
                "b4",
            ),
            // U8: 8 bytes → 4 short pair sums
            (
                BucketWidth::U8,
                pack_u8x8([100, 200, 50, 50, 0, 0, 0, 0]),
                pack_u16x4([300, 100, 0, 0]),
                "u8",
            ),
            // U16: 4 shorts → 2 int pair sums
            (
                BucketWidth::U16,
                pack_u16x4([1000, 2000, 3000, 4000]),
                pack_u32x2(3000, 7000),
                "u16",
            ),
            // U32: 2 ints → 1 u64 sum
            (
                BucketWidth::U32,
                pack_u32x2(100000, 200000),
                300000,
                "u32",
            ),
        ];
        for &(width, input, expected, label) in cases {
            let mut data = [input];
            swar_step(&mut data, width);
            assert_eq!(data[0], expected, "{label}: got {:#018x}", data[0]);
        }
    }

    #[test]
    fn test_swar_step_b4_max_pair_sum() {
        // Two 15s: sum = 30, which fits in U8 (max 255).
        let mut data = [pack_b4x16([
            15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])];
        swar_step(&mut data, BucketWidth::B4);
        assert_eq!(data[0] & 0xFF, 30);
    }

    // -----------------------------------------------------------------------
    // change_scale tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_change_scale() {
        let cases: &[(i32, i32, i32, i32, &str)] = &[
            (0, 4, 10, 0, "fits"),
            (0, 10, 10, 1, "exact boundary"),
            (0, 39, 10, 2, "double"),
            (-10, 10, 10, 2, "negative indices"),
            (5, 5, 10, 0, "zero span"),
        ];
        for &(low, high, cap, expected, label) in cases {
            assert_eq!(
                change_scale(HighLow { low, high }, cap),
                expected,
                "{label}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Adaptive merge (scalar fallback) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bucket_downscale_scalar_preserves_total_no_overflow() {
        // Two values at adjacent indices with small counts → scalar merge
        // should sum them without widening.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_literal_mode(false);
        h.update_by_incr(2.0, 3).unwrap(); // index 0
        h.update_by_incr(4.0, 5).unwrap(); // index 1

        let width_before = h.bucket_width();
        assert_total_conserved(&mut h, 1);
        // Small counts (3+5=8 ≤ 15) → should stay at B4.
        assert_eq!(h.bucket_width(), width_before);
    }

    #[test]
    fn test_bucket_downscale_scalar_preserves_total_with_overflow() {
        // Fill enough that pair sums exceed B4 max (15).
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 10).unwrap(); // index 0, count 10
        h.update_by_incr(4.0, 10).unwrap(); // index 1, count 10

        assert_total_conserved(&mut h, 1);
        // 10+10=20 > 15 → must widen to U8.
        assert_eq!(h.bucket_width(), BucketWidth::U8);
    }

    #[test]
    fn test_do_downscale_multi_step_preserves_total() {
        // Insert 4 values at separate indices, then downscale by 3.
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        for i in 0..4 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        let total_before = bucket_total(&mut h);
        assert_eq!(total_before, 4);
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        h.do_downscale(3).unwrap();

        let total_after = bucket_total(&mut h);
        assert_eq!(total_after, 4, "total changed after 3-step downscale");
        // All counts are 1, pair sums ≤ 2 → should stay at B4.
        assert_eq!(h.bucket_width(), BucketWidth::B4);
    }

    #[test]
    fn test_do_downscale_multi_step_through_alignment_boundary() {
        // Start with base aligned to 16, downscale 5+ times so base
        // goes from even to odd and back. Verify totals survive.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_literal_mode(false);
        for i in 0..8 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        let total_before = bucket_total(&mut h);
        assert_eq!(total_before, 8);

        // 5 steps: base starts at e.g. -16 >> 5 = -1 (odd), so the
        // 5th step must use scalar fallback.
        h.do_downscale(5).unwrap();

        let total_after = bucket_total(&mut h);
        assert_eq!(total_after, 8, "total changed after 5-step downscale");
    }

    #[test]
    fn test_do_downscale_odd_base_preserves_total() {
        // Downscale through odd-base steps using SWAR-shift.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_literal_mode(false);
        for i in 0..4 {
            h.update(2.0_f64.powi(i)).unwrap();
        }

        // At B1, base = -64. After 6 steps: base = -64 >> 6 = -1 (odd).
        // Step 7 uses the odd SWAR-shift merge.
        assert_total_conserved(&mut h, 7);
    }

    // -----------------------------------------------------------------------
    // bucket_widen at odd base
    // -----------------------------------------------------------------------

    #[test]
    fn test_bucket_widen_odd_base_uses_scalar() {
        // Construct a scenario where base is odd, then verify widen works.
        // Start at max scale so we have room to downscale.
        let mut h: Histogram<16> = Histogram::with_scale(8);
        h.update_by_incr(1.5, 5).unwrap();
        h.update_by_incr(1.6, 7).unwrap();

        let total_before = bucket_total(&mut h);

        // Downscale until base is odd (at most 15 steps to stay above MIN_SCALE).
        let mut tries = 0;
        while h.index_base & 1 == 0 && tries < 15 {
            h.do_downscale(1).unwrap();
            tries += 1;
        }

        if h.index_base & 1 != 0 {
            // Now force a widen at odd base.
            let width_before = h.bucket_width();
            if width_before != BucketWidth::U64 {
                h.bucket_widen(1).unwrap();
                assert!(
                    h.bucket_width() > width_before,
                    "width should increase: {:?} → {:?}",
                    width_before,
                    h.bucket_width()
                );

                let total_after = bucket_total(&mut h);
                assert_eq!(
                    total_before, total_after,
                    "bucket total changed on odd-base widen"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Speculative merge: width preservation across counter magnitudes
    // -----------------------------------------------------------------------

    #[test]
    fn test_speculative_merge_width_behavior() {
        // B4 sparse: many single-count buckets, pair sums ≤ 2 → stays B4
        {
            let mut h: Histogram<16> = Histogram::with_scale(0)
                .with_min_bucket_width(BucketWidth::B4)
                .with_literal_mode(false);
            for i in 0..16 {
                h.update(2.0_f64.powi(i)).unwrap();
            }
            assert_eq!(h.bucket_width(), BucketWidth::B4);
            assert_total_conserved(&mut h, 1);
            assert_eq!(h.bucket_width(), BucketWidth::B4, "b4 sparse stays");
        }

        // U8 dense: 200+200=400 > 255 → widens to U16
        {
            let mut h: Histogram<16> = Histogram::with_scale(0);
            h.update_by_incr(2.0, 200).unwrap();
            assert_eq!(h.bucket_width(), BucketWidth::U8);
            h.update_by_incr(4.0, 200).unwrap();
            assert_total_conserved(&mut h, 1);
            assert_eq!(h.bucket_width(), BucketWidth::U16, "u8 dense widens to u16");
        }

        // U8 sparse: 100+50=150 ≤ 255 → stays U8
        {
            let mut h: Histogram<16> = Histogram::with_scale(0);
            h.update_by_incr(2.0, 100).unwrap();
            assert_eq!(h.bucket_width(), BucketWidth::U8);
            h.update_by_incr(4.0, 50).unwrap();
            assert_total_conserved(&mut h, 1);
            assert_eq!(h.bucket_width(), BucketWidth::U8, "u8 sparse stays");
        }
    }

    // -----------------------------------------------------------------------
    // Sum conservation stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sum_conservation_through_full_widen_chain() {
        // Fill a histogram with enough count magnitude to force widening
        // at every level: B4(max 15) → U8(255) → U16(65535) → U32 → U64.
        // Adjacent pairs sum to 1000, forcing overflow at B4, U8.
        let mut h: Histogram<16> = Histogram::with_scale(8);
        h.update_by_incr(1.5, 500).unwrap();
        h.update_by_incr(1.6, 500).unwrap();
        // Start at U16 (500 > 255).
        assert_eq!(bucket_total(&mut h), 1000);

        // Add more to push into U32 territory.
        h.update_by_incr(1.7, 65000).unwrap();
        h.update_by_incr(1.8, 65000).unwrap();

        // Downscale up to 10 steps, verify total at each.
        assert_total_conserved(&mut h, 10);
    }

    #[test]
    fn test_sum_conservation_scalar_path() {
        // Force the scalar path and check totals at each step.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_literal_mode(false);
        for i in 0..10 {
            h.update(2.0_f64.powi(i)).unwrap();
        }

        // Downscale 8 times — should cross the odd-base boundary
        // multiple times, exercising scalar and SWAR paths alternately.
        assert_total_conserved(&mut h, 8);
    }

    #[test]
    fn test_sum_conservation_large_counts() {
        // High counts that force widening at every merge.
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 15).unwrap(); // fills B4 to max
        h.update_by_incr(4.0, 15).unwrap();
        h.update_by_incr(8.0, 15).unwrap();
        h.update_by_incr(16.0, 15).unwrap();
        assert_eq!(bucket_total(&mut h), 60);

        assert_total_conserved(&mut h, 6);
    }

    // -----------------------------------------------------------------------
    // Narrow function: slot ordering correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_narrow_u8_to_b4_preserves_slot_order() {
        // Verify that byte[i] maps to nibble[i], not a permuted position.
        for i in 0..8u8 {
            let mut bytes = [0u8; 8];
            bytes[i as usize] = (i + 1).min(15);
            let input = pack_u8x8(bytes);
            let result = narrow_word(input, BucketWidth::B4);

            // Extract nibble i from the result (low 32 bits).
            let nibble = (result >> (i as u64 * 4)) & 0xF;
            assert_eq!(
                nibble,
                (i + 1).min(15) as u64,
                "nibble {i}: expected {}, got {nibble}",
                (i + 1).min(15)
            );

            // All other nibbles should be zero.
            for j in 0..8u8 {
                if j != i {
                    let other = (result >> (j as u64 * 4)) & 0xF;
                    assert_eq!(
                        other, 0,
                        "nibble {j} should be 0 when only nibble {i} is set, got {other}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_narrow_u16_to_u8_preserves_slot_order() {
        for i in 0..4u16 {
            let mut shorts = [0u16; 4];
            shorts[i as usize] = (i + 1).min(255);
            let input = pack_u16x4(shorts);
            let result = narrow_word(input, BucketWidth::U8);

            let byte = (result >> (i as u64 * 8)) & 0xFF;
            assert_eq!(
                byte,
                (i + 1).min(255) as u64,
                "byte {i}: expected {}, got {byte}",
                (i + 1).min(255)
            );
        }
    }

    #[test]
    fn test_narrow_u32_to_u16_preserves_slot_order() {
        for i in 0..2u32 {
            let lo = if i == 0 { 42 } else { 0 };
            let hi = if i == 1 { 42 } else { 0 };
            let input = pack_u32x2(lo, hi);
            let result = narrow_word(input, BucketWidth::U16);

            let short = (result >> (i as u64 * 16)) & 0xFFFF;
            assert_eq!(short, 42, "short {i}: expected 42, got {short}");
        }
    }

    // -----------------------------------------------------------------------
    // swar_has_overflow: boundary values
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_has_overflow_multi_word() {
        // Overflow only in the last word — must still be detected.
        let data = [
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 16]),
        ];
        assert!(swar_has_overflow(&data, BucketWidth::B4));
    }

    // -----------------------------------------------------------------------
    // Reproducer: bucket total integrity through adaptive downscale
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_downscale_sequential_inserts_small_pool() {
        // Reproducer: insert 1.0..=8.0 into Histogram<8>.
        // At B1 with 6 bucket words (384 slots), the index span forces
        // repeated downscaling. Bucket totals must stay consistent.
        let mut h: Histogram<8> = Histogram::new();
        for i in 1..=8 {
            let v = i as f64;
            h.update(v).unwrap();
            let total = bucket_total(&mut h);
            assert_eq!(
                total,
                h.count(),
                "After inserting {v}: bucket total ({total}) != count ({})\n  \
                 scale={} width={:?}",
                h.count(),
                h.scale(),
                h.bucket_width()
            );
        }
    }

    #[test]
    fn test_adaptive_downscale_wide_span_small_pool() {
        // Wide value range in a small pool — forces multi-step downscale.
        let mut h: Histogram<8> = Histogram::new();
        let values = [0.001, 1.0, 1000.0, 0.5, 50.0, 0.01, 100.0, 10.0];
        for (vi, &v) in values.iter().enumerate() {
            h.update(v).unwrap();
            let total = bucket_total(&mut h);
            assert_eq!(
                total,
                h.count(),
                "After values[{vi}]={v}: bucket total ({total}) != count ({})\n  \
                 scale={} width={:?}",
                h.count(),
                h.scale(),
                h.bucket_width()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests (formerly in regression_stat_widen)
    // -----------------------------------------------------------------------

    /// Downscales `steps` times, asserting the bucket total is preserved
    /// at each step.
    fn assert_total_conserved<const N: usize>(h: &mut Histogram<N>, steps: i32) {
        let total = bucket_total(h);
        for step in 1..=steps {
            h.do_downscale(1).unwrap();
            let current = bucket_total(h);
            assert_eq!(
                current,
                total,
                "total changed at step {step}: {current} != {total}, \
                 width={:?}",
                h.bucket_width()
            );
        }
    }

    /// Helper: build two same-size histograms from ops, merge, and
    /// assert count and bucket-total invariants.
    fn build_histogram<const N: usize>(ops: &[(f64, u64)]) -> Histogram<N> {
        let mut h = Histogram::<N>::new();
        for &(v, incr) in ops {
            h.update_by_incr(v, incr).unwrap();
        }
        h
    }

    /// Helper: build a histogram from plain f64 values (each inserted once).
    fn build_from_values<const N: usize>(values: &[f64]) -> Histogram<N> {
        let mut h = Histogram::<N>::new();
        for &v in values {
            h.update(v).unwrap();
        }
        h
    }

    /// Helper: build a bucket-mode (non-literal) histogram from plain f64 values.
    fn build_bucket<const N: usize>(values: &[f64]) -> Histogram<N> {
        let mut h = Histogram::<N>::new().with_literal_mode(false);
        for &v in values {
            h.update(v).unwrap();
        }
        h
    }

    fn assert_merge_result<const N: usize>(
        h: &mut Histogram<N>,
        left: &[(f64, u64)],
        right: &[(f64, u64)],
        label: &str,
    ) {
        let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
        assert_eq!(h.count(), expected, "{label}: count mismatch");
        let bt = bucket_total(h);
        assert!(bt <= h.count(), "{label}: bt={bt} > count={}", h.count());
    }

    fn merge_check<const N: usize>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
        let (mut h1, h2) = (build_histogram::<N>(left), build_histogram::<N>(right));
        h1.merge_from(&h2).unwrap();
        assert_merge_result(&mut h1, left, right, label);
    }

    /// Helper: build two different-size histograms from ops, merge via
    /// `merge_from_other`, and assert count and bucket-total invariants.
    fn merge_check_cross<const N: usize, const M: usize>(
        left: &[(f64, u64)],
        right: &[(f64, u64)],
        label: &str,
    ) {
        let (mut h1, h2) = (build_histogram::<N>(left), build_histogram::<M>(right));
        h1.merge_from_other(&h2).unwrap();
        assert_merge_result(&mut h1, left, right, label);
    }

    #[test]
    fn test_merge_needs_downscale_in_raw() {
        let mut h1 = Histogram::<8>::new();
        h1.update(1.0).unwrap();

        let mut h2 = Histogram::<8>::new();
        h2.update(1e30).unwrap();
        h2.update(1e-30).unwrap();

        let h2_count = h2.count();
        let h2_sum = h2.sum();
        let h2_min = h2.min();
        let h2_max = h2.max();
        let h2_scale = h2.scale();
        let b2 = h2.positive();
        h1.merge_from_raw(
            &Stats {
                count: h2_count,
                sum: h2_sum,
                min: h2_min,
                max: h2_max,
            },
            &BucketDescriptor {
                scale: h2_scale,
                offset: b2.offset(),
                len: b2.len(),
            },
            &|i| b2.at(i),
        )
        .unwrap();
        assert_eq!(h1.count(), 3);
        assert_eq!(bucket_total(&mut h1), 3);
    }

    /// Regression: large weighted inserts of subnormal + normal value
    /// trigger bucket_widen during downscale, corrupting bucket totals
    /// when scalar_merge_step produced len > cap.
    #[test]
    fn test_weighted_subnormal_merge_bucket_total() {
        let v1 = f64::from_le_bytes([32, 0, 66, 0, 0, 98, 65, 3]); // ~5.44e-293, subnormal as f32
        let v2 = f64::from_le_bytes([0, 32, 0, 66, 0, 98, 65, 64]); // ~34.77

        let left_ops: Vec<(f64, u64)> =
            vec![(v1, 3), (v2, 1), (v1, 12), (v2, 4), (v1, 192), (v2, 64)];
        let right_ops: Vec<(f64, u64)> = vec![(v1, 3072), (v2, 1024)];

        merge_check::<8>(&left_ops, &right_ops, "same N=8");
        merge_check::<16>(&left_ops, &right_ops, "same N=16");
        merge_check_cross::<8, 16>(&left_ops, &right_ops, "cross 8←16");
        merge_check_cross::<16, 8>(&left_ops, &right_ops, "cross 16←8");
    }

    /// Regression: three values with a subnormal, split across merge,
    /// with echo-amplified increments.
    #[test]
    fn test_three_vals_with_subnormal_echo() {
        let v1 = f64::from_le_bytes([22, 22, 0, 237, 237, 59, 59, 59]); // ~2.25e-23
        let v2 = f64::from_le_bytes([59, 59, 1, 0, 59, 31, 0, 0]); // ~1.70e-310, subnormal as f32
        let v3 = f64::from_le_bytes([0, 59, 237, 237, 64, 0, 122, 64]); // ~416.0

        let left: Vec<(f64, u64)> = vec![(v1, 300), (v2, 5)];
        let right: Vec<(f64, u64)> = vec![
            (v3, 5),
            (v1, 1200),
            (v2, 20),
            (v3, 20),
            (v1, 19200),
            (v2, 320),
            (v3, 320),
        ];

        merge_check::<8>(&left, &right, "same 8");
        merge_check::<16>(&left, &right, "same 16");
        merge_check_cross::<8, 16>(&left, &right, "cross 8←16");
        merge_check_cross::<16, 8>(&left, &right, "cross 16←8");
    }

    #[test]
    fn test_merge_p64_bucket_total_exceeds_count() {
        let mut h0 = Histogram::<8>::new();
        let mut h1 = Histogram::<8>::new();

        h1.update_by_incr(2.8396262443943004e+238, 40).unwrap();
        h0.update_by_incr(2.635549485807631e-82, 1).unwrap();

        // Step 3: merge h0 into h1
        h1.merge_from(&h0).unwrap();
        assert_eq!(h1.count(), 41);

        // Step 4: merge h1 into h0
        if h0.merge_from(&h1).is_ok() {
            let count = h0.count();
            let bt = bucket_total(&mut h0);
            assert!(bt <= count, "bucket total ({bt}) exceeds count ({count})");
        }
    }

    #[test]
    fn test_merge_p32_bucket_len_after_merge_chain() {
        use crate::Mapping;

        let v0: f64 = 5.653943197254256e-308;
        let v1: f64 = 2.740490672504645e-61;

        let mut h0 = Histogram::<8>::new();
        let mut h1 = Histogram::<8>::new();

        h0.update_by_incr(v0, 1).unwrap();
        h1.update_by_incr(v1, 1).unwrap();

        // Merge chain: h0→h1, h0→h1, h1→h0
        h1.merge_from(&h0).unwrap();
        h1.merge_from(&h0).unwrap();
        h0.merge_from(&h1).unwrap();

        // Insert many zeros
        for _ in 0..150 {
            h0.update_by_incr(0.0, 1).unwrap();
        }

        // Verify bucket structure
        let scale = h0.scale();
        let mapping = Mapping::new(scale).unwrap();

        // All non-zero values should map to indices at the current scale
        let idx0 = mapping.map_to_index(v0);
        let idx1 = mapping.map_to_index(v1);
        let exp_min = idx0.min(idx1);
        let exp_max = idx0.max(idx1);
        let exp_len = (exp_max - exp_min + 1) as u32;

        let b = h0.positive();
        assert_eq!(
            b.offset(),
            exp_min,
            "offset mismatch: got {} expected {} (scale={})",
            b.offset(),
            exp_min,
            scale
        );
        assert_eq!(
            b.len(),
            exp_len,
            "len mismatch: got {} expected {} (scale={}, idx0={}, idx1={})",
            b.len(),
            exp_len,
            scale,
            idx0,
            idx1
        );

        // No trailing/leading zero buckets
        if !b.is_empty() {
            assert!(b.at(0) > 0, "leading zero bucket");
            assert!(b.at(b.len() - 1) > 0, "trailing zero bucket");
        }
    }

    // -----------------------------------------------------------------------
    // Literal mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_literal_mode_default() {
        let h: Histogram<8> = Histogram::new();
        assert!(h.is_literal());
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_literal_mode_stores_values() {
        let mut h: Histogram<8> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(2.0).unwrap();
        h.update(4.0).unwrap();
        assert!(h.is_literal());
        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 7.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 4.0);
    }

    #[test]
    fn test_literal_mode_capacity() {
        // Histogram<8>: all 8 words available for literals.
        let mut h: Histogram<8> = Histogram::new();
        for i in 0..8 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert!(h.is_literal(), "should still be literal with 8 values");
        assert_eq!(h.count(), 8);

        // 9th value should trigger promotion.
        h.update(256.0).unwrap();
        assert!(!h.is_literal(), "should promote on 9th value");
        assert_eq!(h.count(), 9);
    }

    #[test]
    fn test_literal_mode_opt_out() {
        let mut h: Histogram<8> = Histogram::new().with_literal_mode(false);
        assert!(!h.is_literal());
        h.update(1.0).unwrap();
        assert!(!h.is_literal());
    }

    #[test]
    fn test_literal_mode_clear_resets() {
        let mut h: Histogram<8> = Histogram::new();
        // Fill beyond literal capacity (8 slots) to trigger promotion.
        for i in 0..9 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert!(!h.is_literal());
        h.clear();
        assert!(h.is_literal());
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_literal_mode_zero_values() {
        // Zero values don't consume literal slots.
        let mut h: Histogram<8> = Histogram::new();
        h.update(0.0).unwrap();
        h.update(0.0).unwrap();
        h.update(0.0).unwrap();
        assert!(h.is_literal());
        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 0.0);
        // Bucket view should be empty (zeros are tracked in MMSC only).
        assert!(h.positive().is_empty());
    }

    #[test]
    fn test_literal_mode_identical_values() {
        let mut h: Histogram<8> = Histogram::new();
        for _ in 0..6 {
            h.update(42.0).unwrap();
        }
        assert!(h.is_literal());
        assert_eq!(h.count(), 6);
        assert_eq!(h.sum(), 252.0);
        assert_eq!(h.min(), 42.0);
        assert_eq!(h.max(), 42.0);
        // All map to the same bucket → len should be 1.
        assert_eq!(h.positive().len(), 1);
        assert_eq!(h.positive().at(0), 6);
    }

    #[test]
    fn test_literal_mode_bucket_view() {
        // Compare literal-mode virtual view to bucket-mode actual view.
        assert_literal_matches_bucket(&[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_literal_promotion_optimal_scale() {
        // Verify that promotion picks the optimal scale (matching what
        // bucket mode would choose given the same values).
        let values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
        assert_literal_matches_bucket(&values);
    }

    #[test]
    fn test_literal_update_by_incr() {
        let mut h: Histogram<8> = Histogram::new();
        // 3 copies of the same value.
        h.update_by_incr(5.0, 3).unwrap();
        assert!(h.is_literal());
        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 15.0);
    }

    #[test]
    fn test_literal_update_by_incr_overflow() {
        // Histogram<8>: 8 literal slots. incr=9 should promote.
        let mut h: Histogram<8> = Histogram::new();
        h.update_by_incr(3.25, 9).unwrap();
        assert!(!h.is_literal());
        assert_eq!(h.count(), 9);
    }

    #[test]
    fn test_literal_merge_literal_into_literal() {
        let mut a = build_from_values::<8>(&[1.0, 2.0]);
        let b = build_from_values::<8>(&[4.0, 8.0]);
        a.merge_from(&b).unwrap();
        assert_stats(&a, 4, 15.0, 1.0, 8.0);
    }

    #[test]
    fn test_literal_merge_literal_into_bucket() {
        let mut collector = build_bucket::<16>(&[1.0, 2.0]);
        let source = build_from_values::<16>(&[4.0, 8.0]);
        assert!(source.is_literal());
        collector.merge_from(&source).unwrap();
        assert_stats(&collector, 4, 15.0, 1.0, 8.0);
    }

    #[test]
    fn test_literal_merge_bucket_into_literal() {
        let mut collector = build_from_values::<16>(&[1.0]);
        assert!(collector.is_literal());
        let source = build_bucket::<16>(&[4.0, 8.0]);
        collector.merge_from(&source).unwrap();
        assert!(!collector.is_literal());
        assert_stats(&collector, 3, 13.0, 1.0, 8.0);
    }

    #[test]
    fn test_literal_merge_preserves_source() {
        let mut collector = build_bucket::<16>(&[1.0]);
        let source = build_from_values::<16>(&[4.0]);
        assert!(source.is_literal());
        collector.merge_from(&source).unwrap();
        assert!(source.is_literal());
        assert_stats(&source, 1, 4.0, 4.0, 4.0);
    }

    #[test]
    fn test_literal_merge_cross_size() {
        let mut collector = build_bucket::<16>(&[1.0]);
        let source = build_from_values::<8>(&[4.0, 8.0]);
        assert!(source.is_literal());
        collector.merge_from_other(&source).unwrap();
        assert_stats(&collector, 3, 13.0, 1.0, 8.0);
    }

    #[test]
    fn test_merge_literal_source_not_promoted() {
        // Verify that merging a literal source into a bucket destination
        // inserts literal values one by one without promoting the source.
        // The source must remain in literal mode after merge.

        // Same-size merge: literal source into bucket dest.
        let mut dest = build_bucket::<16>(&[1.0, 2.0, 3.0]);
        let source = build_from_values::<16>(&[10.0, 20.0, 30.0]);
        assert!(source.is_literal());
        dest.merge_from(&source).unwrap();
        assert!(source.is_literal(), "same-size merge must not promote source");
        assert_eq!(dest.count(), 6);

        // Cross-size merge: small literal source into large bucket dest.
        let mut big = build_bucket::<16>(&[1.0, 2.0, 3.0]);
        let small = build_from_values::<8>(&[100.0, 200.0]);
        assert!(small.is_literal());
        big.merge_from_other(&small).unwrap();
        assert!(small.is_literal(), "cross-size merge must not promote source");
        assert_eq!(big.count(), 5);

        // Wide-range literal values: ensure even with values spanning
        // many scales, the source stays literal and dest absorbs them
        // correctly through incremental insertion.
        let mut dest2 = build_bucket::<16>(&[1.0]);
        let source2 = build_from_values::<16>(&[1e-200, 1e200]);
        assert!(source2.is_literal());
        dest2.merge_from(&source2).unwrap();
        assert!(source2.is_literal(), "wide-range merge must not promote source");
        assert_eq!(dest2.count(), 3);
        assert_eq!(bucket_total(&mut dest2), 3, "all three non-zero values should be in buckets");
    }

    #[test]
    fn test_literal_equivalence_with_bucket_mode() {
        // Verify that a promoted literal histogram and a bucket-mode
        // histogram produce the same bucket totals for the same inputs.
        let values = [1.5, 2.7, 0.3, 100.0, 42.0, 7.7, 13.0, 55.5, 999.0];
        assert_literal_matches_bucket(&values);
    }

    #[test]
    fn test_literal_subnormal_values() {
        let subnormal = 5.0e-324_f64; // smallest subnormal
        let mut h: Histogram<8> = Histogram::new();
        h.update(subnormal).unwrap();
        h.update(subnormal).unwrap();
        assert!(h.is_literal());
        assert_eq!(h.count(), 2);
        // With f64 stats, subnormals are preserved exactly.
        // Check bucket view to verify data integrity.
        assert_eq!(h.positive().at(0), 2);
    }

    #[test]
    fn test_literal_empty_bucket_view() {
        let mut h: Histogram<8> = Histogram::new();
        assert!(h.is_literal());
        assert!(h.positive().is_empty());
        assert_eq!(h.positive().len(), 0);
    }

    #[test]
    fn test_literal_scale_matches_bucket() {
        assert_literal_matches_bucket(&[1.0, 1024.0]);
    }

    #[test]
    fn test_literal_debug_format() {
        let mut h: Histogram<8> = Histogram::new();
        h.update(1.0).unwrap();
        let debug = format!("{:?}", h);
        assert!(
            debug.contains("literal"),
            "Debug should mention literal mode"
        );
    }
}
