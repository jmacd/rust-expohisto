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

impl BucketWidth {
    /// Returns the bit width of one counter.
    #[inline]
    const fn bits(self) -> usize {
        self as usize
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
        match self {
            Self::B1 => Some(Self::B2),
            Self::B2 => Some(Self::B4),
            Self::B4 => Some(Self::U8),
            Self::U8 => Some(Self::U16),
            Self::U16 => Some(Self::U32),
            Self::U32 => Some(Self::U64),
            Self::U64 => None,
        }
    }

    /// Returns the width `steps` levels wider, or `None` if it would
    /// exceed U64.
    #[inline]
    const fn widen_by(self, steps: i32) -> Option<BucketWidth> {
        let mut w = self;
        let mut i = 0;
        while i < steps {
            match w.wider() {
                Some(next) => w = next,
                None => return None,
            }
            i += 1;
        }
        Some(w)
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    const fn counter_max(self) -> u64 {
        match self {
            Self::B1 => 1,
            Self::B2 => 3,
            Self::B4 => 15,
            Self::U8 => u8::MAX as u64,
            Self::U16 => u16::MAX as u64,
            Self::U32 => u32::MAX as u64,
            Self::U64 => u64::MAX,
        }
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
    #[inline]
    fn bucket_get(&self, slot: usize) -> u64 {
        let data = self.bucket_data();
        match self.bucket_width {
            BucketWidth::B1 => (data[slot / 64] >> (slot % 64)) & 1,
            BucketWidth::B2 => (data[slot / 32] >> ((slot % 32) * 2)) & 0x3,
            BucketWidth::B4 => (data[slot / 16] >> ((slot % 16) * 4)) & 0xF,
            BucketWidth::U8 => {
                let bytes: &[u8] = bytemuck::cast_slice(data);
                bytes[slot] as u64
            }
            BucketWidth::U16 => {
                let s: &[u16] = bytemuck::cast_slice(data);
                s[slot] as u64
            }
            BucketWidth::U32 => {
                let s: &[u32] = bytemuck::cast_slice(data);
                s[slot] as u64
            }
            BucketWidth::U64 => data[slot],
        }
    }

    /// Sets the value at a physical slot index.
    #[inline]
    fn bucket_set(&mut self, slot: usize, value: u64) {
        let width = self.bucket_width;
        let data = self.bucket_data_mut();
        match width {
            BucketWidth::B1 => {
                let word = &mut data[slot / 64];
                let shift = slot % 64;
                *word = (*word & !(1u64 << shift)) | ((value & 1) << shift);
            }
            BucketWidth::B2 => {
                let word = &mut data[slot / 32];
                let shift = (slot % 32) * 2;
                *word = (*word & !(0x3u64 << shift)) | ((value & 0x3) << shift);
            }
            BucketWidth::B4 => {
                let word = &mut data[slot / 16];
                let shift = (slot % 16) * 4;
                *word = (*word & !(0xFu64 << shift)) | ((value & 0xF) << shift);
            }
            BucketWidth::U8 => {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(data);
                bytes[slot] = value as u8;
            }
            BucketWidth::U16 => {
                let s: &mut [u16] = bytemuck::cast_slice_mut(data);
                s[slot] = value as u16;
            }
            BucketWidth::U32 => {
                let s: &mut [u32] = bytemuck::cast_slice_mut(data);
                s[slot] = value as u32;
            }
            BucketWidth::U64 => data[slot] = value,
        }
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
    /// If `index_base` is odd, shifts data up by one slot to restore
    /// even alignment before the SWAR step. When the top slot is
    /// occupied (live range fills capacity), falls back to a scalar
    /// gather-scatter for that one step.
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
            let width = self.bucket_width;
            if width == BucketWidth::U64 {
                return None; // would exceed U64
            }

            let shifted = self.index_base & 1 != 0;

            if shifted {
                if self.top_slot_occupied() {
                    done += self.scalar_merge_step(true)?;
                    continue;
                }
                swar_shift_up_one(self.bucket_data_mut(), width);
            }

            swar_step(self.bucket_data_mut(), width);
            self.bucket_width = width.wider().unwrap();
            self.shift_indices(1);

            if shifted {
                self.clamp_index_end();
            }

            done += 1;
        }

        Some(done)
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

/// Applies a single SWAR pairwise-sum step: for each word, adds the
/// upper half-slots to the lower half-slots, producing sums in the
/// next-wider format.
#[inline]
fn swar_masked_step(data: &mut [u64], shift: u32, mask: u64) {
    for w in data.iter_mut() {
        let x = *w;
        *w = ((x >> shift) & mask) + (x & mask);
    }
}

/// Single SWAR step: sum adjacent counters at the current width into
/// the next wider width, in place.
#[inline]
fn swar_step(data: &mut [u64], width: BucketWidth) {
    match width {
        BucketWidth::B1 => swar_masked_step(data, 1, 0x5555_5555_5555_5555),
        BucketWidth::B2 => swar_masked_step(data, 2, 0x3333_3333_3333_3333),
        BucketWidth::B4 => swar_masked_step(data, 4, 0x0F0F_0F0F_0F0F_0F0F),
        BucketWidth::U8 => swar_masked_step(data, 8, 0x00FF_00FF_00FF_00FF),
        BucketWidth::U16 => swar_masked_step(data, 16, 0x0000_FFFF_0000_FFFF),
        BucketWidth::U32 => {
            // 2 ints → 1 long (no mask needed, full 32-bit halves)
            for w in data.iter_mut() {
                let x = *w;
                *w = (x >> 32) + (x & 0xFFFF_FFFF);
            }
        }
        BucketWidth::U64 => unreachable!("cannot widen past U64"),
    }
}

/// Checks whether any word has bits set in the given overflow mask.
#[inline]
fn any_masked(data: &[u64], mask: u64) -> bool {
    data.iter().any(|&w| w & mask != 0)
}

/// Checks whether any widened pair-sum overflows the original width.
/// Called after `swar_step` has already written the wider sums.
#[inline]
fn swar_has_overflow(data: &[u64], original_width: BucketWidth) -> bool {
    match original_width {
        BucketWidth::B1 => any_masked(data, 0xAAAA_AAAA_AAAA_AAAA),
        BucketWidth::B2 => any_masked(data, 0xCCCC_CCCC_CCCC_CCCC),
        BucketWidth::B4 => any_masked(data, 0xF0F0_F0F0_F0F0_F0F0),
        BucketWidth::U8 => any_masked(data, 0xFF00_FF00_FF00_FF00),
        BucketWidth::U16 => any_masked(data, 0xFFFF_0000_FFFF_0000),
        BucketWidth::U32 => data.iter().any(|&w| w > u32::MAX as u64),
        BucketWidth::U64 => false,
    }
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

/// After a SWAR step that produced no overflow, narrow the widened sums
/// back to the original width and compact words 2:1.
///
/// The data is currently in `wider(original_width)` format with all values
/// fitting in `original_width`. This function bit-compresses each word
/// and packs pairs of words into one, freeing the upper half of the array.
#[inline]
fn swar_narrow_compact(data: &mut [u64], original_width: BucketWidth) {
    match original_width {
        BucketWidth::B1 => compact_with(data, narrow_b2_to_b1),
        BucketWidth::B2 => compact_with(data, narrow_b4_to_b2),
        BucketWidth::B4 => compact_with(data, narrow_u8_to_b4),
        BucketWidth::U8 => compact_with(data, narrow_u16_to_u8),
        BucketWidth::U16 => compact_with(data, narrow_u32_to_u16),
        BucketWidth::U32 => compact_with(data, |w| w & 0xFFFF_FFFF),
        BucketWidth::U64 => unreachable!("cannot narrow past U64"),
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

/// Compress 32 crumbs (each ≤ 1) into 32 bits in the low 32 bits.
#[inline]
fn narrow_b2_to_b1(w: u64) -> u64 {
    compact_lanes(
        w & 0x5555_5555_5555_5555,
        &[
            (1, 0x3333_3333_3333_3333),
            (2, 0x0F0F_0F0F_0F0F_0F0F),
            (4, 0x00FF_00FF_00FF_00FF),
            (8, 0x0000_FFFF_0000_FFFF),
            (16, 0x0000_0000_FFFF_FFFF),
        ],
    )
}

/// Compress 16 nibbles (each ≤ 3) into 16 crumbs in the low 32 bits.
#[inline]
fn narrow_b4_to_b2(w: u64) -> u64 {
    compact_lanes(
        w & 0x3333_3333_3333_3333,
        &[
            (2, 0x0F0F_0F0F_0F0F_0F0F),
            (4, 0x00FF_00FF_00FF_00FF),
            (8, 0x0000_FFFF_0000_FFFF),
            (16, 0x0000_0000_FFFF_FFFF),
        ],
    )
}

/// Compress 8 bytes (each ≤ 15) into 8 nibbles in the low 32 bits.
#[inline]
fn narrow_u8_to_b4(w: u64) -> u64 {
    compact_lanes(
        w & 0x0F0F_0F0F_0F0F_0F0F,
        &[
            (4, 0x00FF_00FF_00FF_00FF),
            (8, 0x0000_FFFF_0000_FFFF),
            (16, 0x0000_0000_FFFF_FFFF),
        ],
    )
}

/// Compress 4 shorts (each ≤ 255) into 4 bytes in the low 32 bits.
#[inline]
fn narrow_u16_to_u8(w: u64) -> u64 {
    compact_lanes(
        w & 0x00FF_00FF_00FF_00FF,
        &[(8, 0x0000_FFFF_0000_FFFF), (16, 0x0000_0000_FFFF_FFFF)],
    )
}

/// Compress 2 ints (each ≤ 65535) into 2 shorts in the low 32 bits.
#[inline]
fn narrow_u32_to_u16(w: u64) -> u64 {
    compact_lanes(w & 0x0000_FFFF_0000_FFFF, &[(16, 0x0000_0000_FFFF_FFFF)])
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
            stats: Stats {
                count: 0,
                sum: 0.0,
                min: 0.0,
                max: 0.0,
            },
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
        self.data.fill(0);
        self.stats = Stats {
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
        };
        self.bucket_width = self.min_bucket_width;
        self.literal = self.literal_enabled;
        self.index_start = 0;
        self.index_end = 0;
        self.index_base = 0;
        self.mapping = Mapping::new(self.limit_scale as i32).unwrap();
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
        self.bucket_width = self.min_bucket_width;
        self.mapping = Mapping::new(self.limit_scale as i32).map_err(|_| Overflow)?;
        self.data.fill(0);
        self.index_base = 0;
        self.index_start = 0;
        self.index_end = 0;

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
        loop {
            let index = self.mapping.map_to_index(value);
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
            let shifted = self.index_base & 1 != 0;
            if shifted {
                if self.top_slot_occupied() {
                    let steps = self.scalar_merge_step(false).ok_or(Overflow)?;
                    self.adjust_scale(steps)?;
                    remaining -= steps;
                    continue;
                }
                let width = self.bucket_width;
                swar_shift_up_one(self.bucket_data_mut(), width);
            }

            let width = self.bucket_width;
            swar_step(self.bucket_data_mut(), width);

            if swar_has_overflow(self.bucket_data(), width) {
                self.bucket_width = width.wider().unwrap();
            } else {
                swar_narrow_compact(self.bucket_data_mut(), width);
            }

            self.shift_indices(1);

            if shifted {
                self.clamp_index_end();
            }

            self.adjust_scale(1)?;
            remaining -= 1;
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
        if other.literal {
            return self.merge_literal_from(other);
        }

        // When self is empty, adopt other's bucket width to avoid
        // unnecessary widening steps during the merge.
        let saved_width = self.bucket_width;
        if !other.buckets_empty() && self.buckets_empty() {
            self.bucket_width = self.bucket_width.max(other.bucket_width);
        }
        let result = self.merge_from_histogram(other);
        if result.is_err() {
            self.bucket_width = saved_width;
        }
        result
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
                    loop {
                        let their_change = buckets.scale - h.mapping.scale();
                        let index = (buckets.offset + i as i32) >> their_change;
                        let result = h.increment_index_by(index, count);
                        if h.handle_incr_result(result)? {
                            break;
                        }
                    }
                }
            }

            h.trim_bucket_range();
            h.commit_stats(new_sum, new_count, stats.min, stats.max);
            Ok(())
        })
    }

    /// Merges a histogram of a different size into this one.
    pub fn merge_from_other<const M: usize>(
        &mut self,
        other: &Histogram<M>,
    ) -> Result<(), Overflow> {
        if other.literal {
            return self.merge_literal_from(other);
        }
        self.merge_from_histogram(other)
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

    fn derived_zero_count<const N: usize>(h: &mut Histogram<N>) -> u64 {
        let non_zero: u64 = {
            let buckets = h.positive();
            (0..buckets.len()).map(|i| buckets.at(i)).sum()
        };
        h.count() - non_zero
    }

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(1.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum(), 1.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
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
        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 7.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 4.0);
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
        assert_eq!(h1.count(), 4);
        assert_eq!(h1.sum(), 10.0);
        assert_eq!(h1.min(), 1.0);
        assert_eq!(h1.max(), 4.0);
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
    fn test_auto_widen_b4_to_u8() {
        let mut h: Histogram<16> = Histogram::new()
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update_by_incr(1.0, 15).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 16);
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
    fn test_auto_widen_u8_to_u16() {
        let mut h: Histogram<16> = Histogram::new()
            .with_literal_mode(false);
        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.update_by_incr(1.0, 239).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), 256);
    }

    #[test]
    fn test_auto_widen_u16_to_u32() {
        let mut h: Histogram<16> = Histogram::new();
        h.update_by_incr(1.0, 16).unwrap();
        h.update_by_incr(1.0, 239).unwrap();
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        h.update_by_incr(1.0, u16::MAX as u64 - 256).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        assert_eq!(h.count(), u16::MAX as u64 + 1);
    }

    #[test]
    fn test_auto_widen_u32_to_u64() {
        // This test needs count > u32::MAX to trigger U32→U64 widening.
        let mut h: Histogram<16> = Histogram::new();
        h.update_by_incr(1.0, 16).unwrap();
        h.update_by_incr(1.0, 239).unwrap();
        h.update(1.0).unwrap(); // U8→U16
        h.update_by_incr(1.0, u16::MAX as u64 - 256).unwrap();
        h.update(1.0).unwrap(); // U16→U32
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        h.update_by_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1))
            .unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        h.update(1.0).unwrap(); // U32→U64
        assert_eq!(h.bucket_width(), BucketWidth::U64);
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
        let mut h: Histogram<16> =
            Histogram::with_max_scale(3)
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
                let mut merged: Histogram<K> = Histogram::new();
                for &v in set_a {
                    merged.update(v).unwrap();
                }
                let mut other: Histogram<K> = Histogram::new();
                for (vi, &v) in set_b.iter().enumerate() {
                    if let Err(e) = other.update(v) {
                        panic!("other.update failed for size={K} sets {i} x {j} val[{vi}]={v}: {e}\n  set_b: {set_b:?}\n  other: {:?}", other);
                    }
                }
                if let Err(e) = merged.merge_from(&other) {
                    panic!("merge_from failed for size={K} sets {i} x {j}: {e}\n  set_a: {set_a:?}\n  set_b: {set_b:?}\n  merged: {:?}\n  other: {:?}", merged, other);
                }

                let mut single: Histogram<K> = Histogram::new();
                for &v in set_a {
                    single.update(v).unwrap();
                }
                for &v in set_b {
                    single.update(v).unwrap();
                }

                assert_eq!(
                    merged.count(),
                    single.count(),
                    "count mismatch for size={K} sets {i} x {j}"
                );
                // Order-of-operations rounding can differ between merged
                // and single paths. Use relative tolerance.
                let ms = merged.sum();
                let ss = single.sum();
                let sum_diff = (ms - ss).abs();
                let denom = ms.abs().max(ss.abs()).max(1e-30);
                assert!(
                    sum_diff / denom < 1e-5,
                    "sum mismatch for size={K} sets {i} x {j}: {} vs {}",
                    ms,
                    ss
                );
                assert_eq!(
                    derived_zero_count(&mut merged),
                    derived_zero_count(&mut single),
                    "zero_count mismatch for size={K} sets {i} x {j}"
                );

                let m_total: u64 = {
                    let mb = merged.positive();
                    (0..mb.len()).map(|k| mb.at(k)).sum()
                };
                let s_total: u64 = {
                    let sb = single.positive();
                    (0..sb.len()).map(|k| sb.at(k)).sum()
                };
                assert_eq!(
                    m_total, s_total,
                    "bucket total mismatch for size={K} sets {i} x {j}"
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

        let mut other: Histogram<8> = Histogram::new();
        for &v in set_b {
            other.update(v).unwrap();
            let bv = other.positive();
            let btotal: u64 = (0..bv.len()).map(|k| bv.at(k)).sum();
            let non_zero_count = other.count() - derived_zero_count(&mut other);
            assert_eq!(
                btotal, non_zero_count,
                "bucket total mismatch after inserting {}",
                v
            );
        }

        let set_a: &[f64] = &[1.0];
        let mut merged: Histogram<8> = Histogram::new();
        for &v in set_a {
            merged.update(v).unwrap();
        }
        merged.merge_from(&other).unwrap();

        let mut single: Histogram<8> = Histogram::new();
        for &v in set_a {
            single.update(v).unwrap();
        }
        for &v in set_b {
            single.update(v).unwrap();
        }

        let mb = merged.positive();
        let sb = single.positive();
        let m_total: u64 = (0..mb.len()).map(|k| mb.at(k)).sum();
        let s_total: u64 = (0..sb.len()).map(|k| sb.at(k)).sum();
        assert_eq!(
            m_total, s_total,
            "bucket total mismatch: merged={} single={}",
            m_total, s_total
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

    #[test]
    fn test_narrow_u8_to_b4_zeroes() {
        assert_eq!(narrow_u8_to_b4(0), 0);
    }

    #[test]
    fn test_narrow_u8_to_b4_all_ones() {
        // 8 bytes, each = 1: should produce 8 nibbles each = 1 in low 32 bits
        let input = pack_u8x8([1, 1, 1, 1, 1, 1, 1, 1]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u8_to_b4_max_values() {
        // 8 bytes, each = 15 (max B4): should produce 8 nibbles each = 15
        let input = pack_u8x8([15, 15, 15, 15, 15, 15, 15, 15]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u8_to_b4_ascending() {
        // 8 bytes: [0, 1, 2, 3, 4, 5, 6, 7] → 8 nibbles in order
        let input = pack_u8x8([0, 1, 2, 3, 4, 5, 6, 7]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u8_to_b4_scattered() {
        // Specific pattern to test bit-compress ordering
        let input = pack_u8x8([15, 0, 8, 0, 3, 0, 1, 0]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([15, 0, 8, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u16_to_u8_zeroes() {
        assert_eq!(narrow_u16_to_u8(0), 0);
    }

    #[test]
    fn test_narrow_u16_to_u8_ascending() {
        // 4 shorts: [10, 20, 30, 40] → 4 bytes in low 32 bits
        let input = pack_u16x4([10, 20, 30, 40]);
        let result = narrow_u16_to_u8(input);
        let expected = pack_u8x8([10, 20, 30, 40, 0, 0, 0, 0]) & 0xFFFF_FFFF;
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u16_to_u8_max_values() {
        // 4 shorts, each = 255 (max U8)
        let input = pack_u16x4([255, 255, 255, 255]);
        let result = narrow_u16_to_u8(input);
        let expected = pack_u8x8([255, 255, 255, 255, 0, 0, 0, 0]) & 0xFFFF_FFFF;
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u32_to_u16_zeroes() {
        assert_eq!(narrow_u32_to_u16(0), 0);
    }

    #[test]
    fn test_narrow_u32_to_u16_values() {
        // 2 ints: [1000, 2000] → 2 shorts in low 32 bits
        let input = pack_u32x2(1000, 2000);
        let result = narrow_u32_to_u16(input);
        let expected = (1000u64) | (2000u64 << 16);
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    #[test]
    fn test_narrow_u32_to_u16_max_values() {
        let input = pack_u32x2(65535, 65535);
        let result = narrow_u32_to_u16(input);
        let expected = (65535u64) | (65535u64 << 16);
        assert_eq!(
            result, expected,
            "got {:#018x}, expected {:#018x}",
            result, expected
        );
    }

    // -----------------------------------------------------------------------
    // swar_narrow_compact end-to-end tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_narrow_compact_b4_two_words() {
        // Two words of U8 data (result of swar_step on B4), each byte ≤ 15.
        // Word 0: slots [0..7] = [1,2,3,4,5,6,7,8]
        // Word 1: slots [8..15] = [9,10,11,12,13,14,15,0]
        // After compact: one word of 16 nibbles in original B4 format.
        let mut data = [
            pack_u8x8([1, 2, 3, 4, 5, 6, 7, 8]),
            pack_u8x8([9, 10, 11, 12, 13, 14, 15, 0]),
        ];
        swar_narrow_compact(&mut data, BucketWidth::B4);

        // Expect one word with 16 nibbles: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
        let expected = pack_b4x16([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]);
        assert_eq!(
            data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}",
            data[0], expected
        );
        assert_eq!(data[1], 0, "word 1 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_b4_four_words() {
        // Four words of U8 data → two words of B4 data.
        let mut data = [
            pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]),
            pack_u8x8([3, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 4]),
        ];
        swar_narrow_compact(&mut data, BucketWidth::B4);

        let expected0 = pack_b4x16([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
        let expected1 = pack_b4x16([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]);
        assert_eq!(
            data[0], expected0,
            "word 0: got {:#018x}, expected {:#018x}",
            data[0], expected0
        );
        assert_eq!(
            data[1], expected1,
            "word 1: got {:#018x}, expected {:#018x}",
            data[1], expected1
        );
        assert_eq!(data[2], 0, "word 2 should be zeroed");
        assert_eq!(data[3], 0, "word 3 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u8_two_words() {
        // Two words of U16 data (result of swar_step on U8), each ≤ 255.
        // Word 0: [10, 20, 30, 40]  Word 1: [50, 60, 70, 80]
        // After compact: one word of 8 bytes.
        let mut data = [pack_u16x4([10, 20, 30, 40]), pack_u16x4([50, 60, 70, 80])];
        swar_narrow_compact(&mut data, BucketWidth::U8);

        let expected = pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(
            data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}",
            data[0], expected
        );
        assert_eq!(data[1], 0, "word 1 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u16_two_words() {
        // Two words of U32 data, each ≤ 65535.
        let mut data = [pack_u32x2(100, 200), pack_u32x2(300, 400)];
        swar_narrow_compact(&mut data, BucketWidth::U16);

        let expected = pack_u16x4([100, 200, 300, 400]);
        assert_eq!(
            data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}",
            data[0], expected
        );
        assert_eq!(data[1], 0, "word 1 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u32_two_words() {
        // Two words of U64 data, each ≤ u32::MAX.
        let mut data = [1000u64, 2000u64];
        swar_narrow_compact(&mut data, BucketWidth::U32);

        let expected = pack_u32x2(1000, 2000);
        assert_eq!(
            data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}",
            data[0], expected
        );
        assert_eq!(data[1], 0, "word 1 should be zeroed");
    }

    // -----------------------------------------------------------------------
    // swar_has_overflow tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_has_overflow_b4_no_overflow() {
        // All byte sums ≤ 15 → no overflow
        let data = [pack_u8x8([15, 0, 8, 3, 1, 14, 7, 0])];
        assert!(!swar_has_overflow(&data, BucketWidth::B4));
    }

    #[test]
    fn test_swar_has_overflow_b4_overflow() {
        // One byte = 16 → overflow
        let data = [pack_u8x8([15, 0, 16, 0, 0, 0, 0, 0])];
        assert!(swar_has_overflow(&data, BucketWidth::B4));
    }

    #[test]
    fn test_swar_has_overflow_u8_no_overflow() {
        let data = [pack_u16x4([255, 0, 128, 1])];
        assert!(!swar_has_overflow(&data, BucketWidth::U8));
    }

    #[test]
    fn test_swar_has_overflow_u8_overflow() {
        let data = [pack_u16x4([256, 0, 0, 0])];
        assert!(swar_has_overflow(&data, BucketWidth::U8));
    }

    #[test]
    fn test_swar_has_overflow_u16_no_overflow() {
        let data = [pack_u32x2(65535, 0)];
        assert!(!swar_has_overflow(&data, BucketWidth::U16));
    }

    #[test]
    fn test_swar_has_overflow_u16_overflow() {
        let data = [pack_u32x2(65536, 0)];
        assert!(swar_has_overflow(&data, BucketWidth::U16));
    }

    #[test]
    fn test_swar_has_overflow_u32_no_overflow() {
        let data = [u32::MAX as u64];
        assert!(!swar_has_overflow(&data, BucketWidth::U32));
    }

    #[test]
    fn test_swar_has_overflow_u32_overflow() {
        let data = [u32::MAX as u64 + 1];
        assert!(swar_has_overflow(&data, BucketWidth::U32));
    }

    // -----------------------------------------------------------------------
    // Full SWAR pipeline: swar_step → overflow check → narrow_compact
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_step_then_narrow_compact_b4_roundtrip() {
        // Start with 2 words of B4 data (32 nibbles).
        // Pair sums: nibbles [0]+[1], [2]+[3], ... → 16 values in U8
        // If all sums ≤ 15, narrow back to 1 word of B4 (16 nibbles).
        //
        // Word 0: nibbles [1,2,3,4, 0,0,0,0, 0,0,0,0, 0,0,0,0]
        // Word 1: nibbles [0,0,0,0, 0,0,0,0, 0,0,0,0, 5,0,6,0]
        let mut data = [
            pack_b4x16([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pack_b4x16([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 6, 0]),
        ];

        // Step 1: SWAR pairwise sum (B4 → U8)
        swar_step(&mut data, BucketWidth::B4);

        // After swar_step, each word has 8 bytes = pairwise sums:
        // Word 0: bytes [1+2, 3+4, 0+0, 0+0, 0+0, 0+0, 0+0, 0+0] = [3,7,0,0,0,0,0,0]
        // Word 1: bytes [0+0, 0+0, 0+0, 0+0, 0+0, 0+0, 5+0, 6+0] = [0,0,0,0,0,0,5,6]
        assert_eq!(
            data[0],
            pack_u8x8([3, 7, 0, 0, 0, 0, 0, 0]),
            "swar_step word 0: got {:#018x}",
            data[0]
        );
        assert_eq!(
            data[1],
            pack_u8x8([0, 0, 0, 0, 0, 0, 5, 6]),
            "swar_step word 1: got {:#018x}",
            data[1]
        );

        // Step 2: No overflow (all ≤ 15)
        assert!(!swar_has_overflow(&data, BucketWidth::B4));

        // Step 3: Narrow + compact (U8 → B4, 2 words → 1 word)
        swar_narrow_compact(&mut data, BucketWidth::B4);

        // Result: 1 word of 16 nibbles = [3,7,0,0,0,0,0,0, 0,0,0,0,0,0,5,6]
        let expected = pack_b4x16([3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6]);
        assert_eq!(
            data[0], expected,
            "compact result: got {:#018x}, expected {:#018x}",
            data[0], expected
        );
        assert_eq!(data[1], 0, "freed word should be zero");
    }

    #[test]
    fn test_swar_step_then_narrow_compact_b4_overflow() {
        // Pair sums > 15 → overflow detected, keep widened result.
        // Word 0: nibbles [8, 9, ...] → sum = 17, overflows B4.
        let mut data = [
            pack_b4x16([8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pack_b4x16([0; 16]),
        ];

        swar_step(&mut data, BucketWidth::B4);
        // Word 0 byte 0 = 8+9 = 17 > 15
        assert!(swar_has_overflow(&data, BucketWidth::B4));
        // Do NOT compact — the widened U8 data is the final result.
        assert_eq!(data[0] & 0xFF, 17, "first byte sum should be 17");
    }

    #[test]
    fn test_swar_step_then_narrow_compact_u8_roundtrip() {
        // 2 words of U8 data (16 bytes). Pair sums each ≤ 255.
        let mut data = [
            pack_u8x8([100, 50, 30, 20, 10, 5, 3, 1]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
        ];

        swar_step(&mut data, BucketWidth::U8);
        // Pair sums: [150, 50, 15, 4, 0, 0, 0, 0] as U16
        assert!(!swar_has_overflow(&data, BucketWidth::U8));

        swar_narrow_compact(&mut data, BucketWidth::U8);
        // 1 word of 8 bytes: [150, 50, 15, 4, 0, 0, 0, 0]
        assert_eq!(
            data[0],
            pack_u8x8([150, 50, 15, 4, 0, 0, 0, 0]),
            "compact result: got {:#018x}",
            data[0]
        );
        assert_eq!(data[1], 0);
    }

    #[test]
    fn test_swar_step_then_narrow_compact_u16_roundtrip() {
        let mut data = [pack_u16x4([1000, 2000, 3000, 4000]), pack_u16x4([0; 4])];

        swar_step(&mut data, BucketWidth::U16);
        // Pair sums: [3000, 7000, 0, 0] as U32
        assert!(!swar_has_overflow(&data, BucketWidth::U16));

        swar_narrow_compact(&mut data, BucketWidth::U16);
        // 1 word: [3000, 7000, 0, 0] as U16
        assert_eq!(
            data[0],
            pack_u16x4([3000, 7000, 0, 0]),
            "compact result: got {:#018x}",
            data[0]
        );
        assert_eq!(data[1], 0);
    }

    #[test]
    fn test_swar_step_then_narrow_compact_u32_roundtrip() {
        let mut data = [pack_u32x2(100_000, 200_000), pack_u32x2(0, 0)];

        swar_step(&mut data, BucketWidth::U32);
        // Sum = 300_000 as U64
        assert!(!swar_has_overflow(&data, BucketWidth::U32));

        swar_narrow_compact(&mut data, BucketWidth::U32);
        assert_eq!(
            data[0],
            pack_u32x2(300_000, 0),
            "compact result: got {:#018x}",
            data[0]
        );
        assert_eq!(data[1], 0);
    }

    // -----------------------------------------------------------------------
    // Adaptive merge (do_downscale) integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_do_downscale_merge_stays_b4() {
        // Insert small values at adjacent indices so pair sums ≤ 15.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        // At scale 0, map_to_index(2.0) = 0, map_to_index(4.0) = 1.
        // These are a pair (even, odd) that will sum via SWAR.
        h.update_by_incr(2.0, 5).unwrap(); // index 0, count 5
        h.update_by_incr(4.0, 7).unwrap(); // index 1, count 7
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        // Merge: 5+7=12 ≤ 15, should stay at B4.
        h.do_downscale(1).unwrap();
        assert_eq!(
            h.bucket_width(),
            BucketWidth::B4,
            "width should be preserved when pair sums fit"
        );
    }

    #[test]
    fn test_do_downscale_merge_widens_b4() {
        // Insert values that sum to > 15 at B4.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        h.update_by_incr(2.0, 10).unwrap(); // index 0, count 10
        h.update_by_incr(4.0, 10).unwrap(); // index 1, count 10
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        // Merge: 10+10=20 > 15, should widen to U8.
        h.do_downscale(1).unwrap();
        assert_eq!(
            h.bucket_width(),
            BucketWidth::U8,
            "width should widen when pair sums overflow"
        );
    }

    #[test]
    fn test_do_downscale_preserves_width_when_possible() {
        // Fill histogram with small counts at many indices.
        // Downscale should merge pairs without widening.
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        // Insert 1 at each of several indices (all count=1, sums ≤ 2).
        for i in 0..8 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        let width_before = h.bucket_width();
        h.do_downscale(1).unwrap();
        assert_eq!(
            h.bucket_width(),
            width_before,
            "width should be preserved when pair sums fit"
        );
    }

    #[test]
    fn test_do_downscale_widens_on_overflow() {
        // Fill histogram with counts that will overflow on merge.
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        h.update_by_incr(2.0, 15).unwrap(); // index 0, count 15
        h.update_by_incr(4.0, 15).unwrap(); // index 1, count 15
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        h.do_downscale(1).unwrap();
        // 15+15=30 > 15, must widen to U8.
        assert_eq!(h.bucket_width(), BucketWidth::U8);
    }

    // -----------------------------------------------------------------------
    // Reproducer for the sets 6 x 10 merge mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_sets_6_x_10_bucket_totals() {
        // set_a = [0.5, 1.5, 2.5], set_b = [5.0, 10.0, 15.0, 20.0]
        // Merged via merge_from must equal sequential inserts.
        let set_a = [0.5, 1.5, 2.5];
        let set_b = [5.0, 10.0, 15.0, 20.0];

        let mut merged: Histogram<8> = Histogram::new();
        for &v in &set_a {
            merged.update(v).unwrap();
        }
        let mut other: Histogram<8> = Histogram::new();
        for &v in &set_b {
            other.update(v).unwrap();
        }
        merged.merge_from(&other).unwrap();

        let mut single: Histogram<8> = Histogram::new();
        for &v in &set_a {
            single.update(v).unwrap();
        }
        for &v in &set_b {
            single.update(v).unwrap();
        }

        let mb = merged.positive();
        let sb = single.positive();
        let m_buckets: Vec<u64> = (0..mb.len()).map(|k| mb.at(k)).collect();
        let s_buckets: Vec<u64> = (0..sb.len()).map(|k| sb.at(k)).collect();
        let m_total: u64 = m_buckets.iter().sum();
        let s_total: u64 = s_buckets.iter().sum();

        assert_eq!(m_total, s_total,
            "bucket total mismatch: merged={m_buckets:?} (sum={m_total}) vs single={s_buckets:?} (sum={s_total})");
    }

    // -----------------------------------------------------------------------
    // swar_step isolation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_step_b4_single_word() {
        // 16 nibbles: [1,2, 3,0, 0,0, 15,0, 0,0, 0,0, 0,0, 0,0]
        // Pair sums → 8 bytes: [3, 3, 0, 15, 0, 0, 0, 0]
        let mut data = [pack_b4x16([
            1, 2, 3, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])];
        swar_step(&mut data, BucketWidth::B4);
        assert_eq!(
            data[0],
            pack_u8x8([3, 3, 0, 15, 0, 0, 0, 0]),
            "got {:#018x}",
            data[0]
        );
    }

    #[test]
    fn test_swar_step_u8_single_word() {
        // 8 bytes: [100, 200, 50, 50, 0, 0, 0, 0]
        // Pair sums → 4 shorts: [300, 100, 0, 0]
        let mut data = [pack_u8x8([100, 200, 50, 50, 0, 0, 0, 0])];
        swar_step(&mut data, BucketWidth::U8);
        assert_eq!(
            data[0],
            pack_u16x4([300, 100, 0, 0]),
            "got {:#018x}",
            data[0]
        );
    }

    #[test]
    fn test_swar_step_u16_single_word() {
        // 4 shorts: [1000, 2000, 3000, 4000]
        // Pair sums → 2 ints: [3000, 7000]
        let mut data = [pack_u16x4([1000, 2000, 3000, 4000])];
        swar_step(&mut data, BucketWidth::U16);
        assert_eq!(data[0], pack_u32x2(3000, 7000), "got {:#018x}", data[0]);
    }

    #[test]
    fn test_swar_step_u32_single_word() {
        // 2 ints: [100000, 200000]
        // Sum → 1 u64: 300000
        let mut data = [pack_u32x2(100000, 200000)];
        swar_step(&mut data, BucketWidth::U32);
        assert_eq!(data[0], 300000);
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
    // swar_narrow_compact with odd word counts
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_narrow_compact_b4_one_word() {
        // Single word of U8 data → half a word of B4 data (low 32 bits).
        // The 8 bytes each ≤ 15 become 8 nibbles in the low 32 bits.
        let mut data = [pack_u8x8([3, 7, 0, 15, 0, 0, 5, 6])];
        swar_narrow_compact(&mut data, BucketWidth::B4);
        // With 1 word, step_by(2) produces i=0 only. lo=narrow(data[0]),
        // hi=0 (i+1 >= n). data[0] = lo | (0 << 32) = lo.
        let expected = narrow_u8_to_b4(pack_u8x8([3, 7, 0, 15, 0, 0, 5, 6]));
        assert_eq!(
            data[0], expected,
            "got {:#018x}, expected {:#018x}",
            data[0], expected
        );
    }

    #[test]
    fn test_swar_narrow_compact_b4_three_words() {
        // 3 words of U8 → 2 compacted + zero the freed word.
        // Word 0: [1,0,0,0,0,0,0,0]
        // Word 1: [0,0,0,0,0,0,0,2]
        // Word 2: [3,0,0,0,0,0,0,4]
        let mut data = [
            pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]),
            pack_u8x8([3, 0, 0, 0, 0, 0, 0, 4]),
        ];
        swar_narrow_compact(&mut data, BucketWidth::B4);
        // step_by(2): i=0 → data[0]=narrow(w0)|narrow(w1)<<32
        //             i=2 → data[1]=narrow(w2)|0<<32
        let lo0 = narrow_u8_to_b4(pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]));
        let hi0 = narrow_u8_to_b4(pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]));
        let lo1 = narrow_u8_to_b4(pack_u8x8([3, 0, 0, 0, 0, 0, 0, 4]));
        assert_eq!(data[0], lo0 | (hi0 << 32), "word 0: got {:#018x}", data[0]);
        assert_eq!(data[1], lo1, "word 1: got {:#018x}", data[1]);
        assert_eq!(data[2], 0, "word 2 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u8_three_words() {
        let mut data = [
            pack_u16x4([10, 20, 30, 40]),
            pack_u16x4([50, 60, 70, 80]),
            pack_u16x4([255, 0, 128, 1]),
        ];
        swar_narrow_compact(&mut data, BucketWidth::U8);
        assert_eq!(
            data[0],
            pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80]),
            "word 0: got {:#018x}",
            data[0]
        );
        let expected1 = narrow_u16_to_u8(pack_u16x4([255, 0, 128, 1]));
        assert_eq!(data[1], expected1, "word 1: got {:#018x}", data[1]);
        assert_eq!(data[2], 0, "word 2 should be zeroed");
    }

    // -----------------------------------------------------------------------
    // change_scale tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_change_scale_fits() {
        // span 5 in capacity 10 → no change needed
        assert_eq!(change_scale(HighLow { low: 0, high: 4 }, 10), 0);
    }

    #[test]
    fn test_change_scale_exact_boundary() {
        // high - low == size → one shift needed (condition is >=)
        assert_eq!(change_scale(HighLow { low: 0, high: 10 }, 10), 1);
    }

    #[test]
    fn test_change_scale_double() {
        // span = 40, cap = 10 → need 2+ shifts
        // 40 >> 1 = 20 (still > 10), 20 >> 1 = 10 → 2 shifts
        assert_eq!(change_scale(HighLow { low: 0, high: 39 }, 10), 2);
    }

    #[test]
    fn test_change_scale_negative_indices() {
        // low = -10, high = 10 → span = 20
        // cap = 10: (10 - (-10)) = 20 ≥ 10 → shift
        // after: 5 - (-5) = 10 ≥ 10 → shift again
        // after: 2 - (-3) = 5 < 10 → done. 2 shifts.
        assert_eq!(change_scale(HighLow { low: -10, high: 10 }, 10), 2);
    }

    #[test]
    fn test_change_scale_zero_span() {
        assert_eq!(change_scale(HighLow { low: 5, high: 5 }, 10), 0);
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

        let total_before: u64 = h.positive().iter().sum();
        let width_before = h.bucket_width();

        // Drive to odd base by doing SWAR merges until base is odd.
        // Start: base = (first_index) & !15. Let's just forcibly test
        // the scalar path by calling do_downscale multiple times.
        // At scale 0 with index_base = -16 (from index -1 & !15...
        // actually index 0 at scale 0 maps to -1).
        // Let's verify and use scalar directly if possible.

        // Instead, test via do_downscale which will route to SWAR or scalar.
        h.do_downscale(1).unwrap();

        let total_after: u64 = h.positive().iter().sum();
        assert_eq!(
            total_before, total_after,
            "bucket total changed: {total_before} → {total_after}"
        );
        // Small counts (3+5=8 ≤ 15) → should stay at B4.
        assert_eq!(h.bucket_width(), width_before);
    }

    #[test]
    fn test_bucket_downscale_scalar_preserves_total_with_overflow() {
        // Fill enough that pair sums exceed B4 max (15).
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 10).unwrap(); // index 0, count 10
        h.update_by_incr(4.0, 10).unwrap(); // index 1, count 10

        let total_before: u64 = h.positive().iter().sum();

        h.do_downscale(1).unwrap();

        let total_after: u64 = h.positive().iter().sum();
        assert_eq!(total_before, total_after);
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
        let total_before: u64 = h.positive().iter().sum();
        assert_eq!(total_before, 4);
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        h.do_downscale(3).unwrap();

        let total_after: u64 = h.positive().iter().sum();
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
        let total_before: u64 = h.positive().iter().sum();
        assert_eq!(total_before, 8);

        // 5 steps: base starts at e.g. -16 >> 5 = -1 (odd), so the
        // 5th step must use scalar fallback.
        h.do_downscale(5).unwrap();

        let total_after: u64 = h.positive().iter().sum();
        assert_eq!(total_after, 8, "total changed after 5-step downscale");
    }

    #[test]
    fn test_do_downscale_odd_base_preserves_total() {
        // Downscale through odd-base steps using SWAR-shift.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_literal_mode(false);
        for i in 0..4 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        let total_before: u64 = h.positive().iter().sum();

        // At B1, base = -64. After 6 steps: base = -64 >> 6 = -1 (odd).
        // Step 7 uses the odd SWAR-shift merge.
        for _ in 0..7 {
            h.do_downscale(1).unwrap();
        }

        let total_after: u64 = h.positive().iter().sum();
        assert_eq!(
            total_before, total_after,
            "total changed after 7-step downscale through odd base"
        );
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

        let total_before: u64 = h.positive().iter().sum();

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

                let total_after: u64 = h.positive().iter().sum();
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
    fn test_speculative_merge_b4_sparse_stays_b4() {
        // Many single-count buckets. All pair sums ≤ 2, so B4 stays.
        let mut h: Histogram<16> = Histogram::with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        for i in 0..16 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        h.do_downscale(1).unwrap();
        // Each pair sums to at most 2. B4 max is 15. Stays.
        assert_eq!(h.bucket_width(), BucketWidth::B4);
    }

    #[test]
    fn test_speculative_merge_b4_dense_widens_to_u8() {
        // Same bucket hit 10 times, its neighbor 10 times → sum 20 > 15.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        h.update_by_incr(2.0, 10).unwrap();
        h.update_by_incr(4.0, 10).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        h.do_downscale(1).unwrap();
        assert_eq!(
            h.bucket_width(),
            BucketWidth::U8,
            "10+10=20 > 15, must widen"
        );
    }

    #[test]
    fn test_speculative_merge_u8_dense_widens_to_u16() {
        // At U8, max = 255. Two adjacent buckets each with count 200 → 400 > 255.
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 200).unwrap();
        // This first update at B4 will overflow (200 > 15) → widen to U8.
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        h.update_by_incr(4.0, 200).unwrap();
        // Now at U8, do a merge: 200+200=400 > 255 → must widen to U16.
        let total_before: u64 = h.positive().iter().sum();
        h.do_downscale(1).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        let total_after: u64 = h.positive().iter().sum();
        assert_eq!(total_before, total_after);
    }

    #[test]
    fn test_speculative_merge_u8_sparse_stays_u8() {
        // At U8 with counts far below 255, pair sums should fit.
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 100).unwrap();
        // 100 > 15 → widens B4→U8 on counter overflow path.
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        h.update_by_incr(4.0, 50).unwrap();
        // 100+50=150 ≤ 255 → should stay at U8.
        let total_before: u64 = h.positive().iter().sum();
        h.do_downscale(1).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        let total_after: u64 = h.positive().iter().sum();
        assert_eq!(total_before, total_after);
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
        let total: u64 = h.positive().iter().sum();
        assert_eq!(total, 1000);

        // Add more to push into U32 territory.
        h.update_by_incr(1.7, 65000).unwrap();
        h.update_by_incr(1.8, 65000).unwrap();
        let total: u64 = h.positive().iter().sum();

        // Downscale up to 10 steps, verify total at each.
        for step in 1..=10 {
            h.do_downscale(1).unwrap();
            let current: u64 = h.positive().iter().sum();
            assert_eq!(
                current,
                total,
                "total changed at step {step} (width={:?}): {current} != {total}",
                h.bucket_width()
            );
            if h.bucket_width() == BucketWidth::U64 {
                break;
            }
        }
    }

    #[test]
    fn test_sum_conservation_scalar_path() {
        // Force the scalar path and check totals at each step.
        let mut h: Histogram<16> = Histogram::with_scale(0).with_literal_mode(false);
        for i in 0..10 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        let total = 10u64;

        // Downscale 8 times — should cross the odd-base boundary
        // multiple times, exercising scalar and SWAR paths alternately.
        for step in 1..=8 {
            h.do_downscale(1).unwrap();
            let current: u64 = h.positive().iter().sum();
            assert_eq!(
                current,
                total,
                "total changed at step {step}: {current} != {total}, \
                 width={:?} base={}",
                h.bucket_width(),
                h.index_base
            );
        }
    }

    #[test]
    fn test_sum_conservation_large_counts() {
        // High counts that force widening at every merge.
        let mut h: Histogram<16> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 15).unwrap(); // fills B4 to max
        h.update_by_incr(4.0, 15).unwrap();
        h.update_by_incr(8.0, 15).unwrap();
        h.update_by_incr(16.0, 15).unwrap();
        let total: u64 = h.positive().iter().sum();
        assert_eq!(total, 60);

        for step in 1..=6 {
            h.do_downscale(1).unwrap();
            let current: u64 = h.positive().iter().sum();
            assert_eq!(
                current,
                total,
                "total changed at step {step}: {current} != {total}, \
                 width={:?}",
                h.bucket_width()
            );
        }
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
            let result = narrow_u8_to_b4(input);

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
            let result = narrow_u16_to_u8(input);

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
            let result = narrow_u32_to_u16(input);

            let short = (result >> (i as u64 * 16)) & 0xFFFF;
            assert_eq!(short, 42, "short {i}: expected 42, got {short}");
        }
    }

    // -----------------------------------------------------------------------
    // swar_has_overflow: boundary values
    // -----------------------------------------------------------------------

    #[test]
    fn test_swar_has_overflow_b4_boundary_15() {
        // Exactly 15 → not overflow
        let data = [pack_u8x8([15, 15, 15, 15, 15, 15, 15, 15])];
        assert!(!swar_has_overflow(&data, BucketWidth::B4));
    }

    #[test]
    fn test_swar_has_overflow_b4_boundary_16() {
        // Exactly 16 in one slot → overflow
        let data = [pack_u8x8([15, 15, 15, 16, 15, 15, 15, 15])];
        assert!(swar_has_overflow(&data, BucketWidth::B4));
    }

    #[test]
    fn test_swar_has_overflow_u8_boundary_255() {
        let data = [pack_u16x4([255, 255, 255, 255])];
        assert!(!swar_has_overflow(&data, BucketWidth::U8));
    }

    #[test]
    fn test_swar_has_overflow_u8_boundary_256() {
        let data = [pack_u16x4([255, 255, 256, 255])];
        assert!(swar_has_overflow(&data, BucketWidth::U8));
    }

    #[test]
    fn test_swar_has_overflow_u16_boundary_65535() {
        let data = [pack_u32x2(65535, 65535)];
        assert!(!swar_has_overflow(&data, BucketWidth::U16));
    }

    #[test]
    fn test_swar_has_overflow_multi_word_only_last() {
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
            let (total, b_width, b_offset, b_len, b_cap) = {
                let b = h.positive();
                let t: u64 = (0..b.len()).map(|k| b.at(k)).sum();
                (t, b.width(), b.offset(), b.len(), b.capacity())
            };
            assert_eq!(
                total,
                h.count(),
                "After inserting {v}: bucket total ({total}) != count ({})\n  \
                 scale={} width={:?} offset={} len={} cap={}",
                h.count(),
                h.scale(),
                b_width,
                b_offset,
                b_len,
                b_cap
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
            let (total, b_width) = {
                let b = h.positive();
                let t: u64 = (0..b.len()).map(|k| b.at(k)).sum();
                (t, b.width())
            };
            assert_eq!(
                total,
                h.count(),
                "After values[{vi}]={v}: bucket total ({total}) != count ({})\n  \
                 scale={} width={:?}",
                h.count(),
                h.scale(),
                b_width
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests (formerly in regression_stat_widen)
    // -----------------------------------------------------------------------

    /// Helper: count total across all positive buckets.
    fn bucket_total<const N: usize>(h: &mut Histogram<N>) -> u64 {
        let b = h.positive();
        (0..b.len()).map(|i| b.at(i)).sum()
    }

    /// Helper: build two same-size histograms from ops, merge, and
    /// assert count and bucket-total invariants.
    fn merge_check<const N: usize>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
        let mut h1 = Histogram::<N>::new();
        for &(v, incr) in left {
            h1.update_by_incr(v, incr).unwrap();
        }
        let mut h2 = Histogram::<N>::new();
        for &(v, incr) in right {
            h2.update_by_incr(v, incr).unwrap();
        }
        h1.merge_from(&h2).unwrap();
        let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
        assert_eq!(h1.count(), expected, "{label}: count mismatch");
        let bt = bucket_total(&mut h1);
        assert!(bt <= h1.count(), "{label}: bt={bt} > count={}", h1.count());
    }

    /// Helper: build two different-size histograms from ops, merge via
    /// `merge_from_other`, and assert count and bucket-total invariants.
    fn merge_check_cross<const N: usize, const M: usize>(
        left: &[(f64, u64)],
        right: &[(f64, u64)],
        label: &str,
    ) {
        let mut h1 = Histogram::<N>::new();
        for &(v, incr) in left {
            h1.update_by_incr(v, incr).unwrap();
        }
        let mut h2 = Histogram::<M>::new();
        for &(v, incr) in right {
            h2.update_by_incr(v, incr).unwrap();
        }
        h1.merge_from_other(&h2).unwrap();
        let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
        assert_eq!(h1.count(), expected, "{label}: count mismatch");
        let bt = bucket_total(&mut h1);
        assert!(bt <= h1.count(), "{label}: bt={bt} > count={}", h1.count());
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
            let b = h0.positive();
            let bt: u64 = (0..b.len()).map(|i| b.at(i)).sum();
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
        let mut lit: Histogram<8> = Histogram::new();
        let mut bkt: Histogram<8> = Histogram::new().with_literal_mode(false);

        let values = [1.0, 2.0, 4.0];
        for &v in &values {
            lit.update(v).unwrap();
            bkt.update(v).unwrap();
        }

        assert!(lit.is_literal());
        assert!(!bkt.is_literal());

        // Both should report the same scale, offset, len, and counts.
        assert_eq!(lit.scale(), bkt.scale());
        assert_eq!(lit.positive().offset(), bkt.positive().offset());
        assert_eq!(lit.positive().len(), bkt.positive().len());
        for i in 0..lit.positive().len() {
            assert_eq!(
                lit.positive().at(i),
                bkt.positive().at(i),
                "bucket[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_literal_promotion_optimal_scale() {
        // Verify that promotion picks the optimal scale (matching what
        // bucket mode would choose given the same values).
        let mut h: Histogram<8> = Histogram::new();
        // Insert values that span a wide range to force downscaling.
        let values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
        for &v in &values {
            h.update(v).unwrap();
        }
        assert!(h.is_literal());

        // Now promote by inserting one more.
        h.update(256.0).unwrap();
        assert!(!h.is_literal());

        // Compare to bucket-mode histogram with same values.
        let mut bkt: Histogram<8> = Histogram::new().with_literal_mode(false);
        for &v in &values {
            bkt.update(v).unwrap();
        }
        bkt.update(256.0).unwrap();

        // The total counts should match.
        let lit_total: u64 = h.positive().iter().sum();
        let bkt_total: u64 = bkt.positive().iter().sum();
        assert_eq!(lit_total, bkt_total);
        assert_eq!(h.count(), bkt.count());
        assert_eq!(h.sum(), bkt.sum());
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
        let mut a: Histogram<8> = Histogram::new();
        a.update(1.0).unwrap();
        a.update(2.0).unwrap();

        let mut b: Histogram<8> = Histogram::new();
        b.update(4.0).unwrap();
        b.update(8.0).unwrap();

        a.merge_from(&b).unwrap();

        // After merge, `a` may or may not still be literal depending on
        // total count vs capacity. With 4 values and capacity 6, it could
        // remain literal if merge-from-literal inserts one by one. But our
        // implementation promotes `a` first, so `a` is now bucket mode.
        assert_eq!(a.count(), 4);
        assert_eq!(a.sum(), 15.0);
        assert_eq!(a.min(), 1.0);
        assert_eq!(a.max(), 8.0);
    }

    #[test]
    fn test_literal_merge_literal_into_bucket() {
        let mut collector: Histogram<16> = Histogram::new().with_literal_mode(false);
        collector.update(1.0).unwrap();
        collector.update(2.0).unwrap();

        let mut source: Histogram<16> = Histogram::new();
        source.update(4.0).unwrap();
        source.update(8.0).unwrap();
        assert!(source.is_literal());

        collector.merge_from(&source).unwrap();
        assert_eq!(collector.count(), 4);
        assert_eq!(collector.sum(), 15.0);
    }

    #[test]
    fn test_literal_merge_bucket_into_literal() {
        let mut collector: Histogram<16> = Histogram::new();
        collector.update(1.0).unwrap();
        assert!(collector.is_literal());

        let mut source: Histogram<16> = Histogram::new().with_literal_mode(false);
        source.update(4.0).unwrap();
        source.update(8.0).unwrap();

        collector.merge_from(&source).unwrap();
        // collector should have promoted to accept bucket data.
        assert!(!collector.is_literal());
        assert_eq!(collector.count(), 3);
        assert_eq!(collector.sum(), 13.0);
    }

    #[test]
    fn test_literal_merge_preserves_source() {
        // Source stays literal after merge (it's &self).
        let mut collector: Histogram<16> = Histogram::new().with_literal_mode(false);
        collector.update(1.0).unwrap();

        let mut source: Histogram<16> = Histogram::new();
        source.update(4.0).unwrap();
        assert!(source.is_literal());

        collector.merge_from(&source).unwrap();

        // Source should still be literal and unchanged.
        assert!(source.is_literal());
        assert_eq!(source.count(), 1);
        assert_eq!(source.sum(), 4.0);
    }

    #[test]
    fn test_literal_merge_cross_size() {
        // Merge a literal Histogram<8> into a larger Histogram<16>.
        let mut collector: Histogram<16> = Histogram::new().with_literal_mode(false);
        collector.update(1.0).unwrap();

        let mut source: Histogram<8> = Histogram::new();
        source.update(4.0).unwrap();
        source.update(8.0).unwrap();
        assert!(source.is_literal());

        collector.merge_from_other(&source).unwrap();
        assert_eq!(collector.count(), 3);
        assert_eq!(collector.sum(), 13.0);
    }

    #[test]
    fn test_merge_literal_source_not_promoted() {
        // Verify that merging a literal source into a bucket destination
        // inserts literal values one by one without promoting the source.
        // The source must remain in literal mode after merge.

        // Same-size merge: literal source into bucket dest.
        let mut dest: Histogram<16> = Histogram::new().with_literal_mode(false);
        for v in [1.0, 2.0, 3.0] {
            dest.update(v).unwrap();
        }
        assert!(!dest.is_literal());

        let mut source: Histogram<16> = Histogram::new();
        for v in [10.0, 20.0, 30.0] {
            source.update(v).unwrap();
        }
        assert!(source.is_literal());

        dest.merge_from(&source).unwrap();
        assert!(
            source.is_literal(),
            "same-size merge must not promote source"
        );
        assert_eq!(dest.count(), 6);

        // Cross-size merge: small literal source into large bucket dest.
        let mut big: Histogram<16> = Histogram::new().with_literal_mode(false);
        for v in [1.0, 2.0, 3.0] {
            big.update(v).unwrap();
        }

        let mut small: Histogram<8> = Histogram::new();
        for v in [100.0, 200.0] {
            small.update(v).unwrap();
        }
        assert!(small.is_literal());

        big.merge_from_other(&small).unwrap();
        assert!(
            small.is_literal(),
            "cross-size merge must not promote source"
        );
        assert_eq!(big.count(), 5);

        // Wide-range literal values: ensure even with values spanning
        // many scales, the source stays literal and dest absorbs them
        // correctly through incremental insertion.
        let mut dest2: Histogram<16> = Histogram::new().with_literal_mode(false);
        dest2.update(1.0).unwrap();

        let mut source2: Histogram<16> = Histogram::new();
        source2.update(1e-200).unwrap();
        source2.update(1e200).unwrap();
        assert!(source2.is_literal());

        dest2.merge_from(&source2).unwrap();
        assert!(
            source2.is_literal(),
            "wide-range merge must not promote source"
        );
        assert_eq!(dest2.count(), 3);

        // Verify bucket totals match count minus zeros.
        let total: u64 = dest2.positive().iter().sum();
        assert_eq!(total, 3, "all three non-zero values should be in buckets");
    }

    #[test]
    fn test_literal_equivalence_with_bucket_mode() {
        // Verify that a promoted literal histogram and a bucket-mode
        // histogram produce the same bucket totals for the same inputs.
        let values = [1.5, 2.7, 0.3, 100.0, 42.0, 7.7, 13.0, 55.5];

        let mut lit: Histogram<8> = Histogram::new();
        let mut bkt: Histogram<8> = Histogram::new().with_literal_mode(false);

        for &v in &values {
            lit.update(v).unwrap();
            bkt.update(v).unwrap();
        }

        // All 8 values fit in literal mode.
        assert!(lit.is_literal());

        // Force promotion by inserting one more.
        lit.update(999.0).unwrap();
        bkt.update(999.0).unwrap();
        assert!(!lit.is_literal());

        let lit_total: u64 = lit.positive().iter().sum();
        let bkt_total: u64 = bkt.positive().iter().sum();
        assert_eq!(lit_total, bkt_total, "bucket totals should match");
        assert_eq!(lit.count(), bkt.count());
        assert_eq!(lit.sum(), bkt.sum());
        assert_eq!(lit.min(), bkt.min());
        assert_eq!(lit.max(), bkt.max());
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
        let values = [1.0, 1024.0]; // wide range
        let mut lit: Histogram<8> = Histogram::new();
        let mut bkt: Histogram<8> = Histogram::new().with_literal_mode(false);

        for &v in &values {
            lit.update(v).unwrap();
            bkt.update(v).unwrap();
        }
        assert!(lit.is_literal());

        // Effective scales should match.
        assert_eq!(lit.scale(), bkt.scale());
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
