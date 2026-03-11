// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram with a unified flat memory layout.
//!
//! `Histogram<N, P>` stores everything in fixed struct fields plus a `[u64; N]`
//! data pool. The pool is split between MMSC fields (min/max/sum/count) at the
//! front — whose width is controlled by the precision tier `P` — and bucket
//! counters in the remainder.
//!
//! Bucket counters start at 1-bit and widen through the chain
//! 1→2→4→8→16→32→64 bits via combined downscale+widen when a counter
//! saturates. Sub-byte transitions use parallel bit-sum (SWAR) — the
//! popcount algorithm's building blocks.

use core::fmt;
use core::marker::PhantomData;

use crate::mapping::{Mapping, max_scale};
use crate::precision::Precision;

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
// MMSC accessor helpers — read/write stats from the data pool
// ---------------------------------------------------------------------------

/// S32 layout (2 words):
///   word 0: [sum:f32 (lo32)] [count:u32 (hi32)]
///   word 1: [min:f32 (lo32)] [max:f32 (hi32)]
#[inline]
fn read_lo32(word: u64) -> u32 {
    word as u32
}

#[inline]
fn read_hi32(word: u64) -> u32 {
    (word >> 32) as u32
}

#[inline]
fn write_lo32(word: &mut u64, val: u32) {
    *word = (*word & 0xFFFF_FFFF_0000_0000) | val as u64;
}

#[inline]
fn write_hi32(word: &mut u64, val: u32) {
    *word = (*word & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
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
/// `N` is the number of `u64` words in the data pool. The pool holds
/// MMSC fields (min/max/sum/count) at the front — whose width is
/// controlled by the precision tier `P` — and bucket counter data in
/// the remainder.
///
/// `P` selects the precision tier: [`P32`](crate::P32) uses 2 words
/// for MMSC (f32/u32), [`P64`](crate::P64) uses 4 words (f64/u64).
///
/// Bucket counters start at 1-bit and auto-widen in place
/// (1→2→4 bits → u8 → u16 → u32 → u64) via combined downscale+widen
/// when a counter saturates.
///
/// At minimum, `N` should be 8 (64 bytes of pool), giving 6 bucket words
/// at P32 (96 B4 buckets) down to 4 bucket words at P64 (4 U64 buckets).
pub struct Histogram<const N: usize, P: Precision> {
    // -- Fixed metadata (never relocates) --
    mapping: Mapping,
    max_scale: i8,
    min_bucket_width: BucketWidth,
    bucket_width: BucketWidth,
    index_base: i32,
    index_start: i32,
    index_end: i32,

    // -- Data pool: MMSC at front, buckets after --
    data: [u64; N],

    _precision: PhantomData<P>,
}

impl<const N: usize, P: Precision> Clone for Histogram<N, P> {
    fn clone(&self) -> Self {
        Self {
            mapping: self.mapping,
            max_scale: self.max_scale,
            min_bucket_width: self.min_bucket_width,
            bucket_width: self.bucket_width,
            index_base: self.index_base,
            index_start: self.index_start,
            index_end: self.index_end,
            data: self.data,
            _precision: PhantomData,
        }
    }
}

impl<const N: usize, P: Precision> fmt::Debug for Histogram<N, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Histogram")
            .field("scale", &self.scale())
            .field("bucket_width", &self.bucket_width)
            .field("count", &self.count())
            .field("sum", &self.sum())
            .field("min", &self.min())
            .field("max", &self.max())
            .field("zero_count", &self.count().saturating_sub(self.non_zero_count()))
            .field("bucket_len", &self.bucket_len())
            .finish()
    }
}

impl<const N: usize, P: Precision> Default for Histogram<N, P> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MMSC read/write methods
// ---------------------------------------------------------------------------

impl<const N: usize, P: Precision> Histogram<N, P> {
    // -- S32 readers --

    #[inline]
    fn sum_s32(&self) -> f32 {
        f32::from_bits(read_lo32(self.data[0]))
    }

    #[inline]
    fn count_s32(&self) -> u32 {
        read_hi32(self.data[0])
    }

    #[inline]
    fn min_s32(&self) -> f32 {
        f32::from_bits(read_lo32(self.data[1]))
    }

    #[inline]
    fn max_s32(&self) -> f32 {
        f32::from_bits(read_hi32(self.data[1]))
    }

    // -- S32 writers --

    #[inline]
    fn set_sum_s32(&mut self, v: f32) {
        write_lo32(&mut self.data[0], v.to_bits());
    }

    #[inline]
    fn set_count_s32(&mut self, v: u32) {
        write_hi32(&mut self.data[0], v);
    }

    #[inline]
    fn set_min_s32(&mut self, v: f32) {
        write_lo32(&mut self.data[1], v.to_bits());
    }

    #[inline]
    fn set_max_s32(&mut self, v: f32) {
        write_hi32(&mut self.data[1], v.to_bits());
    }

    // -- S64 readers --

    #[inline]
    fn sum_s64(&self) -> f64 {
        f64::from_bits(self.data[0])
    }

    #[inline]
    fn count_s64(&self) -> u64 {
        self.data[1]
    }

    #[inline]
    fn min_s64(&self) -> f64 {
        f64::from_bits(self.data[2])
    }

    #[inline]
    fn max_s64(&self) -> f64 {
        f64::from_bits(self.data[3])
    }

    // -- S64 writers --

    #[inline]
    fn set_sum_s64(&mut self, v: f64) {
        self.data[0] = v.to_bits();
    }

    #[inline]
    fn set_count_s64(&mut self, v: u64) {
        self.data[1] = v;
    }

    #[inline]
    fn set_min_s64(&mut self, v: f64) {
        self.data[2] = v.to_bits();
    }

    #[inline]
    fn set_max_s64(&mut self, v: f64) {
        self.data[3] = v.to_bits();
    }

    // -- Width-dispatched public readers --

    /// Returns the sum of all recorded values as `f64`.
    #[inline]
    pub fn sum(&self) -> f64 {
        if P::STAT_WORDS == 2 {
            self.sum_s32() as f64
        } else {
            self.sum_s64()
        }
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        if P::STAT_WORDS == 2 {
            self.count_s32() as u64
        } else {
            self.count_s64()
        }
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        if P::STAT_WORDS == 2 {
            self.min_s32() as f64
        } else {
            self.min_s64()
        }
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
        if P::STAT_WORDS == 2 {
            self.max_s32() as f64
        } else {
            self.max_s64()
        }
    }

    // -- Width-dispatched internal writers --

    #[inline]
    fn set_sum(&mut self, v: f64) {
        if P::STAT_WORDS == 2 {
            self.set_sum_s32(v as f32);
        } else {
            self.set_sum_s64(v);
        }
    }

    #[inline]
    fn set_count(&mut self, v: u64) {
        if P::STAT_WORDS == 2 {
            self.set_count_s32(v as u32);
        } else {
            self.set_count_s64(v);
        }
    }

    #[inline]
    fn set_min(&mut self, v: f64) {
        if P::STAT_WORDS == 2 {
            self.set_min_s32(v as f32);
        } else {
            self.set_min_s64(v);
        }
    }

    #[inline]
    fn set_max(&mut self, v: f64) {
        if P::STAT_WORDS == 2 {
            self.set_max_s32(v as f32);
        } else {
            self.set_max_s64(v);
        }
    }

    #[inline]
    fn add_sum(&mut self, v: f64) {
        self.set_sum(self.sum() + v);
    }

    /// Checked increment of count by `incr`. Returns `None` on overflow.
    #[inline]
    fn checked_add_count(&self, incr: u64) -> Option<u64> {
        if P::STAT_WORDS == 2 {
            let c = self.count_s32();
            let i = u32::try_from(incr).ok()?;
            c.checked_add(i).map(|v| v as u64)
        } else {
            self.count_s64().checked_add(incr)
        }
    }

    /// Returns the total count stored across all positive buckets.
    fn non_zero_count(&self) -> u64 {
        let mut total = 0u64;
        for pos in 0..self.bucket_len() {
            total = total.saturating_add(self.bucket_at(pos));
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Bucket data access — operates on the bucket slice of the data pool
// ---------------------------------------------------------------------------

impl<const N: usize, P: Precision> Histogram<N, P> {
    /// Returns the start index of bucket data within the data pool.
    #[inline]
    fn bucket_data_start(&self) -> usize {
        P::STAT_WORDS
    }

    /// Returns the number of u64 words available for bucket data.
    #[inline]
    fn bucket_word_count(&self) -> usize {
        N - self.bucket_data_start()
    }

    /// Returns the bucket data as a slice.
    #[inline]
    fn bucket_data(&self) -> &[u64] {
        &self.data[self.bucket_data_start()..]
    }

    /// Returns the bucket data as a mutable slice.
    #[inline]
    fn bucket_data_mut(&mut self) -> &mut [u64] {
        let start = self.bucket_data_start();
        &mut self.data[start..]
    }

    /// Number of logical buckets available at the current width.
    #[inline]
    pub fn bucket_capacity(&self) -> usize {
        self.bucket_width.capacity(self.bucket_word_count())
    }

    /// Returns the number of buckets in use.
    #[inline]
    pub fn bucket_len(&self) -> u32 {
        if self.is_effectively_empty() {
            0
        } else {
            (self.index_end - self.index_start + 1) as u32
        }
    }

    /// Returns true if no buckets have been used.
    #[inline]
    pub fn buckets_empty(&self) -> bool {
        self.bucket_len() == 0
    }

    /// Checks if the bucket range represents no data.
    #[inline]
    fn is_effectively_empty(&self) -> bool {
        if self.index_end != self.index_start {
            return false;
        }
        let cap = self.bucket_capacity() as i32;
        let slot = (self.index_start - self.index_base).rem_euclid(cap);
        self.bucket_get(slot as usize) == 0
    }

    /// Trims leading and trailing zero buckets from the index range.
    fn trim_bucket_range(&mut self) {
        let cap = self.bucket_capacity() as i32;
        while self.index_end > self.index_start {
            let slot = (self.index_end - self.index_base).rem_euclid(cap) as usize;
            if self.bucket_get(slot) != 0 {
                break;
            }
            self.index_end -= 1;
        }
        while self.index_start < self.index_end {
            let slot = (self.index_start - self.index_base).rem_euclid(cap) as usize;
            if self.bucket_get(slot) != 0 {
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
    #[inline]
    pub fn bucket_at(&self, pos: u32) -> u64 {
        let index = self.index_start + pos as i32;
        let cap = self.bucket_capacity() as i32;
        let slot = (index - self.index_base).rem_euclid(cap) as usize;
        self.bucket_get(slot)
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
    #[inline]
    pub fn bucket_offset(&self) -> i32 {
        self.index_start
    }
}

// ---------------------------------------------------------------------------
// BucketView — read-only public view of bucket data
// ---------------------------------------------------------------------------

/// Read-only view of bucket data in a histogram.
#[derive(Debug)]
pub struct BucketView<'a, const N: usize, P: Precision> {
    hist: &'a Histogram<N, P>,
}

impl<const N: usize, P: Precision> BucketView<'_, N, P> {
    /// Returns the offset (smallest index).
    #[inline]
    pub fn offset(&self) -> i32 {
        self.hist.index_start
    }

    /// Number of logical buckets in use.
    #[inline]
    pub fn len(&self) -> u32 {
        self.hist.bucket_len()
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
    #[inline]
    pub fn at(&self, pos: u32) -> u64 {
        self.hist.bucket_at(pos)
    }

    /// Returns an iterator over bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, N, P> {
        BucketsIter {
            hist: self.hist,
            pos: 0,
            len: self.len(),
        }
    }
}

impl<'a, const N: usize, P: Precision> IntoIterator for &'a BucketView<'a, N, P> {
    type Item = u64;
    type IntoIter = BucketsIter<'a, N, P>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over bucket counts.
#[derive(Debug)]
pub struct BucketsIter<'a, const N: usize, P: Precision> {
    hist: &'a Histogram<N, P>,
    pos: u32,
    len: u32,
}

impl<const N: usize, P: Precision> Iterator for BucketsIter<'_, N, P> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let count = self.hist.bucket_at(self.pos);
        self.pos += 1;
        Some(count)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<const N: usize, P: Precision> ExactSizeIterator for BucketsIter<'_, N, P> {}

// ---------------------------------------------------------------------------
// Bucket operations — widen (SWAR) and downscale
// ---------------------------------------------------------------------------

impl<const N: usize, P: Precision> Histogram<N, P> {
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
            self.index_start >>= steps;
            self.index_end >>= steps;
            self.index_base >>= steps;
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

            // Ensure even base for SWAR.
            if shifted {
                let bits = width.bits();
                let n = self.bucket_word_count();
                if n > 0 && self.bucket_data()[n - 1] >> (64 - bits) != 0 {
                    // Top slot occupied — scalar force-widen.
                    done += self.scalar_merge_step(true)?;
                    continue;
                }
                swar_shift_up_one(self.bucket_data_mut(), width);
            }

            swar_step(self.bucket_data_mut(), width);
            self.bucket_width = width.wider().unwrap();
            self.index_start >>= 1;
            self.index_end >>= 1;
            self.index_base >>= 1;

            // swar_shift_up_one pushes the top slot off the array
            // (the assert guarantees it was zero). When the widened
            // capacity exactly equals the new logical length, the
            // off-by-one from right-shifting an odd base causes
            // len > cap. Clamp index_end to keep the range valid.
            if shifted {
                let cap = self.bucket_capacity() as i32;
                let max_end = self.index_start + cap - 1;
                if self.index_end > max_end {
                    self.index_end = max_end;
                }
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
    fn scalar_merge_step(&mut self, force_widen: bool) -> Option<i32> {
        let cap = self.bucket_capacity() as i32;
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
            let slot = (old_idx - self.index_base).rem_euclid(cap) as usize;
            let val = self.bucket_get(slot);
            let out = ((old_idx >> 1) - cur_start) as usize;
            sums[out] = sums[out].saturating_add(val);
            if sums[out] > max_val {
                needs_widen = true;
            }
        }

        let mut target_width = if needs_widen {
            width.wider()?
        } else {
            width
        };

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

        let data = self.bucket_data_mut();
        for w in data.iter_mut() {
            *w = 0;
        }

        self.index_start = cur_start;
        self.index_end = cur_end;
        let spw = target_width.slots_per_word() as i32;
        self.index_base = cur_start & !(spw - 1);

        let new_cap = self.bucket_capacity() as i32;
        for i in 0..cur_len {
            let idx = cur_start + i as i32;
            let slot = (idx - self.index_base).rem_euclid(new_cap) as usize;
            self.bucket_set(slot, sums[i]);
        }

        Some(scale_steps)
    }

    /// Downscales at U64 width by collapsing 2^by adjacent buckets.
    ///
    /// At U64 width, sums use saturating arithmetic and cannot
    /// meaningfully overflow.
    fn bucket_downscale_u64(&mut self, by: i32) {
        debug_assert_eq!(self.bucket_width, BucketWidth::U64);
        debug_assert!(by >= 1);

        if self.is_effectively_empty() {
            self.index_start >>= by;
            self.index_end >>= by;
            self.index_base >>= by;
            return;
        }

        let cap = self.bucket_capacity() as i32;

        let new_start = self.index_start >> by;
        let new_end = self.index_end >> by;
        let new_len = (new_end - new_start + 1) as usize;

        let mut sums = [0u64; 256];
        debug_assert!(new_len <= sums.len());

        for old_idx in self.index_start..=self.index_end {
            let slot = (old_idx - self.index_base).rem_euclid(cap) as usize;
            let val = self.bucket_get(slot);
            let out = ((old_idx >> by) - new_start) as usize;
            sums[out] = sums[out].saturating_add(val);
        }

        let data = self.bucket_data_mut();
        for w in data.iter_mut() {
            *w = 0;
        }

        self.index_start = new_start;
        self.index_end = new_end;
        self.index_base = new_start;

        for i in 0..new_len {
            let idx = new_start + i as i32;
            let slot = (idx - self.index_base).rem_euclid(cap) as usize;
            self.bucket_set(slot, sums[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// SWAR — per-word parallel pairwise summation
// ---------------------------------------------------------------------------

/// Single SWAR step: sum adjacent counters at the current width into
/// the next wider width, in place.
#[inline]
fn swar_step(data: &mut [u64], width: BucketWidth) {
    match width {
        BucketWidth::B1 => {
            // 64 bits → 32 crumbs (2-bit)
            const MASK: u64 = 0x5555_5555_5555_5555;
            for w in data.iter_mut() {
                let x = *w;
                *w = ((x >> 1) & MASK) + (x & MASK);
            }
        }
        BucketWidth::B2 => {
            // 32 crumbs → 16 nibbles
            const MASK: u64 = 0x3333_3333_3333_3333;
            for w in data.iter_mut() {
                let x = *w;
                *w = ((x >> 2) & MASK) + (x & MASK);
            }
        }
        BucketWidth::B4 => {
            // 16 nibbles → 8 bytes
            const MASK: u64 = 0x0F0F_0F0F_0F0F_0F0F;
            for w in data.iter_mut() {
                let x = *w;
                *w = ((x >> 4) & MASK) + (x & MASK);
            }
        }
        BucketWidth::U8 => {
            // 8 bytes → 4 shorts
            const MASK: u64 = 0x00FF_00FF_00FF_00FF;
            for w in data.iter_mut() {
                let x = *w;
                *w = ((x >> 8) & MASK) + (x & MASK);
            }
        }
        BucketWidth::U16 => {
            // 4 shorts → 2 ints
            const MASK: u64 = 0x0000_FFFF_0000_FFFF;
            for w in data.iter_mut() {
                let x = *w;
                *w = ((x >> 16) & MASK) + (x & MASK);
            }
        }
        BucketWidth::U32 => {
            // 2 ints → 1 long
            for w in data.iter_mut() {
                let x = *w;
                *w = (x >> 32) + (x & 0xFFFF_FFFF);
            }
        }
        BucketWidth::U64 => unreachable!("cannot widen past U64"),
    }
}

/// Checks whether any widened pair-sum overflows the original width.
/// Called after `swar_step` has already written the wider sums.
#[inline]
fn swar_has_overflow(data: &[u64], original_width: BucketWidth) -> bool {
    match original_width {
        BucketWidth::B1 => {
            // Sums are in B2 format. Overflow if any crumb > 1.
            const HI: u64 = 0xAAAA_AAAA_AAAA_AAAA;
            data.iter().any(|&w| w & HI != 0)
        }
        BucketWidth::B2 => {
            // Sums are in B4 format. Overflow if any nibble > 3.
            const HI: u64 = 0xCCCC_CCCC_CCCC_CCCC;
            data.iter().any(|&w| w & HI != 0)
        }
        BucketWidth::B4 => {
            // Sums are in U8 format. Overflow if any byte > 15.
            const HI: u64 = 0xF0F0_F0F0_F0F0_F0F0;
            data.iter().any(|&w| w & HI != 0)
        }
        BucketWidth::U8 => {
            // Sums are in U16 format. Overflow if any short > 255.
            const HI: u64 = 0xFF00_FF00_FF00_FF00;
            data.iter().any(|&w| w & HI != 0)
        }
        BucketWidth::U16 => {
            // Sums are in U32 format. Overflow if any int > 65535.
            const HI: u64 = 0xFFFF_0000_FFFF_0000;
            data.iter().any(|&w| w & HI != 0)
        }
        BucketWidth::U32 => {
            // Sums are in U64 format. Overflow if any > u32::MAX.
            data.iter().any(|&w| w > u32::MAX as u64)
        }
        BucketWidth::U64 => false, // saturating add, never "overflows"
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
    let n = data.len();
    match original_width {
        BucketWidth::B1 => {
            // Data is 32 × B2 per word, each ≤ 1. Compress to 64 × B1.
            // Two words of 32 bits each → one word of 64 bits.
            for i in (0..n).step_by(2) {
                let lo = narrow_b2_to_b1(data[i]);
                let hi = if i + 1 < n { narrow_b2_to_b1(data[i + 1]) } else { 0 };
                data[i / 2] = lo | (hi << 32);
            }
            for i in ((n + 1) / 2)..n {
                data[i] = 0;
            }
        }
        BucketWidth::B2 => {
            // Data is 16 × B4 per word, each ≤ 3. Compress to 32 × B2.
            for i in (0..n).step_by(2) {
                let lo = narrow_b4_to_b2(data[i]);
                let hi = if i + 1 < n { narrow_b4_to_b2(data[i + 1]) } else { 0 };
                data[i / 2] = lo | (hi << 32);
            }
            for i in ((n + 1) / 2)..n {
                data[i] = 0;
            }
        }
        BucketWidth::B4 => {
            // Data is 8 × U8 per word, each ≤ 15. Compress to 16 × B4.
            // Two words of 8 nibbles each → one word of 16 nibbles.
            for i in (0..n).step_by(2) {
                let lo = narrow_u8_to_b4(data[i]);
                let hi = if i + 1 < n { narrow_u8_to_b4(data[i + 1]) } else { 0 };
                data[i / 2] = lo | (hi << 32);
            }
            for i in ((n + 1) / 2)..n {
                data[i] = 0;
            }
        }
        BucketWidth::U8 => {
            // Data is 4 × U16 per word, each ≤ 255. Compress to 8 × U8.
            for i in (0..n).step_by(2) {
                let lo = narrow_u16_to_u8(data[i]);
                let hi = if i + 1 < n { narrow_u16_to_u8(data[i + 1]) } else { 0 };
                data[i / 2] = lo | (hi << 32);
            }
            for i in ((n + 1) / 2)..n {
                data[i] = 0;
            }
        }
        BucketWidth::U16 => {
            // Data is 2 × U32 per word, each ≤ 65535. Compress to 4 × U16.
            for i in (0..n).step_by(2) {
                let lo = narrow_u32_to_u16(data[i]);
                let hi = if i + 1 < n { narrow_u32_to_u16(data[i + 1]) } else { 0 };
                data[i / 2] = lo | (hi << 32);
            }
            for i in ((n + 1) / 2)..n {
                data[i] = 0;
            }
        }
        BucketWidth::U32 => {
            // Data is 1 × U64 per word, each ≤ u32::MAX. Compress to 2 × U32.
            for i in (0..n).step_by(2) {
                let lo = data[i] & 0xFFFF_FFFF;
                let hi = if i + 1 < n { data[i + 1] & 0xFFFF_FFFF } else { 0 };
                data[i / 2] = lo | (hi << 32);
            }
            for i in ((n + 1) / 2)..n {
                data[i] = 0;
            }
        }
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

/// Compress 32 crumbs (each ≤ 1) into 32 bits in the low 32 bits.
#[inline]
fn narrow_b2_to_b1(w: u64) -> u64 {
    // Each 2-bit field has value 0 or 1. Pack pairs into single bits.
    let x = w & 0x5555_5555_5555_5555;  // mask: keep low bit of each crumb
    let x = x | (x >> 1);               // crumb pairs → bits
    let x = x & 0x3333_3333_3333_3333;  // keep merged pairs
    let x = x | (x >> 2);               // nibble compaction
    let x = x & 0x0F0F_0F0F_0F0F_0F0F;  // keep merged nibbles
    let x = x | (x >> 4);               // byte compaction
    let x = x & 0x00FF_00FF_00FF_00FF;  // keep merged bytes
    let x = x | (x >> 8);               // short compaction
    let x = x & 0x0000_FFFF_0000_FFFF;  // keep merged shorts
    let x = x | (x >> 16);              // 32-bit compaction
    x & 0x0000_0000_FFFF_FFFF
}

/// Compress 16 nibbles (each ≤ 3) into 16 crumbs in the low 32 bits.
#[inline]
fn narrow_b4_to_b2(w: u64) -> u64 {
    // Each nibble has value 0..3. Pack pairs into 2-bit crumbs.
    let x = w & 0x3333_3333_3333_3333;  // mask: keep low 2 bits of each nibble
    let x = x | (x >> 2);               // nibble pairs → crumbs
    let x = x & 0x0F0F_0F0F_0F0F_0F0F;  // keep merged nibbles
    let x = x | (x >> 4);               // byte compaction
    let x = x & 0x00FF_00FF_00FF_00FF;  // keep merged bytes
    let x = x | (x >> 8);               // short compaction
    let x = x & 0x0000_FFFF_0000_FFFF;  // keep merged shorts
    let x = x | (x >> 16);              // 32-bit compaction
    x & 0x0000_0000_FFFF_FFFF
}

/// Compress 8 bytes (each ≤ 15) into 8 nibbles in the low 32 bits.
#[inline]
fn narrow_u8_to_b4(w: u64) -> u64 {
    // Each byte has value 0..15. Pack adjacent byte pairs into byte,
    // then adjacent byte pairs into shorts, etc.
    let x = w & 0x0F0F_0F0F_0F0F_0F0F; // mask (redundant safety)
    let x = x | (x >> 4);               // nibble pairs → bytes
    let x = x & 0x00FF_00FF_00FF_00FF;  // keep merged bytes
    let x = x | (x >> 8);               // byte pairs → shorts
    let x = x & 0x0000_FFFF_0000_FFFF;  // keep merged shorts
    let x = x | (x >> 16);              // short pairs → 32 bits
    x & 0x0000_0000_FFFF_FFFF
}

/// Compress 4 shorts (each ≤ 255) into 4 bytes in the low 32 bits.
#[inline]
fn narrow_u16_to_u8(w: u64) -> u64 {
    let x = w & 0x00FF_00FF_00FF_00FF;  // mask (redundant safety)
    let x = x | (x >> 8);               // short pairs → ints
    let x = x & 0x0000_FFFF_0000_FFFF;  // keep merged ints
    let x = x | (x >> 16);              // int pairs → 32 bits
    x & 0x0000_0000_FFFF_FFFF
}

/// Compress 2 ints (each ≤ 65535) into 2 shorts in the low 32 bits.
#[inline]
fn narrow_u32_to_u16(w: u64) -> u64 {
    let x = w & 0x0000_FFFF_0000_FFFF;  // mask (redundant safety)
    let x = x | (x >> 16);              // int pairs → 32 bits
    x & 0x0000_0000_FFFF_FFFF
}

// ---------------------------------------------------------------------------
// Histogram<N, P> — construction and public API
// ---------------------------------------------------------------------------

impl<const N: usize, P: Precision> Histogram<N, P> {
    /// Creates a new histogram at the maximum supported scale.
    #[inline]
    pub fn new() -> Self {
        let scale = max_scale();
        Self {
            mapping: Mapping::new(scale).unwrap(),
            max_scale: scale as i8,
            min_bucket_width: BucketWidth::B1,
            bucket_width: BucketWidth::B1,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            data: [0u64; N],
            _precision: PhantomData,
        }
    }

    /// Creates a new histogram with an upper bound on scale.
    #[inline]
    pub fn with_max_scale(scale: i32) -> Self {
        let scale = scale.min(max_scale());
        Self {
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale as i8,
            min_bucket_width: BucketWidth::B1,
            bucket_width: BucketWidth::B1,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            data: [0u64; N],
            _precision: PhantomData,
        }
    }

    /// Creates a new histogram at the specified scale.
    #[inline]
    pub fn with_scale(scale: i32) -> Self {
        Self {
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale as i8,
            min_bucket_width: BucketWidth::B1,
            bucket_width: BucketWidth::B1,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            data: [0u64; N],
            _precision: PhantomData,
        }
    }

    /// Sets the minimum (initial) bucket counter width.
    ///
    /// By default, counters start at 4-bit (B4). Setting a higher floor
    /// (e.g. `BucketWidth::U8`) trades bucket capacity for avoiding the
    /// CPU cost of sub-byte bit-level indexing and SWAR widening.
    ///
    /// This also becomes the width used after `clear()`.
    #[inline]
    pub fn with_min_bucket_width(mut self, width: BucketWidth) -> Self {
        self.min_bucket_width = width;
        self.bucket_width = width;
        self
    }

    /// Returns the current scale.
    #[inline]
    pub fn scale(&self) -> i32 {
        if self.non_zero_count() == 0 {
            0
        } else {
            self.mapping.scale()
        }
    }

    /// Returns the maximum scale this histogram will use on reset.
    #[inline]
    pub fn max_scale(&self) -> i32 {
        self.max_scale as i32
    }

    /// Returns the current bucket counter width.
    #[inline]
    pub fn bucket_width(&self) -> BucketWidth {
        self.bucket_width
    }

    /// Returns a read-only view of the positive buckets.
    #[inline]
    pub fn positive(&self) -> BucketView<'_, N, P> {
        BucketView { hist: self }
    }

    /// Clears the histogram, resetting to initial state.
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.bucket_width = self.min_bucket_width;
        self.index_start = 0;
        self.index_end = 0;
        self.index_base = 0;
        self.mapping = Mapping::new(self.max_scale as i32).unwrap();
    }

    /// Swaps contents with another histogram.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Records a single value.
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

        // Pre-validate count overflow before mutating any state.
        let new_count = self.checked_add_count(incr).ok_or(Overflow)?;

        if value != 0.0 {
            let snapshot = self.clone();
            match self.update_buckets(value, incr) {
                Ok(()) => {}
                Err(e) => {
                    *self = snapshot;
                    return Err(e);
                }
            }
            self.add_sum(value * incr as f64);
        }

        // Commit count and min/max after all fallible work succeeds.
        if self.count() == 0 {
            self.set_min(value);
            self.set_max(value);
        } else {
            if value < self.min() {
                self.set_min(value);
            }
            if value > self.max() {
                self.set_max(value);
            }
        }
        self.set_count(new_count);
        Ok(())
    }

    /// Updates buckets for a positive value.
    fn update_buckets(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        loop {
            let index = self.mapping.map_to_index(value);

            match self.increment_index_by(index, incr) {
                IncrResult::Ok => return Ok(()),
                IncrResult::NeedsDownscale(hl) => {
                    let change = change_scale(hl, self.bucket_capacity() as i32);
                    if change > 0 {
                        self.do_downscale(change)?;
                    } else {
                        // Span fits but index is outside [base, base+cap)
                        // at sub-U64. Force widen so width increases
                        // toward U64 where wrapping is allowed.
                        if self.bucket_width != BucketWidth::U64 {
                            let by = self.bucket_widen(1).ok_or(Overflow)?;
                            let new_scale = self.mapping.scale() - by;
                            self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                        } else {
                            self.do_downscale(1)?;
                        }
                    }
                }
                IncrResult::CounterOverflow => {
                    // Counter saturated — widen by 1 step.
                    let by = self.bucket_widen(1).ok_or(Overflow)?;
                    let new_scale = self.mapping.scale() - by;
                    self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                }
            }
        }
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
    pub fn do_downscale(&mut self, change: i32) -> Result<(), Overflow> {
        if change <= 0 {
            return Ok(());
        }

        if self.is_effectively_empty() {
            self.index_start >>= change;
            self.index_end >>= change;
            self.index_base >>= change;
            let new_scale = self.mapping.scale() - change;
            self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
            return Ok(());
        }

        let mut remaining = change;

        // Phase 1: Adaptive SWAR merge at sub-U64 widths.
        while remaining > 0 && self.bucket_width != BucketWidth::U64 {
            // Ensure even base for correct SWAR pairing.
            let shifted = self.index_base & 1 != 0;
            if shifted {
                let width = self.bucket_width;
                let bits = width.bits();
                let n = self.bucket_word_count();
                if n > 0 && self.bucket_data()[n - 1] >> (64 - bits) != 0 {
                    // Top slot occupied — can't shift. Scalar merge.
                    let steps = self.scalar_merge_step(false).ok_or(Overflow)?;
                    let new_scale = self.mapping.scale() - steps;
                    self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                    remaining -= steps;
                    continue;
                }
                swar_shift_up_one(self.bucket_data_mut(), width);
            }

            let width = self.bucket_width;
            swar_step(self.bucket_data_mut(), width);

            if swar_has_overflow(self.bucket_data(), width) {
                // Overflow — accept the wider format.
                self.bucket_width = width.wider().unwrap();
            } else {
                // No overflow — narrow back to preserve capacity.
                swar_narrow_compact(self.bucket_data_mut(), width);
            }

            self.index_start >>= 1;
            self.index_end >>= 1;
            self.index_base >>= 1;

            // Same clamp as bucket_widen: after shift_up_one + >>= 1,
            // the odd-base shift can cause len > cap by 1.
            if shifted {
                let cap = self.bucket_capacity() as i32;
                let max_end = self.index_start + cap - 1;
                if self.index_end > max_end {
                    self.index_end = max_end;
                }
            }

            let new_scale = self.mapping.scale() - 1;
            self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
            remaining -= 1;
        }

        // Phase 2: At U64, scatter-write for remaining steps.
        if remaining > 0 {
            debug_assert_eq!(self.bucket_width, BucketWidth::U64);
            self.bucket_downscale_u64(remaining);
            let new_scale = self.mapping.scale() - remaining;
            self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
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
                let bi = (idx - self.index_base).rem_euclid(max_size);
                self.bucket_set(bi as usize, 0);
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
                let bi = (idx - self.index_base).rem_euclid(max_size);
                self.bucket_set(bi as usize, 0);
            }
            self.index_end = index;
        }

        let bucket_index = (index - self.index_base).rem_euclid(max_size);

        if !self.bucket_try_increment(bucket_index as usize, incr) {
            return IncrResult::CounterOverflow;
        }

        IncrResult::Ok
    }

    // -- Merge --

    /// Merges another histogram (same N) into this one.
    pub fn merge_from(&mut self, other: &Self) -> Result<(), Overflow> {
        if other.count() == 0 {
            return Ok(());
        }

        let new_count = self.checked_add_count(other.count()).ok_or(Overflow)?;

        // Save state so we can roll back if the merge fails partway.
        let snapshot = self.clone();

        let new_sum = self.sum() + other.sum();

        let result = (|| {
            if !other.buckets_empty() {
                if self.buckets_empty() {
                    self.bucket_width = self.bucket_width.max(other.bucket_width);
                }

                let min_scale = self.mapping.scale().min(other.scale());
                let cap = self.bucket_capacity() as i32;

                let hlp = self.high_low_at_scale(min_scale)
                    .merge(Self::high_low_at_scale_of(other, other.mapping.scale(), min_scale));

                let min_scale = min_scale - change_scale(hlp, cap);

                self.downscale_to(min_scale)?;

                self.merge_buckets_from(other, other.mapping.scale())?;

                self.trim_bucket_range();
            }

            // Commit stats.
            if self.count() == 0 {
                self.set_min(other.min());
                self.set_max(other.max());
            } else {
                if other.min() < self.min() {
                    self.set_min(other.min());
                }
                if other.max() > self.max() {
                    self.set_max(other.max());
                }
            }
            self.set_sum(new_sum);
            self.set_count(new_count);
            Ok(())
        })();

        if result.is_err() {
            *self = snapshot;
        }
        result
    }

    /// Merges from raw histogram data, enabling cross-size merging.
    pub fn merge_from_raw(
        &mut self,
        other_count: u64,
        other_sum: f64,
        other_min: f64,
        other_max: f64,
        other_scale: i32,
        other_offset: i32,
        other_len: u32,
        other_at: &dyn Fn(u32) -> u64,
    ) -> Result<(), Overflow> {
        if other_count == 0 {
            return Ok(());
        }

        let new_count = self.checked_add_count(other_count).ok_or(Overflow)?;

        // Save state so we can roll back if the merge fails partway.
        let snapshot = self.clone();

        let new_sum = self.sum() + other_sum;

        let result = (|| {
            if other_len > 0 {
                let other_end = other_offset + other_len as i32 - 1;
                let cap = self.bucket_capacity() as i32;
                let min_scale = self.mapping.scale().min(other_scale);

                let self_hl = if self.buckets_empty() {
                    HighLow::empty()
                } else {
                    let shift = self.mapping.scale() - min_scale;
                    HighLow {
                        low: self.index_start >> shift,
                        high: self.index_end >> shift,
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
                let min_scale = min_scale - change_scale(hlp, cap);

                self.downscale_to(min_scale)?;

                for i in 0..other_len {
                    let count = other_at(i);
                    if count == 0 {
                        continue;
                    }
                    loop {
                        let their_change = other_scale - self.mapping.scale();
                        let index = (other_offset + i as i32) >> their_change;

                        match self.increment_index_by(index, count) {
                            IncrResult::Ok => break,
                            IncrResult::CounterOverflow => {
                                let by = self.bucket_widen(1).ok_or(Overflow)?;
                                let new_scale = self.mapping.scale() - by;
                                self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                            }
                            IncrResult::NeedsDownscale(_) => {
                                if self.bucket_width != BucketWidth::U64 {
                                    let by = self.bucket_widen(1).ok_or(Overflow)?;
                                    let new_scale = self.mapping.scale() - by;
                                    self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                                } else {
                                    self.do_downscale(1)?;
                                }
                            }
                        }
                    }
                }
            }

            self.trim_bucket_range();

            // Commit stats.
            if self.count() == 0 {
                self.set_min(other_min);
                self.set_max(other_max);
            } else {
                if other_min < self.min() {
                    self.set_min(other_min);
                }
                if other_max > self.max() {
                    self.set_max(other_max);
                }
            }
            self.set_sum(new_sum);
            self.set_count(new_count);
            Ok(())
        })();

        if result.is_err() {
            *self = snapshot;
        }
        result
    }

    /// Merges a histogram of a different size into this one.
    pub fn merge_from_other<const M: usize>(
        &mut self,
        other: &Histogram<M, P>,
    ) -> Result<(), Overflow> {
        self.merge_from_raw(
            other.count(),
            other.sum(),
            other.min(),
            other.max(),
            other.scale(),
            other.bucket_offset(),
            other.bucket_len(),
            &|i| other.bucket_at(i),
        )
    }

    fn high_low_at_scale(&self, scale: i32) -> HighLow {
        Self::high_low_at_scale_of(self, self.mapping.scale(), scale)
    }

    fn high_low_at_scale_of(hist: &Histogram<N, P>, current_scale: i32, target_scale: i32) -> HighLow {
        if hist.buckets_empty() {
            return HighLow::empty();
        }
        let shift = current_scale - target_scale;
        HighLow {
            low: hist.index_start >> shift,
            high: hist.index_end >> shift,
        }
    }

    fn merge_buckets_from(&mut self, other: &Histogram<N, P>, other_scale: i32) -> Result<(), Overflow> {
        let their_offset = other.bucket_offset();

        for i in 0..other.bucket_len() {
            let count = other.bucket_at(i);
            if count == 0 {
                continue;
            }
            loop {
                let their_change = other_scale - self.mapping.scale();
                let index = (their_offset + i as i32) >> their_change;

                match self.increment_index_by(index, count) {
                    IncrResult::Ok => break,
                    IncrResult::CounterOverflow => {
                        let by = self.bucket_widen(1).ok_or(Overflow)?;
                        let new_scale = self.mapping.scale() - by;
                        self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                    }
                    IncrResult::NeedsDownscale(hl) => {
                        let change = change_scale(hl, self.bucket_capacity() as i32);
                        if change > 0 {
                            self.do_downscale(change)?;
                        } else {
                            // Span fits but alignment doesn't at sub-U64.
                            // Force unconditional widen so width actually
                            // increases toward U64.
                            if self.bucket_width != BucketWidth::U64 {
                                let by = self.bucket_widen(1).ok_or(Overflow)?;
                                let new_scale = self.mapping.scale() - by;
                                self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                            } else {
                                self.do_downscale(1)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::{P32, P64, Precision};

    fn derived_zero_count<const N: usize, P: Precision>(h: &Histogram<N, P>) -> u64 {
        let buckets = h.positive();
        let non_zero: u64 = (0..buckets.len()).map(|i| buckets.at(i)).sum();
        h.count() - non_zero
    }

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<16, P32> = Histogram::new();
        h.update(1.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum(), 1.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
        assert_eq!(derived_zero_count(&h), 0);
        assert_eq!(h.bucket_width(), BucketWidth::B1);
    }

    #[test]
    fn test_histogram_zero() {
        let mut h: Histogram<16, P32> = Histogram::new();
        h.update(0.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(derived_zero_count(&h), 1);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_multiple() {
        let mut h: Histogram<16, P32> = Histogram::new();
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
        let mut h: Histogram<8, P32> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(1000.0).unwrap();
        assert_eq!(h.count(), 2);
        assert!(h.scale() < max_scale());
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1: Histogram<16, P32> = Histogram::new();
        let mut h2: Histogram<16, P32> = Histogram::new();
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
        let mut h: Histogram<16, P32> = Histogram::new();
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        h.update(1.5).unwrap();
        h.update(100.0).unwrap();
        h.update(1e10).unwrap();

        let buckets = h.positive();
        assert!(buckets.len() >= 2,
            "expected at least 2 buckets, got {} at scale {}",
            buckets.len(), h.scale());
    }

    #[test]
    fn test_auto_widen_b4_to_u8() {
        let mut h: Histogram<16, P32> = Histogram::new().with_min_bucket_width(BucketWidth::B4);
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update_by_incr(1.0, 15).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 16);
    }

    #[test]
    fn test_auto_widen_b4_to_u8_from_b4_start() {
        let mut h: Histogram<16, P32> = Histogram::new().with_min_bucket_width(BucketWidth::B4);
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
        let mut h: Histogram<16, P32> = Histogram::new();
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
        let mut h: Histogram<16, P32> = Histogram::new();
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
        // Use P64 because this test needs count > u32::MAX.
        let mut h: Histogram<16, P64> = Histogram::new();
        h.update_by_incr(1.0, 16).unwrap();
        h.update_by_incr(1.0, 239).unwrap();
        h.update(1.0).unwrap(); // U8→U16
        h.update_by_incr(1.0, u16::MAX as u64 - 256).unwrap();
        h.update(1.0).unwrap(); // U16→U32
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        h.update_by_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1)).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        h.update(1.0).unwrap(); // U32→U64
        assert_eq!(h.bucket_width(), BucketWidth::U64);
    }

    #[test]
    fn test_bucket_count_halves_on_widen() {
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        let initial_cap = h.bucket_capacity();
        assert_eq!(initial_cap, 14 * 16); // 224

        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.bucket_capacity(), 14 * 8); // 112
    }

    #[test]
    fn test_clear_resets_to_b4() {
        let mut h: Histogram<16, P32> = Histogram::with_max_scale(3).with_min_bucket_width(BucketWidth::B4);
        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.clear();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale() {
        let h: Histogram<16, P32> = Histogram::with_max_scale(3);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale_clamps() {
        let h: Histogram<16, P32> = Histogram::with_max_scale(100);
        assert_eq!(h.max_scale(), max_scale());
    }

    #[test]
    fn test_with_max_scale_records_at_limited_scale() {
        let mut limited: Histogram<16, P32> = Histogram::with_max_scale(3);
        let mut unlimited: Histogram<16, P32> = Histogram::new();
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
    fn test_clear_resets_to_max_scale() {
        let mut h: Histogram<16, P32> = Histogram::with_max_scale(3);
        h.update(0.001).unwrap();
        h.update(1000.0).unwrap();
        assert!(h.scale() <= 3);
        h.clear();
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);
        h.update(1.0).unwrap();
        assert_eq!(h.scale(), 3);
    }

    #[test]
    fn test_widen_preserves_data() {
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
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
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let hardcoded_sets: &[&[f64]] = &[
            &[], &[0.0], &[1.0], &[0.0, 0.0], &[1.0, 1.0], &[1.0, 2.0],
            &[0.5, 1.5, 2.5], &[0.001, 1.0, 20.0], &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 1.0, 2.0, 0.0], &[5.0, 10.0, 15.0, 20.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            &[0.01, 0.1, 1.0, 10.0], &[0.0, 20.0], &[1.0, 19.0],
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

        test_merge_equivalence_for_size::<8, P32>(&test_sets);
        test_merge_equivalence_for_size::<12, P32>(&test_sets);
        test_merge_equivalence_for_size::<16, P32>(&test_sets);
        test_merge_equivalence_for_size::<20, P32>(&test_sets);
    }

    fn test_merge_equivalence_for_size<const K: usize, P: Precision>(test_sets: &[Vec<f64>]) {
        for (i, set_a) in test_sets.iter().enumerate() {
            for (j, set_b) in test_sets.iter().enumerate() {
                let mut merged: Histogram<K, P> = Histogram::new();
                for &v in set_a {
                    merged.update(v).unwrap();
                }
                let mut other: Histogram<K, P> = Histogram::new();
                for (vi, &v) in set_b.iter().enumerate() {
                    if let Err(e) = other.update(v) {
                        panic!("other.update failed for size={K} sets {i} x {j} val[{vi}]={v}: {e}\n  set_b: {set_b:?}\n  other: {:?}", other);
                    }
                }
                if let Err(e) = merged.merge_from(&other) {
                    panic!("merge_from failed for size={K} sets {i} x {j}: {e}\n  set_a: {set_a:?}\n  set_b: {set_b:?}\n  merged: {:?}\n  other: {:?}", merged, other);
                }

                let mut single: Histogram<K, P> = Histogram::new();
                for &v in set_a {
                    single.update(v).unwrap();
                }
                for &v in set_b {
                    single.update(v).unwrap();
                }

                assert_eq!(merged.count(), single.count(),
                    "count mismatch for size={K} sets {i} x {j}");
                // P32 uses f32 for sum, so order-of-operations rounding
                // can differ between merged and single paths. Use relative
                // tolerance appropriate for f32 precision.
                let ms = merged.sum();
                let ss = single.sum();
                let sum_diff = (ms - ss).abs();
                let denom = ms.abs().max(ss.abs()).max(1e-30);
                assert!(sum_diff / denom < 1e-5,
                    "sum mismatch for size={K} sets {i} x {j}: {} vs {}",
                    ms, ss);
                assert_eq!(derived_zero_count(&merged), derived_zero_count(&single),
                    "zero_count mismatch for size={K} sets {i} x {j}");

                let mb = merged.positive();
                let sb = single.positive();
                let m_total: u64 = (0..mb.len()).map(|k| mb.at(k)).sum();
                let s_total: u64 = (0..sb.len()).map(|k| sb.at(k)).sum();
                if m_total != s_total {
                    eprintln!("FAIL size={K} sets {i} x {j}");
                    eprintln!("  set_a: {:?}", set_a);
                    eprintln!("  set_b: {:?}", set_b);
                    eprintln!("  merged: width={:?} start={} end={} base={}",
                        mb.width(), merged.index_start, merged.index_end, merged.index_base);
                    eprintln!("  single: width={:?} start={} end={} base={}",
                        sb.width(), single.index_start, single.index_end, single.index_base);
                    eprintln!("  merged buckets (len={}): {:?}", mb.len(),
                        (0..mb.len()).map(|k| mb.at(k)).collect::<Vec<_>>());
                    eprintln!("  single buckets (len={}): {:?}", sb.len(),
                        (0..sb.len()).map(|k| sb.at(k)).collect::<Vec<_>>());
                }
                assert_eq!(m_total, s_total,
                    "bucket total mismatch for size={K} sets {i} x {j}");
            }
        }
    }

    #[test]
    fn test_debug_merge_sets_2_x_35() {
        // Reproduces "bucket total mismatch for size=8 sets 2 x 35"
        let set_b: &[f64] = &[18.896147780359236, 19.038540970281623, 15.726266735088323,
                               19.97053274796744, 16.963914020801518];

        // First check: does 'other' itself have consistent bucket totals?
        let mut other: Histogram<8, P32> = Histogram::new();
        for (vi, &v) in set_b.iter().enumerate() {
            other.update(v).unwrap();
            let bv = other.positive();
            let btotal: u64 = (0..bv.len()).map(|k| bv.at(k)).sum();
            let non_zero_count = other.count() - derived_zero_count(&other);
            eprintln!("Other after val[{}]={}: scale={} width={:?} start={} end={} base={} cap={} btotal={} expected={}",
                vi, v, other.scale(), bv.width(),
                other.index_start, other.index_end, other.index_base,
                other.bucket_capacity(), btotal, non_zero_count);
            if btotal != non_zero_count {
                eprintln!("  ** BUCKET TOTAL MISMATCH IN OTHER ** raw data: {:?}", &other.data);
                eprintln!("  buckets: {:?}", (0..bv.len()).map(|k| bv.at(k)).collect::<Vec<_>>());
            }
        }

        let set_a: &[f64] = &[1.0];
        let mut merged: Histogram<8, P32> = Histogram::new();
        for &v in set_a {
            merged.update(v).unwrap();
        }
        merged.merge_from(&other).unwrap();

        let mut single: Histogram<8, P32> = Histogram::new();
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
        eprintln!("Merged buckets: {:?}", (0..mb.len()).map(|k| mb.at(k)).collect::<Vec<_>>());
        eprintln!("Single buckets: {:?}", (0..sb.len()).map(|k| sb.at(k)).collect::<Vec<_>>());
        assert_eq!(m_total, s_total,
            "bucket total mismatch: merged={} single={}", m_total, s_total);
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

        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        h.update(1.0).unwrap();
        h.update(max_f64).unwrap();
        // At P32, f64::MAX overflows f32, so sum is already infinite.
        // At P64, it would still be finite. Don't assert finiteness.
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

        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        h.update(subnormal).unwrap();
        h.update(min_normal).unwrap();
        assert_eq!(h.count(), 2);
        assert_eq!(h.positive().len(), 1);
    }

    #[test]
    fn test_exhaustive_u8_overflow() {
        // Insert 8 values spanning a wide index range at scale 0, each
        // with count 255. Starting at B1 with 320 slots (Histogram<8, P32>),
        // counters widen B1→B2→B4→U8 (255 fits in U8), but the larger
        // initial capacity means the span still fits without reaching U64.
        let mut h: Histogram<8, P32> = Histogram::with_scale(0);
        let num_buckets = 8;
        for i in 0..num_buckets {
            let val = 2.0_f64.powi(i * 8);
            h.update_by_incr(val, 255).unwrap();
        }
        // With B1 start, U8 has enough capacity for the span.
        assert!(h.bucket_width() >= BucketWidth::U8,
            "expected at least U8, got {:?}", h.bucket_width());
        assert_eq!(h.count(), num_buckets as u64 * 255);
        // Adding one more should still be fine at U64 (no further widen needed).
        h.update(1.0).unwrap();
        assert_eq!(h.count(), num_buckets as u64 * 255 + 1);
    }

    #[test]
    fn test_successive_sub_byte_widening() {
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);

        h.update(1.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        for count in 2..=15u64 {
            h.update(1.0).unwrap();
            assert_eq!(h.count(), count);
            assert_eq!(h.bucket_width(), BucketWidth::B4,
                "expected B4 at count {count}");
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        let num_buckets = 8;
        let values: Vec<f64> = (1..=num_buckets)
            .map(|k| 2.0_f64.powi(k))
            .collect();

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
        assert!((h.sum() - expected_sum).abs() < 1e-6,
            "sum mismatch: got {} expected {}", h.sum(), expected_sum);
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
        let mut collector: Histogram<16, P32> = Histogram::new();
        let mut source: Histogram<8, P32> = Histogram::new();

        source.update(1.0).unwrap();
        source.update(2.0).unwrap();
        source.update(4.0).unwrap();
        source.update(0.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        assert_eq!(collector.count(), 4);
        assert_eq!(derived_zero_count(&collector), 1);
        assert!((collector.sum() - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_merge_multiple_sources() {
        let mut collector: Histogram<20, P32> = Histogram::new();

        for batch in 0..5 {
            let mut src: Histogram<16, P32> = Histogram::new();
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
        let mut collector: Histogram<16, P32> = Histogram::with_scale(0);
        let mut source: Histogram<16, P32> = Histogram::with_scale(0);

        source.update(1.0).unwrap();
        source.update(2.0).unwrap();
        source.update(4.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        let mut direct: Histogram<16, P32> = Histogram::with_scale(0);
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
        let mut collector: Histogram<16, P32> = Histogram::new();
        collector.update(1.0).unwrap();

        let empty: Histogram<8, P32> = Histogram::new();
        collector.merge_from_other(&empty).unwrap();

        assert_eq!(collector.count(), 1);
        assert_eq!(collector.sum(), 1.0);
    }

    #[test]
    fn test_merge_into_empty() {
        let mut collector: Histogram<16, P32> = Histogram::new();
        let mut source: Histogram<8, P32> = Histogram::new();
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
        use crate::precision::{P32, P64};

        #[test]
        fn test_capacity_p32() {
            let h: Histogram<16, P32> = Histogram::new();
            assert_eq!(h.bucket_word_count(), 14);
            assert_eq!(h.bucket_capacity(), 896); // 14 * 64 at B1
        }

        #[test]
        fn test_capacity_p64() {
            let h: Histogram<16, P64> = Histogram::new();
            assert_eq!(h.bucket_word_count(), 12);
            assert_eq!(h.bucket_capacity(), 768); // 12 * 64 at B1
        }

        #[test]
        fn test_minimum_n() {
            let h: Histogram<8, P32> = Histogram::new();
            assert_eq!(h.bucket_word_count(), 6);
            assert_eq!(h.bucket_capacity(), 384); // 6 * 64 at B1
        }

        #[test]
        fn test_struct_size() {
            use core::mem;
            let size = mem::size_of::<Histogram<16, P32>>();
            eprintln!("Histogram<16, P32> size: {} bytes", size);
            // Fixed fields + 128 bytes of data
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
        for i in 0..16 {
            w |= (n[i] as u64 & 0xF) << (i * 4);
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
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
    }

    #[test]
    fn test_narrow_u8_to_b4_max_values() {
        // 8 bytes, each = 15 (max B4): should produce 8 nibbles each = 15
        let input = pack_u8x8([15, 15, 15, 15, 15, 15, 15, 15]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
    }

    #[test]
    fn test_narrow_u8_to_b4_ascending() {
        // 8 bytes: [0, 1, 2, 3, 4, 5, 6, 7] → 8 nibbles in order
        let input = pack_u8x8([0, 1, 2, 3, 4, 5, 6, 7]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
    }

    #[test]
    fn test_narrow_u8_to_b4_scattered() {
        // Specific pattern to test bit-compress ordering
        let input = pack_u8x8([15, 0, 8, 0, 3, 0, 1, 0]);
        let result = narrow_u8_to_b4(input);
        let expected = pack_b4x16([15, 0, 8, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
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
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
    }

    #[test]
    fn test_narrow_u16_to_u8_max_values() {
        // 4 shorts, each = 255 (max U8)
        let input = pack_u16x4([255, 255, 255, 255]);
        let result = narrow_u16_to_u8(input);
        let expected = pack_u8x8([255, 255, 255, 255, 0, 0, 0, 0]) & 0xFFFF_FFFF;
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
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
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
    }

    #[test]
    fn test_narrow_u32_to_u16_max_values() {
        let input = pack_u32x2(65535, 65535);
        let result = narrow_u32_to_u16(input);
        let expected = (65535u64) | (65535u64 << 16);
        assert_eq!(result, expected,
            "got {:#018x}, expected {:#018x}", result, expected);
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
        assert_eq!(data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}", data[0], expected);
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
        assert_eq!(data[0], expected0,
            "word 0: got {:#018x}, expected {:#018x}", data[0], expected0);
        assert_eq!(data[1], expected1,
            "word 1: got {:#018x}, expected {:#018x}", data[1], expected1);
        assert_eq!(data[2], 0, "word 2 should be zeroed");
        assert_eq!(data[3], 0, "word 3 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u8_two_words() {
        // Two words of U16 data (result of swar_step on U8), each ≤ 255.
        // Word 0: [10, 20, 30, 40]  Word 1: [50, 60, 70, 80]
        // After compact: one word of 8 bytes.
        let mut data = [
            pack_u16x4([10, 20, 30, 40]),
            pack_u16x4([50, 60, 70, 80]),
        ];
        swar_narrow_compact(&mut data, BucketWidth::U8);

        let expected = pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}", data[0], expected);
        assert_eq!(data[1], 0, "word 1 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u16_two_words() {
        // Two words of U32 data, each ≤ 65535.
        let mut data = [
            pack_u32x2(100, 200),
            pack_u32x2(300, 400),
        ];
        swar_narrow_compact(&mut data, BucketWidth::U16);

        let expected = pack_u16x4([100, 200, 300, 400]);
        assert_eq!(data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}", data[0], expected);
        assert_eq!(data[1], 0, "word 1 should be zeroed");
    }

    #[test]
    fn test_swar_narrow_compact_u32_two_words() {
        // Two words of U64 data, each ≤ u32::MAX.
        let mut data = [1000u64, 2000u64];
        swar_narrow_compact(&mut data, BucketWidth::U32);

        let expected = pack_u32x2(1000, 2000);
        assert_eq!(data[0], expected,
            "word 0: got {:#018x}, expected {:#018x}", data[0], expected);
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
        assert_eq!(data[0], pack_u8x8([3, 7, 0, 0, 0, 0, 0, 0]),
            "swar_step word 0: got {:#018x}", data[0]);
        assert_eq!(data[1], pack_u8x8([0, 0, 0, 0, 0, 0, 5, 6]),
            "swar_step word 1: got {:#018x}", data[1]);

        // Step 2: No overflow (all ≤ 15)
        assert!(!swar_has_overflow(&data, BucketWidth::B4));

        // Step 3: Narrow + compact (U8 → B4, 2 words → 1 word)
        swar_narrow_compact(&mut data, BucketWidth::B4);

        // Result: 1 word of 16 nibbles = [3,7,0,0,0,0,0,0, 0,0,0,0,0,0,5,6]
        let expected = pack_b4x16([3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6]);
        assert_eq!(data[0], expected,
            "compact result: got {:#018x}, expected {:#018x}", data[0], expected);
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
        assert_eq!(data[0], pack_u8x8([150, 50, 15, 4, 0, 0, 0, 0]),
            "compact result: got {:#018x}", data[0]);
        assert_eq!(data[1], 0);
    }

    #[test]
    fn test_swar_step_then_narrow_compact_u16_roundtrip() {
        let mut data = [
            pack_u16x4([1000, 2000, 3000, 4000]),
            pack_u16x4([0; 4]),
        ];

        swar_step(&mut data, BucketWidth::U16);
        // Pair sums: [3000, 7000, 0, 0] as U32
        assert!(!swar_has_overflow(&data, BucketWidth::U16));

        swar_narrow_compact(&mut data, BucketWidth::U16);
        // 1 word: [3000, 7000, 0, 0] as U16
        assert_eq!(data[0], pack_u16x4([3000, 7000, 0, 0]),
            "compact result: got {:#018x}", data[0]);
        assert_eq!(data[1], 0);
    }

    #[test]
    fn test_swar_step_then_narrow_compact_u32_roundtrip() {
        let mut data = [pack_u32x2(100_000, 200_000), pack_u32x2(0, 0)];

        swar_step(&mut data, BucketWidth::U32);
        // Sum = 300_000 as U64
        assert!(!swar_has_overflow(&data, BucketWidth::U32));

        swar_narrow_compact(&mut data, BucketWidth::U32);
        assert_eq!(data[0], pack_u32x2(300_000, 0),
            "compact result: got {:#018x}", data[0]);
        assert_eq!(data[1], 0);
    }

    // -----------------------------------------------------------------------
    // Adaptive merge (do_downscale) integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_do_downscale_merge_stays_b4() {
        // Insert small values at adjacent indices so pair sums ≤ 15.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        // At scale 0, map_to_index(2.0) = 0, map_to_index(4.0) = 1.
        // These are a pair (even, odd) that will sum via SWAR.
        h.update_by_incr(2.0, 5).unwrap();  // index 0, count 5
        h.update_by_incr(4.0, 7).unwrap();  // index 1, count 7
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        // Merge: 5+7=12 ≤ 15, should stay at B4.
        h.do_downscale(1).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4,
            "width should be preserved when pair sums fit");
    }

    #[test]
    fn test_do_downscale_merge_widens_b4() {
        // Insert values that sum to > 15 at B4.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        h.update_by_incr(2.0, 10).unwrap();  // index 0, count 10
        h.update_by_incr(4.0, 10).unwrap();  // index 1, count 10
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        // Merge: 10+10=20 > 15, should widen to U8.
        h.do_downscale(1).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8,
            "width should widen when pair sums overflow");
    }

    #[test]
    fn test_do_downscale_preserves_width_when_possible() {
        // Fill histogram with small counts at many indices.
        // Downscale should merge pairs without widening.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        // Insert 1 at each of several indices (all count=1, sums ≤ 2).
        for i in 0..8 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        let width_before = h.bucket_width();
        h.do_downscale(1).unwrap();
        assert_eq!(h.bucket_width(), width_before,
            "width should be preserved when pair sums fit");
    }

    #[test]
    fn test_do_downscale_widens_on_overflow() {
        // Fill histogram with counts that will overflow on merge.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        h.update_by_incr(2.0, 15).unwrap();  // index 0, count 15
        h.update_by_incr(4.0, 15).unwrap();  // index 1, count 15
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

        let mut merged: Histogram<8, P32> = Histogram::new();
        for &v in &set_a { merged.update(v).unwrap(); }
        let mut other: Histogram<8, P32> = Histogram::new();
        for &v in &set_b { other.update(v).unwrap(); }
        merged.merge_from(&other).unwrap();

        let mut single: Histogram<8, P32> = Histogram::new();
        for &v in &set_a { single.update(v).unwrap(); }
        for &v in &set_b { single.update(v).unwrap(); }

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
        let mut data = [pack_b4x16([1,2, 3,0, 0,0, 15,0, 0,0, 0,0, 0,0, 0,0])];
        swar_step(&mut data, BucketWidth::B4);
        assert_eq!(data[0], pack_u8x8([3, 3, 0, 15, 0, 0, 0, 0]),
            "got {:#018x}", data[0]);
    }

    #[test]
    fn test_swar_step_u8_single_word() {
        // 8 bytes: [100, 200, 50, 50, 0, 0, 0, 0]
        // Pair sums → 4 shorts: [300, 100, 0, 0]
        let mut data = [pack_u8x8([100, 200, 50, 50, 0, 0, 0, 0])];
        swar_step(&mut data, BucketWidth::U8);
        assert_eq!(data[0], pack_u16x4([300, 100, 0, 0]),
            "got {:#018x}", data[0]);
    }

    #[test]
    fn test_swar_step_u16_single_word() {
        // 4 shorts: [1000, 2000, 3000, 4000]
        // Pair sums → 2 ints: [3000, 7000]
        let mut data = [pack_u16x4([1000, 2000, 3000, 4000])];
        swar_step(&mut data, BucketWidth::U16);
        assert_eq!(data[0], pack_u32x2(3000, 7000),
            "got {:#018x}", data[0]);
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
        let mut data = [pack_b4x16([15,15, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0])];
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
        assert_eq!(data[0], expected,
            "got {:#018x}, expected {:#018x}", data[0], expected);
    }

    #[test]
    fn test_swar_narrow_compact_b4_three_words() {
        // 3 words of U8 → 2 compacted + zero the freed word.
        // Word 0: [1,0,0,0,0,0,0,0]
        // Word 1: [0,0,0,0,0,0,0,2]
        // Word 2: [3,0,0,0,0,0,0,4]
        let mut data = [
            pack_u8x8([1,0,0,0,0,0,0,0]),
            pack_u8x8([0,0,0,0,0,0,0,2]),
            pack_u8x8([3,0,0,0,0,0,0,4]),
        ];
        swar_narrow_compact(&mut data, BucketWidth::B4);
        // step_by(2): i=0 → data[0]=narrow(w0)|narrow(w1)<<32
        //             i=2 → data[1]=narrow(w2)|0<<32
        let lo0 = narrow_u8_to_b4(pack_u8x8([1,0,0,0,0,0,0,0]));
        let hi0 = narrow_u8_to_b4(pack_u8x8([0,0,0,0,0,0,0,2]));
        let lo1 = narrow_u8_to_b4(pack_u8x8([3,0,0,0,0,0,0,4]));
        assert_eq!(data[0], lo0 | (hi0 << 32),
            "word 0: got {:#018x}", data[0]);
        assert_eq!(data[1], lo1,
            "word 1: got {:#018x}", data[1]);
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
        assert_eq!(data[0], pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80]),
            "word 0: got {:#018x}", data[0]);
        let expected1 = narrow_u16_to_u8(pack_u16x4([255, 0, 128, 1]));
        assert_eq!(data[1], expected1,
            "word 1: got {:#018x}", data[1]);
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 3).unwrap();  // index 0
        h.update_by_incr(4.0, 5).unwrap();  // index 1

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
        assert_eq!(total_before, total_after,
            "bucket total changed: {total_before} → {total_after}");
        // Small counts (3+5=8 ≤ 15) → should stay at B4.
        assert_eq!(h.bucket_width(), width_before);
    }

    #[test]
    fn test_bucket_downscale_scalar_preserves_total_with_overflow() {
        // Fill enough that pair sums exceed B4 max (15).
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 10).unwrap();  // index 0, count 10
        h.update_by_incr(4.0, 10).unwrap();  // index 1, count 10

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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
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
        assert_eq!(total_before, total_after,
            "total changed after 7-step downscale through odd base");
    }

    // -----------------------------------------------------------------------
    // bucket_widen at odd base
    // -----------------------------------------------------------------------

    #[test]
    fn test_bucket_widen_odd_base_uses_scalar() {
        // Construct a scenario where base is odd, then verify widen works.
        // Start at max scale so we have room to downscale.
        let mut h: Histogram<16, P32> = Histogram::with_scale(8);
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
                assert!(h.bucket_width() > width_before,
                    "width should increase: {:?} → {:?}", width_before, h.bucket_width());

                let total_after: u64 = h.positive().iter().sum();
                assert_eq!(total_before, total_after,
                    "bucket total changed on odd-base widen");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Speculative merge: width preservation across counter magnitudes
    // -----------------------------------------------------------------------

    #[test]
    fn test_speculative_merge_b4_sparse_stays_b4() {
        // Many single-count buckets. All pair sums ≤ 2, so B4 stays.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0).with_min_bucket_width(BucketWidth::B4);
        h.update_by_incr(2.0, 10).unwrap();
        h.update_by_incr(4.0, 10).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);

        h.do_downscale(1).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8,
            "10+10=20 > 15, must widen");
    }

    #[test]
    fn test_speculative_merge_u8_dense_widens_to_u16() {
        // At U8, max = 255. Two adjacent buckets each with count 200 → 400 > 255.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
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
        let mut h: Histogram<16, P32> = Histogram::with_scale(8);
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
            assert_eq!(current, total,
                "total changed at step {step} (width={:?}): {current} != {total}",
                h.bucket_width());
            if h.bucket_width() == BucketWidth::U64 {
                break;
            }
        }
    }

    #[test]
    fn test_sum_conservation_scalar_path() {
        // Force the scalar path and check totals at each step.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        for i in 0..10 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        let total = 10u64;

        // Downscale 8 times — should cross the odd-base boundary
        // multiple times, exercising scalar and SWAR paths alternately.
        for step in 1..=8 {
            h.do_downscale(1).unwrap();
            let current: u64 = h.positive().iter().sum();
            assert_eq!(current, total,
                "total changed at step {step}: {current} != {total}, \
                 width={:?} base={}",
                h.bucket_width(), h.index_base);
        }
    }

    #[test]
    fn test_sum_conservation_large_counts() {
        // High counts that force widening at every merge.
        let mut h: Histogram<16, P32> = Histogram::with_scale(0);
        h.update_by_incr(2.0, 15).unwrap(); // fills B4 to max
        h.update_by_incr(4.0, 15).unwrap();
        h.update_by_incr(8.0, 15).unwrap();
        h.update_by_incr(16.0, 15).unwrap();
        let total: u64 = h.positive().iter().sum();
        assert_eq!(total, 60);

        for step in 1..=6 {
            h.do_downscale(1).unwrap();
            let current: u64 = h.positive().iter().sum();
            assert_eq!(current, total,
                "total changed at step {step}: {current} != {total}, \
                 width={:?}", h.bucket_width());
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
            assert_eq!(nibble, (i + 1).min(15) as u64,
                "nibble {i}: expected {}, got {nibble}", (i + 1).min(15));

            // All other nibbles should be zero.
            for j in 0..8u8 {
                if j != i {
                    let other = (result >> (j as u64 * 4)) & 0xF;
                    assert_eq!(other, 0,
                        "nibble {j} should be 0 when only nibble {i} is set, got {other}");
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
            assert_eq!(byte, (i + 1).min(255) as u64,
                "byte {i}: expected {}, got {byte}", (i + 1).min(255));
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
    fn test_swar_has_overflow_u16_boundary_65536() {
        let data = [pack_u32x2(65536, 0)];
        assert!(swar_has_overflow(&data, BucketWidth::U16));
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
        // Reproducer: insert 1.0..=8.0 into Histogram<8, P32>.
        // At B1 with 6 bucket words (384 slots), the index span forces
        // repeated downscaling. Bucket totals must stay consistent.
        let mut h: Histogram<8, P32> = Histogram::new();
        for i in 1..=8 {
            let v = i as f64;
            h.update(v).unwrap();
            let b = h.positive();
            let total: u64 = (0..b.len()).map(|k| b.at(k)).sum();
            assert_eq!(total, h.count(),
                "After inserting {v}: bucket total ({total}) != count ({})\n  \
                 scale={} width={:?} offset={} len={} cap={}",
                h.count(), h.scale(), b.width(), b.offset(), b.len(), b.capacity());
        }
    }

    #[test]
    fn test_adaptive_downscale_wide_span_small_pool() {
        // Wide value range in a small pool — forces multi-step downscale.
        let mut h: Histogram<8, P32> = Histogram::new();
        let values = [0.001, 1.0, 1000.0, 0.5, 50.0, 0.01, 100.0, 10.0];
        for (vi, &v) in values.iter().enumerate() {
            h.update(v).unwrap();
            let b = h.positive();
            let total: u64 = (0..b.len()).map(|k| b.at(k)).sum();
            assert_eq!(total, h.count(),
                "After values[{vi}]={v}: bucket total ({total}) != count ({})\n  \
                 scale={} width={:?}",
                h.count(), h.scale(), b.width());
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests (formerly in regression_stat_widen)
    // -----------------------------------------------------------------------

    /// Helper: count total across all positive buckets.
    fn bucket_total<const N: usize, P: Precision>(h: &Histogram<N, P>) -> u64 {
        let b = h.positive();
        (0..b.len()).map(|i| b.at(i)).sum()
    }

    #[test]
    fn test_merge_needs_downscale_in_raw() {
        let mut h1 = Histogram::<8, P32>::new();
        h1.update(1.0).unwrap();

        let mut h2 = Histogram::<8, P32>::new();
        h2.update(1e30).unwrap();
        h2.update(1e-30).unwrap();

        let b2 = h2.positive();
        h1.merge_from_raw(
            h2.count(),
            h2.sum(),
            h2.min(),
            h2.max(),
            h2.scale(),
            b2.offset(),
            b2.len(),
            &|i| b2.at(i),
        )
        .unwrap();
        assert_eq!(h1.count(), 3);
        assert_eq!(bucket_total(&h1), 3);
    }

    /// Regression: large weighted inserts of subnormal + normal value
    /// trigger bucket_widen during downscale, corrupting bucket totals
    /// when scalar_merge_step produced len > cap.
    #[test]
    fn test_weighted_subnormal_merge_bucket_total() {
        let v1 = f64::from_le_bytes([32, 0, 66, 0, 0, 98, 65, 3]); // ~5.44e-293, subnormal as f32
        let v2 = f64::from_le_bytes([0, 32, 0, 66, 0, 98, 65, 64]); // ~34.77

        let left_ops: Vec<(f64, u64)> = vec![
            (v1, 3), (v2, 1), (v1, 12), (v2, 4), (v1, 192), (v2, 64),
        ];
        let right_ops: Vec<(f64, u64)> = vec![
            (v1, 3072), (v2, 1024),
        ];

        fn check<const N: usize, P: Precision>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
            let mut h1 = Histogram::<N, P>::new();
            for &(v, incr) in left {
                h1.update_by_incr(v, incr).unwrap();
            }
            let mut h2 = Histogram::<N, P>::new();
            for &(v, incr) in right {
                h2.update_by_incr(v, incr).unwrap();
            }
            h1.merge_from(&h2).unwrap();
            let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
            assert_eq!(h1.count(), expected, "{label}: count mismatch");
            let bt = { let b = h1.positive(); (0..b.len()).map(|i| b.at(i)).sum::<u64>() };
            assert!(bt <= h1.count(), "{label}: bucket total ({bt}) exceeds count ({})", h1.count());
        }

        fn check_cross<const N: usize, P: Precision, const M: usize>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
            let mut h1 = Histogram::<N, P>::new();
            for &(v, incr) in left {
                h1.update_by_incr(v, incr).unwrap();
            }
            let mut h2 = Histogram::<M, P>::new();
            for &(v, incr) in right {
                h2.update_by_incr(v, incr).unwrap();
            }
            h1.merge_from_other(&h2).unwrap();
            let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
            assert_eq!(h1.count(), expected, "{label}: count mismatch");
            let bt = { let b = h1.positive(); (0..b.len()).map(|i| b.at(i)).sum::<u64>() };
            assert!(bt <= h1.count(), "{label}: bucket total ({bt}) exceeds count ({})", h1.count());
        }

        check::<8, P32>(&left_ops, &right_ops, "same N=8");
        check::<16, P32>(&left_ops, &right_ops, "same N=16");
        check_cross::<8, P32, 16>(&left_ops, &right_ops, "cross 8←16");
        check_cross::<16, P32, 8>(&left_ops, &right_ops, "cross 16←8");
    }

    /// Regression: three values with a subnormal, split across merge,
    /// with echo-amplified increments.
    #[test]
    fn test_three_vals_with_subnormal_echo() {
        let v1 = f64::from_le_bytes([22, 22, 0, 237, 237, 59, 59, 59]); // ~2.25e-23
        let v2 = f64::from_le_bytes([59, 59, 1, 0, 59, 31, 0, 0]);     // ~1.70e-310, subnormal as f32
        let v3 = f64::from_le_bytes([0, 59, 237, 237, 64, 0, 122, 64]); // ~416.0

        let left: Vec<(f64, u64)> = vec![(v1, 300), (v2, 5)];
        let right: Vec<(f64, u64)> = vec![
            (v3, 5), (v1, 1200), (v2, 20), (v3, 20), (v1, 19200), (v2, 320), (v3, 320),
        ];

        fn check<const N: usize, P: Precision>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
            let mut h1 = Histogram::<N, P>::new();
            for &(v, incr) in left {
                h1.update_by_incr(v, incr).unwrap();
            }
            let mut h2 = Histogram::<N, P>::new();
            for &(v, incr) in right {
                h2.update_by_incr(v, incr).unwrap();
            }
            h1.merge_from(&h2).unwrap();
            let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
            assert_eq!(h1.count(), expected, "{label}: count");
            let bt = { let b = h1.positive(); (0..b.len()).map(|i| b.at(i)).sum::<u64>() };
            assert!(bt <= h1.count(), "{label}: bt={bt} > count={}", h1.count());
        }

        fn check_cross<const N: usize, P: Precision, const M: usize>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
            let mut h1 = Histogram::<N, P>::new();
            for &(v, incr) in left {
                h1.update_by_incr(v, incr).unwrap();
            }
            let mut h2 = Histogram::<M, P>::new();
            for &(v, incr) in right {
                h2.update_by_incr(v, incr).unwrap();
            }
            h1.merge_from_other(&h2).unwrap();
            let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
            assert_eq!(h1.count(), expected, "{label}: count");
            let bt = { let b = h1.positive(); (0..b.len()).map(|i| b.at(i)).sum::<u64>() };
            assert!(bt <= h1.count(), "{label}: bt={bt} > count={}", h1.count());
        }

        check::<8, P32>(&left, &right, "same 8");
        check::<16, P32>(&left, &right, "same 16");
        check_cross::<8, P32, 16>(&left, &right, "cross 8←16");
        check_cross::<16, P32, 8>(&left, &right, "cross 16←8");
    }

    #[test]
    fn test_merge_p64_bucket_total_exceeds_count() {
        let mut h0 = Histogram::<8, P64>::new();
        let mut h1 = Histogram::<8, P64>::new();

        h1.update_by_incr(2.8396262443943004e+238, 40).unwrap();
        h0.update_by_incr(2.635549485807631e-82, 1).unwrap();

        // Step 3: merge h0 into h1
        h1.merge_from(&h0).unwrap();
        assert_eq!(h1.count(), 41);

        // Step 4: merge h1 into h0
        if h0.merge_from(&h1).is_ok() {
            let b = h0.positive();
            let bt: u64 = (0..b.len()).map(|i| b.at(i)).sum();
            assert!(bt <= h0.count(),
                "bucket total ({bt}) exceeds count ({})", h0.count());
        }
    }

    #[test]
    fn test_merge_p32_bucket_len_after_merge_chain() {
        use crate::Mapping;

        let v0: f64 = 5.653943197254256e-308;
        let v1: f64 = 2.740490672504645e-61;

        let mut h0 = Histogram::<8, P32>::new();
        let mut h1 = Histogram::<8, P32>::new();

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
        let b = h0.positive();
        let scale = h0.scale();
        let mapping = Mapping::new(scale).unwrap();

        // All non-zero values should map to indices at the current scale
        let idx0 = mapping.map_to_index(v0);
        let idx1 = mapping.map_to_index(v1);
        let exp_min = idx0.min(idx1);
        let exp_max = idx0.max(idx1);
        let exp_len = (exp_max - exp_min + 1) as u32;

        assert_eq!(b.offset(), exp_min,
            "offset mismatch: got {} expected {} (scale={})", b.offset(), exp_min, scale);
        assert_eq!(b.len(), exp_len,
            "len mismatch: got {} expected {} (scale={}, idx0={}, idx1={})",
            b.len(), exp_len, scale, idx0, idx1);

        // No trailing/leading zero buckets
        if b.len() > 0 {
            assert!(b.at(0) > 0, "leading zero bucket");
            assert!(b.at(b.len() - 1) > 0, "trailing zero bucket");
        }
    }
}
