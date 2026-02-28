// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram with a unified flat memory layout.
//!
//! `Histogram<N>` stores everything in fixed struct fields plus a `[u64; N]`
//! data pool. The pool is split between auto-widening MMZSC fields
//! (min/max/sum/count/zero_count) at the front and bucket counters in the
//! remainder. Both the MMZSC fields and bucket counters widen in place
//! when they saturate, without allocation.
//!
//! Bucket counters start at 1-bit and widen through the chain
//! 1→2→4→8→16→32→64 bits via combined downscale+widen when a counter
//! saturates. Sub-byte transitions use parallel bit-sum (SWAR) — the
//! popcount algorithm's building blocks.

use core::fmt;

use crate::mapping::{Mapping, max_scale};

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
/// Counters start at 4-bit (maximizing initial bucket count) and widen
/// in place through the chain: 4→8→16→32→64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BucketWidth {
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

    /// Returns the next wider counter width, or `None` if already at u64.
    #[inline]
    const fn wider(self) -> Option<BucketWidth> {
        match self {
            Self::B4 => Some(Self::U8),
            Self::U8 => Some(Self::U16),
            Self::U16 => Some(Self::U32),
            Self::U32 => Some(Self::U64),
            Self::U64 => None,
        }
    }

    /// Returns true if this width is sub-byte (packed within bytes).
    #[inline]
    const fn is_sub_byte(self) -> bool {
        (self as u8) < 8
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    const fn counter_max(self) -> u64 {
        match self {
            Self::B4 => 15,
            Self::U8 => u8::MAX as u64,
            Self::U16 => u16::MAX as u64,
            Self::U32 => u32::MAX as u64,
            Self::U64 => u64::MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// StatWidth — width of MMZSC fields in the data pool
// ---------------------------------------------------------------------------

/// Width of the MMZSC (min/max/sum/count/zero_count) fields.
///
/// Starts at S32 (20 bytes packed into 3 u64 words) and auto-widens to
/// S64 (40 bytes in 5 u64 words) when count or zero_count exceeds `u32::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StatWidth {
    /// 4-byte fields: f32 for sum/min/max, u32 for count/zero_count.
    /// Packed into 3 u64 words.
    S32 = 4,
    /// 8-byte fields: f64 for sum/min/max, u64 for count/zero_count.
    /// Uses 5 u64 words.
    S64 = 8,
}

impl StatWidth {
    /// Number of u64 words consumed by MMZSC fields at this width.
    #[inline]
    const fn words(self) -> usize {
        match self {
            Self::S32 => 3,
            Self::S64 => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// MMZSC accessor helpers — read/write stats from the data pool
// ---------------------------------------------------------------------------

/// S32 layout (3 words):
///   word 0: [sum:f32 (lo32)] [count:u32 (hi32)]
///   word 1: [zero_count:u32 (lo32)] [min:f32 (hi32)]
///   word 2: [max:f32 (lo32)] [pad:32]
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
/// auto-widening MMZSC fields (min/max/sum/count/zero_count) at the front
/// and bucket counter data in the remainder.
///
/// MMZSC fields start at 4-byte width (3 words) and auto-widen to 8-byte
/// (5 words) when count or zero_count exceeds `u32::MAX`.
///
/// Bucket counters start at 1-bit and auto-widen in place
/// (1→2→4 bits → u8 → u16 → u32 → u64) via combined downscale+widen
/// when a counter saturates.
///
/// At minimum, `N` should be 8 (64 bytes of pool), giving 5 bucket words
/// at S32 (80 B4 buckets) down to 3 bucket words at S64 (3 U64 buckets).
#[derive(Clone)]
pub struct Histogram<const N: usize> {
    // -- Fixed metadata (never relocates) --
    mapping: Mapping,
    max_scale: i8,
    min_bucket_width: BucketWidth,
    bucket_width: BucketWidth,
    stat_width: StatWidth,
    index_base: i32,
    index_start: i32,
    index_end: i32,

    // -- Data pool: MMZSC at front, buckets after --
    data: [u64; N],
}

impl<const N: usize> fmt::Debug for Histogram<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Histogram")
            .field("scale", &self.scale())
            .field("stat_width", &self.stat_width)
            .field("bucket_width", &self.bucket_width)
            .field("count", &self.count())
            .field("sum", &self.sum())
            .field("min", &self.min())
            .field("max", &self.max())
            .field("zero_count", &self.zero_count())
            .field("bucket_len", &self.bucket_len())
            .finish()
    }
}

impl<const N: usize> Default for Histogram<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MMZSC read/write methods
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
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
    fn zero_count_s32(&self) -> u32 {
        read_lo32(self.data[1])
    }

    #[inline]
    fn min_s32(&self) -> f32 {
        f32::from_bits(read_hi32(self.data[1]))
    }

    #[inline]
    fn max_s32(&self) -> f32 {
        f32::from_bits(read_lo32(self.data[2]))
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
    fn set_zero_count_s32(&mut self, v: u32) {
        write_lo32(&mut self.data[1], v);
    }

    #[inline]
    fn set_min_s32(&mut self, v: f32) {
        write_hi32(&mut self.data[1], v.to_bits());
    }

    #[inline]
    fn set_max_s32(&mut self, v: f32) {
        write_lo32(&mut self.data[2], v.to_bits());
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
    fn zero_count_s64(&self) -> u64 {
        self.data[2]
    }

    #[inline]
    fn min_s64(&self) -> f64 {
        f64::from_bits(self.data[3])
    }

    #[inline]
    fn max_s64(&self) -> f64 {
        f64::from_bits(self.data[4])
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
    fn set_zero_count_s64(&mut self, v: u64) {
        self.data[2] = v;
    }

    #[inline]
    fn set_min_s64(&mut self, v: f64) {
        self.data[3] = v.to_bits();
    }

    #[inline]
    fn set_max_s64(&mut self, v: f64) {
        self.data[4] = v.to_bits();
    }

    // -- Width-dispatched public readers --

    /// Returns the sum of all recorded values as `f64`.
    #[inline]
    pub fn sum(&self) -> f64 {
        match self.stat_width {
            StatWidth::S32 => self.sum_s32() as f64,
            StatWidth::S64 => self.sum_s64(),
        }
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        match self.stat_width {
            StatWidth::S32 => self.count_s32() as u64,
            StatWidth::S64 => self.count_s64(),
        }
    }

    /// Returns the count of zero values.
    #[inline]
    pub fn zero_count(&self) -> u64 {
        match self.stat_width {
            StatWidth::S32 => self.zero_count_s32() as u64,
            StatWidth::S64 => self.zero_count_s64(),
        }
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        match self.stat_width {
            StatWidth::S32 => self.min_s32() as f64,
            StatWidth::S64 => self.min_s64(),
        }
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
        match self.stat_width {
            StatWidth::S32 => self.max_s32() as f64,
            StatWidth::S64 => self.max_s64(),
        }
    }

    // -- Width-dispatched internal writers --

    #[inline]
    fn set_sum(&mut self, v: f64) {
        match self.stat_width {
            StatWidth::S32 => self.set_sum_s32(v as f32),
            StatWidth::S64 => self.set_sum_s64(v),
        }
    }

    #[inline]
    fn set_count(&mut self, v: u64) {
        match self.stat_width {
            StatWidth::S32 => self.set_count_s32(v as u32),
            StatWidth::S64 => self.set_count_s64(v),
        }
    }

    #[inline]
    fn set_zero_count(&mut self, v: u64) {
        match self.stat_width {
            StatWidth::S32 => self.set_zero_count_s32(v as u32),
            StatWidth::S64 => self.set_zero_count_s64(v),
        }
    }

    #[inline]
    fn set_min(&mut self, v: f64) {
        match self.stat_width {
            StatWidth::S32 => self.set_min_s32(v as f32),
            StatWidth::S64 => self.set_min_s64(v),
        }
    }

    #[inline]
    fn set_max(&mut self, v: f64) {
        match self.stat_width {
            StatWidth::S32 => self.set_max_s32(v as f32),
            StatWidth::S64 => self.set_max_s64(v),
        }
    }

    #[inline]
    fn add_sum(&mut self, v: f64) {
        self.set_sum(self.sum() + v);
    }

    /// Checked increment of count by `incr`. Returns `None` on overflow.
    #[inline]
    fn checked_add_count(&self, incr: u64) -> Option<u64> {
        match self.stat_width {
            StatWidth::S32 => {
                let c = self.count_s32();
                let i = u32::try_from(incr).ok()?;
                c.checked_add(i).map(|v| v as u64)
            }
            StatWidth::S64 => self.count_s64().checked_add(incr),
        }
    }

    /// Checked increment of zero_count by `incr`. Returns `None` on overflow.
    #[inline]
    fn checked_add_zero_count(&self, incr: u64) -> Option<u64> {
        match self.stat_width {
            StatWidth::S32 => {
                let c = self.zero_count_s32();
                let i = u32::try_from(incr).ok()?;
                c.checked_add(i).map(|v| v as u64)
            }
            StatWidth::S64 => self.zero_count_s64().checked_add(incr),
        }
    }
}

// ---------------------------------------------------------------------------
// Bucket data access — operates on the bucket slice of the data pool
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Returns the start index of bucket data within the data pool.
    #[inline]
    fn bucket_data_start(&self) -> usize {
        self.stat_width.words()
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
        let mut slot = self.index_start - self.index_base;
        if slot < 0 {
            slot += self.bucket_capacity() as i32;
        }
        self.bucket_get(slot as usize) == 0
    }

    /// Gets the value at a physical slot index.
    #[inline]
    fn bucket_get(&self, slot: usize) -> u64 {
        let data = self.bucket_data();
        match self.bucket_width {
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
        let bias = (self.index_base - self.index_start) as u32;
        let cap = self.bucket_capacity() as u32;
        let mut idx = pos;
        if idx < bias {
            idx += cap;
        }
        idx -= bias;
        self.bucket_get(idx as usize)
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
pub struct BucketView<'a, const N: usize> {
    hist: &'a Histogram<N>,
}

impl<const N: usize> BucketView<'_, N> {
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

impl<const N: usize> ExactSizeIterator for BucketsIter<'_, N> {}

// ---------------------------------------------------------------------------
// Bucket operations — rotate, downscale, widen
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Rotates the circular buffer so that index_start == index_base.
    fn bucket_rotate(&mut self) {
        let bias = (self.index_base - self.index_start) as usize;
        if bias == 0 {
            return;
        }
        let cap = self.bucket_capacity();
        debug_assert!(bias < cap, "rotate bias {} >= capacity {}", bias, cap);

        let width = self.bucket_width;
        if width.is_sub_byte() {
            let bit_shift = bias * width.bits();
            let data = self.bucket_data_mut();
            bit_rotate_right(data, bit_shift);
        } else {
            let data = self.bucket_data_mut();
            match width {
                BucketWidth::U8 => {
                    let s: &mut [u8] = bytemuck::cast_slice_mut(data);
                    s[..cap].rotate_right(bias);
                }
                BucketWidth::U16 => {
                    let s: &mut [u16] = bytemuck::cast_slice_mut(data);
                    s[..cap].rotate_right(bias);
                }
                BucketWidth::U32 => {
                    let s: &mut [u32] = bytemuck::cast_slice_mut(data);
                    s[..cap].rotate_right(bias);
                }
                BucketWidth::U64 => {
                    data[..cap].rotate_right(bias);
                }
                _ => unreachable!(),
            }
        }
        self.index_base = self.index_start;
    }

    /// Downscales by collapsing 2^by adjacent buckets into 1.
    ///
    /// Operates in place: linearizes the circular buffer, then checks
    /// for overflow in a dry-run pass before merging. After rotation
    /// the output position always trails the input, so the merge is
    /// safe to perform without a copy.
    ///
    /// Returns `false` if combining buckets would overflow the counter
    /// type. On failure the buffer is rotated but data is intact.
    fn bucket_downscale(&mut self, by: i32) -> bool {
        if self.is_effectively_empty() {
            self.index_start >>= by;
            self.index_end >>= by;
            self.index_base = self.index_start;
            return true;
        }

        self.bucket_rotate();

        let size = (1 + self.index_end - self.index_start) as usize;
        let each = 1usize << by;
        let max = self.bucket_width.counter_max();

        // Pass 1: check that no group sum overflows the counter width.
        {
            let mut inpos = 0usize;
            let mut pos = self.index_start;
            while inpos < size && pos <= self.index_end {
                let mod_val = (pos as i64).rem_euclid(each as i64) as usize;
                let mut group_sum = 0u64;
                let mut j = mod_val;
                while j < each && inpos < size {
                    group_sum += self.bucket_get(inpos);
                    inpos += 1;
                    pos += 1;
                    j += 1;
                }
                if group_sum > max {
                    return false;
                }
            }
        }

        // Pass 2: merge in place (output always trails input).
        let mut inpos = 0usize;
        let mut outpos = 0usize;
        let mut pos = self.index_start;

        while inpos < size && pos <= self.index_end {
            let mod_val = (pos as i64).rem_euclid(each as i64) as usize;
            let mut group_sum = 0u64;
            let mut j = mod_val;
            while j < each && inpos < size {
                group_sum += self.bucket_get(inpos);
                inpos += 1;
                pos += 1;
                j += 1;
            }
            self.bucket_set(outpos, group_sum);
            outpos += 1;
        }

        // Zero unused slots.
        let cap = self.bucket_capacity();
        for s in outpos..cap {
            self.bucket_set(s, 0);
        }

        self.index_start >>= by;
        self.index_end >>= by;
        self.index_base = self.index_start;

        true
    }

    /// In-place widen: linearize the circular buffer, group-sum adjacent
    /// counters, then reinterpret the memory as the next wider counter type.
    ///
    /// This operation is infallible because pairwise sums of values at width W
    /// always fit in width 2W (e.g. max B4 pair-sum = 30 ≤ U8 max 255).
    /// No clone needed.
    ///
    /// Returns `None` if already at u64 width (no further widening possible).
    /// Returns `Some(by)` with the actual downscale amount on success.
    fn bucket_widen_in_place(&mut self) -> Option<i32> {
        let new_width = self.bucket_width.wider()?;

        self.bucket_rotate();

        let used = if self.is_effectively_empty() {
            0
        } else {
            (self.index_end - self.index_start + 1) as usize
        };

        let new_cap = new_width.capacity(self.bucket_word_count());

        // Determine downscale amount.
        let by: i32 = if used == 0 {
            1
        } else {
            let mut b = 1;
            loop {
                let ds_start = self.index_start >> b;
                let ds_end = self.index_end >> b;
                if (ds_end - ds_start + 1) as usize <= new_cap {
                    break;
                }
                b += 1;
            }
            b
        };

        if self.bucket_width == BucketWidth::B4 && by == 1 {
            // B4→U8: one SWAR step (4-bit→8-bit)
            let data = self.bucket_data_mut();
            parallel_pairwise_sum(data, BucketWidth::B4);

            let new_used = (used + 1) / 2;
            self.bucket_width = new_width;
            for s in new_used..new_cap {
                self.bucket_set(s, 0);
            }

            self.index_start >>= by;
            self.index_end >>= by;
            self.index_base = self.index_start;
            Some(by)
        } else {
            // General path: sequential group-sum. Output position always
            // trails input position, so we can safely overwrite in place.
            let old_width = self.bucket_width;
            let mut out_slot = 0usize;
            let mut i = 0usize;
            while i < used {
                let old_index = self.index_start + i as i32;
                let new_index = old_index >> by;
                let mut sum = self.bucket_get(i);
                i += 1;
                while i < used {
                    let next_old_index = self.index_start + i as i32;
                    if (next_old_index >> by) != new_index {
                        break;
                    }
                    sum = sum.saturating_add(self.bucket_get(i));
                    i += 1;
                }
                self.bucket_width = new_width;
                self.bucket_set(out_slot, sum);
                self.bucket_width = old_width;
                out_slot += 1;
            }

            self.bucket_width = new_width;
            for s in out_slot..new_cap {
                self.bucket_set(s, 0);
            }

            self.index_start >>= by;
            self.index_end >>= by;
            self.index_base = self.index_start;
            Some(by)
        }
    }
}

// ---------------------------------------------------------------------------
// Bit-level rotate and SWAR helpers (free functions on slices)
// ---------------------------------------------------------------------------

/// Bit-level rotate right of a `[u64]` slice by `shift` bits.
///
/// Matches the semantics of `[T]::rotate_right`: the element (bit) at
/// flat position `p` moves to `(p + shift) % (len*64)`.
fn bit_rotate_right(data: &mut [u64], shift: usize) {
    let n = data.len();
    let total_bits = n * 64;
    let shift = shift % total_bits;
    if shift == 0 {
        return;
    }

    let word_shift = shift / 64;
    let bit_shift = (shift % 64) as u32;

    if word_shift > 0 {
        data.rotate_right(word_shift);
    }

    if bit_shift > 0 {
        let saved = data[n - 1] >> (64 - bit_shift);
        for i in (1..n).rev() {
            data[i] = (data[i] << bit_shift) | (data[i - 1] >> (64 - bit_shift));
        }
        data[0] = (data[0] << bit_shift) | saved;
    }
}

/// Parallel pairwise sum: sum adjacent N-bit fields into 2N-bit fields.
#[inline]
fn parallel_pairwise_sum(data: &mut [u64], width: BucketWidth) {
    match width {
        BucketWidth::B4 => {
            const MASK: u64 = 0x0F0F_0F0F_0F0F_0F0F;
            for w in data.iter_mut() {
                let x = *w;
                *w = ((x >> 4) & MASK) + (x & MASK);
            }
        }
        _ => unreachable!("parallel_pairwise_sum only for B4 width"),
    }
}

// ---------------------------------------------------------------------------
// Stat widening — S32 → S64
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Widens MMZSC fields from S32 to S64.
    ///
    /// Shifts bucket data right by 2 words to make room for wider
    /// stat fields, downscaling buckets first if needed. Operates
    /// entirely in place.
    fn stat_widen(&mut self) -> Result<(), Overflow> {
        debug_assert_eq!(self.stat_width, StatWidth::S32);

        // Read current S32 values before the layout changes.
        let sum = self.sum();
        let count = self.count();
        let zero_count = self.zero_count();
        let min = self.min();
        let max = self.max();

        let old_start = StatWidth::S32.words(); // 3
        let new_start = StatWidth::S64.words(); // 5
        let new_bucket_words = N - new_start;

        // Downscale or rotate buckets so they fit in the smaller area.
        if !self.is_effectively_empty() {
            let new_cap = self.bucket_width.capacity(new_bucket_words);
            let used = (self.index_end - self.index_start + 1) as usize;
            if used > new_cap {
                let mut ds = 1;
                while (used + (1 << ds) - 1) >> ds > new_cap {
                    ds += 1;
                }
                if !self.bucket_downscale(ds) {
                    return Err(Overflow);
                }
            } else {
                self.bucket_rotate();
            }
        }

        // Shift bucket data right by 2 words within the data pool.
        let shift_words = new_bucket_words.min(N - old_start);
        self.data.copy_within(old_start..old_start + shift_words, new_start);

        // Zero the gap (words old_start..new_start).
        for i in old_start..new_start {
            self.data[i] = 0;
        }

        // Write MMZSC at S64.
        self.stat_width = StatWidth::S64;
        self.set_sum_s64(sum);
        self.set_count_s64(count);
        self.set_zero_count_s64(zero_count);
        self.set_min_s64(min);
        self.set_max_s64(max);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Histogram<N> — construction and public API
// ---------------------------------------------------------------------------

impl<const N: usize> Histogram<N> {
    /// Creates a new histogram at the maximum supported scale.
    #[inline]
    pub fn new() -> Self {
        let scale = max_scale();
        Self {
            mapping: Mapping::new(scale).unwrap(),
            max_scale: scale as i8,
            min_bucket_width: BucketWidth::B4,
            bucket_width: BucketWidth::B4,
            stat_width: StatWidth::S32,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            data: [0u64; N],
        }
    }

    /// Creates a new histogram with an upper bound on scale.
    #[inline]
    pub fn with_max_scale(scale: i32) -> Self {
        let scale = scale.min(max_scale());
        Self {
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale as i8,
            min_bucket_width: BucketWidth::B4,
            bucket_width: BucketWidth::B4,
            stat_width: StatWidth::S32,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            data: [0u64; N],
        }
    }

    /// Creates a new histogram at the specified scale.
    #[inline]
    pub fn with_scale(scale: i32) -> Self {
        Self {
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale as i8,
            min_bucket_width: BucketWidth::B4,
            bucket_width: BucketWidth::B4,
            stat_width: StatWidth::S32,
            index_base: 0,
            index_start: 0,
            index_end: 0,
            data: [0u64; N],
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
        if self.count() == self.zero_count() {
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

    /// Returns the current MMZSC field width.
    #[inline]
    pub fn stat_width(&self) -> StatWidth {
        self.stat_width
    }

    /// Returns a read-only view of the positive buckets.
    #[inline]
    pub fn positive(&self) -> BucketView<'_, N> {
        BucketView { hist: self }
    }

    /// Clears the histogram, resetting to initial state.
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.bucket_width = self.min_bucket_width;
        self.stat_width = StatWidth::S32;
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
        let new_count = match self.checked_add_count(incr) {
            Some(c) => c,
            None => {
                // S32 overflow — try widening stats to S64.
                if self.stat_width == StatWidth::S32 {
                    self.stat_widen()?;
                    self.checked_add_count(incr).ok_or(Overflow)?
                } else {
                    return Err(Overflow);
                }
            }
        };

        if value == 0.0 {
            let new_zc = match self.checked_add_zero_count(incr) {
                Some(c) => c,
                None => {
                    if self.stat_width == StatWidth::S32 {
                        self.stat_widen()?;
                        self.checked_add_zero_count(incr).ok_or(Overflow)?
                    } else {
                        return Err(Overflow);
                    }
                }
            };
            self.set_zero_count(new_zc);
        } else {
            self.update_buckets(value, incr)?;
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
                    if !self.downscale(change) {
                        if !self.widen_and_retry_downscale(change) {
                            return Err(Overflow);
                        }
                    }
                }
                IncrResult::CounterOverflow => {
                    let by = match self.bucket_widen_in_place() {
                        Some(by) => by,
                        None => return Err(Overflow),
                    };
                    let new_scale = self.mapping.scale() - by;
                    self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                }
            }
        }
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
            self.index_base = index;
        } else if index < self.index_start {
            let span = self.index_end.saturating_sub(index);
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.index_end,
                });
            }
            for idx in index..self.index_start {
                let mut bi = idx - self.index_base;
                if bi < 0 { bi += max_size; }
                self.bucket_set(bi as usize, 0);
            }
            self.index_start = index;
        } else if index > self.index_end {
            let span = index.saturating_sub(self.index_start);
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: self.index_start,
                    high: index,
                });
            }
            for idx in (self.index_end + 1)..=index {
                let mut bi = idx - self.index_base;
                if bi < 0 { bi += max_size; }
                self.bucket_set(bi as usize, 0);
            }
            self.index_end = index;
        }

        let mut bucket_index = index - self.index_base;
        if bucket_index < 0 {
            bucket_index += max_size;
        }

        if !self.bucket_try_increment(bucket_index as usize, incr) {
            return IncrResult::CounterOverflow;
        }

        IncrResult::Ok
    }

    /// Downscales the histogram by the given amount.
    fn downscale(&mut self, change: i32) -> bool {
        if change == 0 {
            return true;
        }
        debug_assert!(change > 0, "cannot upscale");

        let new_scale = self.mapping.scale() - change;
        if !self.bucket_downscale(change) {
            return false;
        }
        self.mapping = match Mapping::new(new_scale) {
            Ok(m) => m,
            Err(_) => return false,
        };
        true
    }

    /// Attempts to widen then downscale.
    fn widen_and_retry_downscale(&mut self, total_change: i32) -> bool {
        let by = match self.bucket_widen_in_place() {
            Some(by) => by,
            None => return false,
        };

        let new_scale = self.mapping.scale() - by;
        self.mapping = match Mapping::new(new_scale) {
            Ok(m) => m,
            Err(_) => return false,
        };

        let remaining = total_change - by;
        if remaining > 0 {
            if !self.bucket_downscale(remaining) {
                return self.widen_and_retry_downscale(remaining);
            }
            let new_scale = self.mapping.scale() - remaining;
            self.mapping = match Mapping::new(new_scale) {
                Ok(m) => m,
                Err(_) => return false,
            };
        }
        true
    }

    // -- Merge --

    /// Merges another histogram (same N) into this one.
    pub fn merge_from(&mut self, other: &Self) -> Result<(), Overflow> {
        if other.count() == 0 {
            return Ok(());
        }

        let new_count = self.count().checked_add(other.count()).ok_or(Overflow)?;
        let new_zero_count = self.zero_count().checked_add(other.zero_count()).ok_or(Overflow)?;

        // Stat-widen if needed.
        if self.stat_width == StatWidth::S32 {
            if new_count > u32::MAX as u64 || new_zero_count > u32::MAX as u64 {
                self.stat_widen()?;
            }
        }

        if !other.buckets_empty() {
            if self.buckets_empty() {
                self.bucket_width = self.bucket_width.max(other.bucket_width);
            }

            let min_scale = self.mapping.scale().min(other.scale());
            let cap = self.bucket_capacity() as i32;

            let hlp = self.high_low_at_scale(min_scale)
                .merge(Self::high_low_at_scale_of(other, other.mapping.scale(), min_scale));

            let min_scale = min_scale - change_scale(hlp, cap);

            if !self.downscale_to(min_scale) {
                if !self.widen_and_retry_downscale(self.mapping.scale() - min_scale) {
                    return Err(Overflow);
                }
            }

            self.merge_buckets_from(other, other.mapping.scale())?;
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
        self.set_sum(self.sum() + other.sum());
        self.set_count(new_count);
        self.set_zero_count(new_zero_count);
        Ok(())
    }

    /// Merges from raw histogram data, enabling cross-size merging.
    pub fn merge_from_raw(
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
    ) -> Result<(), Overflow> {
        if other_count == 0 {
            return Ok(());
        }

        let new_count = self.count().checked_add(other_count).ok_or(Overflow)?;
        let new_zero_count = self.zero_count().checked_add(other_zero_count).ok_or(Overflow)?;

        if self.stat_width == StatWidth::S32 {
            if new_count > u32::MAX as u64 || new_zero_count > u32::MAX as u64 {
                self.stat_widen()?;
            }
        }

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

            if !self.downscale_to(min_scale) {
                if !self.widen_and_retry_downscale(self.mapping.scale() - min_scale) {
                    return Err(Overflow);
                }
            }

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
                            let by = match self.bucket_widen_in_place() {
                                Some(by) => by,
                                None => return Err(Overflow),
                            };
                            let new_scale = self.mapping.scale() - by;
                            self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                        }
                        IncrResult::NeedsDownscale(_) => {
                            debug_assert!(false, "incorrect merge scale in merge_from_raw");
                            return Err(Overflow);
                        }
                    }
                }
            }
        }

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
        self.set_sum(self.sum() + other_sum);
        self.set_count(new_count);
        self.set_zero_count(new_zero_count);
        Ok(())
    }

    /// Merges a histogram of a different size into this one.
    pub fn merge_from_other<const M: usize>(
        &mut self,
        other: &Histogram<M>,
    ) -> Result<(), Overflow> {
        self.merge_from_raw(
            other.count(),
            other.zero_count(),
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

    fn high_low_at_scale_of(hist: &Histogram<N>, current_scale: i32, target_scale: i32) -> HighLow {
        if hist.buckets_empty() {
            return HighLow::empty();
        }
        let shift = current_scale - target_scale;
        HighLow {
            low: hist.index_start >> shift,
            high: hist.index_end >> shift,
        }
    }

    fn merge_buckets_from(&mut self, other: &Histogram<N>, other_scale: i32) -> Result<(), Overflow> {
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
                        let by = match self.bucket_widen_in_place() {
                            Some(by) => by,
                            None => return Err(Overflow),
                        };
                        let new_scale = self.mapping.scale() - by;
                        self.mapping = Mapping::new(new_scale).map_err(|_| Overflow)?;
                    }
                    IncrResult::NeedsDownscale(hl) => {
                        let change = change_scale(hl, self.bucket_capacity() as i32);
                        if !self.downscale(change) {
                            if !self.widen_and_retry_downscale(change) {
                                return Err(Overflow);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn downscale_to(&mut self, target_scale: i32) -> bool {
        let change = self.mapping.scale() - target_scale;
        if change <= 0 {
            return true;
        }
        self.downscale(change)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(1.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum(), 1.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
        assert_eq!(h.zero_count(), 0);
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_eq!(h.stat_width(), StatWidth::S32);
    }

    #[test]
    fn test_histogram_zero() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(0.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.zero_count(), 1);
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
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_eq!(h.stat_width(), StatWidth::S32);
    }

    #[test]
    fn test_buckets_at() {
        let mut h: Histogram<16> = Histogram::with_scale(0);
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
        let mut h: Histogram<16> = Histogram::new();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update_by_incr(1.0, 15).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 16);
    }

    #[test]
    fn test_auto_widen_b4_to_u8_from_b4_start() {
        let mut h: Histogram<16> = Histogram::new();
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
        let mut h: Histogram<16> = Histogram::new();
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
        let mut h: Histogram<16> = Histogram::new();
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
        let mut h: Histogram<16> = Histogram::with_scale(0);
        let initial_cap = h.bucket_capacity();
        assert_eq!(initial_cap, 13 * 16); // 208

        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.bucket_capacity(), 13 * 8); // 104
    }

    #[test]
    fn test_clear_resets_to_b4() {
        let mut h: Histogram<16> = Histogram::with_max_scale(3);
        h.update_by_incr(1.0, 16).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.clear();
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale() {
        let h: Histogram<16> = Histogram::with_max_scale(3);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale_clamps() {
        let h: Histogram<16> = Histogram::with_max_scale(100);
        assert_eq!(h.max_scale(), max_scale());
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
    fn test_clear_resets_to_max_scale() {
        let mut h: Histogram<16> = Histogram::with_max_scale(3);
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
                for &v in set_b {
                    other.update(v).unwrap();
                }
                merged.merge_from(&other).unwrap();

                let mut single: Histogram<K> = Histogram::new();
                for &v in set_a {
                    single.update(v).unwrap();
                }
                for &v in set_b {
                    single.update(v).unwrap();
                }

                assert_eq!(merged.count(), single.count(),
                    "count mismatch for size={K} sets {i} x {j}");
                // S32 uses f32 for sum, so order-of-operations rounding
                // can differ between merged and single paths. Use relative
                // tolerance appropriate for f32 precision.
                let ms = merged.sum();
                let ss = single.sum();
                let sum_diff = (ms - ss).abs();
                let denom = ms.abs().max(ss.abs()).max(1e-30);
                assert!(sum_diff / denom < 1e-5,
                    "sum mismatch for size={K} sets {i} x {j}: {} vs {}",
                    ms, ss);
                assert_eq!(merged.zero_count(), single.zero_count(),
                    "zero_count mismatch for size={K} sets {i} x {j}");

                let mb = merged.positive();
                let sb = single.positive();
                let m_total: u64 = (0..mb.len()).map(|k| mb.at(k)).sum();
                let s_total: u64 = (0..sb.len()).map(|k| sb.at(k)).sum();
                assert_eq!(m_total, s_total,
                    "bucket total mismatch for size={K} sets {i} x {j}");
            }
        }
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
        // At S32, f64::MAX overflows f32, so sum is already infinite.
        // At S64, it would still be finite. Don't assert finiteness.
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
        let mut h: Histogram<8> = Histogram::with_scale(0);
        let num_buckets = 8;
        for i in 0..num_buckets {
            let val = 2.0_f64.powi(i * 8);
            h.update_by_incr(val, 255).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), num_buckets as u64 * 255);
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), num_buckets as u64 * 255 + 1);
    }

    #[test]
    fn test_successive_sub_byte_widening() {
        let mut h: Histogram<16> = Histogram::with_scale(0);

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
        let mut h: Histogram<16> = Histogram::with_scale(0);
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
    // Stat widening tests (S32 → S64)
    // -----------------------------------------------------------------------

    #[test]
    fn test_stat_widen_on_count_overflow() {
        let mut h: Histogram<16> = Histogram::new();
        assert_eq!(h.stat_width(), StatWidth::S32);

        h.update_by_incr(1.0, u32::MAX as u64 - 1).unwrap();
        assert_eq!(h.stat_width(), StatWidth::S32);

        h.update(1.0).unwrap();
        assert_eq!(h.count(), u32::MAX as u64);
        assert_eq!(h.stat_width(), StatWidth::S32);

        h.update(1.0).unwrap();
        assert_eq!(h.count(), u32::MAX as u64 + 1);
        assert_eq!(h.stat_width(), StatWidth::S64);
    }

    #[test]
    fn test_stat_widen_on_zero_count_overflow() {
        let mut h: Histogram<16> = Histogram::new();
        h.update_by_incr(0.0, u32::MAX as u64).unwrap();
        assert_eq!(h.stat_width(), StatWidth::S32);

        h.update(0.0).unwrap();
        assert_eq!(h.zero_count(), u32::MAX as u64 + 1);
        assert_eq!(h.stat_width(), StatWidth::S64);
    }

    #[test]
    fn test_stat_widen_preserves_values() {
        let mut h: Histogram<16> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(2.0).unwrap();
        h.update(4.0).unwrap();
        h.update(0.0).unwrap();

        let sum_before = h.sum();
        let count_before = h.count();
        let zc_before = h.zero_count();

        h.update_by_incr(1.0, u32::MAX as u64 - count_before).unwrap();
        assert_eq!(h.stat_width(), StatWidth::S32);
        h.update(1.0).unwrap();
        assert_eq!(h.stat_width(), StatWidth::S64);

        assert_eq!(h.zero_count(), zc_before);
        assert!(h.sum() > sum_before);
    }

    #[test]
    fn test_stat_widen_bucket_capacity_shrinks() {
        let h32: Histogram<16> = Histogram::new();
        assert_eq!(h32.stat_width(), StatWidth::S32);
        let cap_s32 = h32.bucket_capacity();

        let mut h64: Histogram<16> = Histogram::new();
        h64.update_by_incr(1.0, u32::MAX as u64).unwrap();
        h64.update(1.0).unwrap();
        assert_eq!(h64.stat_width(), StatWidth::S64);
        let cap_s64 = h64.bucket_capacity();

        assert!(cap_s64 < cap_s32,
            "S64 cap {} should be < S32 cap {}", cap_s64, cap_s32);
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
        assert_eq!(collector.zero_count(), 1);
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
        fn test_capacity_at_s32() {
            let h: Histogram<16> = Histogram::new();
            assert_eq!(h.stat_width(), StatWidth::S32);
            assert_eq!(h.bucket_word_count(), 13);
            assert_eq!(h.bucket_capacity(), 208);
        }

        #[test]
        fn test_capacity_at_s64() {
            let mut h: Histogram<16> = Histogram::new();
            h.update_by_incr(1.0, u32::MAX as u64).unwrap();
            h.update(1.0).unwrap();
            assert_eq!(h.stat_width(), StatWidth::S64);
            assert_eq!(h.bucket_word_count(), 11);
        }

        #[test]
        fn test_minimum_n() {
            let h: Histogram<8> = Histogram::new();
            assert_eq!(h.bucket_word_count(), 5);
            assert_eq!(h.bucket_capacity(), 80);
        }

        #[test]
        fn test_struct_size() {
            use core::mem;
            let size = mem::size_of::<Histogram<16>>();
            eprintln!("Histogram<16> size: {} bytes", size);
            // Fixed fields + 128 bytes of data
        }
    }
}
