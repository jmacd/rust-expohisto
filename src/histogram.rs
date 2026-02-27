// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram implementation.
//!
//! The histogram uses a fixed-size byte array for bucket storage, starting
//! with u8 counters and widening in place (u8 → u16 → u32 → u64) via
//! combined downscale+widen when a counter saturates.

use core::fmt;

use crate::mapping::{Mapping, max_scale};
use crate::precision::{HistCount, HistFloat, Precision};

/// Error returned when a histogram operation would overflow its
/// precision tier's count or bucket counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("histogram counter overflow")
    }
}

/// The current width of bucket counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BucketWidth {
    /// 1-byte counters (max 255 per bucket).
    U8 = 1,
    /// 2-byte counters (max 65,535 per bucket).
    U16 = 2,
    /// 4-byte counters (max ~4 billion per bucket).
    U32 = 4,
    /// 8-byte counters.
    U64 = 8,
}

impl BucketWidth {
    /// Returns the byte size of one counter at this width.
    #[inline]
    const fn bytes(self) -> usize {
        self as usize
    }

    /// Returns the number of buckets that fit in `byte_count` bytes.
    #[inline]
    const fn capacity(self, byte_count: usize) -> usize {
        byte_count / self.bytes()
    }

    /// Returns the next wider counter width, or `None` if already at u64.
    #[inline]
    const fn wider(self) -> Option<BucketWidth> {
        match self {
            Self::U8 => Some(Self::U16),
            Self::U16 => Some(Self::U32),
            Self::U32 => Some(Self::U64),
            Self::U64 => None,
        }
    }

    /// Returns the maximum value storable in one counter at this width.
    #[inline]
    const fn counter_max(self) -> u64 {
        match self {
            Self::U8 => u8::MAX as u64,
            Self::U16 => u16::MAX as u64,
            Self::U32 => u32::MAX as u64,
            Self::U64 => u64::MAX,
        }
    }
}

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

/// Fixed-size bucket storage using a byte array with runtime counter width.
///
/// The backing `[u64; N]` array is reinterpreted as `[u8]`, `[u16]`,
/// `[u32]`, or `[u64]` via `bytemuck` depending on the current [`BucketWidth`].
/// When a counter saturates and the width is less than u64, the caller
/// performs an in-place widen+downscale to double the counter width while
/// halving the bucket count.
///
/// `N` is the number of `u64` words of bucket storage.
#[derive(Clone)]
#[repr(C)]
pub struct Buckets<const N: usize> {
    /// Raw bucket storage, naturally 8-byte aligned via `[u64]`.
    data: [u64; N],

    /// Current counter width.
    width: BucketWidth,

    /// Index of the 0th position in the backing array.
    index_base: i32,

    /// Smallest index value represented.
    index_start: i32,

    /// Largest index value represented.
    index_end: i32,
}

impl<const N: usize> Buckets<N> {
    /// Creates empty buckets with u8 counters.
    #[inline]
    pub fn new() -> Self {
        Self {
            data: [0u64; N],
            width: BucketWidth::U8,
            index_base: 0,
            index_start: 0,
            index_end: 0,
        }
    }

    /// Returns the byte slice of the backing storage.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    /// Returns the mutable byte slice of the backing storage.
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.data)
    }

    /// Returns the current counter width.
    #[inline]
    pub fn width(&self) -> BucketWidth {
        self.width
    }

    /// Returns the offset (smallest index).
    #[inline]
    pub fn offset(&self) -> i32 {
        self.index_start
    }

    /// Number of logical buckets available at the current width.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.width.capacity(N * 8)
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
        if self.index_end != self.index_start {
            return false;
        }
        // Compute the physical slot for index_start, handling the
        // circular buffer offset.
        let mut slot = self.index_start - self.index_base;
        if slot < 0 {
            slot += self.capacity() as i32;
        }
        self.get(slot as usize) == 0
    }

    /// Returns the count at position `pos` (0-indexed from offset).
    #[inline]
    pub fn at(&self, pos: u32) -> u64 {
        let bias = (self.index_base - self.index_start) as u32;
        let cap = self.capacity() as u32;

        let mut idx = pos;
        if idx < bias {
            idx += cap;
        }
        idx -= bias;

        self.get(idx as usize)
    }

    /// Gets the value at a physical slot index.
    #[inline]
    fn get(&self, slot: usize) -> u64 {
        match self.width {
            BucketWidth::U8 => self.as_bytes()[slot] as u64,
            BucketWidth::U16 => {
                let s: &[u16] = bytemuck::cast_slice(&self.data);
                s[slot] as u64
            }
            BucketWidth::U32 => {
                let s: &[u32] = bytemuck::cast_slice(&self.data);
                s[slot] as u64
            }
            BucketWidth::U64 => self.data[slot],
        }
    }

    /// Sets the value at a physical slot index.
    #[inline]
    fn set(&mut self, slot: usize, value: u64) {
        match self.width {
            BucketWidth::U8 => self.as_bytes_mut()[slot] = value as u8,
            BucketWidth::U16 => {
                let s: &mut [u16] = bytemuck::cast_slice_mut(&mut self.data);
                s[slot] = value as u16;
            }
            BucketWidth::U32 => {
                let s: &mut [u32] = bytemuck::cast_slice_mut(&mut self.data);
                s[slot] = value as u32;
            }
            BucketWidth::U64 => self.data[slot] = value,
        }
    }

    /// Returns the maximum value storable in one counter at the current width.
    #[inline]
    fn counter_max(&self) -> u64 {
        match self.width {
            BucketWidth::U8 => u8::MAX as u64,
            BucketWidth::U16 => u16::MAX as u64,
            BucketWidth::U32 => u32::MAX as u64,
            BucketWidth::U64 => u64::MAX,
        }
    }

    /// Attempts to add `incr` to a physical slot. Returns false on overflow.
    #[inline]
    fn try_increment(&mut self, slot: usize, incr: u64) -> bool {
        let val = self.get(slot);
        let new_val = match val.checked_add(incr) {
            Some(v) if v <= self.counter_max() => v,
            _ => return false,
        };
        self.set(slot, new_val);
        true
    }

    /// Empties a slot and returns its count.
    #[inline]
    fn empty_slot(&mut self, slot: usize) -> u64 {
        let v = self.get(slot);
        self.set(slot, 0);
        v
    }

    /// Clears all bucket counts and resets to u8 width.
    #[inline]
    pub fn clear(&mut self) {
        self.index_start = 0;
        self.index_end = 0;
        self.index_base = 0;
        self.width = BucketWidth::U8;
        self.data.fill(0);
    }

    /// Rotates the circular buffer so that index_start == index_base.
    fn rotate(&mut self) {
        let bias = (self.index_base - self.index_start) as usize;
        if bias == 0 {
            return;
        }
        let cap = self.capacity();
        debug_assert!(bias < cap, "rotate bias {} >= capacity {}", bias, cap);
        match self.width {
            BucketWidth::U8 => {
                let s = self.as_bytes_mut();
                s[..cap].rotate_right(bias);
            }
            BucketWidth::U16 => {
                let s: &mut [u16] = bytemuck::cast_slice_mut(&mut self.data);
                s[..cap].rotate_right(bias);
            }
            BucketWidth::U32 => {
                let s: &mut [u32] = bytemuck::cast_slice_mut(&mut self.data);
                s[..cap].rotate_right(bias);
            }
            BucketWidth::U64 => {
                self.data[..cap].rotate_right(bias);
            }
        }
        self.index_base = self.index_start;
    }

    /// Downscales by collapsing 2^by adjacent buckets into 1.
    ///
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
                if !self.relocate(outpos as usize, inpos as usize) {
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

    /// Moves count from src slot to dest slot. Returns false on overflow.
    fn relocate(&mut self, dest: usize, src: usize) -> bool {
        if dest == src {
            return true;
        }
        let count = self.empty_slot(src);
        self.try_increment(dest, count)
    }

    /// In-place widen: linearize the circular buffer, group-sum adjacent
    /// counters, then reinterpret the memory as the next wider counter type.
    ///
    /// This is a combined downscale + counter-widen operation that preserves
    /// the total byte budget. Bucket count halves (or quarters), counter
    /// width doubles.
    ///
    /// Usually downscales by 1 (pairwise grouping). When the span is at
    /// maximum capacity with odd `index_start`, downscale-by-1 would produce
    /// more output groups than new-width capacity allows, so we bump to
    /// downscale-by-2 (groups of 4).
    ///
    /// Returns `None` if already at u64 width (no further widening possible).
    /// Returns `Some(by)` with the actual downscale amount on success.
    fn widen_in_place(&mut self) -> Option<i32> {
        let new_width = match self.width.wider() {
            Some(w) => w,
            None => return None,
        };

        // Step 1: linearize the circular buffer.
        self.rotate();

        let used = if self.is_effectively_empty() {
            0
        } else {
            (self.index_end - self.index_start + 1) as usize
        };

        let new_cap = new_width.capacity(N * 8);

        // Step 2: determine downscale amount.
        // Usually 1. When fully packed with odd index_start, the
        // downscaled span can exceed new_cap, so bump to 2.
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

        // Step 3: group-sum and write as wider type.
        //
        // We read old-width values left-to-right and write new-width values
        // left-to-right. Output never overtakes input because each output
        // element occupies at most the same byte span as its input group.
        let old_width = self.width;
        let mut out_slot = 0usize;
        let mut i = 0usize;
        while i < used {
            let old_index = self.index_start + i as i32;
            let new_index = old_index >> by;
            let mut sum = self.get(i);
            i += 1;
            // Consume partners that map to the same new_index.
            while i < used {
                let next_old_index = self.index_start + i as i32;
                if (next_old_index >> by) != new_index {
                    break;
                }
                sum = sum.checked_add(self.get(i)).expect(
                    "widen group sum overflowed u64; bucket data is corrupt",
                );
                i += 1;
            }
            debug_assert!(
                sum <= new_width.counter_max(),
                "widen group sum {} exceeds new width {:?} max {}",
                sum, new_width, new_width.counter_max(),
            );
            // Write the accumulated sum at the new width.
            // Temporarily switch width for set(), then back for get().
            self.width = new_width;
            self.set(out_slot, sum);
            self.width = old_width;
            out_slot += 1;
        }

        // Zero remaining slots at the new width.
        self.width = new_width;
        for s in out_slot..new_cap {
            self.set(s, 0);
        }

        // Step 4: update indices.
        self.index_start >>= by;
        self.index_end >>= by;
        self.index_base = self.index_start;

        Some(by)
    }

    /// Returns an iterator over the bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, N> {
        BucketsIter {
            buckets: self,
            pos: 0,
            len: self.len(),
        }
    }
}

impl<const N: usize> Default for Buckets<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> core::fmt::Debug for Buckets<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buckets")
            .field("width", &self.width)
            .field("index_start", &self.index_start)
            .field("index_end", &self.index_end)
            .field("len", &self.len())
            .finish()
    }
}

/// Iterator over bucket counts.
#[derive(Debug)]
pub struct BucketsIter<'a, const N: usize> {
    buckets: &'a Buckets<N>,
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

impl<const N: usize> ExactSizeIterator for BucketsIter<'_, N> {}

impl<'a, const N: usize> IntoIterator for &'a Buckets<N> {
    type Item = u64;
    type IntoIter = BucketsIter<'a, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An allocation-free exponential histogram for non-negative values.
///
/// `P` selects the precision tier for statistics (see [`Precision`]):
/// - [`P64`](crate::precision::P64): `f64` / `u64` (default, full precision)
/// - [`P32`](crate::precision::P32): `f32` / `u32` (half the overhead)
/// - [`P16`](crate::precision::P16): `f16` / `u16` (requires `half` feature)
///
/// `N` is the number of `u64` words of bucket storage (total bytes = N×8).
/// Buckets start as u8 counters and automatically widen in place
/// (u8 → u16 → u32 → u64) via combined downscale+widen when a counter
/// saturates. At u64 width, counter overflow returns `false`.
#[derive(Debug, Clone)]
pub struct Histogram<P: Precision, const N: usize> {
    // Statistics (min, max, sum, zero_count, count)
    sum: P::Float,
    count: P::Count,
    zero_count: P::Count,
    min: P::Float,
    max: P::Float,

    // Mapping (scale-dependent index calculation)
    mapping: Mapping,

    // Upper bound on scale, used as the initial/reset scale.
    max_scale: i32,

    // Positive value buckets
    positive: Buckets<N>,
}

impl<P: Precision, const N: usize> Default for Histogram<P, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Precision, const N: usize> Histogram<P, N> {
    /// Creates a new histogram at the maximum supported scale.
    /// Buckets start at u8 width.
    #[inline]
    pub fn new() -> Self {
        let scale = max_scale();
        Self {
            sum: P::Float::zero(),
            count: P::Count::zero(),
            zero_count: P::Count::zero(),
            min: P::Float::zero(),
            max: P::Float::zero(),
            mapping: Mapping::new(scale).unwrap(),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Creates a new histogram with an upper bound on scale.
    #[inline]
    pub fn with_max_scale(scale: i32) -> Self {
        let scale = scale.min(max_scale());
        Self {
            sum: P::Float::zero(),
            count: P::Count::zero(),
            zero_count: P::Count::zero(),
            min: P::Float::zero(),
            max: P::Float::zero(),
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Creates a new histogram at the specified scale.
    #[inline]
    pub fn with_scale(scale: i32) -> Self {
        Self {
            sum: P::Float::zero(),
            count: P::Count::zero(),
            zero_count: P::Count::zero(),
            min: P::Float::zero(),
            max: P::Float::zero(),
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Returns the sum of all recorded values.
    #[inline]
    pub fn sum(&self) -> P::Float {
        self.sum
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> P::Count {
        self.count
    }

    /// Returns the count of zero values.
    #[inline]
    pub fn zero_count(&self) -> P::Count {
        self.zero_count
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> P::Float {
        self.min
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> P::Float {
        self.max
    }

    /// Returns the current scale.
    #[inline]
    pub fn scale(&self) -> i32 {
        if self.count == self.zero_count {
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

    /// Returns the current bucket counter width.
    #[inline]
    pub fn bucket_width(&self) -> BucketWidth {
        self.positive.width()
    }

    /// Returns a reference to the positive buckets.
    #[inline]
    pub fn positive(&self) -> &Buckets<N> {
        &self.positive
    }

    /// Clears the histogram, resetting to initial state.
    ///
    /// Scale resets to max_scale, bucket width resets to u8.
    pub fn clear(&mut self) {
        self.positive.clear();
        self.sum = P::Float::zero();
        self.count = P::Count::zero();
        self.zero_count = P::Count::zero();
        self.min = P::Float::zero();
        self.max = P::Float::zero();
        self.mapping = Mapping::new(self.max_scale).unwrap();
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
        let new_count = self.count.checked_add_u64(incr).ok_or(Overflow)?;
        let fval = P::Float::from_f64(value);

        if value == 0.0 {
            let new_zc = self.zero_count.checked_add_u64(incr).ok_or(Overflow)?;
            // All checks passed — commit stats.
            self.zero_count = new_zc;
        } else {
            // Attempt bucket operation first; only commit stats on success.
            self.update_buckets(value, incr)?;
            self.sum += P::Float::from_f64(value * incr as f64);
        }

        // Commit count and min/max only after all fallible work succeeds.
        if self.count == P::Count::zero() {
            self.min = fval;
            self.max = fval;
        } else {
            self.min = self.min.min_of(fval);
            self.max = self.max.max_of(fval);
        }
        self.count = new_count;
        Ok(())
    }

    /// Updates buckets for a positive value.
    ///
    /// On counter overflow at width < u64, performs in-place widen+downscale
    /// and retries. Returns `Err(Overflow)` only when u64 counters overflow.
    fn update_buckets(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        loop {
            let index = self.mapping.map_to_index(value);

            match self.increment_index_by(index, incr) {
                IncrResult::Ok => return Ok(()),
                IncrResult::NeedsDownscale(hl) => {
                    let change = change_scale(hl, self.positive.capacity() as i32);
                    if !self.downscale(change) {
                        // Downscale failed due to counter overflow — try widening.
                        if !self.widen_and_retry_downscale(change) {
                            return Err(Overflow);
                        }
                    }
                    // Loop: re-map at the new scale and try again.
                }
                IncrResult::CounterOverflow => {
                    // Counter overflow — try widening in place.
                    let by = match self.positive.widen_in_place() {
                        Some(by) => by,
                        None => return Err(Overflow), // Already at u64, truly overflowed.
                    };
                    let new_scale = self.mapping.scale() - by;
                    self.mapping = Mapping::new(new_scale)
                        .map_err(|_| Overflow)?;
                    // Loop: re-map at the new scale and try again.
                }
            }
        }
    }

    /// Attempts to increment at the given index.
    fn increment_index_by(&mut self, index: i32, incr: u64) -> IncrResult {
        if incr == 0 {
            return IncrResult::Ok;
        }

        let max_size = self.positive.capacity() as i32;

        if self.positive.is_empty() {
            self.positive.index_start = index;
            self.positive.index_end = index;
            self.positive.index_base = index;
        } else if index < self.positive.index_start {
            // Use saturating_sub to avoid i32 overflow on extreme index ranges.
            let span = self.positive.index_end.saturating_sub(index);
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.positive.index_end,
                });
            }
            self.positive.index_start = index;
        } else if index > self.positive.index_end {
            let span = index.saturating_sub(self.positive.index_start);
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
            bucket_index += max_size;
        }

        if !self.positive.try_increment(bucket_index as usize, incr) {
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
        self.mapping = match Mapping::new(new_scale) {
            Ok(m) => m,
            Err(_) => return false, // Scale out of range.
        };
        true
    }

    /// Attempts to widen then downscale. Used when a regular downscale would
    /// overflow at the current counter width.
    fn widen_and_retry_downscale(&mut self, total_change: i32) -> bool {
        // Widen in place (which itself downscales by at least 1).
        let by = match self.positive.widen_in_place() {
            Some(by) => by,
            None => return false, // Already at u64.
        };
        let new_scale = self.mapping.scale() - by;
        self.mapping = match Mapping::new(new_scale) {
            Ok(m) => m,
            Err(_) => return false, // Scale out of range.
        };

        // Apply remaining downscale if needed.
        let remaining = total_change - by;
        if remaining > 0 {
            if !self.positive.downscale(remaining) {
                // Still overflows — try widening again recursively.
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

    /// Merges another histogram (same P and N) into this one.
    ///
    /// Returns `Err(Overflow)` if counters overflow.
    pub fn merge_from(&mut self, other: &Self) -> Result<(), Overflow> {
        if other.count == P::Count::zero() {
            return Ok(());
        }

        // Pre-validate count overflow before mutating any state.
        let new_count = self.count.checked_add(other.count).ok_or(Overflow)?;
        let new_zero_count = self.zero_count.checked_add(other.zero_count).ok_or(Overflow)?;

        // Attempt bucket operations first; stats are committed only on success.
        if !other.positive.is_empty() {
            let min_scale = self.mapping.scale().min(other.scale());
            let cap = self.positive.capacity() as i32;

            let hlp = self.high_low_at_scale(min_scale)
                .merge(Self::high_low_at_scale_of(&other.positive, other.scale(), min_scale));

            let min_scale = min_scale - change_scale(hlp, cap);

            if !self.downscale_to(min_scale) {
                if !self.widen_and_retry_downscale(self.mapping.scale() - min_scale) {
                    return Err(Overflow);
                }
            }
            self.merge_buckets_from(&other.positive, other.scale(), self.mapping.scale())?;
        }

        // All fallible work succeeded — commit stats.
        if self.count == P::Count::zero() {
            self.min = other.min;
            self.max = other.max;
        } else {
            self.min = self.min.min_of(other.min);
            self.max = self.max.max_of(other.max);
        }
        self.sum += other.sum;
        self.count = new_count;
        self.zero_count = new_zero_count;
        Ok(())
    }

    /// Merges from raw histogram data, enabling cross-size and cross-precision
    /// merging.
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

        // Pre-validate count overflow before mutating any state.
        let new_count = self.count.checked_add_u64(other_count).ok_or(Overflow)?;
        let new_zero_count = self.zero_count.checked_add_u64(other_zero_count).ok_or(Overflow)?;

        // Attempt bucket operations first; stats are committed only on success.
        if other_len > 0 {
            let other_end = other_offset + other_len as i32 - 1;
            let cap = self.positive.capacity() as i32;
            let min_scale = self.mapping.scale().min(other_scale);

            let self_hl = if self.positive.is_empty() {
                HighLow::empty()
            } else {
                let shift = self.mapping.scale() - min_scale;
                debug_assert!(shift >= 0, "self scale shift negative");
                HighLow {
                    low: self.positive.index_start >> shift,
                    high: self.positive.index_end >> shift,
                }
            };
            let other_hl = {
                let shift = other_scale - min_scale;
                debug_assert!(shift >= 0, "other scale shift negative");
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
                    // Recompute index each attempt since widening changes our scale.
                    let their_change = other_scale - self.mapping.scale();
                    debug_assert!(their_change >= 0, "merge scale invariant violated: other_scale={} self_scale={}", other_scale, self.mapping.scale());
                    let index = (other_offset + i as i32) >> their_change;

                    match self.increment_index_by(index, count) {
                        IncrResult::Ok => break,
                        IncrResult::CounterOverflow => {
                            let by = match self.positive.widen_in_place() {
                                Some(by) => by,
                                None => return Err(Overflow),
                            };
                            let new_scale = self.mapping.scale() - by;
                            self.mapping = Mapping::new(new_scale)
                                .map_err(|_| Overflow)?;
                            // Loop back to recompute index at new scale and retry.
                        }
                        IncrResult::NeedsDownscale(_) => {
                            debug_assert!(false, "incorrect merge scale in merge_from_raw");
                            return Err(Overflow);
                        }
                    }
                }
            }
        }

        // All fallible work succeeded — commit stats.
        let other_min_f = P::Float::from_f64(other_min);
        let other_max_f = P::Float::from_f64(other_max);
        if self.count == P::Count::zero() {
            self.min = other_min_f;
            self.max = other_max_f;
        } else {
            self.min = self.min.min_of(other_min_f);
            self.max = self.max.max_of(other_max_f);
        }
        self.sum += P::Float::from_f64(other_sum);
        self.count = new_count;
        self.zero_count = new_zero_count;
        Ok(())
    }

    /// Merges a histogram of a different precision tier into this one.
    ///
    /// This enables aggregating e.g. `Histogram<P32, M>` data into a
    /// `Histogram<P64, N>` collector.
    pub fn merge_from_other<Q: Precision, const M: usize>(
        &mut self,
        other: &Histogram<Q, M>,
    ) -> Result<(), Overflow> {
        self.merge_from_raw(
            other.count().to_u64(),
            other.zero_count().to_u64(),
            other.sum().to_f64(),
            other.min().to_f64(),
            other.max().to_f64(),
            other.scale(),
            other.positive().offset(),
            other.positive().len(),
            &|i| other.positive().at(i),
        )
    }

    fn high_low_at_scale(&self, scale: i32) -> HighLow {
        Self::high_low_at_scale_of(&self.positive, self.mapping.scale(), scale)
    }

    fn high_low_at_scale_of(buckets: &Buckets<N>, current_scale: i32, target_scale: i32) -> HighLow {
        if buckets.is_empty() {
            return HighLow::empty();
        }
        let shift = current_scale - target_scale;
        debug_assert!(shift >= 0, "high_low_at_scale_of: current_scale {} < target_scale {}", current_scale, target_scale);
        HighLow {
            low: buckets.index_start >> shift,
            high: buckets.index_end >> shift,
        }
    }

    fn merge_buckets_from(&mut self, other_buckets: &Buckets<N>, other_scale: i32, _target_scale: i32) -> Result<(), Overflow> {
        let their_offset = other_buckets.offset();

        for i in 0..other_buckets.len() {
            let count = other_buckets.at(i);
            if count == 0 {
                continue;
            }

            loop {
                // Recompute index each attempt since widening changes our scale.
                let their_change = other_scale - self.mapping.scale();
                debug_assert!(their_change >= 0, "merge scale invariant violated: other_scale={} self_scale={}", other_scale, self.mapping.scale());
                let index = (their_offset + i as i32) >> their_change;

                match self.increment_index_by(index, count) {
                    IncrResult::Ok => break,
                    IncrResult::CounterOverflow => {
                        let by = match self.positive.widen_in_place() {
                            Some(by) => by,
                            None => return Err(Overflow),
                        };
                        let new_scale = self.mapping.scale() - by;
                        self.mapping = Mapping::new(new_scale)
                            .map_err(|_| Overflow)?;
                        // Loop back to recompute index at new scale and retry.
                    }
                    IncrResult::NeedsDownscale(_) => {
                        debug_assert!(false, "unexpected downscale in merge");
                        return Err(Overflow);
                    }
                }
            }
        }
        Ok(())
    }

    /// Downscales to the given target scale.
    fn downscale_to(&mut self, target_scale: i32) -> bool {
        let change = self.mapping.scale() - target_scale;
        if change <= 0 {
            return true;
        }
        self.downscale(change)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::P64;

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<P64, 2> = Histogram::new();

        h.update(1.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum(), 1.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
        assert_eq!(h.zero_count(), 0);
        assert_eq!(h.bucket_width(), BucketWidth::U8);
    }

    #[test]
    fn test_histogram_zero() {
        let mut h: Histogram<P64, 2> = Histogram::new();

        h.update(0.0).unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.zero_count(), 1);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_multiple() {
        let mut h: Histogram<P64, 2> = Histogram::new();

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
        // 1 word (8 bytes) = 8 u8 buckets initially. Wide value range forces downscale.
        let mut h: Histogram<P64, 1> = Histogram::new();

        h.update(1.0).unwrap();
        h.update(1000.0).unwrap();

        assert_eq!(h.count(), 2);
        assert!(h.scale() < max_scale());
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1: Histogram<P64, 2> = Histogram::new();
        let mut h2: Histogram<P64, 2> = Histogram::new();

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
        let mut h: Histogram<P64, 2> = Histogram::new();

        h.update(1.0).unwrap();
        h.update(2.0).unwrap();
        h.clear();

        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.scale(), 0);
        assert_eq!(h.bucket_width(), BucketWidth::U8);
    }

    #[test]
    fn test_buckets_at() {
        // Scale 0: each power-of-2 is a bucket
        let mut h: Histogram<P64, 2> = Histogram::with_scale(0);
        h.update(1.5).unwrap();
        h.update(1.7).unwrap();
        h.update(3.0).unwrap();

        let buckets = h.positive();
        assert!(buckets.len() >= 2);
    }

    #[test]
    fn test_auto_widen_u8_to_u16() {
        let mut h: Histogram<P64, 2> = Histogram::new();
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        // Fill one bucket to u8::MAX.
        h.update_by_incr(1.0, 255).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        // One more triggers in-place widen+downscale → u16.
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), 256);
    }

    #[test]
    fn test_auto_widen_u16_to_u32() {
        let mut h: Histogram<P64, 2> = Histogram::new();

        // Force to u16 first.
        h.update_by_incr(1.0, 255).unwrap();
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        // Fill to u16::MAX.
        h.update_by_incr(1.0, u16::MAX as u64 - 256).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        // One more triggers widen → u32.
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        assert_eq!(h.count(), u16::MAX as u64 + 1);
    }

    #[test]
    fn test_auto_widen_u32_to_u64() {
        let mut h: Histogram<P64, 2> = Histogram::new();

        // Force to u32.
        h.update_by_incr(1.0, 255).unwrap();
        h.update(1.0).unwrap();
        h.update_by_incr(1.0, u16::MAX as u64 - 256).unwrap();
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);

        // Fill to u32::MAX.
        h.update_by_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1)).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U32);

        // One more triggers widen → u64.
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U64);
    }

    #[test]
    fn test_bucket_count_halves_on_widen() {
        // 2 words (16 bytes): 16 u8 buckets → 8 u16 → 4 u32 → 2 u64
        let mut h: Histogram<P64, 2> = Histogram::with_scale(0);
        assert_eq!(h.positive.capacity(), 16);

        // Fill a u8 bucket to overflow.
        h.update_by_incr(1.0, 255).unwrap();
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.positive.capacity(), 8);
    }

    #[test]
    fn test_clear_resets_to_u8() {
        let mut h: Histogram<P64, 2> = Histogram::with_max_scale(3);
        h.update_by_incr(1.0, 255).unwrap();
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        h.clear();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale() {
        let h: Histogram<P64, 2> = Histogram::with_max_scale(3);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale_clamps() {
        let h: Histogram<P64, 2> = Histogram::with_max_scale(100);
        assert_eq!(h.max_scale(), max_scale());
    }

    #[test]
    fn test_with_max_scale_records_at_limited_scale() {
        let mut limited: Histogram<P64, 2> = Histogram::with_max_scale(3);
        let mut unlimited: Histogram<P64, 2> = Histogram::new();

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
        let mut h: Histogram<P64, 2> = Histogram::with_max_scale(3);
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
        let mut h: Histogram<P64, 2> = Histogram::with_scale(0);

        // Insert into distinct buckets at scale 0.
        h.update_by_incr(1.0, 100).unwrap(); // index -1
        h.update_by_incr(2.0, 50).unwrap();  // index 0
        h.update_by_incr(4.0, 200).unwrap(); // index 1

        let count_before = h.count();
        let sum_before = h.sum();

        // Force widen by overflowing the 4.0 bucket (at 200, add 55+1=56).
        h.update_by_incr(4.0, 55).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.update(4.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        assert_eq!(h.count(), count_before + 56);
        assert!((h.sum() - (sum_before + 56.0 * 4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_merge_equivalence_comprehensive() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

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

        let mut test_sets: Vec<Vec<f64>> = hardcoded_sets.iter().map(|s| s.to_vec()).collect();

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let size = rng.gen_range(0..=10);
            let set: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..20.0)).collect();
            test_sets.push(set);
        }

        test_merge_equivalence_for_size::<1>(&test_sets);
        test_merge_equivalence_for_size::<2>(&test_sets);
        test_merge_equivalence_for_size::<3>(&test_sets);
        test_merge_equivalence_for_size::<4>(&test_sets);
        test_merge_equivalence_for_size::<1>(&test_sets);
    }

    fn test_merge_equivalence_for_size<const N: usize>(test_sets: &[Vec<f64>]) {
        for (i, set_a) in test_sets.iter().enumerate() {
            for (j, set_b) in test_sets.iter().enumerate() {
                let mut merged: Histogram<P64, N> = Histogram::new();
                for &v in set_a {
                    merged.update(v).unwrap();
                }
                let mut other: Histogram<P64, N> = Histogram::new();
                for &v in set_b {
                    other.update(v).unwrap();
                }
                merged.merge_from(&other).unwrap();

                let mut single: Histogram<P64, N> = Histogram::new();
                for &v in set_a {
                    single.update(v).unwrap();
                }
                for &v in set_b {
                    single.update(v).unwrap();
                }

                assert_eq!(merged.count(), single.count(),
                    "count mismatch for size={N} sets {i} x {j}");
                let sum_diff = (merged.sum() - single.sum()).abs();
                assert!(sum_diff < 1e-10,
                    "sum mismatch for size={N} sets {i} x {j}: {} vs {}",
                    merged.sum(), single.sum());
                assert_eq!(merged.zero_count(), single.zero_count(),
                    "zero_count mismatch for size={N} sets {i} x {j}");
                assert_eq!(merged.min(), single.min(),
                    "min mismatch for size={N} sets {i} x {j}");
                assert_eq!(merged.max(), single.max(),
                    "max mismatch for size={N} sets {i} x {j}");
                assert_eq!(merged.scale(), single.scale(),
                    "scale mismatch for size={N} sets {i} x {j}");

                let mb = merged.positive();
                let sb = single.positive();
                assert_eq!(mb.offset(), sb.offset(),
                    "offset mismatch for size={N} sets {i} x {j}");
                assert_eq!(mb.len(), sb.len(),
                    "bucket len mismatch for size={N} sets {i} x {j}");
                for k in 0..mb.len() {
                    assert_eq!(mb.at(k), sb.at(k),
                        "bucket[{k}] mismatch for size={N} sets {i} x {j}");
                }
            }
        }
    }

    #[test]
    fn test_edge_values_inf() {
        use crate::mapping::Mapping;

        let max_f64: f64 = f64::MAX;
        let inf: f64 = f64::INFINITY;

        // +Inf shares bucket with MAX at scale 0.
        let m0 = Mapping::new(0).unwrap();
        let idx_max = m0.map_to_index(max_f64);
        let idx_inf = m0.map_to_index(inf);
        assert_eq!(idx_max, idx_inf);

        // Histogram accepts +Inf: sum/max become Inf.
        let mut h: Histogram<P64, 2> = Histogram::with_scale(0);
        h.update(1.0).unwrap();
        h.update(max_f64).unwrap();
        assert!(h.sum().is_finite());

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

        // At scale 0, subnormals and MIN_VALUE share a bucket.
        let m0 = Mapping::new(0).unwrap();
        assert_eq!(m0.map_to_index(subnormal), m0.map_to_index(min_normal));

        // Histogram accepts subnormals.
        let mut h: Histogram<P64, 2> = Histogram::with_scale(0);
        h.update(subnormal).unwrap();
        h.update(min_normal).unwrap();
        assert_eq!(h.count(), 2);
        assert_eq!(h.positive().len(), 1);
    }

    #[test]
    fn test_exhaustive_u8_overflow() {
        // At scale 0, every u8 overflow results in in-place widen+downscale.
        let mut h: Histogram<P64, 1> = Histogram::with_scale(0);
        // 1 word (8 bytes) = 8 u8 slots. Fill all to 255.
        for i in 0..8 {
            let val = 2.0_f64.powi(i);
            h.update_by_incr(val, 255).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 8 * 255);

        // One more to any bucket triggers widen.
        h.update(1.0).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), 8 * 255 + 1);
    }

    // -----------------------------------------------------------------------
    // P32 precision tier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_p32_basic() {
        use crate::precision::P32;

        let mut h: Histogram<P32, 2> = Histogram::new();
        h.update(1.0).unwrap();
        h.update(2.0).unwrap();
        h.update(4.0).unwrap();

        assert_eq!(h.count(), 3);
        // f32 sum
        assert!((h.sum().to_f64() - 7.0).abs() < 1e-5);
        assert!((h.min().to_f64() - 1.0).abs() < 1e-5);
        assert!((h.max().to_f64() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_p32_count_saturates() {
        use crate::precision::P32;

        let mut h: Histogram<P32, 2> = Histogram::new();
        // Fill count to u32::MAX.
        h.update_by_incr(1.0, u32::MAX as u64).unwrap();
        assert_eq!(h.count(), u32::MAX);

        // One more should fail with Overflow.
        let result = h.update(1.0);
        assert_eq!(result, Err(Overflow));

        // Stats must be unchanged after failed update.
        assert_eq!(h.count(), u32::MAX);
        assert_eq!(h.sum(), u32::MAX as f32);
        assert_eq!(h.min(), 1.0f32);
        assert_eq!(h.max(), 1.0f32);
    }

    #[test]
    fn test_p32_zero_count_saturates() {
        use crate::precision::P32;

        let mut h: Histogram<P32, 2> = Histogram::new();
        h.update_by_incr(0.0, u32::MAX as u64).unwrap();
        assert_eq!(h.zero_count(), u32::MAX);

        let result = h.update(0.0);
        assert_eq!(result, Err(Overflow));
    }

    #[test]
    fn test_p32_update_by_incr_too_large() {
        use crate::precision::P32;

        let mut h: Histogram<P32, 2> = Histogram::new();
        // incr exceeds u32::MAX — should fail immediately.
        let result = h.update_by_incr(1.0, u32::MAX as u64 + 1);
        assert_eq!(result, Err(Overflow));
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_p32_merge_saturates() {
        use crate::precision::P32;

        let mut h1: Histogram<P32, 2> = Histogram::new();
        let mut h2: Histogram<P32, 2> = Histogram::new();

        h1.update_by_incr(1.0, u32::MAX as u64 - 1).unwrap();
        h2.update_by_incr(2.0, 2).unwrap();

        // Capture pre-merge state.
        let count_before = h1.count();
        let sum_before = h1.sum();
        let min_before = h1.min();
        let max_before = h1.max();
        let zc_before = h1.zero_count();

        // Merging would overflow count.
        let result = h1.merge_from(&h2);
        assert_eq!(result, Err(Overflow));

        // Stats must be unchanged after failed merge.
        assert_eq!(h1.count(), count_before);
        assert_eq!(h1.sum(), sum_before);
        assert_eq!(h1.min(), min_before);
        assert_eq!(h1.max(), max_before);
        assert_eq!(h1.zero_count(), zc_before);
    }

    // -----------------------------------------------------------------------
    // Cross-precision merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_p32_into_p64() {
        use crate::precision::P32;

        let mut collector: Histogram<P64, 2> = Histogram::new();
        let mut source: Histogram<P32, 2> = Histogram::new();

        source.update(1.0).unwrap();
        source.update(2.0).unwrap();
        source.update(4.0).unwrap();
        source.update(0.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        assert_eq!(collector.count(), 4);
        assert_eq!(collector.zero_count(), 1);
        // sum may differ slightly due to f32→f64 round-trip
        assert!((collector.sum() - 7.0).abs() < 1e-5);
        assert!((collector.min() - 0.0).abs() < 1e-10);
        assert!((collector.max() - 4.0).abs() < 1e-5);
        assert_eq!(collector.positive().len(), source.positive().len());
    }

    #[test]
    fn test_merge_multiple_p32_into_p64() {
        use crate::precision::P32;

        let mut collector: Histogram<P64, 4> = Histogram::new();

        // Simulate multiple P32 sources being aggregated.
        for batch in 0..5 {
            let mut src: Histogram<P32, 4> = Histogram::new();
            for i in 0..10 {
                src.update((batch * 10 + i) as f64 * 0.1 + 0.1).unwrap();
            }
            collector.merge_from_other(&src).unwrap();
        }

        assert_eq!(collector.count(), 50);
        assert!(collector.sum() > 0.0);
    }

    #[test]
    fn test_merge_p32_into_p64_preserves_buckets() {
        use crate::precision::P32;

        // Both at scale 0 for predictable bucket layout.
        let mut collector: Histogram<P64, 2> = Histogram::with_scale(0);
        let mut source: Histogram<P32, 2> = Histogram::with_scale(0);

        source.update(1.0).unwrap();
        source.update(2.0).unwrap();
        source.update(4.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        // Verify bucket-by-bucket equivalence with a direct P64 histogram.
        let mut direct: Histogram<P64, 2> = Histogram::with_scale(0);
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
    fn test_p32_single_update_u32max() {
        use crate::precision::P32;

        let mut h: Histogram<P32, 2> = Histogram::new();
        h.update_by_incr(1.0, u32::MAX as u64).unwrap();

        let mut h2: Histogram<P32, 2> = Histogram::new();
        h2.update_by_incr(1.0, u32::MAX as u64).unwrap();
    }

    #[test]
    fn test_merge_p32_exceeding_u32_count_into_p64() {
        use crate::precision::P32;

        let mut collector: Histogram<P64, 2> = Histogram::new();

        let mut s1: Histogram<P32, 2> = Histogram::new();
        let mut s2: Histogram<P32, 2> = Histogram::new();
        s1.update_by_incr(1.0, u32::MAX as u64).unwrap();
        s2.update_by_incr(1.0, u32::MAX as u64).unwrap();

        collector.merge_from_other(&s1).unwrap();
        collector.merge_from_other(&s2).unwrap();

        assert_eq!(collector.count(), 2 * u32::MAX as u64);
    }

    #[test]
    fn test_merge_p32_with_different_sizes_into_p64() {
        use crate::precision::P32;

        let mut collector: Histogram<P64, 4> = Histogram::new();
        let mut source: Histogram<P32, 2> = Histogram::new();

        source.update(0.5).unwrap();
        source.update(100.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        assert_eq!(collector.count(), 2);
    }

    #[test]
    fn test_merge_empty_p32_into_p64() {
        use crate::precision::P32;

        let mut collector: Histogram<P64, 2> = Histogram::new();
        collector.update(1.0).unwrap();

        let empty: Histogram<P32, 2> = Histogram::new();
        collector.merge_from_other(&empty).unwrap();

        assert_eq!(collector.count(), 1);
        assert_eq!(collector.sum(), 1.0);
    }

    #[test]
    fn test_merge_p32_into_empty_p64() {
        use crate::precision::P32;

        let mut collector: Histogram<P64, 2> = Histogram::new();
        let mut source: Histogram<P32, 2> = Histogram::new();
        source.update(5.0).unwrap();

        collector.merge_from_other(&source).unwrap();

        assert_eq!(collector.count(), 1);
        assert!((collector.sum() - 5.0).abs() < 1e-5);
        assert!((collector.min() - 5.0).abs() < 1e-5);
        assert!((collector.max() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_merge_p64_into_p32_saturates() {
        use crate::precision::P32;

        let mut collector: Histogram<P32, 2> = Histogram::new();
        let mut source: Histogram<P64, 2> = Histogram::new();
        source.update_by_incr(1.0, u32::MAX as u64 + 1).unwrap();

        // P32 collector can't hold count > u32::MAX.
        let result = collector.merge_from_other(&source);
        assert_eq!(result, Err(Overflow));
    }

    // -----------------------------------------------------------------------
    // Experimental tests: unified flat layout design exploration
    //
    // Goal: A Histogram is declared as [u64; N] — a single fixed-size block.
    // MMZSC (min, max, zero_count, sum, count) starts at 4-byte width and
    // can widen to 8-byte width independently of bucket counter widening.
    //
    // Layout (4-byte MMZSC, header = 3 words = 24 bytes):
    //   Word 0: [scale:i8][mmzsc_w:u8][bucket_w:u8][spare:u8][min:f32]
    //   Word 1: [max:f32][sum:f32]
    //   Word 2: [count:u32][zero_count:u32]
    //   Words 3..N: bucket storage = (N-3)*8 bytes
    //
    // Layout (8-byte MMZSC, header = 6 words = 48 bytes):
    //   Word 0: [scale:i8][mmzsc_w:u8][bucket_w:u8][spare:5 bytes]
    //   Word 1: [min:f64]
    //   Word 2: [max:f64]
    //   Word 3: [sum:f64]
    //   Word 4: [count:u64]
    //   Word 5: [zero_count:u64]
    //   Words 6..N: bucket storage = (N-6)*8 bytes
    //
    // Bucket metadata (index_base, index_start, index_end) live as separate
    // struct fields outside the data array (they are small and fixed-size).
    // -----------------------------------------------------------------------

    /// Capacity calculation for the proposed flat layout.
    mod flat_layout {
        /// Header words for 4-byte MMZSC mode.
        const HEADER_W32: usize = 3;
        /// Header words for 8-byte MMZSC mode.
        const HEADER_W64: usize = 6;

        /// Bucket bytes available for a given N and MMZSC width.
        const fn bucket_bytes(n: usize, mmzsc_wide: bool) -> usize {
            let header = if mmzsc_wide { HEADER_W64 } else { HEADER_W32 };
            (n - header) * 8
        }

        /// Number of buckets at a given bucket counter width.
        const fn bucket_count(n: usize, mmzsc_wide: bool, counter_bytes: usize) -> usize {
            bucket_bytes(n, mmzsc_wide) / counter_bytes
        }

        /// How many downscale steps to go from `from` buckets to at most `to`
        /// bucket capacity.
        const fn downscales_needed(from: usize, to: usize) -> u32 {
            let mut current = from;
            let mut steps = 0u32;
            while current > to {
                current = (current + 1) / 2; // ceil(current/2)
                steps += 1;
            }
            steps
        }

        #[test]
        fn test_capacity_256_bytes() {
            // N=32 words = 256 bytes total.
            let n = 32;

            // 4-byte MMZSC mode
            assert_eq!(bucket_bytes(n, false), 232);
            assert_eq!(bucket_count(n, false, 1), 232); // u8
            assert_eq!(bucket_count(n, false, 2), 116); // u16
            assert_eq!(bucket_count(n, false, 4), 58);  // u32
            assert_eq!(bucket_count(n, false, 8), 29);  // u64

            // 8-byte MMZSC mode
            assert_eq!(bucket_bytes(n, true), 208);
            assert_eq!(bucket_count(n, true, 1), 208);  // u8
            assert_eq!(bucket_count(n, true, 2), 104);  // u16
            assert_eq!(bucket_count(n, true, 4), 52);   // u32
            assert_eq!(bucket_count(n, true, 8), 26);   // u64
        }

        #[test]
        fn test_capacity_128_bytes() {
            // N=16 words = 128 bytes total.
            let n = 16;

            assert_eq!(bucket_bytes(n, false), 104);
            assert_eq!(bucket_count(n, false, 1), 104);
            assert_eq!(bucket_count(n, false, 4), 26);
            assert_eq!(bucket_count(n, false, 8), 13);

            assert_eq!(bucket_bytes(n, true), 80);
            assert_eq!(bucket_count(n, true, 1), 80);
            assert_eq!(bucket_count(n, true, 4), 20);
            assert_eq!(bucket_count(n, true, 8), 10);
        }

        #[test]
        fn test_capacity_64_bytes() {
            // N=8 words = 64 bytes total — smallest practical.
            let n = 8;

            assert_eq!(bucket_bytes(n, false), 40);
            assert_eq!(bucket_count(n, false, 1), 40);
            assert_eq!(bucket_count(n, false, 4), 10);
            assert_eq!(bucket_count(n, false, 8), 5);

            assert_eq!(bucket_bytes(n, true), 16);
            assert_eq!(bucket_count(n, true, 1), 16);
            assert_eq!(bucket_count(n, true, 4), 4);
            assert_eq!(bucket_count(n, true, 8), 2);
        }

        /// When MMZSC widens from 4-byte to 8-byte, bucket area shrinks by
        /// 3 words (24 bytes). This test explores how many downscales are
        /// needed at each bucket width for common N values.
        #[test]
        fn test_mmzsc_widen_downscale_requirements() {
            // For each N: at each bucket counter width, compute:
            //   old_cap = max buckets in 4-byte MMZSC mode
            //   new_cap = max buckets in 8-byte MMZSC mode
            //   downscales = how many halvings to fit old_cap into new_cap
            struct Case {
                n: usize,
                label: &'static str,
            }
            let cases = [
                Case { n: 8, label: "64B" },
                Case { n: 16, label: "128B" },
                Case { n: 32, label: "256B" },
                Case { n: 64, label: "512B" },
            ];

            for case in &cases {
                for &(bw, bw_name) in &[(1, "u8"), (2, "u16"), (4, "u32"), (8, "u64")] {
                    let old_cap = bucket_count(case.n, false, bw);
                    let new_cap = bucket_count(case.n, true, bw);
                    let ds = downscales_needed(old_cap, new_cap);
                    eprintln!(
                        "{:>4} N={:>2} bw={}: old_cap={:>3} new_cap={:>3} downscales={}",
                        case.label, case.n, bw_name, old_cap, new_cap, ds
                    );
                    // The key invariant: we should never need more than 2
                    // downscales to accommodate the MMZSC widening.
                    assert!(
                        ds <= 2,
                        "{} bw={}: needs {} downscales (old={}, new={})",
                        case.label, bw_name, ds, old_cap, new_cap
                    );
                }
            }
        }

        /// Verify that after the worst-case downscaling for MMZSC widening,
        /// we still have a usable number of buckets.
        #[test]
        fn test_min_buckets_after_mmzsc_widen() {
            // N=32 (256 bytes) — the reference case in user's description.
            let n = 32;

            // Worst case: u32 buckets, fully packed at 58, need to fit in 52.
            let old = bucket_count(n, false, 4); // 58
            let new = bucket_count(n, true, 4);  // 52
            assert_eq!(old, 58);
            assert_eq!(new, 52);

            // One downscale: 58 -> ceil(58/2) = 29. 29 <= 52 ✓
            // So we need exactly 1 downscale, losing ~1 unit of scale
            // resolution. The result (29 buckets in 52 slots) has room.
            assert_eq!(downscales_needed(58, 52), 1);

            // For u8 buckets: 232 -> 208. Need 1 downscale: 232 -> 116 ≤ 208
            assert_eq!(downscales_needed(232, 208), 1);

            // For u64 buckets: 29 -> 26. Need 1 downscale: 29 -> 15 ≤ 26
            assert_eq!(downscales_needed(29, 26), 1);

            // N=8 (64B), u64 is the worst: 5 -> 2, needs 2 downscales
            // 5 -> 3 -> 2
            let n8 = 8;
            assert_eq!(downscales_needed(
                bucket_count(n8, false, 8),
                bucket_count(n8, true, 8)
            ), 2);
        }

        /// Verify struct sizes of current implementation to understand overhead.
        #[test]
        fn test_current_struct_sizes() {
            use core::mem;
            use super::*;

            // Mapping: i8 scale + f64 inverse_factor (+ optional f64 scale_factor)
            let mapping_size = mem::size_of::<Mapping>();
            eprintln!("Mapping: {} bytes", mapping_size);
            // Should be 16 or 24 depending on features
            assert!(mapping_size <= 24);

            // Buckets<N>: [u64; N] + BucketWidth(u8) + 3*i32
            // = N*8 + 1 + 12 = N*8 + 13, padded to 8-byte alignment = N*8 + 16
            let b2 = mem::size_of::<Buckets<2>>();
            let b4 = mem::size_of::<Buckets<4>>();
            eprintln!("Buckets<2>: {} bytes (data=16 + meta={})", b2, b2 - 16);
            eprintln!("Buckets<4>: {} bytes (data=32 + meta={})", b4, b4 - 32);
            let meta_overhead = b2 - 16;
            assert_eq!(b4 - 32, meta_overhead, "metadata overhead should be constant");

            // Histogram<P64, 2>: 5*f64/u64 + Mapping + max_scale(i32) + Buckets<2>
            let h64_2 = mem::size_of::<Histogram<P64, 2>>();
            let h32_2 = mem::size_of::<Histogram<crate::precision::P32, 2>>();
            eprintln!("Histogram<P64, 2>: {} bytes", h64_2);
            eprintln!("Histogram<P32, 2>: {} bytes", h32_2);
            // P64: 5*8=40 MMZSC + Mapping + i32 + Buckets<2>
            // P32: 5*4=20 MMZSC + Mapping + i32 + Buckets<2>
            // Difference should be ~20 bytes (5*(8-4))
            let diff = h64_2 - h32_2;
            eprintln!("P64-P32 size difference: {} bytes (expected ~20)", diff);

            // In the proposed flat layout, Histogram would be:
            //   data: [u64; N]  (N*8 bytes)
            //   + bucket metadata: width(1) + index_base(4) + index_start(4) + index_end(4) = 13
            //   + mapping metadata: i8 scale + f64 inverse_factor ≈ 16
            //   + mmzsc_width flag: 1 byte
            //   + max_scale: i8
            // Total metadata outside data: ~32 bytes
            // Total struct: N*8 + 32 bytes (vs current which mixes everything)
            let proposed_256 = 32 * 8 + 32;
            eprintln!("\nProposed flat layout (N=32): {} bytes total", proposed_256);
            eprintln!("  data: {} bytes", 32 * 8);
            eprintln!("  external metadata: ~32 bytes");
        }

        /// The MMZSC widening operation: simulate what happens when count
        /// overflows u32 and we need to promote all 5 MMZSC fields to 8-byte.
        #[test]
        fn test_mmzsc_widen_simulation() {
            // Simulate: N=32 (256 bytes), bucket width = u8, MMZSC is 4-byte.
            // Bucket area: (32-3)*8 = 232 bytes = 232 u8 buckets.
            // After MMZSC widen: bucket area: (32-6)*8 = 208 bytes = 208 u8 buckets.
            // Need to shift data right by 3 words. Any bucket data in
            // the first 24 bytes of the old bucket area (words 3-5) must be
            // relocated into the remaining 208 bytes.
            //
            // Actually, the operation is:
            // 1. Read MMZSC as f32/u32 from header
            // 2. Downscale buckets if needed (from cap 232 to cap 208 for u8)
            //    - bucket data in words 3..32 must shrink to fit words 6..32
            // 3. Move/compact bucket data to start at word 6
            // 4. Write MMZSC as f64/u64 into new header (words 0..6)

            let n: usize = 32;
            let old_bucket_bytes = (n - 3) * 8; // 232
            let new_bucket_bytes = (n - 6) * 8; // 208

            // At u8 width: 232 -> 208 buckets.
            // If we have 200 active buckets, no downscale needed (200 ≤ 208).
            assert!(200 <= new_bucket_bytes / 1);

            // If we have all 232 active, need 1 downscale: 232 -> 116 ≤ 208.
            assert!(232 > 208);
            assert!((232 + 1) / 2 <= 208);

            // At u32 width: 58 -> 52 buckets.
            // If 58 active, need 1 downscale: 58 -> 29 ≤ 52.
            assert!((58 + 1) / 2 <= 52);

            // At u64 width: 29 -> 26 buckets.
            // If 29 active, need 1 downscale: 29 -> 15 ≤ 26.
            // But this also downscales the bucket INDICES.
            assert!((29 + 1) / 2 <= 26);
        }

        /// Explore what sizes of N make the flat layout practical.
        /// Below some threshold, the header eats too much of the budget.
        #[test]
        fn test_minimum_viable_n() {
            // N=7: 56 bytes total
            // 4-byte MMZSC: (7-3)*8 = 32 bucket bytes = 4 u64 buckets. Marginal.
            // 8-byte MMZSC: (7-6)*8 = 8 bucket bytes = 1 u64 bucket. Barely usable.
            assert_eq!(bucket_count(7, false, 8), 4);
            assert_eq!(bucket_count(7, true, 8), 1);

            // N=8: 64 bytes total — minimum for 8-byte MMZSC with 2+ u64 buckets.
            assert_eq!(bucket_count(8, true, 8), 2);

            // N=10: 80 bytes total
            assert_eq!(bucket_count(10, false, 1), 56);  // 56 u8 buckets
            assert_eq!(bucket_count(10, true, 1), 32);   // 32 u8 buckets
            assert_eq!(bucket_count(10, true, 8), 4);    // 4 u64 buckets

            // N=16: 128 bytes — nice round number, practical capacity.
            assert_eq!(bucket_count(16, false, 1), 104);
            assert_eq!(bucket_count(16, true, 1), 80);
            assert_eq!(bucket_count(16, true, 8), 10);

            // The sweet spot: enough u8 buckets for fine-grained histograms,
            // still usable after MMZSC widening + bucket widening to u64.
            // N=32 seems ideal: 232 u8 → 26 u64 after both widenings.
        }

        /// When MMZSC widens, the bucket data needs to be physically
        /// moved in the array. This test verifies the memmove-like operation.
        #[test]
        fn test_bucket_data_shift() {
            // Simulate the data array for N=32.
            let mut data = [0u64; 32];

            // Fill "header" (words 0-2) with MMZSC in 4-byte format.
            // Word 0: [scale=5:i8][mmzsc_w=0:u8][bucket_w=0:u8][0:u8][min:f32]
            // Just mark them with sentinel values.
            data[0] = 0xAAAA_AAAA_AAAA_AAAA;
            data[1] = 0xBBBB_BBBB_BBBB_BBBB;
            data[2] = 0xCCCC_CCCC_CCCC_CCCC;

            // Fill bucket data (words 3-31) with recognizable pattern.
            for i in 3..32 {
                data[i] = i as u64 * 100;
            }

            // === Simulate MMZSC widening ===
            // 1. Extract MMZSC values (would decode f32/u32 from header)
            let _old_header = [data[0], data[1], data[2]];

            // 2. Compact bucket data: shift words 3..32 -> 6..32
            //    But first, if needed, downscale bucket data to fit
            //    in (32-6)=26 words instead of (32-3)=29 words.
            //
            //    For this test, assume no downscale needed (active < new_cap).
            //    Just shift the data.

            // Move bucket data from [3..32] to [6..32].
            // We're moving 26 words (the new bucket area can hold 26 words).
            // Source starts at word 3, destination at word 6.
            // We copy forward to avoid overlap issues (src < dst).
            // Actually, we need to copy *backward* since dst > src.
            let old_start = 3usize;
            let new_start = 6usize;
            let new_bucket_words = 32 - new_start; // 26
            // Copy from end to avoid overwriting.
            for i in (0..new_bucket_words).rev() {
                data[new_start + i] = data[old_start + i];
            }
            // Clear the gap (words 3-5 are now the extended header).
            for i in old_start..new_start {
                data[i] = 0;
            }

            // 3. Write new MMZSC in 8-byte format into words 0-5.
            //    (would encode f64/u64 values; just verify positions)
            // Word 0: metadata (scale, flags) - keep as-is for now
            // Words 1-5: min:f64, max:f64, sum:f64, count:u64, zero_count:u64

            // Verify bucket data survived at new positions.
            for i in 0..new_bucket_words {
                let expected = (old_start + i) as u64 * 100;
                assert_eq!(
                    data[new_start + i], expected,
                    "bucket word {} should be {} but got {}",
                    i, expected, data[new_start + i]
                );
            }
        }

        /// The combined bucket-widen + MMZSC-widen scenario.
        /// When bucket counters are at u32 and count overflows u32,
        /// both widenings might need to happen.
        #[test]
        fn test_combined_widen_scenario_256b() {
            let n = 32usize;

            // State: bucket_width=u32, mmzsc_width=32
            // Bucket area: (32-3)*8 = 232 bytes = 58 u32 counters
            let buckets_before = bucket_count(n, false, 4);
            assert_eq!(buckets_before, 58);

            // Event: count overflows u32.
            // Action: widen MMZSC from 4-byte to 8-byte.
            // New bucket area: (32-6)*8 = 208 bytes = 52 u32 counters
            let buckets_after_mmzsc = bucket_count(n, true, 4);
            assert_eq!(buckets_after_mmzsc, 52);

            // If we had 58 active buckets, we need 1 downscale to fit in 52.
            // 58 -> ceil(58/2) = 29. 29 ≤ 52. ✓
            // Scale decreases by 1 (acceptable loss).
            assert!(downscales_needed(58, 52) <= 1);

            // Now if a bucket counter later overflows u32:
            // Action: widen bucket counters u32 -> u64.
            // New bucket area stays at 208 bytes = 26 u64 counters.
            let buckets_after_both = bucket_count(n, true, 8);
            assert_eq!(buckets_after_both, 26);

            // From 29 active u32 -> 26 u64 slots. Bucket widen already halves:
            // 29 -> ceil(29/2) = 15. 15 ≤ 26. ✓
            assert!(downscales_needed(29, 26) <= 1);
        }

        /// Validate that the current Histogram struct overhead matches
        /// expectations, establishing a baseline for the flat layout to improve.
        #[test]
        fn test_overhead_comparison() {
            use core::mem;
            use super::*;

            // Current: Histogram<P64, 29> should give 29*8=232 bytes of bucket
            // data, which is equivalent to the 4-byte MMZSC mode's bucket area
            // in a 256-byte flat layout.
            let h64_29 = mem::size_of::<Histogram<P64, 29>>();
            eprintln!("Current Histogram<P64, 29>: {} bytes", h64_29);
            // This includes: 5*8 MMZSC + Mapping + max_scale + Buckets<29>
            // = 40 + ~16 + 4 + (232 + 16) = ~308 bytes

            // Proposed flat: total = 32*8 + ~32 external = ~288 bytes
            // for the same 232 bytes of bucket data in 4-byte MMZSC mode.
            // Better: 288 vs ~308.
            //
            // But the real win: the flat layout can START at 232 u8 buckets
            // with 4-byte MMZSC, giving much finer resolution initially.
            // The current design always uses P64 or P32 — it can't transition.

            let proposed_flat = 32 * 8 + 32;
            eprintln!("Proposed flat (N=32): {} bytes", proposed_flat);
            eprintln!("Savings: {} bytes", h64_29 as i64 - proposed_flat as i64);
        }
    }
}
