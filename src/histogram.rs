// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free exponential histogram implementation.
//!
//! The histogram uses a fixed-size byte array for bucket storage, starting
//! with u8 counters and widening in place (u8 → u16 → u32 → u64) via
//! combined downscale+widen when a counter saturates.

use crate::mapping::{Mapping, max_scale};

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
        self.index_end == self.index_start && self.get(0) == 0
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
                sum += self.get(i);
                i += 1;
            }
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
/// `N` is the number of `u64` words of bucket storage (total bytes = N×8).
/// Buckets start as u8 counters and automatically widen in place
/// (u8 → u16 → u32 → u64) via combined downscale+widen when a counter
/// saturates. At u64 width, counter overflow returns `false`.
#[derive(Debug, Clone)]
pub struct Histogram<const N: usize> {
    // Statistics (min, max, sum, zero_count, count)
    sum: f64,
    count: u64,
    zero_count: u64,
    min: f64,
    max: f64,

    // Mapping (scale-dependent index calculation)
    mapping: Mapping,

    // Upper bound on scale, used as the initial/reset scale.
    max_scale: i32,

    // Positive value buckets
    positive: Buckets<N>,
}

impl<const N: usize> Default for Histogram<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Histogram<N> {
    /// Creates a new histogram at the maximum supported scale.
    /// Buckets start at u8 width.
    #[inline]
    pub fn new() -> Self {
        let scale = max_scale();
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
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
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Creates a new histogram at the specified scale.
    #[inline]
    pub fn with_scale(scale: i32) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mapping: Mapping::new(scale).expect("invalid scale"),
            max_scale: scale,
            positive: Buckets::new(),
        }
    }

    /// Returns the sum of all recorded values.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the count of zero values.
    #[inline]
    pub fn zero_count(&self) -> u64 {
        self.zero_count
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
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
        self.sum = 0.0;
        self.count = 0;
        self.zero_count = 0;
        self.min = 0.0;
        self.max = 0.0;
        self.mapping = Mapping::new(self.max_scale).unwrap();
    }

    /// Swaps contents with another histogram.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(self, other);
    }

    /// Records a single value.
    #[inline]
    pub fn update(&mut self, value: f64) -> bool {
        self.update_by_incr(value, 1)
    }

    /// Records a value with a specified increment.
    ///
    /// Returns `false` only if u64 counters overflow.
    pub fn update_by_incr(&mut self, value: f64, incr: u64) -> bool {
        debug_assert!(value >= 0.0, "Histogram only accepts non-negative values");

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
            return true;
        }

        self.sum += value * incr as f64;
        self.update_buckets(value, incr)
    }

    /// Updates buckets for a positive value.
    ///
    /// On counter overflow at width < u64, performs in-place widen+downscale
    /// and retries. Returns `false` only when u64 counters overflow.
    fn update_buckets(&mut self, value: f64, incr: u64) -> bool {
        loop {
            let index = self.mapping.map_to_index(value);

            match self.increment_index_by(index, incr) {
                IncrResult::Ok => return true,
                IncrResult::NeedsDownscale(hl) => {
                    let change = change_scale(hl, self.positive.capacity() as i32);
                    if !self.downscale(change) {
                        // Downscale failed due to counter overflow — try widening.
                        if !self.widen_and_retry_downscale(change) {
                            return false;
                        }
                    }
                    // Loop: re-map at the new scale and try again.
                }
                IncrResult::CounterOverflow => {
                    // Counter overflow — try widening in place.
                    let by = match self.positive.widen_in_place() {
                        Some(by) => by,
                        None => return false, // Already at u64, truly overflowed.
                    };
                    let new_scale = self.mapping.scale() - by;
                    self.mapping = Mapping::new(new_scale)
                        .expect("invalid scale after widen");
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
            let span = self.positive.index_end - index;
            if span >= max_size {
                return IncrResult::NeedsDownscale(HighLow {
                    low: index,
                    high: self.positive.index_end,
                });
            }
            self.positive.index_start = index;
        } else if index > self.positive.index_end {
            let span = index - self.positive.index_start;
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
        self.mapping = Mapping::new(new_scale).expect("invalid scale after downscale");
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
        self.mapping = Mapping::new(new_scale).expect("invalid scale after widen");

        // Apply remaining downscale if needed.
        let remaining = total_change - by;
        if remaining > 0 {
            if !self.positive.downscale(remaining) {
                // Still overflows — try widening again recursively.
                return self.widen_and_retry_downscale(remaining);
            }
            let new_scale = self.mapping.scale() - remaining;
            self.mapping = Mapping::new(new_scale).expect("invalid scale after downscale");
        }
        true
    }

    /// Merges another histogram (same N) into this one.
    ///
    /// Returns `false` if counters overflow.
    pub fn merge_from(&mut self, other: &Self) -> bool {
        if other.count == 0 {
            return true;
        }

        let new_count = match self.count.checked_add(other.count) {
            Some(c) => c,
            None => return false,
        };
        let new_zero_count = match self.zero_count.checked_add(other.zero_count) {
            Some(c) => c,
            None => return false,
        };

        if self.count == 0 {
            self.min = other.min;
            self.max = other.max;
        } else {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }

        self.sum += other.sum;
        self.count = new_count;
        self.zero_count = new_zero_count;

        if other.positive.is_empty() {
            return true;
        }

        let min_scale = self.scale().min(other.scale());
        let cap = self.positive.capacity() as i32;

        let hlp = self.high_low_at_scale(min_scale)
            .merge(Self::high_low_at_scale_of(&other.positive, other.scale(), min_scale));

        let min_scale = min_scale - change_scale(hlp, cap);

        if !self.downscale_to(min_scale) {
            return false;
        }
        self.merge_buckets_from(&other.positive, other.scale(), min_scale)
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
    ) -> bool {
        if other_count == 0 {
            return true;
        }

        let new_count = match self.count.checked_add(other_count) {
            Some(c) => c,
            None => return false,
        };
        let new_zero_count = match self.zero_count.checked_add(other_zero_count) {
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

        self.sum += other_sum;
        self.count = new_count;
        self.zero_count = new_zero_count;

        if other_len == 0 {
            return true;
        }

        let other_end = other_offset + other_len as i32 - 1;
        let cap = self.positive.capacity() as i32;
        let min_scale = self.scale().min(other_scale);

        let self_hl = if self.positive.is_empty() {
            HighLow::empty()
        } else {
            let shift = self.scale() - min_scale;
            HighLow {
                low: self.positive.index_start >> shift,
                high: self.positive.index_end >> shift,
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
            return false;
        }

        let their_change = other_scale - min_scale;
        for i in 0..other_len {
            let count = other_at(i);
            if count == 0 {
                continue;
            }
            let index = (other_offset + i as i32) >> their_change;
            match self.increment_index_by(index, count) {
                IncrResult::Ok => {}
                IncrResult::CounterOverflow => return false,
                IncrResult::NeedsDownscale(_) => {
                    debug_assert!(false, "incorrect merge scale in merge_from_raw");
                    return false;
                }
            }
        }
        true
    }

    fn high_low_at_scale(&self, scale: i32) -> HighLow {
        Self::high_low_at_scale_of(&self.positive, self.scale(), scale)
    }

    fn high_low_at_scale_of(buckets: &Buckets<N>, current_scale: i32, target_scale: i32) -> HighLow {
        if buckets.is_empty() {
            return HighLow::empty();
        }
        let shift = current_scale - target_scale;
        HighLow {
            low: buckets.index_start >> shift,
            high: buckets.index_end >> shift,
        }
    }

    fn merge_buckets_from(&mut self, other_buckets: &Buckets<N>, other_scale: i32, target_scale: i32) -> bool {
        let their_offset = other_buckets.offset();
        let their_change = other_scale - target_scale;

        for i in 0..other_buckets.len() {
            let count = other_buckets.at(i);
            if count == 0 {
                continue;
            }
            let index = (their_offset + i as i32) >> their_change;
            match self.increment_index_by(index, count) {
                IncrResult::Ok => {}
                IncrResult::CounterOverflow => return false,
                IncrResult::NeedsDownscale(_) => {
                    debug_assert!(false, "incorrect merge scale");
                    return false;
                }
            }
        }
        true
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

    #[test]
    fn test_histogram_basic() {
        let mut h: Histogram<2> = Histogram::new();

        h.update(1.0);
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum(), 1.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 1.0);
        assert_eq!(h.zero_count(), 0);
        assert_eq!(h.bucket_width(), BucketWidth::U8);
    }

    #[test]
    fn test_histogram_zero() {
        let mut h: Histogram<2> = Histogram::new();

        h.update(0.0);
        assert_eq!(h.count(), 1);
        assert_eq!(h.zero_count(), 1);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_multiple() {
        let mut h: Histogram<2> = Histogram::new();

        h.update(1.0);
        h.update(2.0);
        h.update(4.0);

        assert_eq!(h.count(), 3);
        assert_eq!(h.sum(), 7.0);
        assert_eq!(h.min(), 1.0);
        assert_eq!(h.max(), 4.0);
    }

    #[test]
    fn test_histogram_downscale() {
        // 1 word (8 bytes) = 8 u8 buckets initially. Wide value range forces downscale.
        let mut h: Histogram<1> = Histogram::new();

        h.update(1.0);
        h.update(1000.0);

        assert_eq!(h.count(), 2);
        assert!(h.scale() < max_scale());
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1: Histogram<2> = Histogram::new();
        let mut h2: Histogram<2> = Histogram::new();

        h1.update(1.0);
        h1.update(2.0);

        h2.update(3.0);
        h2.update(4.0);

        h1.merge_from(&h2);

        assert_eq!(h1.count(), 4);
        assert_eq!(h1.sum(), 10.0);
        assert_eq!(h1.min(), 1.0);
        assert_eq!(h1.max(), 4.0);
    }

    #[test]
    fn test_histogram_clear() {
        let mut h: Histogram<2> = Histogram::new();

        h.update(1.0);
        h.update(2.0);
        h.clear();

        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.scale(), 0);
        assert_eq!(h.bucket_width(), BucketWidth::U8);
    }

    #[test]
    fn test_buckets_at() {
        // Scale 0: each power-of-2 is a bucket
        let mut h: Histogram<2> = Histogram::with_scale(0);
        h.update(1.5);
        h.update(1.7);
        h.update(3.0);

        let buckets = h.positive();
        assert!(buckets.len() >= 2);
    }

    #[test]
    fn test_auto_widen_u8_to_u16() {
        let mut h: Histogram<2> = Histogram::new();
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        // Fill one bucket to u8::MAX.
        assert!(h.update_by_incr(1.0, 255));
        assert_eq!(h.bucket_width(), BucketWidth::U8);

        // One more triggers in-place widen+downscale → u16.
        assert!(h.update(1.0));
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), 256);
    }

    #[test]
    fn test_auto_widen_u16_to_u32() {
        let mut h: Histogram<2> = Histogram::new();

        // Force to u16 first.
        assert!(h.update_by_incr(1.0, 255));
        assert!(h.update(1.0));
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        // Fill to u16::MAX.
        assert!(h.update_by_incr(1.0, u16::MAX as u64 - 256));
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        // One more triggers widen → u32.
        assert!(h.update(1.0));
        assert_eq!(h.bucket_width(), BucketWidth::U32);
        assert_eq!(h.count(), u16::MAX as u64 + 1);
    }

    #[test]
    fn test_auto_widen_u32_to_u64() {
        let mut h: Histogram<2> = Histogram::new();

        // Force to u32.
        h.update_by_incr(1.0, 255);
        h.update(1.0);
        h.update_by_incr(1.0, u16::MAX as u64 - 256);
        h.update(1.0);
        assert_eq!(h.bucket_width(), BucketWidth::U32);

        // Fill to u32::MAX.
        h.update_by_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1));
        assert_eq!(h.bucket_width(), BucketWidth::U32);

        // One more triggers widen → u64.
        assert!(h.update(1.0));
        assert_eq!(h.bucket_width(), BucketWidth::U64);
    }

    #[test]
    fn test_bucket_count_halves_on_widen() {
        // 2 words (16 bytes): 16 u8 buckets → 8 u16 → 4 u32 → 2 u64
        let mut h: Histogram<2> = Histogram::with_scale(0);
        assert_eq!(h.positive.capacity(), 16);

        // Fill a u8 bucket to overflow.
        h.update_by_incr(1.0, 255);
        h.update(1.0);
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.positive.capacity(), 8);
    }

    #[test]
    fn test_clear_resets_to_u8() {
        let mut h: Histogram<2> = Histogram::with_max_scale(3);
        h.update_by_incr(1.0, 255);
        h.update(1.0);
        assert_eq!(h.bucket_width(), BucketWidth::U16);

        h.clear();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale() {
        let h: Histogram<2> = Histogram::with_max_scale(3);
        assert_eq!(h.max_scale(), 3);
    }

    #[test]
    fn test_with_max_scale_clamps() {
        let h: Histogram<2> = Histogram::with_max_scale(100);
        assert_eq!(h.max_scale(), max_scale());
    }

    #[test]
    fn test_with_max_scale_records_at_limited_scale() {
        let mut limited: Histogram<2> = Histogram::with_max_scale(3);
        let mut unlimited: Histogram<2> = Histogram::new();

        limited.update(1.0);
        limited.update(1.001);
        unlimited.update(1.0);
        unlimited.update(1.001);

        assert!(limited.scale() <= 3);
        if max_scale() > 3 {
            assert!(unlimited.scale() > limited.scale());
        }
    }

    #[test]
    fn test_clear_resets_to_max_scale() {
        let mut h: Histogram<2> = Histogram::with_max_scale(3);
        h.update(0.001);
        h.update(1000.0);
        assert!(h.scale() <= 3);

        h.clear();
        assert_eq!(h.count(), 0);
        assert_eq!(h.max_scale(), 3);

        h.update(1.0);
        assert_eq!(h.scale(), 3);
    }

    #[test]
    fn test_widen_preserves_data() {
        let mut h: Histogram<2> = Histogram::with_scale(0);

        // Insert into distinct buckets at scale 0.
        h.update_by_incr(1.0, 100); // index -1
        h.update_by_incr(2.0, 50);  // index 0
        h.update_by_incr(4.0, 200); // index 1

        let count_before = h.count();
        let sum_before = h.sum();

        // Force widen by overflowing the 4.0 bucket (at 200, add 55+1=56).
        h.update_by_incr(4.0, 55);
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.update(4.0);
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
                let mut merged: Histogram<N> = Histogram::new();
                for &v in set_a {
                    merged.update(v);
                }
                let mut other: Histogram<N> = Histogram::new();
                for &v in set_b {
                    other.update(v);
                }
                merged.merge_from(&other);

                let mut single: Histogram<N> = Histogram::new();
                for &v in set_a {
                    single.update(v);
                }
                for &v in set_b {
                    single.update(v);
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
        let mut h: Histogram<2> = Histogram::with_scale(0);
        h.update(1.0);
        h.update(max_f64);
        assert!(h.sum().is_finite());

        h.update(inf);
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
        let mut h: Histogram<2> = Histogram::with_scale(0);
        h.update(subnormal);
        h.update(min_normal);
        assert_eq!(h.count(), 2);
        assert_eq!(h.positive().len(), 1);
    }

    #[test]
    fn test_exhaustive_u8_overflow() {
        // At scale 0, every u8 overflow results in in-place widen+downscale.
        let mut h: Histogram<1> = Histogram::with_scale(0);
        // 1 word (8 bytes) = 8 u8 slots. Fill all to 255.
        for i in 0..8 {
            let val = 2.0_f64.powi(i);
            h.update_by_incr(val, 255);
        }
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        assert_eq!(h.count(), 8 * 255);

        // One more to any bucket triggers widen.
        h.update(1.0);
        assert_eq!(h.bucket_width(), BucketWidth::U16);
        assert_eq!(h.count(), 8 * 255 + 1);
    }
}
