// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixed-scale, collapse-left histogram with a relative-error guarantee.
//!
//! [`Sketch<N>`] is a sibling of [`HistogramNN`](super::HistogramNN) that
//! trades auto-scaling for a **guaranteed relative error** in the style of
//! DDSketch. Unlike `HistogramNN`, the scale never changes: it is fixed at
//! construction. When a value would not fit the current bucket window — or
//! when a counter must widen and the window therefore shrinks — the
//! structure does **not** downscale (which would lose precision over the
//! whole range). Instead it **collapses the left (small-value) side**: the
//! window slides so the largest values keep their resolution, and the
//! dropped small buckets are folded into the lowest retained slot, which
//! becomes an inaccurate *underflow placeholder*.
//!
//! ## Guarantee
//!
//! Let `b = 2^(2^-scale)` and `α = (b - 1) / (b + 1)`. Every retained
//! bucket has worst-case relative error `≤ α`. Because collapsing only
//! removes buckets on the low side, the accurate window always covers the
//! top of the observed distribution. For any quantile whose true value is
//! above the underflow bucket's upper boundary, the estimate is within a
//! factor `α` — exactly DDSketch's collapsing-lowest guarantee, but using
//! the OTel base-`2^(2^-scale)` bucket structure (so the result is still a
//! valid OTel exponential histogram).

use crate::float64::{get_biased_exponent, get_significand, unbias_exponent, NAN_INF_BIASED};
use crate::mapping::{table_scale, Scale, ScaleError};

use super::width::Width;
use super::{Error, Stats};

/// Outcome of a single placement attempt.
enum Place {
    /// The increment was applied.
    Done,
    /// A counter overflowed; widen to fit `total`, then retry.
    Widen(u64),
    /// The value's word `widx` is above the window and does not fit;
    /// slide the window up, then retry.
    SlideUp(i32),
}

/// A fixed-scale exponential histogram with DDSketch-style collapse-left
/// behavior and a relative-error guarantee.
///
/// `N` is the data-pool size in `u64` words (same meaning as
/// [`HistogramNN`](super::HistogramNN)). The scale is fixed at
/// construction and never changes.
pub struct Sketch<const N: usize> {
    scale: Scale,
    width: Width,
    min_width: Width,

    word_base: i32,
    word_start: i32,
    word_end: i32,

    /// True once the lowest retained slot has absorbed folded mass and is
    /// therefore an inaccurate underflow placeholder.
    collapsed: bool,

    /// Sum of all bucket counts (excludes zero observations).
    live_total: u64,

    stats: Stats,

    data: [u64; N],
}

impl<const N: usize> Clone for Sketch<N> {
    fn clone(&self) -> Self {
        Self {
            scale: self.scale,
            width: self.width,
            min_width: self.min_width,
            word_base: self.word_base,
            word_start: self.word_start,
            word_end: self.word_end,
            collapsed: self.collapsed,
            live_total: self.live_total,
            stats: self.stats,
            data: self.data,
        }
    }
}

impl<const N: usize> Default for Sketch<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Sketch<N> {
    /// Creates a new sketch at the maximum table scale and `B1` width.
    ///
    /// # Panics
    ///
    /// Panics if `N < 1` or `N > 250`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        assert!(N >= 1, "requires >= 1 u64 word");
        assert!(N <= 250, "requires <= 250 u64 words");
        let scale = Scale::new(table_scale()).expect("table scale is valid");
        Self {
            scale,
            width: Width::B1,
            min_width: Width::B1,
            word_base: 0,
            word_start: 0,
            word_end: 0,
            collapsed: false,
            live_total: 0,
            stats: Stats::EMPTY,
            data: [0u64; N],
        }
    }

    /// Sets the fixed scale.
    ///
    /// The relative-error guarantee is `α = (b - 1) / (b + 1)` with
    /// `b = 2^(2^-scale)`.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError::InvalidScale`] if `scale` is outside
    /// [`MIN_SCALE`](crate::MIN_SCALE)..=[`table_scale()`](crate::table_scale).
    #[inline]
    pub fn with_scale(mut self, scale: i32) -> Result<Self, ScaleError> {
        self.scale = Scale::new(scale)?;
        Ok(self)
    }

    /// Sets the minimum (starting) counter width.
    #[inline]
    #[must_use]
    pub fn with_min_width(mut self, width: Width) -> Self {
        self.min_width = width;
        self.width = width;
        self
    }

    /// Returns the fixed scale.
    #[inline]
    pub fn scale(&self) -> i32 {
        self.scale.scale()
    }

    /// Returns the current counter width.
    #[inline]
    pub const fn width(&self) -> Width {
        self.width
    }

    /// Returns the total observation count (including zeros).
    #[inline]
    pub const fn count(&self) -> u64 {
        self.stats.count
    }

    /// Returns the arithmetic sum of observed values.
    #[inline]
    pub const fn sum(&self) -> f64 {
        if self.stats.count == 0 {
            0.0
        } else {
            self.stats.sum
        }
    }

    /// Returns the minimum observed value (0.0 if empty).
    #[inline]
    pub fn min(&self) -> f64 {
        if self.stats.count == 0 {
            0.0
        } else {
            self.stats.min
        }
    }

    /// Returns the maximum observed value (0.0 if empty).
    #[inline]
    pub fn max(&self) -> f64 {
        if self.stats.count == 0 {
            0.0
        } else {
            self.stats.max
        }
    }

    /// Returns true once the lowest slot is an underflow placeholder.
    #[inline]
    pub const fn collapsed(&self) -> bool {
        self.collapsed
    }

    /// Returns true if no non-zero values have been bucketed.
    #[inline]
    pub const fn buckets_empty(&self) -> bool {
        self.live_total == 0
    }

    /// Slot index of the lowest slot in the window (the underflow slot
    /// once `collapsed`).
    #[inline]
    fn low_slot(&self) -> i32 {
        self.width.word_to_slot_index(self.word_start)
    }

    /// Slot index of the highest slot in the window.
    #[inline]
    fn high_slot(&self) -> i32 {
        self.width.word_to_slot_index(self.word_end + 1) - 1
    }

    /// Count held in the underflow placeholder (0 if not collapsed).
    #[inline]
    pub fn underflow_count(&self) -> u64 {
        if self.collapsed {
            Self::read_bucket(&self.data, self.width, self.word_base, self.low_slot())
        } else {
            0
        }
    }

    /// Records a single observation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extreme`] if the value is NaN, ±Inf, or negative.
    /// Returns [`Error::Overflow`] if the total count would exceed `u64::MAX`.
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<(), Error> {
        self.record_incr(value, 1)
    }

    /// Records a value with a specified increment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extreme`] if the value is NaN, ±Inf, or negative.
    /// Returns [`Error::Overflow`] if the total count would exceed `u64::MAX`.
    pub fn record_incr(&mut self, value: f64, incr: u64) -> Result<(), Error> {
        let mut biased_exp = get_biased_exponent(value);
        let mut significand = get_significand(value);

        let new_count = self.stats.count.checked_add(incr).ok_or(Error::Overflow)?;

        match biased_exp {
            0 => {
                if significand == 0 {
                    // Zero: counted but not bucketed (matches HistogramNN).
                    self.stats.count = new_count;
                    return Ok(());
                } else if value.is_sign_negative() {
                    return Err(Error::Extreme);
                } else {
                    biased_exp = 1;
                    significand = 1;
                }
            }
            NAN_INF_BIASED => return Err(Error::Extreme),
            _ => {
                if value.is_sign_negative() {
                    return Err(Error::Extreme);
                }
            }
        }

        let base2_exp = unbias_exponent(biased_exp);
        let slot = self.scale.map_decomposed(significand, base2_exp);

        self.place(slot, incr);

        self.stats.min = self.stats.min.min(value);
        self.stats.max = self.stats.max.max(value);
        self.stats.sum += value * incr as f64;
        self.stats.count = new_count;
        Ok(())
    }

    /// Reads the counter at `slot` from `buf` under `(width, base)`.
    #[inline]
    fn read_bucket(buf: &[u64; N], width: Width, base: i32, slot: i32) -> u64 {
        let addr = width.slot_addr(slot);
        let idx = addr.data_index(N, base);
        addr.retrieve_counter(buf[idx])
    }

    /// Adds `incr` to the counter at `slot`, assuming it fits `width`.
    #[inline]
    fn write_add(buf: &mut [u64; N], width: Width, base: i32, slot: i32, incr: u64) {
        let addr = width.slot_addr(slot);
        let idx = addr.data_index(N, base);
        let cur = addr.retrieve_counter(buf[idx]);
        buf[idx] = addr.update_counter_in_word(buf[idx], cur + incr);
    }

    /// Adds `incr` to `slot` in the live buffer, or reports the would-be
    /// total on overflow.
    #[inline]
    fn bucket_try_add(&mut self, slot: i32, incr: u64) -> Result<(), u64> {
        let addr = self.width.slot_addr(slot);
        let idx = addr.data_index(N, self.word_base);
        let cur = addr.retrieve_counter(self.data[idx]);
        let total = cur + incr;
        if total > self.width.counter_max() {
            return Err(total);
        }
        self.data[idx] = addr.update_counter_in_word(self.data[idx], total);
        Ok(())
    }

    /// Zero-fills the physical words for the logical range `lo..=hi`.
    #[inline]
    fn zero_fill(&mut self, lo: i32, hi: i32) {
        let mut w = lo;
        while w <= hi {
            let idx = (w - self.word_base).rem_euclid(N as i32) as usize;
            self.data[idx] = 0;
            w += 1;
        }
    }

    /// Places `incr` at bucket `slot`, performing collapse/widen as needed.
    fn place(&mut self, slot: i32, incr: u64) {
        if self.buckets_empty() {
            // Initialize at a width that fits the first increment.
            let w = self.min_width.max(Width::from_max_value(incr));
            self.width = w;
            let widx = w.slot_to_word_index(slot);
            self.word_base = widx;
            self.word_start = widx;
            self.word_end = widx;
            Self::write_add(&mut self.data, w, self.word_base, slot, incr);
            self.live_total += incr;
            return;
        }

        loop {
            match self.try_place_once(slot, incr) {
                Place::Done => return,
                Place::Widen(total) => {
                    let want = Width::from_max_value(total);
                    self.collapse_and_repack(want, self.high_slot());
                }
                Place::SlideUp(widx) => {
                    let top = self.width.word_to_slot_index(widx + 1) - 1;
                    self.collapse_and_repack(self.width, top);
                }
            }
        }
    }

    /// One placement attempt at the current width. Mutates the window for
    /// in-range / adjacent inserts; signals collapse/widen otherwise.
    fn try_place_once(&mut self, slot: i32, incr: u64) -> Place {
        let widx = self.width.slot_to_word_index(slot);
        let cap_words = N as i32;

        // Already collapsed and the value is at or below the underflow
        // placeholder: fold it in.
        if self.collapsed && slot <= self.low_slot() {
            return self.add_or_widen(self.low_slot(), incr);
        }

        if widx >= self.word_start && widx <= self.word_end {
            return self.add_or_widen(slot, incr);
        }

        if widx > self.word_end {
            if widx - self.word_start < cap_words {
                self.zero_fill(self.word_end + 1, widx);
                self.word_end = widx;
                return self.add_or_widen(slot, incr);
            }
            return Place::SlideUp(widx);
        }

        // widx < word_start
        if self.word_end - widx < cap_words {
            self.zero_fill(widx, self.word_start - 1);
            self.word_start = widx;
            return self.add_or_widen(slot, incr);
        }

        // Too far below the top to keep accurately: fold into the lowest
        // slot, which becomes the underflow placeholder.
        self.collapsed = true;
        self.add_or_widen(self.low_slot(), incr)
    }

    /// Adds to a slot, updating `live_total`, or returns `Widen` on
    /// counter overflow.
    #[inline]
    fn add_or_widen(&mut self, slot: i32, incr: u64) -> Place {
        match self.bucket_try_add(slot, incr) {
            Ok(()) => {
                self.live_total += incr;
                Place::Done
            }
            Err(total) => Place::Widen(total),
        }
    }

    /// Rebuilds the window keeping the top slots ending at `top_slot`,
    /// folding everything below the retained range into the lowest slot.
    ///
    /// `want_width` is the minimum output width (widened further if the
    /// folded sum or a kept counter requires it). The scale is unchanged.
    fn collapse_and_repack(&mut self, want_width: Width, top_slot: i32) {
        let old_w = self.width;
        let old_base = self.word_base;
        let old_lo = old_w.word_to_slot_index(self.word_start);
        let old_hi = old_w.word_to_slot_index(self.word_end + 1) - 1;
        let clone = self.data;

        // Pick the output width and window together: widening shrinks
        // capacity, which can fold more mass, which can require more
        // width. Iterate to a fixed point (bounded by U64). The window is
        // anchored at the top word so it is always exactly N words wide.
        let mut w = want_width;
        let low;
        let new_start;
        let new_end;
        let folded_below;
        loop {
            let we = w.slot_to_word_index(top_slot);
            let ws = we - (N as i32 - 1);
            let lw = w.word_to_slot_index(ws);

            let mut at_or_below = 0u64;
            let mut above_max = 0u64;
            let mut below = 0u64;
            let mut s = old_lo;
            while s <= old_hi {
                let c = Self::read_bucket(&clone, old_w, old_base, s);
                if c != 0 {
                    if s <= lw {
                        at_or_below += c;
                        if s < lw {
                            below += c;
                        }
                    } else if c > above_max {
                        above_max = c;
                    }
                }
                s += 1;
            }

            let need = w
                .max(Width::from_max_value(at_or_below))
                .max(Width::from_max_value(above_max));
            if need == w {
                low = lw;
                new_start = ws;
                new_end = we;
                folded_below = below;
                break;
            }
            w = need;
        }

        // Scatter into a fresh zeroed buffer at the new width/anchor.
        self.data = [0u64; N];
        let new_base = new_start;
        let mut s = old_lo;
        while s <= old_hi {
            let c = Self::read_bucket(&clone, old_w, old_base, s);
            if c != 0 {
                let dest = if s < low { low } else { s };
                Self::write_add(&mut self.data, w, new_base, dest, c);
            }
            s += 1;
        }

        self.width = w;
        self.word_base = new_base;
        self.word_start = new_start;
        self.word_end = new_end;
        debug_assert!(
            self.word_end - self.word_start < N as i32,
            "collapse produced a window wider than N words"
        );
        if folded_below > 0 {
            self.collapsed = true;
        }
    }

    /// Calls `f(slot_index, count)` for each non-zero bucket, low to high.
    pub fn for_each_bucket(&self, mut f: impl FnMut(i32, u64)) {
        if self.buckets_empty() {
            return;
        }
        let lo = self.low_slot();
        let hi = self.high_slot();
        let mut s = lo;
        while s <= hi {
            let c = Self::read_bucket(&self.data, self.width, self.word_base, s);
            if c != 0 {
                f(s, c);
            }
            s += 1;
        }
    }

    /// Returns the slot index of the lowest non-zero bucket, or `None`.
    pub fn offset(&self) -> Option<i32> {
        let mut first = None;
        self.for_each_bucket(|s, _| {
            if first.is_none() {
                first = Some(s);
            }
        });
        first
    }
}

#[cfg(feature = "quantile")]
impl<const N: usize> Sketch<N> {
    /// Estimates the value at quantile `q` in `[0.0, 1.0]`.
    ///
    /// `q <= 0.0` returns `min`, `q >= 1.0` returns `max`. For interior
    /// quantiles this returns the relative-error-optimal representative of
    /// the rank-selected bucket, `2·L·U / (L + U)` for that bucket's
    /// boundaries `L <= U`.
    ///
    /// # Guarantee
    ///
    /// For any `q` whose true value lies above the underflow placeholder
    /// (rank above [`underflow_count`](Self::underflow_count) plus the zero
    /// count), the estimate is within `α = (b - 1)/(b + 1)`
    /// (`b = 2^(2^-scale)`) of the true value.
    ///
    /// Returns `NaN` for an empty sketch.
    pub fn quantile(&self, q: f64) -> f64 {
        let n = self.stats.count;
        if n == 0 {
            return f64::NAN;
        }
        if q <= 0.0 {
            return self.min();
        }
        if q >= 1.0 {
            return self.max();
        }

        let target = q * n as f64;

        // Zero observations contribute CDF mass at value 0.0 first.
        let zero_count = n - self.live_total;
        let mut cum = zero_count as f64;
        if cum >= target {
            return 0.0;
        }

        let mut chosen: Option<i32> = None;
        self.for_each_bucket(|idx, c| {
            if chosen.is_none() {
                cum += c as f64;
                if cum >= target {
                    chosen = Some(idx);
                }
            }
        });

        chosen.map_or_else(|| self.max(), |idx| self.representative(idx))
    }

    /// Relative-error-optimal representative of bucket `idx`: the point
    /// `2LU/(L+U)` whose worst-case relative error over `[L, U]` is exactly
    /// `α`. Computed to avoid overflow when `L·U` is huge.
    fn representative(&self, idx: i32) -> f64 {
        let l = self.scale.lower_boundary(idx).unwrap_or_else(|_| self.min());
        let u = self
            .scale
            .lower_boundary(idx + 1)
            .unwrap_or_else(|_| self.max());
        if l <= 0.0 {
            u
        } else {
            l * 2.0 / (1.0 + l / u)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative error bound `α = (b - 1)/(b + 1)`, `b = 2^(2^-scale)`.
    fn alpha(scale: i32) -> f64 {
        let b = 2f64.powf(2f64.powi(-scale));
        (b - 1.0) / (b + 1.0)
    }

    fn bucket_lower(scale: i32, index: i32) -> f64 {
        // Lower boundary of bucket `index` at the given scale.
        2f64.powf((index as f64) * 2f64.powi(-scale))
    }

    #[test]
    fn basic_stats() {
        let mut s: Sketch<16> = Sketch::new().with_scale(4).unwrap();
        s.update(1.5).unwrap();
        s.update(2.7).unwrap();
        s.update(100.0).unwrap();
        assert_eq!(s.count(), 3);
        assert_eq!(s.scale(), 4);
        assert_eq!(s.min(), 1.5);
        assert_eq!(s.max(), 100.0);
        assert!((s.sum() - 104.2).abs() < 1e-9);
    }

    #[test]
    fn rejects_extremes() {
        let mut s: Sketch<8> = Sketch::new();
        assert_eq!(s.update(f64::NAN), Err(Error::Extreme));
        assert_eq!(s.update(f64::INFINITY), Err(Error::Extreme));
        assert_eq!(s.update(-1.0), Err(Error::Extreme));
        // Zero is counted but not bucketed.
        s.update(0.0).unwrap();
        assert_eq!(s.count(), 1);
        assert!(s.buckets_empty());
    }

    #[test]
    fn scale_is_fixed_under_wide_range() {
        // HistogramNN would downscale here; Sketch must not.
        let mut s: Sketch<8> = Sketch::new().with_scale(3).unwrap();
        for e in -40..40 {
            s.update(2f64.powi(e)).unwrap();
        }
        assert_eq!(s.scale(), 3, "scale must stay fixed");
    }

    #[test]
    fn count_preserved_under_collapse() {
        let mut s: Sketch<4> = Sketch::new().with_scale(4).unwrap();
        let mut total = 0u64;
        for e in -60..60 {
            let v = 2f64.powi(e);
            s.update(v).unwrap();
            total += 1;
        }
        assert_eq!(s.count(), total);
        assert!(s.collapsed(), "wide range must force collapse");

        // The sum of all bucket counts plus zeros equals the total.
        let mut bucket_sum = 0u64;
        s.for_each_bucket(|_, c| bucket_sum += c);
        assert_eq!(bucket_sum, total, "no counts lost during collapse");
    }

    #[test]
    fn top_of_distribution_stays_accurate() {
        // The largest value must land in an accurate (non-underflow)
        // bucket whose boundaries bracket it within the guarantee.
        let scale = 4;
        let mut s: Sketch<4> = Sketch::new().with_scale(scale).unwrap();
        for e in -60..60 {
            s.update(2f64.powi(e)).unwrap();
        }
        let maxv = s.max();
        assert_eq!(maxv, 2f64.powi(59));

        // Find the top bucket; it must bracket the max value.
        let mut top = None;
        s.for_each_bucket(|idx, _| top = Some(idx));
        let top = top.unwrap();
        let lo = bucket_lower(scale, top);
        let hi = bucket_lower(scale, top + 1);
        assert!(lo <= maxv && maxv <= hi, "max {maxv} not in [{lo},{hi}]");

        // Relative width within the guarantee.
        let a = alpha(scale);
        assert!((hi - lo) / (hi + lo) <= a + 1e-12);
    }

    #[test]
    fn below_window_folds_into_underflow() {
        let scale = 4;
        let mut s: Sketch<2> = Sketch::new().with_scale(scale).unwrap();
        // Establish a window around large values.
        for _ in 0..3 {
            s.update(1e6).unwrap();
        }
        let before = s.collapsed();
        // A tiny value is far below the window; it must fold into the
        // underflow placeholder rather than change the scale.
        s.update(1e-6).unwrap();
        assert_eq!(s.scale(), scale);
        assert!(s.collapsed() || !before);
        assert!(s.underflow_count() >= 1, "tiny value folded into underflow");
        assert_eq!(s.max(), 1e6);
        assert_eq!(s.min(), 1e-6, "min still tracked exactly");
    }

    #[test]
    fn spread_widen_preserves_counts_and_scale() {
        // Many observations of the same value force counter widening
        // (B1 -> ...). Scale must stay fixed and the count exact.
        let mut s: Sketch<8> = Sketch::new().with_scale(6).unwrap();
        let n = 5000u64;
        for _ in 0..n {
            s.update(42.0).unwrap();
        }
        assert_eq!(s.count(), n);
        assert_eq!(s.scale(), 6, "widening must not change scale");
        assert!(s.width() > Width::B1, "counter must have widened");

        let mut bucket_sum = 0u64;
        s.for_each_bucket(|_, c| bucket_sum += c);
        assert_eq!(bucket_sum, n);
    }

    #[test]
    fn widen_then_collapse_keeps_top() {
        // Fill many buckets at a narrow width, then drive counts up so a
        // widen halves capacity and forces a left-collapse.
        let scale = 5;
        let mut s: Sketch<4> = Sketch::new().with_scale(scale).unwrap();

        // A spread of distinct values to populate many buckets.
        for i in 1..=200 {
            s.update(i as f64).unwrap();
        }
        // Then hammer the largest value to widen counters.
        for _ in 0..100_000 {
            s.update(200.0).unwrap();
        }
        assert_eq!(s.scale(), scale);
        assert_eq!(s.max(), 200.0);

        // Total preserved.
        let mut bucket_sum = 0u64;
        s.for_each_bucket(|_, c| bucket_sum += c);
        assert_eq!(bucket_sum, s.count());

        // The top bucket brackets the max.
        let mut top = None;
        s.for_each_bucket(|idx, _| top = Some(idx));
        let top = top.unwrap();
        let lo = bucket_lower(scale, top);
        let hi = bucket_lower(scale, top + 1);
        assert!(lo <= 200.0 && 200.0 <= hi);
    }

    #[test]
    fn accurate_until_window_full() {
        // While everything fits, no collapse and every bucket is accurate.
        let scale = 2;
        let mut s: Sketch<16> = Sketch::new().with_scale(scale).unwrap();
        // 16 words * 64 slots = 1024 B1 slots; scale 2 covers a huge range.
        for i in 1..=50 {
            s.update(i as f64).unwrap();
        }
        assert!(!s.collapsed(), "small range must not collapse");
        assert_eq!(s.underflow_count(), 0);

        // Every populated bucket brackets its contributing values.
        let a = alpha(scale);
        s.for_each_bucket(|idx, _| {
            let lo = bucket_lower(scale, idx);
            let hi = bucket_lower(scale, idx + 1);
            assert!((hi - lo) / (hi + lo) <= a + 1e-12);
        });
    }

    /// Before any collapse, the structure is a ring buffer anchored at a
    /// fixed `word_base` that expands in BOTH directions (wrapping). This
    /// confirms anchoring did not turn growth into a one-directional or
    /// re-anchoring scheme.
    #[test]
    fn ring_expands_both_directions_before_collapse() {
        // Scale 6 at B1: 64 slots/word = 1 octave/word, so distinct
        // octaves occupy distinct words. N=8 words fits 7 octaves.
        let mut s: Sketch<8> = Sketch::new().with_scale(6).unwrap();

        // First insert establishes the anchor.
        s.update(2f64.powi(0)).unwrap();
        let base0 = s.word_base;

        // Expand upward and downward around the anchor, staying in N words.
        for e in [1, 2, 3, -1, -2, -3] {
            s.update(2f64.powi(e)).unwrap();
        }

        // Anchor unchanged → genuine ring buffer (not re-anchored on grow).
        assert_eq!(s.word_base, base0, "word_base must stay fixed while growing");
        // Expanded strictly in both directions (wrapped around the anchor).
        assert!(s.word_start < base0, "did not expand downward past anchor");
        assert!(s.word_end > base0, "did not expand upward past anchor");
        // Still within one pool, no collapse, counters never widened.
        assert!((s.word_end - s.word_start) < 8, "window exceeded N words");
        assert!(!s.collapsed());
        assert_eq!(s.underflow_count(), 0);
        assert_eq!(s.width(), Width::B1);
        assert_eq!(s.count(), 7);

        let mut bs = 0u64;
        s.for_each_bucket(|_, c| bs += c);
        assert_eq!(bs, 7, "counts lost while wrapping");
    }

    /// Heavy interleaved widen + collapse with a tiny pool. The window
    /// must never exceed N words and no counts may be lost — this is what
    /// catches ring-buffer overflow from mis-sized collapse windows.
    #[test]
    fn stress_invariants_tiny_pool() {
        const N: usize = 2;
        let scale = 4;
        let mut s: Sketch<N> = Sketch::new().with_scale(scale).unwrap();

        // Deterministic LCG; map to a wide dynamic range with repeats so
        // both counter widening and left-collapse fire repeatedly.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };

        let mut bucket_sum_check = 0u64;
        for i in 0..50_000u64 {
            let r = next();
            // Exponent spread across ~120 octaves; occasional hot repeats.
            let exp = ((r >> 40) % 240) as i32 - 120;
            let frac = 1.0 + ((r >> 8) & 0xffff) as f64 / 65536.0;
            let v = frac * 2f64.powi(exp);
            s.update(v).unwrap();
            bucket_sum_check += 1;

            if i % 997 == 0 {
                // Window never exceeds N words.
                let span_words = s.word_end - s.word_start + 1;
                assert!(
                    span_words >= 1 && span_words <= N as i32,
                    "window span {span_words} out of [1,{N}] at i={i}",
                );
                // No counts lost.
                let mut bs = 0u64;
                s.for_each_bucket(|_, c| bs += c);
                assert_eq!(bs, bucket_sum_check, "count loss at i={i}");
            }
        }

        // Final tallies.
        assert_eq!(s.count(), 50_000);
        let mut bs = 0u64;
        s.for_each_bucket(|_, c| bs += c);
        assert_eq!(bs, 50_000);
        assert_eq!(s.scale(), scale, "scale must stay fixed throughout");

        // Top bucket brackets the max.
        let maxv = s.max();
        let mut top = None;
        s.for_each_bucket(|idx, _| top = Some(idx));
        let top = top.unwrap();
        let lo = bucket_lower(scale, top);
        let hi = bucket_lower(scale, top + 1);
        assert!(lo <= maxv && maxv <= hi, "max {maxv} not in top bucket [{lo},{hi}]");
    }

    /// Empirically verifies the DDSketch-style relative-error guarantee:
    /// for every quantile whose true value lies above the underflow
    /// placeholder, the estimate is within `α` of the brute-force truth.
    #[cfg(feature = "quantile")]
    fn check_guarantee<const M: usize>(s: &Sketch<M>, raw: &mut [f64], a: f64) {
        let n = raw.len();
        raw.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let mut live = 0u64;
        s.for_each_bucket(|_, c| live += c);
        let zero_count = s.count() - live;
        // Ranks at or below this are in the inaccurate (underflow/zero)
        // region and excluded from the guarantee.
        let floor_rank = zero_count + s.underflow_count();

        let mut tested = 0usize;
        let mut worst = 0.0f64;
        for i in 1..1000 {
            let q = i as f64 / 1000.0;
            let rank = (q * n as f64).ceil() as u64; // 1-based
            if rank <= floor_rank {
                continue;
            }
            let true_q = raw[(rank as usize - 1).min(n - 1)];
            if true_q <= 0.0 {
                continue;
            }
            let est = s.quantile(q);
            let rel = (est - true_q).abs() / true_q;
            worst = worst.max(rel);
            assert!(
                rel <= a * 1.0001 + 1e-12,
                "q={q} rank={rank} est={est} true={true_q} rel={rel:.6} > α={a:.6}",
            );
            tested += 1;
        }
        eprintln!(
            "guarantee: N={M} scale={} α={a:.4} collapsed={} underflow={} tested={tested} worst_rel={worst:.4}",
            s.scale(),
            s.collapsed(),
            s.underflow_count(),
        );
        assert!(tested > 50, "guarantee test too sparse: {tested} points");
    }

    /// Wide range + fine scale + small pool: forces left-collapse. The
    /// guarantee must hold for the accurate (upper) part of the
    /// distribution despite a large underflow mass.
    #[cfg(feature = "quantile")]
    #[test]
    fn guarantee_holds_under_collapse() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};

        let scale = 4;
        let mut s: Sketch<16> = Sketch::new().with_scale(scale).unwrap();
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        // Log-uniform over 24 octaves: a controlled, very wide spread.
        let octaves = Uniform::new(0.0f64, 24.0);
        let n = 200_000usize;
        let mut raw = Vec::with_capacity(n);
        for _ in 0..n {
            let v = 2f64.powf(octaves.sample(&mut rng));
            raw.push(v);
            s.update(v).unwrap();
        }

        assert!(s.collapsed(), "wide range must collapse");
        assert!(s.underflow_count() > 0);
        check_guarantee(&s, &mut raw, alpha(scale));
    }

    /// Bounded range + large pool: no collapse, so the guarantee must hold
    /// across the entire distribution.
    #[cfg(feature = "quantile")]
    #[test]
    fn guarantee_holds_without_collapse() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};

        let scale = 4;
        let mut s: Sketch<64> = Sketch::new().with_scale(scale).unwrap();
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let dist = Uniform::new(1.0f64, 8.0); // ~3 octaves, fits the window
        let n = 200_000usize;
        let mut raw = Vec::with_capacity(n);
        for _ in 0..n {
            let v = dist.sample(&mut rng);
            raw.push(v);
            s.update(v).unwrap();
        }

        assert!(!s.collapsed(), "bounded range must not collapse");
        assert_eq!(s.underflow_count(), 0);
        check_guarantee(&s, &mut raw, alpha(scale));
    }

    /// The optimal representative must beat the naive lower-boundary
    /// estimator: its worst-case relative error is `α`, not `≈ 2α`.
    #[cfg(feature = "quantile")]
    #[test]
    fn representative_achieves_alpha_not_two_alpha() {
        let scale = 3;
        let s: Sketch<8> = Sketch::new().with_scale(scale).unwrap();
        let a = alpha(scale);
        // Probe every bucket across a wide index range: the representative
        // must lie within α of both boundaries.
        for idx in -200..200 {
            let l = bucket_lower(scale, idx);
            let u = bucket_lower(scale, idx + 1);
            let r = s.representative(idx);
            assert!((r - l) / l <= a + 1e-9, "idx {idx}: lower err exceeds α");
            assert!((u - r) / u <= a + 1e-9, "idx {idx}: upper err exceeds α");
        }
    }
}
