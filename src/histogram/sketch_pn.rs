// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixed-scale collapse-left histogram for values of any sign.
//!
//! [`SketchPN<K, L>`] pairs two [`Sketch`]es — one for positive values
//! (`K` words) and one for the magnitudes of negative values (`L` words) —
//! plus a zero count, giving the DDSketch-style relative-error guarantee
//! over the whole real line. Both ranges share the same fixed scale, so —
//! unlike [`HistogramPN`](super::HistogramPN) — no scale synchronization is
//! ever needed.

use core::fmt;

use crate::float64::{get_biased_exponent, get_significand, NAN_INF_BIASED};
use crate::mapping::ScaleError;

use super::sketch::{Sketch, SketchBucketView};
use super::width::Width;
use super::{Error, Stats};

/// A fixed-scale, collapse-left exponential histogram for values of any
/// sign.
///
/// `K` is the positive-range pool size in `u64` words; `L` is the
/// negative-range pool size. Both ranges share one fixed scale and carry
/// the relative-error guarantee of [`Sketch`].
///
/// # Examples
///
/// ```
/// use otel_expohisto::SketchPN;
///
/// let mut s: SketchPN<8, 4> = SketchPN::new().with_scale(4).unwrap();
/// s.update(100.0).unwrap();
/// s.update(-0.5).unwrap();
/// s.update(0.0).unwrap();
///
/// let v = s.view();
/// assert_eq!(v.stats().count, 3);
/// assert_eq!(v.zero_count(), 1);
/// assert!(!v.positive().is_empty());
/// assert!(!v.negative().is_empty());
/// ```
pub struct SketchPN<const K: usize, const L: usize> {
    positive: Sketch<K>,
    negative: Sketch<L>,

    sum: f64,
    min: f64,
    max: f64,
    zero_count: u64,
}

impl<const K: usize, const L: usize> Clone for SketchPN<K, L> {
    fn clone(&self) -> Self {
        Self {
            positive: self.positive.clone(),
            negative: self.negative.clone(),
            sum: self.sum,
            min: self.min,
            max: self.max,
            zero_count: self.zero_count,
        }
    }
}

impl<const K: usize, const L: usize> Default for SketchPN<K, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize, const L: usize> fmt::Debug for SketchPN<K, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.aggregate_stats();
        f.debug_struct("SketchPN")
            .field("scale", &self.view().scale())
            .field("count", &stats.count)
            .field("sum", &stats.sum)
            .field("min", &stats.min)
            .field("max", &stats.max)
            .field("zero_count", &self.zero_count)
            .finish()
    }
}

impl<const K: usize, const L: usize> SketchPN<K, L> {
    /// Creates a new sketch at the maximum table scale and `B1` width.
    ///
    /// # Panics
    ///
    /// Panics if either `K` or `L` is `0` or greater than `250`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            positive: Sketch::new(),
            negative: Sketch::new(),
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            zero_count: 0,
        }
    }

    /// Sets the fixed scale for both ranges.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError::InvalidScale`] if `scale` is outside
    /// [`MIN_SCALE`](crate::MIN_SCALE)..=[`table_scale()`](crate::table_scale).
    #[inline]
    pub fn with_scale(mut self, scale: i32) -> Result<Self, ScaleError> {
        self.positive = self.positive.with_scale(scale)?;
        self.negative = self.negative.with_scale(scale)?;
        Ok(self)
    }

    /// Sets the fixed scale for both ranges to the coarsest value whose
    /// worst-case relative error does not exceed `target`.
    ///
    /// See [`Sketch::with_relative_error`]. Requires the `std` feature.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError::InvalidScale`] if `target` is not in `(0, 1)`
    /// or no supported scale is fine enough to meet it.
    #[cfg(feature = "std")]
    pub fn with_relative_error(mut self, target: f64) -> Result<Self, ScaleError> {
        self.positive = self.positive.with_relative_error(target)?;
        self.negative = self.negative.with_relative_error(target)?;
        Ok(self)
    }

    /// Sets the minimum (starting) counter width for both ranges.
    #[inline]
    #[must_use]
    pub fn with_min_width(mut self, width: Width) -> Self {
        self.positive = self.positive.with_min_width(width);
        self.negative = self.negative.with_min_width(width);
        self
    }

    /// Returns the fixed scale (shared by both ranges).
    #[inline]
    pub fn scale(&self) -> i32 {
        self.positive.scale()
    }

    /// Returns the total observation count (positive, negative, and zero).
    #[inline]
    pub fn count(&self) -> u64 {
        self.total_count()
    }

    /// Returns the minimum observed value (0.0 if empty).
    #[inline]
    pub fn min(&self) -> f64 {
        self.aggregate_stats().min
    }

    /// Returns the maximum observed value (0.0 if empty).
    #[inline]
    pub fn max(&self) -> f64 {
        self.aggregate_stats().max
    }

    /// Returns the arithmetic sum of observed values.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.aggregate_stats().sum
    }

    /// Returns a read-only, OTel-export-shaped view.
    #[inline]
    pub fn view(&self) -> SketchPNView<'_, K, L> {
        SketchPNView { pn: self }
    }

    /// Returns a read-only view of the positive-range buckets.
    #[inline]
    pub fn positive(&self) -> &Sketch<K> {
        &self.positive
    }

    /// Returns a read-only view of the negative-range buckets.
    ///
    /// The bucket indices describe the distribution of `|value|` over all
    /// recorded negative values.
    #[inline]
    pub fn negative(&self) -> &Sketch<L> {
        &self.negative
    }

    /// Records a single value (positive, negative, or zero).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extreme`] if the value is NaN, +Inf, or -Inf.
    /// Returns [`Error::Overflow`] if the total count would exceed `u64::MAX`.
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<(), Error> {
        self.record_incr(value, 1)
    }

    /// Records a value with a specified increment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extreme`] if the value is NaN, +Inf, or -Inf.
    /// Returns [`Error::Overflow`] if the total count would exceed `u64::MAX`.
    pub fn record_incr(&mut self, value: f64, incr: u64) -> Result<(), Error> {
        let biased_exp = get_biased_exponent(value);
        let significand = get_significand(value);

        self.total_count().checked_add(incr).ok_or(Error::Overflow)?;

        match biased_exp {
            0 if significand == 0 => {
                // Both +0.0 and -0.0 are treated as zero.
                self.zero_count = self.zero_count.checked_add(incr).ok_or(Error::Overflow)?;
                self.min = self.min.min(0.0);
                self.max = self.max.max(0.0);
                return Ok(());
            }
            NAN_INF_BIASED => return Err(Error::Extreme),
            _ => {}
        }

        // Pre-validation guarantees the inner calls succeed: NaN/Inf are
        // rejected, zero is handled, and the total-count check above bounds
        // each sub-sketch's individual count.
        if value.is_sign_negative() {
            self.negative.record_incr(-value, incr)?;
        } else {
            self.positive.record_incr(value, incr)?;
        }

        self.sum += value * incr as f64;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        Ok(())
    }

    /// Merges another sketch into this one. Pool sizes may differ.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ScaleMismatch`] if the two sketches were
    /// constructed at different scales. Returns [`Error::Overflow`] if the
    /// combined total count would exceed `u64::MAX`.
    pub fn merge_from<const K2: usize, const L2: usize>(
        &mut self,
        other: &SketchPN<K2, L2>,
    ) -> Result<(), Error> {
        let other_total = other.total_count();
        if other_total == 0 {
            return Ok(());
        }
        if self.scale() != other.scale() {
            return Err(Error::ScaleMismatch);
        }
        self.total_count()
            .checked_add(other_total)
            .ok_or(Error::Overflow)?;

        self.positive.merge_from(&other.positive)?;
        self.negative.merge_from(&other.negative)?;

        self.sum += other.sum;
        self.zero_count = self.zero_count.saturating_add(other.zero_count);
        if other.min != f64::INFINITY {
            self.min = self.min.min(other.min);
        }
        if other.max != f64::NEG_INFINITY {
            self.max = self.max.max(other.max);
        }
        Ok(())
    }

    /// Total count across both ranges and zeros.
    #[inline]
    fn total_count(&self) -> u64 {
        self.positive
            .count()
            .saturating_add(self.negative.count())
            .saturating_add(self.zero_count)
    }

    /// Aggregate stats from both ranges and zeros.
    fn aggregate_stats(&self) -> Stats {
        let total = self.total_count();
        if total == 0 {
            return Stats {
                count: 0,
                sum: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        Stats {
            count: total,
            sum: self.sum,
            min: if self.min == f64::INFINITY {
                0.0
            } else {
                self.min
            },
            max: if self.max == f64::NEG_INFINITY {
                0.0
            } else {
                self.max
            },
        }
    }
}

/// Read-only, OTel-export-shaped view of a [`SketchPN`].
///
/// Created by [`SketchPN::view`]. The negative range's bucket indices
/// describe `|value|`, so an exporter negates them for the OTel negative
/// buckets.
#[derive(Debug)]
pub struct SketchPNView<'a, const K: usize, const L: usize> {
    pn: &'a SketchPN<K, L>,
}

impl<const K: usize, const L: usize> SketchPNView<'_, K, L> {
    /// Returns the scale. Returns 0 when no non-zero value is recorded.
    #[inline]
    pub fn scale(&self) -> i32 {
        if self.pn.positive.buckets_empty() && self.pn.negative.buckets_empty() {
            0
        } else {
            self.pn.scale()
        }
    }

    /// Returns the aggregate statistics (count, sum, min, max). When the
    /// sketch is empty, min, max and sum are reported as 0.0.
    #[inline]
    pub fn stats(&self) -> Stats {
        self.pn.aggregate_stats()
    }

    /// Returns the count of exactly-zero observations.
    #[inline]
    pub fn zero_count(&self) -> u64 {
        self.pn.zero_count
    }

    /// Returns a contiguous read-only view of the positive buckets.
    #[inline]
    pub fn positive(&self) -> SketchBucketView<'_, K> {
        self.pn.positive.buckets()
    }

    /// Returns a contiguous read-only view of the negative buckets
    /// (indices describe `|value|`).
    #[inline]
    pub fn negative(&self) -> SketchBucketView<'_, L> {
        self.pn.negative.buckets()
    }

    /// Returns true if either range has collapsed (its lowest bucket is an
    /// underflow placeholder).
    #[inline]
    pub fn collapsed(&self) -> bool {
        self.pn.positive.collapsed() || self.pn.negative.collapsed()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    fn live_sum<const N: usize>(s: &Sketch<N>) -> u64 {
        let mut t = 0u64;
        s.for_each_bucket(|_, c| t += c);
        t
    }

    #[test]
    fn empty() {
        let s: SketchPN<8, 4> = SketchPN::new().with_scale(4).unwrap();
        let v = s.view();
        assert_eq!(v.scale(), 0);
        assert_eq!(v.stats().count, 0);
        assert_eq!(v.stats().min, 0.0);
        assert_eq!(v.stats().max, 0.0);
        assert_eq!(v.zero_count(), 0);
        assert!(v.positive().is_empty());
        assert!(v.negative().is_empty());
        assert!(!v.collapsed());
    }

    #[test]
    fn dispatch_by_sign_and_stats() {
        let mut s: SketchPN<8, 8> = SketchPN::new().with_scale(4).unwrap();
        s.update(100.0).unwrap();
        s.update(-2.0).unwrap();
        s.update(-8.0).unwrap();
        s.update(0.0).unwrap();
        s.update(-0.0).unwrap();

        let v = s.view();
        assert_eq!(v.scale(), 4);
        let st = v.stats();
        assert_eq!(st.count, 5);
        assert_eq!(v.zero_count(), 2, "+0.0 and -0.0 both count as zero");
        assert_eq!(st.min, -8.0, "most-negative value is the min");
        assert_eq!(st.max, 100.0);
        assert!((st.sum - (100.0 - 2.0 - 8.0)).abs() < 1e-9);

        // One positive bucket, two negative buckets.
        assert_eq!(live_sum(s.positive()), 1);
        assert_eq!(live_sum(s.negative()), 2);
    }

    #[test]
    fn rejects_extremes() {
        let mut s: SketchPN<4, 4> = SketchPN::new();
        assert_eq!(s.update(f64::NAN), Err(Error::Extreme));
        assert_eq!(s.update(f64::INFINITY), Err(Error::Extreme));
        assert_eq!(s.update(f64::NEG_INFINITY), Err(Error::Extreme));
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn negative_range_collapses_independently() {
        // A wide negative spread collapses the negative range while the
        // positive range stays exact.
        let scale = 4;
        let mut s: SketchPN<16, 2> = SketchPN::new().with_scale(scale).unwrap();
        for e in -30..30 {
            s.update(-(2f64.powi(e))).unwrap();
        }
        s.update(5.0).unwrap();
        let v = s.view();
        assert!(v.negative().width() >= Width::B1);
        assert!(s.negative().collapsed(), "wide negative range collapses");
        assert!(!s.positive().collapsed(), "single positive value is exact");
        assert!(v.collapsed());
        // No counts lost.
        assert_eq!(
            live_sum(s.positive()) + live_sum(s.negative()) + v.zero_count(),
            s.count()
        );
        assert_eq!(s.max(), 5.0);
    }

    #[test]
    fn merge_combines_both_ranges_and_zeros() {
        let scale = 5;
        let mut a: SketchPN<8, 8> = SketchPN::new().with_scale(scale).unwrap();
        let mut b: SketchPN<8, 8> = SketchPN::new().with_scale(scale).unwrap();

        for i in 1..=50 {
            a.update(i as f64).unwrap();
            b.update(-(i as f64)).unwrap();
        }
        a.update(0.0).unwrap();
        b.update(0.0).unwrap();
        b.update(0.0).unwrap();

        a.merge_from(&b).unwrap();
        let v = a.view();
        assert_eq!(v.stats().count, 100 + 3);
        assert_eq!(v.zero_count(), 3);
        assert_eq!(v.stats().min, -50.0);
        assert_eq!(v.stats().max, 50.0);
        assert_eq!(
            live_sum(a.positive()) + live_sum(a.negative()) + v.zero_count(),
            a.count(),
        );
        assert_eq!(live_sum(a.positive()), 50);
        assert_eq!(live_sum(a.negative()), 50);
    }

    #[test]
    fn merge_different_pool_sizes() {
        let scale = 4;
        let mut a: SketchPN<8, 4> = SketchPN::new().with_scale(scale).unwrap();
        let mut b: SketchPN<16, 2> = SketchPN::new().with_scale(scale).unwrap();
        let mut total = 0u64;
        for i in 1..=200 {
            a.update(i as f64).unwrap();
            a.update(-(i as f64)).unwrap();
            b.update((i as f64) * 0.25).unwrap();
            b.update(-(i as f64) * 0.25).unwrap();
            total += 4;
        }
        a.merge_from(&b).unwrap();
        assert_eq!(a.count(), total);
        assert_eq!(
            live_sum(a.positive()) + live_sum(a.negative()),
            total,
            "no counts lost across pool sizes",
        );
        assert_eq!(a.view().stats().max, 200.0);
        assert_eq!(a.view().stats().min, -200.0);
    }

    #[test]
    fn merge_scale_mismatch_errs() {
        let mut a: SketchPN<8, 8> = SketchPN::new().with_scale(4).unwrap();
        let mut b: SketchPN<8, 8> = SketchPN::new().with_scale(5).unwrap();
        a.update(1.0).unwrap();
        b.update(1.0).unwrap();
        assert_eq!(a.merge_from(&b), Err(Error::ScaleMismatch));
        // Mismatched but empty source is a no-op.
        let c: SketchPN<8, 8> = SketchPN::new().with_scale(7).unwrap();
        assert_eq!(a.merge_from(&c), Ok(()));
    }

    #[cfg(feature = "std")]
    #[test]
    fn with_relative_error_sets_both_ranges() {
        let s: SketchPN<8, 8> = SketchPN::new().with_relative_error(0.1).unwrap();
        assert_eq!(s.scale(), 2);
        assert_eq!(s.positive().scale(), 2);
        assert_eq!(s.negative().scale(), 2);
        assert!(SketchPN::<8, 8>::new().with_relative_error(0.0).is_err());
    }
}
