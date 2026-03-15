// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Promoted read-only view of a histogram.

use crate::mapping::Mapping;

use super::bucket_view::BucketView;
use super::quantile::QuantileIter;
use super::Histogram;

/// Promoted read-only view of a histogram's data.
///
/// Created by [`Histogram::view`], which promotes from literal mode if
/// needed. All accessors take `&self`, so a `HistogramView` can be
/// shared freely once obtained.
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
/// assert!(v.sum() > 4.0);
/// println!("scale = {}, buckets = {}", v.scale(), v.positive().len());
/// ```
#[derive(Debug)]
pub struct HistogramView<'a, const N: usize> {
    pub(super) hist: &'a Histogram<N>,
}

impl<const N: usize> HistogramView<'_, N> {
    /// Returns the current scale.
    ///
    /// Returns 0 when no non-zero values have been recorded.
    #[inline]
    pub fn scale(&self) -> i32 {
        if self.hist.non_zero_count() == 0 {
            0
        } else {
            self.hist.mapping.scale()
        }
    }

    /// Returns the count of all recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        self.hist.stats.count
    }

    /// Returns the sum of all recorded values as `f64`.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.hist.stats.sum
    }

    /// Returns the minimum recorded value, or 0.0 if empty.
    #[inline]
    pub fn min(&self) -> f64 {
        self.hist.stats.min
    }

    /// Returns the maximum recorded value, or 0.0 if empty.
    #[inline]
    pub fn max(&self) -> f64 {
        self.hist.stats.max
    }

    /// Returns a read-only view of the positive buckets.
    #[inline]
    pub fn positive(&self) -> BucketView<'_, N> {
        BucketView { hist: self.hist }
    }

    /// Returns an iterator that estimates values at the requested quantiles.
    ///
    /// Each quantile must be in `[0.0, 1.0]` and the slice must be sorted
    /// in non-decreasing order. By definition, quantile 0.0 yields
    /// [`min()`](Self::min) and quantile 1.0 yields [`max()`](Self::max).
    ///
    /// The iterator walks the histogram's CDF exactly once, using linear
    /// interpolation within the bucket that straddles each threshold.
    /// Zero-valued observations contribute CDF mass at value 0.0 before
    /// any positive buckets.
    ///
    /// # Panics
    ///
    /// Debug-asserts that every quantile is in `[0.0, 1.0]` and that the
    /// slice is sorted.
    pub fn quantiles<'a>(&'a self, quantiles: &'a [f64]) -> QuantileIter<'a, N> {
        debug_assert!(
            quantiles.windows(2).all(|w| w[0] <= w[1]),
            "quantiles must be sorted in non-decreasing order"
        );
        debug_assert!(
            quantiles.iter().all(|&q| (0.0..=1.0).contains(&q)),
            "quantiles must be in [0.0, 1.0]"
        );

        let total_count = self.count();
        let nz = self.hist.non_zero_count();
        let zero_count = total_count.saturating_sub(nz);
        let min = self.min();
        let max = self.max();
        let mapping = if nz == 0 {
            Mapping::new(0).unwrap()
        } else {
            self.hist.mapping
        };
        let bucket_len = self.hist.range_len();
        let offset = self.hist.index_start;

        QuantileIter::new(
            self.hist, mapping, quantiles, bucket_len, offset, total_count, zero_count, min, max,
        )
    }
}
