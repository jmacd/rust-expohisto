// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Promoted read-only view of a histogram.

#[cfg(feature = "quantile")]
use crate::mapping::Scale;

#[cfg(feature = "quantile")]
use super::quantile::QuantileIter;

use super::width::{SlotAddr, Width};
use super::{Histogram, Stats};

/// Read-only view of a histogram's data.
///
/// Created by [`Histogram::view`], which may promote from literal mode
/// to bucket mode internally. All accessors take `&self`, so a
/// `HistogramView` can be shared freely once obtained.
///
/// ```
/// use otel_expohisto::Histogram;
///
/// let mut h: Histogram<16> = Histogram::new();
/// h.update(1.5).unwrap();
/// h.update(2.7).unwrap();
///
/// let v = h.view();
/// assert_eq!(v.stats().count, 2);
/// assert!(v.stats().sum > 4.0);
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
        if self.hist.buckets_empty() {
            0
        } else {
            self.hist.current.scale.scale()
        }
    }

    /// Returns the aggregate statistics (count, sum, min, max).
    ///
    /// When the histogram is empty (count is 0), min and max are
    /// reported as 0.0.
    #[inline]
    pub const fn stats(&self) -> Stats {
        if self.hist.stats.count == 0 {
            Stats {
                count: 0,
                sum: 0.0,
                min: 0.0,
                max: 0.0,
            }
        } else {
            self.hist.stats
        }
    }

    /// Returns a read-only view of the positive buckets.
    #[inline]
    pub fn positive(&self) -> BucketView<'_, N> {
        BucketView { hist: self.hist }
    }

    /// Returns an iterator that estimates values at the requested quantiles.
    ///
    /// Each quantile must be in `[0.0, 1.0]` and the slice must be sorted
    /// in non-decreasing order. By definition, quantile 0.0 yields the
    /// minimum and quantile 1.0 yields the maximum.
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
    #[cfg(feature = "quantile")]
    pub fn quantiles<'a>(&'a self, quantiles: &'a [f64]) -> QuantileIter<'a, N> {
        debug_assert!(
            quantiles.windows(2).all(|w| w[0] <= w[1]),
            "quantiles must be sorted in non-decreasing order"
        );
        debug_assert!(
            quantiles.iter().all(|&q| (0.0..=1.0).contains(&q)),
            "quantiles must be in [0.0, 1.0]"
        );

        let stats = self.stats();
        let total_count = stats.count;
        let min = stats.min;
        let max = stats.max;

        let bucket_len = self.hist.range_len();
        let offset = self.hist.index_start;

        // Derive zero_count by summing positive buckets and subtracting
        // from total. Any consumer that walks buckets learns this naturally.
        let positive_count: u64 = (0..bucket_len)
            .map(|pos| {
                let index = offset + pos as i32;
                self.hist.bucket_get(self.hist.slot_for(index))
            })
            .sum();
        let zero_count = total_count.saturating_sub(positive_count);

        let scale = if positive_count == 0 {
            // Cannot fail: scale 0 is always valid.
            Scale::new(0).unwrap()
        } else {
            self.hist.current.scale
        };

        QuantileIter::new(
            self.hist,
            scale,
            quantiles,
            bucket_len,
            offset,
            total_count,
            zero_count,
            min,
            max,
        )
    }
}

/// Read-only view of bucket data in a histogram.
#[derive(Debug)]
pub struct BucketView<'a, const N: usize> {
    pub(super) hist: &'a Histogram<N>,
}

impl<const N: usize> BucketView<'_, N> {
    /// Returns the first slot index (bucket offset).
    #[inline]
    pub fn offset(&self) -> i32 {
        self.hist
            .current
            .width
            .word_to_slot_index(self.hist.word_start)
    }

    /// Number of logical buckets in use.
    #[inline]
    pub fn len(&self) -> u32 {
        self.hist.current_slot_count() as u32
    }

    /// Number of logical buckets in use (alias).
    #[inline]
    pub fn bucket_count(&self) -> u32 {
        self.len()
    }

    /// Returns the current counter width.
    #[inline]
    pub fn width(&self) -> Width {
        self.hist.current.width
    }

    /// Returns true if no buckets are in use.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hist.buckets_empty()
    }

    /// Returns an iterator over bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, N> {
        BucketsIter {
            hist: self.hist,
            addr: (!self.is_empty()).then(|| self.hist.start_addr()),
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
    addr: Option<SlotAddr<'a>>,
}

impl<const N: usize> Iterator for BucketsIter<'_, N> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let count = self.addr.as_ref().map(|addr| self.hist.bucket_get(&addr));
        if let Some(addr) = self.addr.take() {
            self.addr = addr.next_addr(self.hist.word_end);
        }
        count
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.addr {
            None => (0, None),
            Some(addr) => {
                let remaining = self.hist.size_hint(addr);
                (remaining, Some(remaining))
            }
        }
    }
}

impl<const N: usize> ExactSizeIterator for BucketsIter<'_, N> {}
