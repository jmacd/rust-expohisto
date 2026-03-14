// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Quantile estimation — CDF walk over bucket view.

use crate::mapping::Mapping;

use super::Histogram;

/// A quantile–value pair estimated from a histogram's bucket distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantileValue {
    /// The requested quantile, in `[0.0, 1.0]`.
    pub quantile: f64,
    /// The estimated value at that quantile.
    pub value: f64,
}

/// Iterator that walks the histogram CDF and yields [`QuantileValue`]s.
///
/// Created by [`Histogram::quantiles`].
///
/// The iterator walks through zero-valued observations first, then through
/// positive buckets in index order, using linear interpolation within the
/// bucket that straddles each quantile threshold.
///
/// By definition, quantile 0.0 yields [`Histogram::min`] and quantile 1.0
/// yields [`Histogram::max`].
#[derive(Debug)]
pub struct QuantileIter<'a, const N: usize> {
    hist: &'a Histogram<N>,
    mapping: Mapping,
    quantiles: &'a [f64],
    qi: usize,

    // Bucket walk state
    bucket_len: u32,
    offset: i32,
    pos: u32,

    // CDF accumulator
    cumulative: u64,
    total_count: u64,
    zero_count: u64,
    zeros_processed: bool,

    min: f64,
    max: f64,
}

impl<const N: usize> Iterator for QuantileIter<'_, N> {
    type Item = QuantileValue;

    fn next(&mut self) -> Option<QuantileValue> {
        let &q = self.quantiles.get(self.qi)?;
        self.qi += 1;

        // Empty histogram — no meaningful estimate.
        if self.total_count == 0 {
            return Some(QuantileValue { quantile: q, value: f64::NAN });
        }

        // Boundary quantiles use exact stats.
        if q <= 0.0 {
            return Some(QuantileValue { quantile: q, value: self.min });
        }
        if q >= 1.0 {
            return Some(QuantileValue { quantile: q, value: self.max });
        }

        let target = q * self.total_count as f64;

        // Account for zero-valued observations (CDF mass at value 0.0).
        if !self.zeros_processed {
            self.cumulative = self.zero_count;
            self.zeros_processed = true;
        }
        if self.cumulative as f64 >= target {
            return Some(QuantileValue { quantile: q, value: 0.0 });
        }

        // Walk positive buckets until cumulative count reaches the target.
        while self.pos < self.bucket_len {
            let index = self.offset + self.pos as i32;
            let count = self.hist.bucket_get(self.hist.slot_for(index));

            if count == 0 {
                self.pos += 1;
                continue;
            }

            let new_cumulative = self.cumulative + count;

            if new_cumulative as f64 >= target {
                let lower = self.mapping.lower_boundary(index).unwrap_or(0.0);
                let upper = self
                    .mapping
                    .lower_boundary(index + 1)
                    .unwrap_or(self.max);
                let fraction = (target - self.cumulative as f64) / count as f64;
                let value = (lower + fraction * (upper - lower)).clamp(self.min, self.max);

                // Don't advance pos/cumulative — next quantile may land
                // in the same bucket.
                return Some(QuantileValue { quantile: q, value });
            }

            self.cumulative = new_cumulative;
            self.pos += 1;
        }

        // All buckets exhausted — return max.
        Some(QuantileValue { quantile: q, value: self.max })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.quantiles.len() - self.qi;
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for QuantileIter<'_, N> {}

impl<const N: usize> Histogram<N> {
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
    pub fn quantiles<'a>(&'a mut self, quantiles: &'a [f64]) -> QuantileIter<'a, N> {
        debug_assert!(
            quantiles.windows(2).all(|w| w[0] <= w[1]),
            "quantiles must be sorted in non-decreasing order"
        );
        debug_assert!(
            quantiles.iter().all(|&q| (0.0..=1.0).contains(&q)),
            "quantiles must be in [0.0, 1.0]"
        );

        // Promote from literal mode if needed.
        self.ensure_promoted();

        let total_count = self.count();
        let nz = self.non_zero_count();
        let zero_count = total_count.saturating_sub(nz);
        let min = self.min();
        let max = self.max();
        let mapping = if nz == 0 {
            Mapping::new(0).unwrap()
        } else {
            self.mapping
        };
        let bucket_len = self.bucket_range_len();
        let offset = self.index_start;

        QuantileIter {
            hist: self,
            mapping,
            quantiles,
            qi: 0,
            bucket_len,
            offset,
            pos: 0,
            cumulative: 0,
            total_count,
            zero_count,
            zeros_processed: false,
            min,
            max,
        }
    }
}
