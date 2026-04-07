// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Quantile estimation — CDF walk over bucket view.
//!
//! This module is gated behind `#[cfg(feature = "quantile")]`.

use crate::mapping::Scale;

use super::view::HistogramView;
use super::HistogramNN;

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
/// Created by [`HistogramView::quantiles`].
///
/// The iterator walks through zero-valued observations first, then through
/// positive buckets in index order, using linear interpolation within the
/// bucket that straddles each quantile threshold.
///
/// By definition, quantile 0.0 yields the minimum and quantile 1.0 yields
/// the maximum.
#[derive(Debug)]
pub struct QuantileIter<'a, const N: usize> {
    hist: &'a HistogramNN<N>,
    scale: Scale,
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

impl<const N: usize> HistogramView<'_, N> {
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

        let bucket_len = self.hist.trimmed_slot_count();
        let offset = if self.hist.buckets_empty() {
            0
        } else {
            self.hist.first_slot()
        };

        // Derive zero_count by subtracting positive bucket sum from total.
        let positive_count: u64 = self.positive().iter().sum();
        let zero_count = total_count.saturating_sub(positive_count);

        let scale = if positive_count == 0 {
            // Cannot fail: scale 0 is always valid.
            Scale::new(0).unwrap()
        } else {
            self.hist.current.scale
        };

        QuantileIter {
            hist: self.hist,
            scale,
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

impl<const N: usize> Iterator for QuantileIter<'_, N> {
    type Item = QuantileValue;

    fn next(&mut self) -> Option<QuantileValue> {
        let &q = self.quantiles.get(self.qi)?;
        self.qi += 1;

        // Empty histogram — no meaningful estimate.
        if self.total_count == 0 {
            return Some(QuantileValue {
                quantile: q,
                value: f64::NAN,
            });
        }

        // Boundary quantiles use exact stats.
        if q <= 0.0 {
            let value = if self.zero_count > 0 { 0.0 } else { self.min };
            return Some(QuantileValue { quantile: q, value });
        }
        if q >= 1.0 {
            return Some(QuantileValue {
                quantile: q,
                value: self.max,
            });
        }

        let target = q * self.total_count as f64;

        // Account for zero-valued observations (CDF mass at value 0.0).
        if !self.zeros_processed {
            self.cumulative = self.zero_count;
            self.zeros_processed = true;
        }
        if self.cumulative as f64 >= target {
            return Some(QuantileValue {
                quantile: q,
                value: 0.0,
            });
        }

        // Walk positive buckets until cumulative count reaches the target.
        while self.pos < self.bucket_len {
            let index = self.offset + self.pos as i32;
            let addr = self.hist.slot_addr(index);
            let count = self.hist.bucket_get(&addr);

            if count == 0 {
                self.pos += 1;
                continue;
            }

            let new_cumulative = self.cumulative + count;

            if new_cumulative as f64 >= target {
                // lower_boundary(index) cannot fail: bucket indices in
                // the histogram always correspond to normal f64 values
                // (subnormals are clamped to MIN_VALUE before mapping).
                //
                // lower_boundary(index + 1) can return Overflow when
                // the uppermost bucket spans the boundary of
                // representable f64.  In that case self.max is the
                // correct upper bound for interpolation.
                let lower = self.scale.lower_boundary(index).unwrap_or(0.0);
                let upper = self.scale.lower_boundary(index + 1).unwrap_or(self.max);
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
        Some(QuantileValue {
            quantile: q,
            value: self.max,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.quantiles.len() - self.qi;
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for QuantileIter<'_, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_empty_histogram() {
        let h: HistogramNN<8> = HistogramNN::new();
        let qs = [0.0, 0.5, 1.0];
        let v = h.view();
        let vals: Vec<_> = v.quantiles(&qs).collect();
        assert_eq!(vals.len(), 3);
        for v in &vals {
            assert!(v.value.is_nan(), "empty histogram should yield NaN");
        }
    }

    #[test]
    fn test_quantile_single_value() {
        let mut h: HistogramNN<8> = HistogramNN::new();
        h.update(42.0).unwrap();
        let qs = [0.0, 0.5, 1.0];
        let v = h.view();
        let vals: Vec<_> = v.quantiles(&qs).collect();
        assert_eq!(vals[0].value, 42.0); // p0 = min
        assert_eq!(vals[2].value, 42.0); // p100 = max
        assert!(
            (vals[1].value - 42.0).abs() < 1.0,
            "p50 = {} should be near 42.0",
            vals[1].value
        );
    }

    #[test]
    fn test_quantile_with_zeros() {
        let mut h: HistogramNN<8> = HistogramNN::new();
        for _ in 0..90 {
            h.update(0.0).unwrap();
        }
        for _ in 0..10 {
            h.update(100.0).unwrap();
        }

        let qs = [0.0, 0.5, 0.89, 0.95, 1.0];
        let v = h.view();
        let vals: Vec<_> = v.quantiles(&qs).collect();
        assert_eq!(vals[0].value, 0.0, "p0 = min = 0");
        assert_eq!(vals[1].value, 0.0, "p50 should be 0 (90% are zeros)");
        assert_eq!(vals[2].value, 0.0, "p89 should still be 0");
        assert!(vals[3].value > 0.0, "p95 should be > 0");
        assert_eq!(vals[4].value, 100.0, "p100 = max");
    }

    /// Tests monotonicity, p0=min, p100=max, clamping, and ExactSizeIterator.
    #[test]
    fn test_quantile_properties() {
        let mut h: HistogramNN<8> = HistogramNN::new();
        for v in 1..=1000 {
            h.update(v as f64).unwrap();
        }

        let qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
        let view = h.view();
        let stats = view.stats();
        let iter = view.quantiles(&qs);
        assert_eq!(iter.len(), qs.len(), "ExactSizeIterator");
        let vals: Vec<_> = iter.collect();

        // p0 = min, p100 = max.
        assert_eq!(vals[0].value, stats.min);
        assert_eq!(vals[qs.len() - 1].value, stats.max);

        // All values clamped to [min, max].
        for v in &vals {
            assert!(
                v.value >= stats.min && v.value <= stats.max,
                "q={} value {} outside [{}, {}]",
                v.quantile,
                v.value,
                stats.min,
                stats.max
            );
        }

        // Monotonically non-decreasing.
        for w in vals.windows(2) {
            assert!(
                w[0].value <= w[1].value,
                "not monotonic: q{}={} > q{}={}",
                w[0].quantile,
                w[0].value,
                w[1].quantile,
                w[1].value,
            );
        }

        // Rough sanity: p50 should be near 500.
        assert!(
            (vals[3].value - 500.0).abs() < 100.0,
            "p50 = {} should be near 500",
            vals[3].value
        );
    }

    /// All-same-value histogram: every quantile should return that value.
    #[test]
    fn test_quantile_all_same_value() {
        let mut h: HistogramNN<8> = HistogramNN::new();
        for _ in 0..100 {
            h.update(7.0).unwrap();
        }
        let qs = [0.0, 0.25, 0.5, 0.75, 1.0];
        let view = h.view();
        let vals: Vec<_> = view.quantiles(&qs).collect();
        for v in &vals {
            assert_eq!(
                v.value, 7.0,
                "all-same histogram: q{}={}",
                v.quantile, v.value
            );
        }
    }

    /// Monotonicity with many closely-spaced quantiles.
    #[test]
    fn test_quantile_fine_grained_monotonicity() {
        let mut h: HistogramNN<16> = HistogramNN::new();
        for v in 1..=500 {
            h.update(v as f64).unwrap();
        }
        let qs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let view = h.view();
        let stats = view.stats();
        let vals: Vec<_> = view.quantiles(&qs).collect();

        assert_eq!(vals[0].value, stats.min);
        assert_eq!(vals[100].value, stats.max);

        for w in vals.windows(2) {
            assert!(
                w[0].value <= w[1].value,
                "not monotonic at q={}: {} > {}",
                w[1].quantile,
                w[0].value,
                w[1].value,
            );
        }
    }

    // -- Distribution-based goodness-of-fit test ------------------------------

    /// Error function via Horner form of the Abramowitz & Stegun
    /// approximation (max error ~1.5 × 10⁻⁷).
    fn erf(x: f64) -> f64 {
        let a = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * a);
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let result = 1.0 - poly * (-a * a).exp();
        if x < 0.0 {
            -result
        } else {
            result
        }
    }

    /// Computes reduced χ²/df of histogram bucket counts vs a theoretical
    /// CDF. Bins with expected count < 5 are merged with neighbours.
    fn reduced_chi_squared<const N: usize>(h: &HistogramNN<N>, cdf: fn(f64) -> f64) -> f64 {
        let histogram_view = h.view();
        let scale = histogram_view.scale();
        let mapping = Scale::new(scale).unwrap();
        let total = histogram_view.stats().count as f64;
        let view = histogram_view.positive();

        // Collect (observed, expected) per bucket, merging on the fly.
        let mut merged: Vec<(f64, f64)> = Vec::new();
        let (mut acc_o, mut acc_e) = (0.0, 0.0);
        let offset = view.offset();
        for (pos, count) in view.iter().enumerate() {
            let index = offset + pos as i32;
            let lower = mapping.lower_boundary(index).unwrap_or(0.0);
            let upper = mapping.lower_boundary(index + 1).unwrap_or(f64::INFINITY);
            acc_o += count as f64;
            acc_e += total * (cdf(upper) - cdf(lower));
            if acc_e >= 5.0 {
                merged.push((acc_o, acc_e));
                acc_o = 0.0;
                acc_e = 0.0;
            }
        }
        if acc_e > 0.0 {
            if let Some(last) = merged.last_mut() {
                last.0 += acc_o;
                last.1 += acc_e;
            }
        }

        let df = merged.len().saturating_sub(1).max(1);
        let chi2: f64 = merged.iter().map(|(o, e)| (o - e).powi(2) / e).sum();
        chi2 / df as f64
    }

    /// Chi-squared goodness-of-fit: validates histogram bucket counts
    /// against three theoretical CDFs and spot-checks the p50 quantile.
    #[test]
    fn test_goodness_of_fit() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Exp, LogNormal, Uniform};

        struct Case {
            name: &'static str,
            seed: u64,
            sample: fn(&mut rand::rngs::StdRng) -> f64,
            cdf: fn(f64) -> f64,
            p50_expected: f64,
        }

        let cases = [
            Case {
                name: "Exponential(1.5)",
                seed: 42,
                sample: |rng| Exp::new(1.5).unwrap().sample(rng),
                cdf: |x| 1.0 - (-1.5 * x).exp(),
                p50_expected: core::f64::consts::LN_2 / 1.5,
            },
            Case {
                name: "LogNormal(2, 0.5)",
                seed: 123,
                sample: |rng| LogNormal::new(2.0, 0.5).unwrap().sample(rng),
                cdf: |x| 0.5 * (1.0 + erf((x.ln() - 2.0) / (0.5 * std::f64::consts::SQRT_2))),
                p50_expected: (2.0_f64).exp(),
            },
            Case {
                name: "Uniform(10, 500)",
                seed: 999,
                sample: |rng| Uniform::new(10.0, 500.0).sample(rng),
                cdf: |x| ((x - 10.0) / 490.0).clamp(0.0, 1.0),
                p50_expected: 255.0,
            },
        ];

        for case in &cases {
            let mut rng = rand::rngs::StdRng::seed_from_u64(case.seed);
            let mut h: HistogramNN<160> = HistogramNN::new();
            for _ in 0..1_000_000 {
                h.update((case.sample)(&mut rng)).unwrap();
            }

            let reduced = reduced_chi_squared(&h, case.cdf);
            eprintln!("{}: χ²/df={reduced:.4}", case.name);
            assert!(
                reduced < 2.0,
                "{}: reduced χ²={reduced:.4} exceeds 2.0",
                case.name,
            );

            // Spot-check p0, p50, p100.
            let qs = [0.0, 0.5, 1.0];
            let view = h.view();
            let stats = view.stats();
            let vals: Vec<_> = view.quantiles(&qs).collect();
            assert_eq!(vals[0].value, stats.min, "{}: p0 must equal min", case.name);
            assert_eq!(
                vals[2].value, stats.max,
                "{}: p100 must equal max",
                case.name
            );

            // Tolerance scales with bucket width: at coarse scales (heavy
            // downscaling from counter widening), buckets are wide and
            // linear interpolation has proportionally more error.
            let scale = view.scale();
            let rel_width = if scale >= 0 {
                2.0_f64.powf(1.0 / (1u64 << scale) as f64) - 1.0
            } else {
                2.0_f64.powi(1 << (-scale)) - 1.0
            };
            let tol = (rel_width * 0.5).max(0.05);

            let p50_err = ((vals[1].value - case.p50_expected) / case.p50_expected).abs();
            assert!(
                p50_err < tol,
                "{}: p50={:.4} expected={:.4} err={:.2}% (tol={:.1}%, scale={})",
                case.name,
                vals[1].value,
                case.p50_expected,
                p50_err * 100.0,
                tol * 100.0,
                scale,
            );
        }
    }
}
