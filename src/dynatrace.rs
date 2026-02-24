// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Dynatrace lookup table-based mapping for exponential histograms.
//!
//! This algorithm uses N linear buckets (one per log bucket) and two branch
//! corrections instead of NewRelic's 2N linear buckets with one branch correction.
//! Trade-off: ~50% smaller index table at the cost of one extra comparison.
//!
//! A single table is compiled at the highest requested scale (TABLE_SCALE).
//! Lower scales are derived by right-shifting: `map_at_S(v) = map_at_H(v) >> (H - S)`.

use crate::float64::{get_normal_base2, get_significand};
use crate::lookup::{BOUNDARIES, TABLE_SCALE, derive_index_table};

/// Number of log buckets at TABLE_SCALE.
const N: usize = 1 << (TABLE_SCALE as usize);

/// Significand shift for N linear buckets: one less bit of resolution than NR.
const SIGNIFICAND_SHIFT: u32 = 52 - TABLE_SCALE as u32;

/// Returns the linear-to-log index table (N entries), derived from
/// BOUNDARIES on first use.
#[inline]
fn index_table() -> &'static [u16] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| derive_index_table(N, SIGNIFICAND_SHIFT))
}

/// Maps a positive f64 value to a bucket index.
///
/// Uses N linear buckets and two branch corrections.
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);
    debug_assert!(value > 0.0);
    debug_assert!(value.is_finite());

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    let index = index_table();
    let linear_idx = (significand >> SIGNIFICAND_SHIFT) as usize;
    let approx = index[linear_idx] as usize;

    let mut bucket = approx as i32;
    if significand >= BOUNDARIES[approx + 1] { bucket += 1; }
    if significand >= BOUNDARIES[approx + 2] { bucket += 1; }

    let fine_index = (exponent << TABLE_SCALE) + bucket - 1;
    fine_index >> (TABLE_SCALE - scale)
}

/// Returns the native scale (resolution) of the lookup table.
#[inline]
pub const fn table_scale() -> i32 {
    TABLE_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powers_of_two() {
        for scale in 1..=TABLE_SCALE {
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index(value, scale);
                assert_eq!(
                    actual, expected,
                    "power of two mismatch at scale={}, exp={}: got {}, expected {}",
                    scale, exp, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_basic_values() {
        let scale = TABLE_SCALE.min(4);

        assert_eq!(map_to_index(1.0, scale), -1);

        let expected = (1 << scale) - 1;
        assert_eq!(map_to_index(2.0, scale), expected);

        let idx = map_to_index(1.5, scale);
        let max_idx = (1 << scale) - 1;
        assert!(
            idx >= 0 && idx < max_idx,
            "1.5 should be in [0, {}), got {}",
            max_idx,
            idx
        );
    }

    #[test]
    fn test_table_scale() {
        #[cfg(feature = "dynatrace-4")]
        assert!(TABLE_SCALE >= 4);
        #[cfg(feature = "dynatrace-6")]
        assert!(TABLE_SCALE >= 6);
        #[cfg(feature = "dynatrace-8")]
        assert!(TABLE_SCALE >= 8);
        #[cfg(feature = "dynatrace-10")]
        assert!(TABLE_SCALE >= 10);
        #[cfg(feature = "dynatrace-12")]
        assert!(TABLE_SCALE >= 12);
        #[cfg(feature = "dynatrace-14")]
        assert!(TABLE_SCALE >= 14);
    }

    #[test]
    fn test_all_scales_consistent() {
        let test_values: &[f64] = &[1.1, 1.5, 1.9, 2.5, 3.3, 7.7, 0.3, 0.7, 100.0, 1e-10, 1e10];

        for scale in 1..TABLE_SCALE {
            for &v in test_values {
                let direct = map_to_index(v, scale);
                let fine = map_to_index(v, TABLE_SCALE);
                let shifted = fine >> (TABLE_SCALE - scale);

                assert_eq!(
                    direct, shifted,
                    "scale {} mismatch for value {}: direct={}, shifted={}",
                    scale, v, direct, shifted
                );
            }
        }
    }

    #[test]
    fn test_matches_newrelic() {
        #[cfg(any(
            feature = "newrelic-4",
            feature = "newrelic-6",
            feature = "newrelic-8",
            feature = "newrelic-10",
            feature = "newrelic-12",
            feature = "newrelic-14"
        ))]
        {
            let test_values: &[f64] = &[
                1e-300, 1e-100, 1e-10, 0.001, 0.1, 0.5, 1.0, 1.5, 2.0,
                core::f64::consts::PI, 10.0, 100.0, 1e10, 1e100, 1e300,
                1.0000000000001, 1.9999999999999, 0.9999999999999,
            ];

            for scale in 1..=TABLE_SCALE {
                for &v in test_values {
                    let dt_idx = map_to_index(v, scale);
                    let nr_idx = crate::newrelic::map_to_index(v, scale);
                    assert_eq!(
                        dt_idx, nr_idx,
                        "dynatrace vs newrelic mismatch at scale={}, value={}: dt={}, nr={}",
                        scale, v, dt_idx, nr_idx
                    );
                }
            }
        }
    }
}
