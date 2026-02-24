// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! NewRelic lookup table-based mapping for exponential histograms.
//!
//! This algorithm uses pre-computed lookup tables for exact bucket mapping
//! without floating-point precision errors. A single table is compiled at
//! the highest requested scale (TABLE_SCALE). Lower scales are derived by
//! right-shifting the result: `map_at_S(v) = map_at_H(v) >> (H - S)`.

use crate::float64::{get_normal_base2, get_significand};
use crate::lookup::{BOUNDARIES, NR_INDEX, NR_SIGNIFICAND_SHIFT, TABLE_SCALE};

/// Maps a positive f64 value to a bucket index using pre-computed
/// lookup tables at TABLE_SCALE, then right-shifts to the requested scale.
///
/// Upper-inclusive semantics are baked into the boundary table:
/// `BOUNDARIES[1] = 1` ensures that `significand == 0` (exact powers of two)
/// naturally fails the `>=` check, placing them in the bucket below.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0, finite)
/// * `scale` - The histogram scale (must be in 1..=TABLE_SCALE)
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);
    debug_assert!(value > 0.0);
    debug_assert!(value.is_finite());

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    let linear_idx = (significand >> NR_SIGNIFICAND_SHIFT) as usize;
    let approx = NR_INDEX[linear_idx] as usize;
    let bucket = if significand >= BOUNDARIES[approx + 1] {
        approx + 1
    } else {
        approx
    } as i32;

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
        // TABLE_SCALE should be >= the feature selected (may be higher
        // when multiple table features are enabled)
        #[cfg(feature = "newrelic-4")]
        assert!(TABLE_SCALE >= 4);
        #[cfg(feature = "newrelic-6")]
        assert!(TABLE_SCALE >= 6);
        #[cfg(feature = "newrelic-8")]
        assert!(TABLE_SCALE >= 8);
        #[cfg(feature = "newrelic-10")]
        assert!(TABLE_SCALE >= 10);
        #[cfg(feature = "newrelic-12")]
        assert!(TABLE_SCALE >= 12);
        #[cfg(feature = "newrelic-14")]
        assert!(TABLE_SCALE >= 14);
    }

    #[test]
    fn test_all_scales_consistent() {
        // Verify that lower scales give the same result as
        // computing at TABLE_SCALE and right-shifting.
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
}
