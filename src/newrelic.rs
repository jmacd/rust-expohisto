// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! NewRelic lookup table-based mapping for exponential histograms.
//!
//! This algorithm uses pre-computed lookup tables for exact bucket mapping
//! without floating-point precision errors. Tables are generated at
//! compile time for every scale from 1 to TABLE_SCALE, so each scale
//! has its own compact arrays — no runtime shifting is needed.
//!
//! Only scales <= TABLE_SCALE are supported. Higher scales require a larger
//! table or a different algorithm (e.g., logarithm).

use crate::float64::{get_normal_base2, get_significand};

/// Per-scale lookup tables for the newrelic algorithm.
#[derive(Debug)]
pub struct NewrelicScaleMapping {
    pub significand_shift: u32,
    pub log_bucket_index: &'static [u16],
    pub log_bucket_end: &'static [u64],
}

// Include the generated lookup tables (provides TABLE_SCALE and SCALE_MAPPINGS)
include!(concat!(env!("OUT_DIR"), "/newrelic_tables.rs"));

/// Returns the `NewrelicScaleMapping` for the given positive scale.
#[inline]
pub fn get_scale_mapping(scale: i32) -> &'static NewrelicScaleMapping {
    debug_assert!(scale >= 1 && scale <= TABLE_SCALE);
    &SCALE_MAPPINGS[(scale - 1) as usize]
}

/// Maps a positive f64 value to a bucket index using the pre-computed
/// per-scale lookup tables.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0, finite)
/// * `scale` - The histogram scale (must be in 1..=TABLE_SCALE)
/// * `sm` - The per-scale mapping tables (from `get_scale_mapping`)
#[inline]
pub fn map_to_index(value: f64, scale: i32, sm: &NewrelicScaleMapping) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);
    debug_assert!(value > 0.0);
    debug_assert!(value.is_finite());

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    // Direct lookup at the target scale — no shifting needed
    let linear_idx = (significand >> sm.significand_shift) as usize;
    let approx_bucket = sm.log_bucket_index[linear_idx] as usize;
    let bucket = if significand >= sm.log_bucket_end[approx_bucket] {
        approx_bucket + 1
    } else {
        approx_bucket
    } as i32;

    // Upper-inclusive correction: exact powers of two (significand == 0)
    // must map one bucket lower.
    // See https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261
    (exponent << scale) + bucket - 1 - (significand == 0) as i32
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
        // Powers of two should map to (exp << scale) - 1
        for scale in 1..=TABLE_SCALE {
            let sm = get_scale_mapping(scale);
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index(value, scale, sm);
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
        let sm = get_scale_mapping(scale);

        // 1.0 is 2^0, should map to -1
        assert_eq!(map_to_index(1.0, scale, sm), -1);

        // 2.0 is 2^1, should map to (1 << scale) - 1
        let expected = (1 << scale) - 1;
        assert_eq!(map_to_index(2.0, scale, sm), expected);

        // Values between 1 and 2 should be in buckets 0..(1 << scale) - 1
        let idx = map_to_index(1.5, scale, sm);
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
        // TABLE_SCALE should match the feature selected
        #[cfg(feature = "newrelic-4")]
        assert_eq!(TABLE_SCALE, 4);
        #[cfg(feature = "newrelic-6")]
        assert_eq!(TABLE_SCALE, 6);
        #[cfg(feature = "newrelic-8")]
        assert_eq!(TABLE_SCALE, 8);
        #[cfg(feature = "newrelic-10")]
        assert_eq!(TABLE_SCALE, 10);
        #[cfg(feature = "newrelic-12")]
        assert_eq!(TABLE_SCALE, 12);
        #[cfg(feature = "newrelic-14")]
        assert_eq!(TABLE_SCALE, 14);
    }

    #[test]
    fn test_all_scales_consistent() {
        // Verify that lower-scale tables give the same result as
        // computing at TABLE_SCALE and right-shifting.
        let sm_fine = get_scale_mapping(TABLE_SCALE);
        let test_values: &[f64] = &[1.1, 1.5, 1.9, 2.5, 3.3, 7.7, 0.3, 0.7, 100.0, 1e-10, 1e10];

        for scale in 1..TABLE_SCALE {
            let sm = get_scale_mapping(scale);
            for &v in test_values {
                let direct = map_to_index(v, scale, sm);

                // Reference: compute at TABLE_SCALE and shift
                let fine = map_to_index(v, TABLE_SCALE, sm_fine);
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
