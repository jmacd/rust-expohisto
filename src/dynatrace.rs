// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Dynatrace lookup table-based mapping for exponential histograms.
//!
//! This algorithm uses pre-computed lookup tables with N linear buckets
//! (one per log bucket) and two branch corrections instead of NewRelic's
//! 2N linear buckets with one branch correction.
//!
//! Trade-off: ~50% smaller index table at the cost of one extra comparison.
//! On modern CPUs where cache pressure matters, this can be a net win.
//!
//! Tables are generated at compile time for every scale from 1 to TABLE_SCALE,
//! so each scale has its own compact arrays.

use crate::float64::{get_normal_base2, get_significand};

/// Per-scale lookup tables for the Dynatrace algorithm.
#[derive(Debug)]
pub struct DynatraceScaleMapping {
    /// Shift to convert 52-bit significand to N linear buckets: 52 - scale.
    pub significand_shift: u32,
    /// Maps each of N linear buckets to an approximate log bucket index.
    /// Length = 1 << scale.
    pub indices: &'static [i16],
    /// Exact boundary significands, with two sentinels at the end.
    /// Length = (1 << scale) + 2.
    /// boundaries[k] = significand of 2^(k/N).
    pub boundaries: &'static [u64],
}

// Include the generated lookup tables (provides TABLE_SCALE and SCALE_MAPPINGS)
include!(concat!(env!("OUT_DIR"), "/dynatrace_tables.rs"));

/// Returns the `DynatraceScaleMapping` for the given positive scale.
#[inline]
pub fn get_scale_mapping(scale: i32) -> &'static DynatraceScaleMapping {
    debug_assert!(scale >= 1 && scale <= TABLE_SCALE);
    &SCALE_MAPPINGS[(scale - 1) as usize]
}

/// Maps a positive f64 value to a bucket index using the Dynatrace
/// two-branch correction algorithm.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0, finite)
/// * `scale` - The histogram scale (must be in 1..=TABLE_SCALE)
/// * `sm` - The per-scale mapping tables (from `get_scale_mapping`)
#[inline]
pub fn map_to_index(value: f64, scale: i32, sm: &DynatraceScaleMapping) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);
    debug_assert!(value > 0.0);
    debug_assert!(value.is_finite());

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    // Exact power-of-two: significand is 0, index is (exp << scale) - 1
    if significand == 0 {
        return (exponent << scale) - 1;
    }

    // Look up the rough bucket from N equidistant linear buckets
    let linear_idx = (significand >> sm.significand_shift) as usize;
    let rough = sm.indices[linear_idx] as usize;

    // Two-branch correction: the rough index may be off by up to 2
    let mut offset = rough;
    if significand >= sm.boundaries[rough + 1] {
        offset += 1;
    }
    if significand >= sm.boundaries[rough + 2] {
        offset += 1;
    }

    // Dynatrace offset is 0-based (0 = first sub-bucket), so no -1 needed.
    // The power-of-two special case above handles upper-inclusive boundaries.
    (exponent << scale) + offset as i32
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
        #[cfg(feature = "dynatrace-4")]
        assert_eq!(TABLE_SCALE, 4);
        #[cfg(feature = "dynatrace-6")]
        assert_eq!(TABLE_SCALE, 6);
        #[cfg(feature = "dynatrace-8")]
        assert_eq!(TABLE_SCALE, 8);
        #[cfg(feature = "dynatrace-10")]
        assert_eq!(TABLE_SCALE, 10);
        #[cfg(feature = "dynatrace-12")]
        assert_eq!(TABLE_SCALE, 12);
        #[cfg(feature = "dynatrace-14")]
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

    #[test]
    fn test_matches_newrelic() {
        // When both newrelic and dynatrace are compiled (bench-all),
        // verify they produce identical results for all values.
        #[cfg(any(
            feature = "newrelic-4",
            feature = "newrelic-6",
            feature = "newrelic-8",
            feature = "newrelic-10",
            feature = "newrelic-12",
            feature = "newrelic-14"
        ))]
        {
            let max_common = TABLE_SCALE.min(crate::newrelic::table_scale());
            let test_values: &[f64] = &[
                1e-300, 1e-100, 1e-10, 0.001, 0.1, 0.5, 1.0, 1.5, 2.0,
                core::f64::consts::PI, 10.0, 100.0, 1e10, 1e100, 1e300,
                1.0000000000001, 1.9999999999999, 0.9999999999999,
            ];

            for scale in 1..=max_common {
                let dt_sm = get_scale_mapping(scale);
                let nr_sm = crate::newrelic::get_scale_mapping(scale);
                for &v in test_values {
                    let dt_idx = map_to_index(v, scale, dt_sm);
                    let nr_idx = crate::newrelic::map_to_index(v, scale, nr_sm);
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
