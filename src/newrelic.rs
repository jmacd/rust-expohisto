// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! NewRelic lookup table-based mapping for exponential histograms.
//!
//! This algorithm uses pre-computed lookup tables for exact bucket mapping
//! without floating-point precision errors. The tables are generated at
//! compile time for the selected scale.
//!
//! Only scales <= TABLE_SCALE are supported. Higher scales require a larger
//! table or a different algorithm (e.g., logarithm).

// Compile-time check: only one newrelic feature allowed
#[cfg(any(
    all(feature = "newrelic-4", feature = "newrelic-6"),
    all(feature = "newrelic-4", feature = "newrelic-8"),
    all(feature = "newrelic-4", feature = "newrelic-10"),
    all(feature = "newrelic-4", feature = "newrelic-12"),
    all(feature = "newrelic-4", feature = "newrelic-14"),
    all(feature = "newrelic-6", feature = "newrelic-8"),
    all(feature = "newrelic-6", feature = "newrelic-10"),
    all(feature = "newrelic-6", feature = "newrelic-12"),
    all(feature = "newrelic-6", feature = "newrelic-14"),
    all(feature = "newrelic-8", feature = "newrelic-10"),
    all(feature = "newrelic-8", feature = "newrelic-12"),
    all(feature = "newrelic-8", feature = "newrelic-14"),
    all(feature = "newrelic-10", feature = "newrelic-12"),
    all(feature = "newrelic-10", feature = "newrelic-14"),
    all(feature = "newrelic-12", feature = "newrelic-14"),
))]
compile_error!("Only one newrelic-* feature may be enabled. Choose one of: newrelic-4, newrelic-6, newrelic-8, newrelic-10, newrelic-12, newrelic-14");

// Include the generated lookup tables
include!(concat!(env!("OUT_DIR"), "/newrelic_tables.rs"));

use crate::float64::{get_normal_base2, get_significand};

/// Maps a positive f64 value to a bucket index at the requested scale.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0, finite)
/// * `scale` - The histogram scale (must be in 1..=TABLE_SCALE)
///
/// # Returns
/// The bucket index for this value at the given scale.
///
/// # Panics
/// Debug assertion fails if scale > TABLE_SCALE. Use `Mapping::new()` to
/// validate scale before calling this function.
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE, "scale {} exceeds TABLE_SCALE {}", scale, TABLE_SCALE);
    debug_assert!(value > 0.0);
    debug_assert!(value.is_finite());

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    // Exact power-of-two: significand is 0, index is (exp << scale) - 1
    if significand == 0 {
        return (exponent << scale) - 1;
    }

    // Non-power-of-two: use lookup table at native scale, then right-shift
    let subbucket = get_subbucket_index(significand) as i32;
    let native_idx = (exponent << TABLE_SCALE) + subbucket - 1;

    // Right-shift to requested scale (scale <= TABLE_SCALE)
    native_idx >> (TABLE_SCALE - scale)
}

/// Returns the subbucket index for a significand at native TABLE_SCALE resolution.
#[inline]
fn get_subbucket_index(significand: u64) -> u32 {
    // Get the linear bucket index from the top bits of significand
    let linear_idx = (significand >> SIGNIFICAND_SHIFT) as usize;

    // Look up the approximate log bucket
    let approx_bucket = LOG_BUCKET_INDEX[linear_idx] as u32;

    // Check if we're past the bucket boundary
    if significand >= LOG_BUCKET_END[approx_bucket as usize] {
        approx_bucket + 1
    } else {
        approx_bucket
    }
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

        // 1.0 is 2^0, should map to -1
        assert_eq!(map_to_index(1.0, scale), -1);

        // 2.0 is 2^1, should map to (1 << scale) - 1
        let expected = (1 << scale) - 1;
        assert_eq!(map_to_index(2.0, scale), expected);

        // Values between 1 and 2 should be in buckets 0..(1 << scale) - 1
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
}
