// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Lookup table-based index mapping for exponential histograms.
//!
//! This module provides fast integer-only mapping from f64 values to bucket indices
//! using precomputed lookup tables. It is enabled by one of the `lookup-*` features.

// Include the generated lookup tables
include!(concat!(env!("OUT_DIR"), "/lookup_tables.rs"));

use crate::float64::{get_normal_base2, get_significand};

/// Maps a value to a bucket index using the lookup table.
///
/// This function is only valid for scale > 0 and scale <= LOOKUP_SCALE.
/// For scale <= 0, use the exponent-based mapping directly.
///
/// # Arguments
/// * `value` - A positive, finite, non-zero f64 value
/// * `scale` - The histogram scale (must be > 0 and <= LOOKUP_SCALE)
///
/// # Returns
/// The bucket index for the value at the given scale.
#[inline]
pub fn map_to_index_lookup(value: f64, scale: i32) -> i32 {
    debug_assert!(scale > 0 && scale <= LOOKUP_SCALE);

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    // Power of two: significand is 0, index is (exp << scale) - 1
    if significand == 0 {
        return (exponent << scale) - 1;
    }

    // Get subbucket index at full LOOKUP_SCALE resolution
    let subbucket = get_subbucket_index_full(significand);

    // Compute full index at LOOKUP_SCALE, then shift down to requested scale.
    // This ensures the -1 (for upper-inclusive boundaries) is correctly
    // accounted for before the arithmetic right shift.
    let scale_diff = LOOKUP_SCALE - scale;
    let index_at_max = (exponent << LOOKUP_SCALE) + subbucket as i32 - 1;
    index_at_max >> scale_diff
}

/// Get the subbucket index for a mantissa at full LOOKUP_SCALE resolution.
///
/// For a value in [1, 2) with the given significand, returns the
/// log-scale subbucket index in [0, 2^LOOKUP_SCALE).
#[inline]
fn get_subbucket_index_full(significand: u64) -> u32 {
    // Get the linear bucket index from the top bits of significand
    let linear_idx = (significand >> MANTISSA_SHIFT) as usize;

    // Look up the approximate log bucket
    let approx_bucket = LOG_BUCKET_INDEX[linear_idx] as u32;

    // Check if we're past the bucket boundary
    if significand >= LOG_BUCKET_END[approx_bucket as usize] {
        approx_bucket + 1
    } else {
        approx_bucket
    }
}

/// Returns true if the lookup table supports the given scale.
#[inline]
pub const fn supports_scale(scale: i32) -> bool {
    scale > 0 && scale <= LOOKUP_SCALE
}

/// Returns the maximum scale supported by the compiled lookup table.
#[inline]
pub const fn max_lookup_scale() -> i32 {
    LOOKUP_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::Mapping;

    #[test]
    fn test_lookup_vs_logarithm() {
        // Test that lookup produces same results as logarithm for supported scales
        if LOOKUP_SCALE == 0 {
            // No lookup table compiled
            return;
        }

        let test_values = [
            1.0001, 1.1, 1.5, 1.9, 1.9999,
            2.0, 2.5, 3.0, 4.0, 7.5,
            10.0, 100.0, 1000.0, 1e10, 1e100,
            0.5, 0.1, 0.01, 1e-10, 1e-100,
        ];

        for scale in 1..=LOOKUP_SCALE {
            let m = Mapping::new(scale).unwrap();
            for &v in &test_values {
                let expected = m.map_to_index(v);
                let actual = map_to_index_lookup(v, scale);
                assert_eq!(
                    actual, expected,
                    "mismatch at scale={}, value={}: lookup={}, logarithm={}",
                    scale, v, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_powers_of_two() {
        if LOOKUP_SCALE == 0 {
            return;
        }

        // Powers of two should map to (exp << scale) - 1
        for scale in 1..=LOOKUP_SCALE {
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index_lookup(value, scale);
                assert_eq!(
                    actual, expected,
                    "power of two mismatch at scale={}, exp={}: got {}, expected {}",
                    scale, exp, actual, expected
                );
            }
        }
    }
}
