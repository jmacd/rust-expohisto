// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Pure logarithm-based mapping for exponential histograms.
//!
//! This algorithm computes bucket indices using `floor(ln(value) * scaleFactor)`.
//! It has floating-point precision errors near bucket boundaries but requires
//! no lookup tables and works for all positive scales.

use crate::float64::{get_significand, get_unbiased_exponent};

/// Pre-computed `LOG2_E * 2^scale` for scales 0..=20.
///
/// Index 0 is unused (exponent mapping handles scale <= 0).
const SCALE_FACTORS: [f64; 21] = {
    let mut arr = [0.0; 21];
    let mut i = 1;
    while i <= 20 {
        arr[i] = core::f64::consts::LOG2_E * (1u64 << i) as f64;
        i += 1;
    }
    arr
};

/// Maps a positive f64 value to a bucket index using pure logarithm.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0)
/// * `scale` - The histogram scale (must be > 0)
///
/// # Returns
/// The bucket index for this value at the given scale.
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    debug_assert!(scale > 0);

    let significand = get_significand(value);
    let exp = get_unbiased_exponent(value);

    // Exact power-of-two: significand is 0, index is (exp << scale) - 1.
    // We use the exponent directly rather than ln() to avoid FP imprecision.
    // See https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261
    if significand == 0 {
        return (exp << scale) - 1;
    }

    // General case: use floor(log(value) * scaleFactor), then clamp
    // to the valid range for this exponent to correct FP imprecision
    // near power-of-two boundaries.
    //
    // For a non-power-of-two with exponent E, log2(value) is in
    // (E, E+1), so the index must be in [E << scale, (E+1) << scale - 1].
    let raw =
        crate::float64::floor(crate::float64::ln(value) * SCALE_FACTORS[scale as usize]) as i32;
    let lo = exp << scale;
    let hi = ((exp + 1) << scale) - 1;
    raw.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powers_of_two() {
        // Powers of two should map to (exp << scale) - 1
        for scale in 1..=20 {
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
        let scale = 4;

        // 1.0 is 2^0, should map to -1
        assert_eq!(map_to_index(1.0, scale), -1);

        // 2.0 is 2^1, should map to (1 << 4) - 1 = 15
        assert_eq!(map_to_index(2.0, scale), 15);

        // Values between 1 and 2 should be in buckets 0..15
        let idx = map_to_index(1.5, scale);
        assert!(
            (0..15).contains(&idx),
            "1.5 should be in [0, 15), got {}",
            idx
        );
    }

    #[test]
    fn test_scale_factors() {
        assert_eq!(SCALE_FACTORS[1], core::f64::consts::LOG2_E * 2.0);
        assert_eq!(SCALE_FACTORS[4], core::f64::consts::LOG2_E * 16.0);
        assert_eq!(SCALE_FACTORS[8], core::f64::consts::LOG2_E * 256.0);
    }
}
