// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Pure logarithm-based mapping for exponential histograms.
//!
//! This algorithm computes bucket indices using `floor(ln(value) * scaleFactor)`.
//! It has floating-point precision errors near bucket boundaries but requires
//! no lookup tables and works for all positive scales.

use crate::float64::{get_normal_base2, get_significand};

/// Maps a positive f64 value to a bucket index using pure logarithm.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0, finite)
/// * `scale` - The histogram scale (must be > 0)
/// * `scale_factor` - Pre-computed `LOG2_E * 2^scale`
///
/// # Returns
/// The bucket index for this value at the given scale.
#[inline]
pub fn map_to_index(value: f64, scale: i32, scale_factor: f64) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(value > 0.0);
    debug_assert!(value.is_finite());

    // Exact power-of-two: significand is 0, index is (exp << scale) - 1
    if get_significand(value) == 0 {
        let exp = get_normal_base2(value);
        return (exp << scale) - 1;
    }

    // General case: use floor(log(value) * scaleFactor)
    (value.ln() * scale_factor).floor() as i32
}

/// Computes the scale factor for a given scale.
///
/// This is `LOG2_E * 2^scale`, used to convert natural log to bucket index.
#[inline]
pub const fn scale_factor(scale: i32) -> f64 {
    // LOG2_E * 2^scale
    core::f64::consts::LOG2_E * (1u64 << scale) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powers_of_two() {
        // Powers of two should map to (exp << scale) - 1
        for scale in 1..=20 {
            let sf = scale_factor(scale);

            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index(value, scale, sf);
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
        let sf = scale_factor(scale);

        // 1.0 is 2^0, should map to -1
        assert_eq!(map_to_index(1.0, scale, sf), -1);

        // 2.0 is 2^1, should map to (1 << 4) - 1 = 15
        assert_eq!(map_to_index(2.0, scale, sf), 15);

        // Values between 1 and 2 should be in buckets 0..15
        let idx = map_to_index(1.5, scale, sf);
        assert!(idx >= 0 && idx < 15, "1.5 should be in [0, 15), got {}", idx);
    }

    #[test]
    fn test_scale_factor() {
        assert_eq!(scale_factor(1), core::f64::consts::LOG2_E * 2.0);
        assert_eq!(scale_factor(4), core::f64::consts::LOG2_E * 16.0);
        assert_eq!(scale_factor(8), core::f64::consts::LOG2_E * 256.0);
    }
}
