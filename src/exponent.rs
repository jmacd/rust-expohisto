// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponent-based mapping for exponential histograms (scale <= 0).
//!
//! At these scales, bucket boundaries are exact powers of two and the
//! mapping reduces to extracting the IEEE 754 exponent with a right-shift.
//! This is the simplest and fastest algorithm, always used for non-positive scales.

/// Maps a positive f64 value to a bucket index at a non-positive scale.
#[inline]
pub(crate) fn map_decomposed(significand: u64, base2_exp: i32, scale: i32) -> i32 {
    debug_assert!(scale <= 0);

    // Upper-inclusive correction: exact powers of two (significand == 0)
    // must map one bucket lower.
    let correction = if significand == 0 { -1 } else { 0 };

    // Arithmetic right shift handles negative exponents correctly
    (base2_exp + correction) >> -scale
}

#[cfg(test)]
mod tests {
    use super::super::float64::{get_significand, get_unbiased_exponent};

    fn map_to_index(value: f64, scale: i32) -> i32 {
        super::map_decomposed(get_significand(value), get_unbiased_exponent(value), scale)
    }

    #[test]
    fn test_scale_0() {
        // Powers of 2 map to exponent - 1
        assert_eq!(map_to_index(1.0, 0), -1); // 2^0 -> -1
        assert_eq!(map_to_index(2.0, 0), 0); // 2^1 -> 0
        assert_eq!(map_to_index(4.0, 0), 1); // 2^2 -> 1
        assert_eq!(map_to_index(0.5, 0), -2); // 2^-1 -> -2

        // Non-powers of 2
        assert_eq!(map_to_index(1.5, 0), 0); // 1.5 in (1, 2] -> 0
        assert_eq!(map_to_index(3.0, 0), 1); // 3.0 in (2, 4] -> 1
        assert_eq!(map_to_index(0.75, 0), -1); // 0.75 in (0.5, 1] -> -1
    }

    #[test]
    fn test_negative_scales() {
        for scale in -10..=0 {
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let shift = (-scale) as u32;
                let expected = (exp - 1) >> shift;
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
    fn test_min_scale() {
        let scale = -10;
        // At scale -10, shift = 10, so indices are exponent >> 10
        assert_eq!(map_to_index(0.001, scale), -1);
        assert_eq!(map_to_index(0.5, scale), -1);
        assert_eq!(map_to_index(1.0, scale), -1);
        assert_eq!(map_to_index(1.0001, scale), 0);
        assert_eq!(map_to_index(2.0, scale), 0);
        assert_eq!(map_to_index(1e100, scale), 0);
        assert_eq!(map_to_index(1e308, scale), 0);
    }
}
