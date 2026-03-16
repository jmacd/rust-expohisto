// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponent-based mapping for exponential histograms (scale <= 0).
//!
//! At these scales, bucket boundaries are exact powers of two and the
//! mapping reduces to extracting the IEEE 754 exponent with a right-shift.
//! This is the simplest and fastest algorithm, always used for non-positive scales.

use crate::float64::{
    MAX_NORMAL_EXPONENT, MIN_NORMAL_EXPONENT, MIN_VALUE, get_normal_base2, get_significand,
};
use crate::mapping::MappingError;

/// Maps a positive f64 value to a bucket index at a non-positive scale.
///
/// # Arguments
/// * `value` - A positive f64 value (must be > 0)
/// * `scale` - The histogram scale (must be <= 0)
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    debug_assert!(scale <= 0);

    let shift = (-scale) as u32;

    if value < MIN_VALUE {
        return min_normal_lower_boundary_index(scale);
    }

    // Extract the raw exponent
    let raw_exp = get_normal_base2(value);

    // Upper-inclusive correction: exact powers of two (significand == 0)
    // must map one bucket lower.
    // See https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261
    let correction = if get_significand(value) == 0 { -1 } else { 0 };

    // Arithmetic right shift handles negative exponents correctly
    (raw_exp + correction) >> shift
}

/// Returns the lower boundary of a bucket at the given index.
#[inline]
pub fn lower_boundary(index: i32, scale: i32) -> Result<f64, MappingError> {
    debug_assert!(scale <= 0);

    let shift = (-scale) as u32;

    if index < min_normal_lower_boundary_index(scale) {
        return Err(MappingError::Underflow);
    }

    if index > max_normal_lower_boundary_index(scale) {
        return Err(MappingError::Overflow);
    }

    // 2^(index << shift)
    let exp = index << shift;
    Ok(crate::float64::pow2(exp))
}

#[inline]
pub const fn min_normal_lower_boundary_index(scale: i32) -> i32 {
    let shift = (-scale) as u32;
    let mut idx = MIN_NORMAL_EXPONENT >> shift;
    if shift < 2 {
        // For scales -1 and 0, 2^-1022 is a power-of-two multiple
        idx -= 1;
    }
    idx
}

#[inline]
pub const fn max_normal_lower_boundary_index(scale: i32) -> i32 {
    let shift = (-scale) as u32;
    MAX_NORMAL_EXPONENT >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_0() {
        // Powers of 2 map to exponent - 1
        assert_eq!(map_to_index(1.0, 0), -1);  // 2^0 -> -1
        assert_eq!(map_to_index(2.0, 0), 0);   // 2^1 -> 0
        assert_eq!(map_to_index(4.0, 0), 1);   // 2^2 -> 1
        assert_eq!(map_to_index(0.5, 0), -2);  // 2^-1 -> -2

        // Non-powers of 2
        assert_eq!(map_to_index(1.5, 0), 0);   // 1.5 in (1, 2] -> 0
        assert_eq!(map_to_index(3.0, 0), 1);   // 3.0 in (2, 4] -> 1
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
    fn test_lower_boundary_scale_0() {
        assert_eq!(lower_boundary(0, 0).unwrap(), 1.0);
        assert_eq!(lower_boundary(1, 0).unwrap(), 2.0);
        assert_eq!(lower_boundary(-1, 0).unwrap(), 0.5);
        assert_eq!(lower_boundary(2, 0).unwrap(), 4.0);
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
