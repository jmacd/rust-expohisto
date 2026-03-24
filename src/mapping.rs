// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponential histogram scale mapping functions.
//!
//! This module provides the `Scale` struct which converts f64 values to
//! bucket indices. For scale <= 0, it uses direct exponent mapping. For
//! scale > 0, it uses the compile-time generated lookup table algorithm.
//! Scales above the compiled table scale are rejected by [`Scale::new`].

use crate::float64::{
    MIN_NORMAL_EXPONENT, MIN_VALUE, NAN_INF_BIASED, get_biased_exponent, get_significand,
    unbias_exponent,
};
use core::fmt;

/// Minimum scale for the exponent mapping.
/// At scale -10, values in (0, 1] map to bucket -1 and values in (1, MAX) map to bucket 0.
pub const MIN_SCALE: i32 = -10;

/// Maximum scale supported is the finest resolution.
/// At scale 20, indices require 31 bits of information.
pub const MAX_SCALE: i32 = 20;

/// Error types for mapping operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleError {
    /// The bucket index corresponds to a subnormal value.
    Underflow,
    /// The bucket index corresponds to +Inf.
    Overflow,
    /// Invalid scale parameter.
    InvalidScale,
}

impl fmt::Display for ScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Underflow => f.write_str("bucket index corresponds to a subnormal value"),
            Self::Overflow => f.write_str("bucket index corresponds to +Inf"),
            Self::InvalidScale => f.write_str("invalid scale parameter"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ScaleError {}

/// Returns the maximum scale supported by the mapping.
///
/// This equals the compiled lookup table scale (set by the `scale-N`
/// feature). Exponent mapping (scale ≤ 0) is always available.
#[inline]
pub const fn max_scale() -> i32 {
    crate::lookup::TABLE_SCALE
}

/// Converts values to bucket indices at a given scale.
///
/// Scale fits in −10..=20 and is stored as `i8` for compactness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Scale(i8);

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Scale {
    /// Creates a new mapping for the given scale.
    ///
    /// Returns `ScaleError::InvalidScale` if scale is outside
    /// [`MIN_SCALE`]..=[`max_scale()`].
    pub fn new(scale: i32) -> Result<Self, ScaleError> {
        if !(MIN_SCALE..=max_scale()).contains(&scale) {
            return Err(ScaleError::InvalidScale);
        }

        Ok(Self(scale as i8))
    }

    /// Returns the current scale as `i32`.
    #[inline]
    pub const fn scale(&self) -> i32 {
        self.0 as i32
    }

    /// Maps a f64 value to a bucket index. Ignores sign.
    #[inline]
    pub fn map_to_index(&self, value: f64) -> Option<i32> {
        let scale = self.scale();

        // Extract the raw exponent.
        let mut biased_exp = get_biased_exponent(value);
        let mut significand = get_significand(value);

        // Handle the extreme cases.
        match biased_exp {
            0 => {
                if significand == 0 {
                    // Zero case.
                    return None;
                } else {
                    // Round up to MIN_VALUE.
                    biased_exp = 1;
                    significand = 0;
                }
            }
            NAN_INF_BIASED => {
                // Inf and NaN cases.
                return None;
            }
        }

        let base2_exp = unbias_exponent(biased_exp);

        if scale <= 0 {
            Some(crate::exponent::map_to_index(significand, base2_exp, scale))
        } else {
            Some(crate::lookup::map_to_index(value, scale))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_mapping() {
        assert!(Scale::new(0).is_ok());
        assert!(Scale::new(-10).is_ok());
        assert!(Scale::new(-11).is_err());

        // Positive scales always available via lookup table
        assert!(Scale::new(1).is_ok());

        // All scales up to max_scale() are supported
        for scale in MIN_SCALE..=max_scale() {
            assert!(
                Scale::new(scale).is_ok(),
                "scale {} should be supported",
                scale
            );
        }
        // Scales above max_scale() are rejected
        if max_scale() < MAX_SCALE {
            assert!(Scale::new(max_scale() + 1).is_err());
        }
        assert!(Scale::new(MAX_SCALE + 1).is_err());
    }

    #[test]
    fn test_min_scale() {
        let m = Scale::new(MIN_SCALE).unwrap();
        assert_eq!(m.scale(), MIN_SCALE);

        // At scale -10, shift = 10, so indices are exponent >> 10
        // Values in (0, 1] map to bucket -1
        assert_eq!(m.map_to_index(0.001), -1);
        assert_eq!(m.map_to_index(0.5), -1);
        assert_eq!(m.map_to_index(1.0), -1);

        // Values in (1, MAX) map to bucket 0
        assert_eq!(m.map_to_index(1.0001), 0);
        assert_eq!(m.map_to_index(2.0), 0);
        assert_eq!(m.map_to_index(1e100), 0);
        assert_eq!(m.map_to_index(1e308), 0);
    }

    #[test]
    fn test_map_to_index_scale_0() {
        let m = Scale::new(0).unwrap();
        assert_eq!(m.scale(), 0, "scale should be 0");

        let idx_1 = m.map_to_index(1.0);

        // Powers of 2 map to exponent - 1
        assert_eq!(idx_1, -1, "1.0 should map to -1"); // 2^0 -> -1
        assert_eq!(m.map_to_index(2.0), 0, "2.0 should map to 0"); // 2^1 -> 0
        assert_eq!(m.map_to_index(4.0), 1, "4.0 should map to 1"); // 2^2 -> 1
        assert_eq!(m.map_to_index(0.5), -2, "0.5 should map to -2"); // 2^-1 -> -2

        // Non-powers of 2 map to floor(log2(value))
        assert_eq!(m.map_to_index(1.5), 0, "1.5 should map to 0"); // 1.5 in (1, 2] -> 0
        assert_eq!(m.map_to_index(3.0), 1, "3.0 should map to 1"); // 3.0 in (2, 4] -> 1
        assert_eq!(m.map_to_index(0.75), -1, "0.75 should map to -1"); // 0.75 in (0.5, 1] -> -1
    }

    #[test]
    fn test_map_to_index_positive_scale() {
        let m = Scale::new(1).unwrap();

        // At scale 1, each power-of-2 bucket is split in two
        assert_eq!(m.map_to_index(1.0), -1);
        assert_eq!(m.map_to_index(2.0), 1);
        assert_eq!(m.map_to_index(4.0), 3);
    }

    #[test]
    fn test_powers_of_two_all_scales() {
        // Powers of two should map to (exp << scale) - 1 for all supported scales
        for scale in 1..=max_scale() {
            let m = Scale::new(scale).unwrap();
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = m.map_to_index(value);
                assert_eq!(
                    actual, expected,
                    "power of two mismatch at scale={}, exp={}: got {}, expected {}",
                    scale, exp, actual, expected
                );
            }
        }
    }
}
