// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponential histogram scale mapping functions.
//!
//! This module provides the `Mapping` struct which converts f64 values to
//! bucket indices. For scale <= 0, it uses direct exponent mapping. For
//! scale > 0, it uses the compile-time generated lookup table algorithm.
//! Scales above the compiled table scale are rejected by [`Mapping::new`].

use crate::float64::{
    MIN_NORMAL_EXPONENT, MIN_VALUE,
};
#[cfg(feature = "boundary")]
use crate::float64::MAX_NORMAL_EXPONENT;
use core::fmt;

/// Minimum scale for the exponent mapping.
/// At scale -10, values in (0, 1] map to bucket -1 and values in (1, MAX) map to bucket 0.
pub const MIN_SCALE: i32 = -10;

/// Maximum scale supported is the finest resolution.
/// At scale 20, indices require 31 bits of information.
pub const MAX_SCALE: i32 = 20;

/// Error types for mapping operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingError {
    /// The bucket index corresponds to a subnormal value.
    Underflow,
    /// The bucket index corresponds to +Inf.
    Overflow,
    /// Invalid scale parameter.
    InvalidScale,
    /// Operation requires the `boundary` feature at this scale.
    Unsupported,
}

impl fmt::Display for MappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Underflow => f.write_str("bucket index corresponds to a subnormal value"),
            Self::Overflow => f.write_str("bucket index corresponds to +Inf"),
            Self::InvalidScale => f.write_str("invalid scale parameter"),
            Self::Unsupported => f.write_str("operation requires the `boundary` feature at this scale"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MappingError {}

/// Returns the maximum scale supported by the mapping.
///
/// This equals the compiled lookup table scale (set by the `scale-N`
/// feature). Exponent mapping (scale ≤ 0) is always available.
#[inline]
pub const fn max_scale() -> i32 {
    #[cfg(has_lookup_table)]
    { crate::lookup::TABLE_SCALE }
    #[cfg(not(has_lookup_table))]
    { 0 }
}

/// Converts values to bucket indices at a given scale.
#[derive(Debug, Clone, Copy)]
pub struct Mapping {
    scale: i32,
    /// Pre-computed inverse factor for boundary computation (boundary feature only).
    #[cfg(feature = "boundary")]
    inverse_factor: f64,
}

impl Mapping {
    /// Creates a new mapping for the given scale.
    ///
    /// Returns `MappingError::InvalidScale` if scale is outside
    /// [`MIN_SCALE`]..=[`max_scale()`].
    pub fn new(scale: i32) -> Result<Self, MappingError> {
        if !(MIN_SCALE..=max_scale()).contains(&scale) {
            return Err(MappingError::InvalidScale);
        }

        Ok(Self {
            scale,
            #[cfg(feature = "boundary")]
            inverse_factor: if scale > 0 {
                core::f64::consts::LN_2 / (1u64 << scale) as f64
            } else {
                0.0
            },
        })
    }

    /// Returns the current scale.
    #[inline]
    pub const fn scale(&self) -> i32 {
        self.scale
    }

    /// Maps a positive f64 value to a bucket index.
    ///
    /// Subnormal values (below `0x1p-1022`) are mapped to the same
    /// bucket as `MIN_VALUE` at every scale.
    #[inline]
    pub fn map_to_index(&self, value: f64) -> i32 {
        if self.scale <= 0 {
            crate::exponent::map_to_index(value, self.scale)
        } else if value < MIN_VALUE {
            // All subnormals land in the MIN_VALUE bucket (2^-1022).
            (MIN_NORMAL_EXPONENT << (self.scale)) - 1
        } else {
            self.map_to_index_positive_scale(value)
        }
    }

    /// Mapping for positive scales — delegates to the compiled lookup table.
    #[inline]
    fn map_to_index_positive_scale(&self, value: f64) -> i32 {
        let scale = self.scale;

        #[cfg(has_lookup_table)]
        {
            crate::lookup::map_to_index(value, scale)
        }

        #[cfg(not(has_lookup_table))]
        {
            let _ = (value, scale);
            0
        }
    }

    /// Returns the lower boundary of a bucket at the given index.
    ///
    /// For scale ≤ 0 this is an exact power of two (no libm needed).
    /// For positive scales, this requires the `boundary` feature which
    /// provides the `exp()` function via std or libm.
    #[cfg(feature = "boundary")]
    #[inline]
    pub fn lower_boundary(&self, index: i32) -> Result<f64, MappingError> {
        if self.scale <= 0 {
            crate::exponent::lower_boundary(index, self.scale)
        } else {
            self.lower_boundary_logarithm(index)
        }
    }

    /// Returns the lower boundary of a bucket at the given index.
    ///
    /// Available without the `boundary` feature only for scale ≤ 0
    /// (exact powers of two).
    #[cfg(not(feature = "boundary"))]
    #[inline]
    pub fn lower_boundary(&self, index: i32) -> Result<f64, MappingError> {
        if self.scale <= 0 {
            crate::exponent::lower_boundary(index, self.scale)
        } else {
            Err(MappingError::Unsupported)
        }
    }

    #[cfg(feature = "boundary")]
    fn lower_boundary_logarithm(&self, index: i32) -> Result<f64, MappingError> {
        let scale = self.scale;
        let max_idx = self.max_normal_lower_boundary_index_log();
        let min_idx = self.min_normal_lower_boundary_index_log();

        if index >= max_idx {
            if index == max_idx {
                // Use alternate equation to avoid overflow
                return Ok(2.0 * crate::float64::exp((index - (1 << scale)) as f64 * self.inverse_factor));
            }
            return Err(MappingError::Overflow);
        }

        if index <= min_idx {
            if index == min_idx {
                return Ok(MIN_VALUE);
            } else if index == min_idx - 1 {
                return Ok(crate::float64::exp((index + (1 << scale)) as f64 * self.inverse_factor) / 2.0);
            }
            return Err(MappingError::Underflow);
        }

        Ok(crate::float64::exp(index as f64 * self.inverse_factor))
    }

    // Helper functions for boundary indices

    #[cfg(feature = "boundary")]
    #[inline]
    const fn min_normal_lower_boundary_index_log(&self) -> i32 {
        MIN_NORMAL_EXPONENT << (self.scale)
    }

    #[cfg(feature = "boundary")]
    #[inline]
    const fn max_normal_lower_boundary_index_log(&self) -> i32 {
        ((MAX_NORMAL_EXPONENT + 1) << (self.scale)) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_mapping() {
        assert!(Mapping::new(0).is_ok());
        assert!(Mapping::new(-10).is_ok());
        assert!(Mapping::new(-11).is_err());

        // Positive scales require a lookup table
        if max_scale() > 0 {
            assert!(Mapping::new(1).is_ok());
        } else {
            assert!(Mapping::new(1).is_err());
        }

        // All scales up to max_scale() are supported
        for scale in MIN_SCALE..=max_scale() {
            assert!(Mapping::new(scale).is_ok(), "scale {} should be supported", scale);
        }
        // Scales above max_scale() are rejected
        if max_scale() < MAX_SCALE {
            assert!(Mapping::new(max_scale() + 1).is_err());
        }
        assert!(Mapping::new(MAX_SCALE + 1).is_err());
    }

    #[test]
    fn test_min_scale() {
        let m = Mapping::new(MIN_SCALE).unwrap();
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

        // Boundaries
        assert_eq!(m.lower_boundary(0).unwrap(), 1.0);
        assert!(m.lower_boundary(1).is_err()); // Overflow
    }

    #[test]
    fn test_map_to_index_scale_0() {
        let m = Mapping::new(0).unwrap();
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
        if max_scale() < 1 {
            return; // No lookup table compiled in
        }
        let m = Mapping::new(1).unwrap();

        // At scale 1, each power-of-2 bucket is split in two
        assert_eq!(m.map_to_index(1.0), -1);
        assert_eq!(m.map_to_index(2.0), 1);
        assert_eq!(m.map_to_index(4.0), 3);
    }

    #[test]
    fn test_lower_boundary_scale_0() {
        let m = Mapping::new(0).unwrap();

        // At scale 0, lower_boundary(index) = 2^index
        assert_eq!(m.lower_boundary(0).unwrap(), 1.0); // 2^0 = 1
        assert_eq!(m.lower_boundary(1).unwrap(), 2.0); // 2^1 = 2
        assert_eq!(m.lower_boundary(-1).unwrap(), 0.5); // 2^-1 = 0.5
        assert_eq!(m.lower_boundary(2).unwrap(), 4.0); // 2^2 = 4
    }

    #[test]
    fn test_powers_of_two_all_scales() {
        // Powers of two should map to (exp << scale) - 1 for all supported scales
        for scale in 1..=max_scale() {
            let m = Mapping::new(scale).unwrap();
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
