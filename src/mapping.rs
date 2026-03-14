// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponential histogram scale mapping functions.
//!
//! This module provides the `Mapping` struct which converts f64 values to
//! bucket indices. For scale <= 0, it uses direct exponent mapping. For
//! scale > 0, it uses the compile-time selected lookup table algorithm
//! (NewRelic or Dynatrace) when available and in range, falling back to
//! the built-in logarithm mapper for higher scales or when no lookup
//! tables are compiled.

use crate::float64::{
    MAX_NORMAL_EXPONENT, MIN_NORMAL_EXPONENT, MIN_VALUE,
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
pub enum MappingError {
    /// The bucket index corresponds to a subnormal value.
    Underflow,
    /// The bucket index corresponds to +Inf.
    Overflow,
    /// Invalid scale parameter.
    InvalidScale,
}

impl fmt::Display for MappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Underflow => f.write_str("bucket index corresponds to a subnormal value"),
            Self::Overflow => f.write_str("bucket index corresponds to +Inf"),
            Self::InvalidScale => f.write_str("invalid scale parameter"),
        }
    }
}

impl std::error::Error for MappingError {}

/// Returns the maximum scale supported by the mapping.
///
/// All scales from 1 to MAX_SCALE (20) are always supported: lookup
/// tables cover scales up to the compiled table scale, and the built-in
/// logarithm mapper handles the rest.
#[inline]
pub const fn max_scale() -> i32 {
    MAX_SCALE
}

/// Converts values to bucket indices at a given scale.
#[derive(Debug, Clone, Copy)]
pub struct Mapping {
    scale: i8,
    // Pre-computed inverse factor for boundary computation
    inverse_factor: f64,
}

impl Mapping {
    /// Creates a new mapping for the given scale.
    ///
    /// Returns `MappingError::InvalidScale` if scale is outside [-10, 20].
    pub fn new(scale: i32) -> Result<Self, MappingError> {
        if !(MIN_SCALE..=MAX_SCALE).contains(&scale) {
            return Err(MappingError::InvalidScale);
        }

        let inverse_factor = if scale > 0 {
            // math.Ldexp(math.Ln2, -scale) = Ln2 * 2^(-scale)
            core::f64::consts::LN_2 / (1u64 << scale) as f64
        } else {
            0.0
        };

        Ok(Self {
            scale: scale as i8,
            inverse_factor,
        })
    }

    /// Returns the current scale.
    #[inline]
    pub fn scale(&self) -> i32 {
        self.scale as i32
    }

    /// Maps a positive f64 value to a bucket index.
    #[inline]
    pub fn map_to_index(&self, value: f64) -> i32 {
        if self.scale <= 0 {
            crate::exponent::map_to_index(value, self.scale as i32)
        } else {
            self.map_to_index_positive_scale(value)
        }
    }

    /// Mapping for positive scales - delegates to selected algorithm.
    ///
    /// Uses lookup tables when available and the scale is within table range,
    /// otherwise falls back to the built-in logarithm mapper.
    #[inline]
    fn map_to_index_positive_scale(&self, value: f64) -> i32 {
        let scale = self.scale as i32;

        #[cfg(all(has_lookup_table, feature = "newrelic"))]
        if scale <= crate::lookup::TABLE_SCALE {
            return crate::newrelic::map_to_index(value, scale);
        }

        #[cfg(all(has_lookup_table, feature = "dynatrace", not(feature = "newrelic")))]
        if scale <= crate::lookup::TABLE_SCALE {
            return crate::dynatrace::map_to_index(value, scale);
        }

        crate::logarithm::map_to_index(value, scale)
    }

    /// Returns the lower boundary of a bucket at the given index.
    #[inline]
    pub fn lower_boundary(&self, index: i32) -> Result<f64, MappingError> {
        if self.scale <= 0 {
            crate::exponent::lower_boundary(index, self.scale as i32)
        } else {
            self.lower_boundary_logarithm(index)
        }
    }

    fn lower_boundary_logarithm(&self, index: i32) -> Result<f64, MappingError> {
        let scale = self.scale as i32;
        let max_idx = self.max_normal_lower_boundary_index_log();
        let min_idx = self.min_normal_lower_boundary_index_log();

        if index >= max_idx {
            if index == max_idx {
                // Use alternate equation to avoid overflow
                return Ok(2.0 * ((index - (1 << scale)) as f64 * self.inverse_factor).exp());
            }
            return Err(MappingError::Overflow);
        }

        if index <= min_idx {
            if index == min_idx {
                return Ok(MIN_VALUE);
            } else if index == min_idx - 1 {
                return Ok(((index + (1 << scale)) as f64 * self.inverse_factor).exp() / 2.0);
            }
            return Err(MappingError::Underflow);
        }

        Ok((index as f64 * self.inverse_factor).exp())
    }

    // Helper functions for boundary indices

    #[inline]
    fn min_normal_lower_boundary_index_log(&self) -> i32 {
        MIN_NORMAL_EXPONENT << (self.scale as i32)
    }

    #[inline]
    fn max_normal_lower_boundary_index_log(&self) -> i32 {
        ((MAX_NORMAL_EXPONENT + 1) << (self.scale as i32)) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_mapping() {
        assert!(Mapping::new(0).is_ok());
        assert!(Mapping::new(1).is_ok());
        assert!(Mapping::new(-10).is_ok());
        assert!(Mapping::new(21).is_err());
        assert!(Mapping::new(-11).is_err());
        
        // All scales up to MAX_SCALE are supported
        for scale in MIN_SCALE..=MAX_SCALE {
            assert!(Mapping::new(scale).is_ok(), "scale {} should be supported", scale);
        }
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
