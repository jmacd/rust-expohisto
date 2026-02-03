// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponential histogram scale mapping functions.
//!
//! This module provides mapping functions that convert floating-point values
//! to bucket indices based on the configured scale.

use crate::float64::{
    get_normal_base2, get_significand, MAX_NORMAL_EXPONENT, MIN_NORMAL_EXPONENT, MIN_VALUE,
};

/// Minimum scale for the exponent mapping (most coarse resolution).
/// At scale -10, bucket indices range from -1 to 1 for normal floats.
pub const MIN_SCALE: i32 = -10;

/// Maximum scale supported (finest resolution).
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

/// A mapping function that converts values to bucket indices.
///
/// This is stored inline (no heap allocation) and provides scale-dependent
/// mapping between f64 values and histogram bucket indices.
#[derive(Debug, Clone, Copy)]
pub struct Mapping {
    // Internal representation optimized for space
    // For scale <= 0: uses exponent mapping
    // For scale > 0: uses logarithm mapping
    scale: i8,
    // Pre-computed scale factor for logarithm mapping (scale > 0)
    scale_factor: f64,
    // Pre-computed inverse factor for boundary computation
    inverse_factor: f64,
}

impl Mapping {
    /// Creates a new mapping for the given scale.
    ///
    /// # Errors
    /// Returns `MappingError::InvalidScale` if scale is outside [MIN_SCALE, MAX_SCALE].
    pub fn new(scale: i32) -> Result<Self, MappingError> {
        if !(MIN_SCALE..=MAX_SCALE).contains(&scale) {
            return Err(MappingError::InvalidScale);
        }

        let scale_factor = if scale > 0 {
            // math.Ldexp(math.Log2E, scale) = Log2E * 2^scale
            core::f64::consts::LOG2_E * (1u64 << scale) as f64
        } else {
            0.0
        };

        let inverse_factor = if scale > 0 {
            // math.Ldexp(math.Ln2, -scale) = Ln2 * 2^(-scale)
            core::f64::consts::LN_2 / (1u64 << scale) as f64
        } else {
            0.0
        };

        Ok(Self {
            scale: scale as i8,
            scale_factor,
            inverse_factor,
        })
    }

    /// Returns the current scale.
    #[inline]
    pub fn scale(&self) -> i32 {
        self.scale as i32
    }

    /// Maps a positive floating-point value to a bucket index.
    ///
    /// # Panics
    /// This function assumes the value is positive, finite, and non-zero.
    #[inline]
    pub fn map_to_index(&self, value: f64) -> i32 {
        if self.scale <= 0 {
            self.map_to_index_exponent(value)
        } else {
            self.map_to_index_logarithm(value)
        }
    }

    /// Exponent-based mapping for scale <= 0.
    #[inline]
    fn map_to_index_exponent(&self, value: f64) -> i32 {
        let shift = (-self.scale) as u32;

        if value < MIN_VALUE {
            return self.min_normal_lower_boundary_index_exp();
        }

        // Extract the raw exponent
        let raw_exp = get_normal_base2(value);

        // Correction for exact powers of two: if significand is 0, subtract 1
        // (significand - 1) >> 52 gives -1 for significand=0, 0 otherwise
        let significand = get_significand(value);
        let correction = if significand == 0 { -1 } else { 0 };

        // Arithmetic right shift handles negative exponents correctly
        (raw_exp + correction) >> shift
    }

    /// Logarithm-based mapping for scale > 0.
    #[inline]
    fn map_to_index_logarithm(&self, value: f64) -> i32 {
        let scale = self.scale as i32;

        if value <= MIN_VALUE {
            return self.min_normal_lower_boundary_index_log() - 1;
        }

        // Exact power-of-two optimization
        if get_significand(value) == 0 {
            let exp = get_normal_base2(value);
            return (exp << scale) - 1;
        }

        // General case: use floor(log(value) * scaleFactor)
        let index = (value.ln() * self.scale_factor).floor() as i32;

        let max_idx = self.max_normal_lower_boundary_index_log();
        if index >= max_idx {
            max_idx
        } else {
            index
        }
    }

    /// Returns the lower boundary of a bucket at the given index.
    #[inline]
    pub fn lower_boundary(&self, index: i32) -> Result<f64, MappingError> {
        if self.scale <= 0 {
            self.lower_boundary_exponent(index)
        } else {
            self.lower_boundary_logarithm(index)
        }
    }

    fn lower_boundary_exponent(&self, index: i32) -> Result<f64, MappingError> {
        let shift = (-self.scale) as u32;

        if index < self.min_normal_lower_boundary_index_exp() {
            return Err(MappingError::Underflow);
        }

        if index > self.max_normal_lower_boundary_index_exp() {
            return Err(MappingError::Overflow);
        }

        // 2^(index << shift)
        let exp = index << shift;
        Ok(libm::ldexp(1.0, exp))
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
    fn min_normal_lower_boundary_index_exp(&self) -> i32 {
        let shift = (-self.scale) as u32;
        let mut idx = MIN_NORMAL_EXPONENT >> shift;
        if shift < 2 {
            // For scales -1 and 0, 2^-1022 is a power-of-two multiple
            idx -= 1;
        }
        idx
    }

    #[inline]
    fn max_normal_lower_boundary_index_exp(&self) -> i32 {
        let shift = (-self.scale) as u32;
        MAX_NORMAL_EXPONENT >> shift
    }

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
        assert!(Mapping::new(20).is_ok());
        assert!(Mapping::new(-10).is_ok());
        assert!(Mapping::new(21).is_err());
        assert!(Mapping::new(-11).is_err());
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
        // Actually: index = floor(log_base(value)) where base = 2 at scale 0
        // For value in (base^i, base^(i+1)], index = i
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
        assert_eq!(m.lower_boundary(0).unwrap(), 1.0);  // 2^0 = 1
        assert_eq!(m.lower_boundary(1).unwrap(), 2.0);  // 2^1 = 2
        assert_eq!(m.lower_boundary(-1).unwrap(), 0.5); // 2^-1 = 0.5
        assert_eq!(m.lower_boundary(2).unwrap(), 4.0);  // 2^2 = 4
    }
}
