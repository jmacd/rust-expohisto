// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exponential histogram scale mapping functions.
//!
//! This module provides mapping functions that convert floating-point values
//! to bucket indices based on the configured scale.

use crate::float64::{
    MAX_NORMAL_EXPONENT, MIN_NORMAL_EXPONENT, MIN_VALUE, get_normal_base2, get_significand,
};

/// Reference implementation using pure logarithm-based mapping.
///
/// This function provides a reference implementation for testing and benchmarking.
/// It uses `floor(log(value) * scaleFactor)` with a special case for exact powers of two.
///
/// # Arguments
/// * `value` - A positive, finite, non-zero f64 value
/// * `scale` - The histogram scale (must be > 0)
/// * `scale_factor` - Pre-computed as `LOG2_E * 2^scale`
///
/// # Returns
/// The bucket index for the value at the given scale.
#[inline]
pub fn map_to_index_lg(value: f64, scale: i32, scale_factor: f64) -> i32 {
    debug_assert!(scale > 0);

    // Exact power-of-two: significand is 0, index is (exp << scale) - 1
    if get_significand(value) == 0 {
        let exp = get_normal_base2(value);
        return (exp << scale) - 1;
    }

    // General case: use floor(log(value) * scaleFactor)
    libm::floor(libm::log(value) * scale_factor) as i32
}

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

        // Use lookup table if available and scale is supported
        #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
        if crate::lookup::supports_scale(scale) {
            return crate::lookup::map_to_index_lookup(value, scale);
        }

        // Exact power-of-two optimization
        if get_significand(value) == 0 {
            let exp = get_normal_base2(value);
            return (exp << scale) - 1;
        }

        // General case: use floor(log(value) * scaleFactor)
        let index = libm::floor(libm::log(value) * self.scale_factor) as i32;

        let max_idx = self.max_normal_lower_boundary_index_log();
        if index >= max_idx { max_idx } else { index }
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
                return Ok(2.0 * libm::exp((index - (1 << scale)) as f64 * self.inverse_factor));
            }
            return Err(MappingError::Overflow);
        }

        if index <= min_idx {
            if index == min_idx {
                return Ok(MIN_VALUE);
            } else if index == min_idx - 1 {
                return Ok(libm::exp((index + (1 << scale)) as f64 * self.inverse_factor) / 2.0);
            }
            return Err(MappingError::Underflow);
        }

        Ok(libm::exp(index as f64 * self.inverse_factor))
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
        assert_eq!(m.lower_boundary(0).unwrap(), 1.0); // 2^0 = 1
        assert_eq!(m.lower_boundary(1).unwrap(), 2.0); // 2^1 = 2
        assert_eq!(m.lower_boundary(-1).unwrap(), 0.5); // 2^-1 = 0.5
        assert_eq!(m.lower_boundary(2).unwrap(), 4.0); // 2^2 = 4
    }

    #[test]
    fn test_lg_matches_mapping() {
        // Test that the reference lg function matches the pure logarithm Mapping implementation.
        // Note: When lookup tables are enabled, Mapping uses lookup which may differ 
        // at bucket boundaries due to table resolution. This test only runs without lookup.
        #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
        {
            // Skip when lookup is enabled - lg matches pure logarithm, not lookup
            return;
        }

        #[cfg(not(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024")))]
        {
            let test_values = [
                1.0001, 1.1, 1.5, 1.9, 1.9999,
                2.0, 2.5, 3.0, 4.0, 7.5,
                10.0, 100.0, 1000.0, 1e10, 1e100,
                0.5, 0.1, 0.01, 1e-10, 1e-100,
            ];

            for scale in 1..=MAX_SCALE {
                let m = Mapping::new(scale).unwrap();
                let scale_factor = core::f64::consts::LOG2_E * (1u64 << scale) as f64;
                
                for &v in &test_values {
                    let expected = m.map_to_index(v);
                    let actual = map_to_index_lg(v, scale, scale_factor);
                    assert_eq!(
                        actual, expected,
                        "lg mismatch at scale={}, value={}: lg={}, mapping={}",
                        scale, v, actual, expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_lg_powers_of_two() {
        // Powers of two should map to (exp << scale) - 1
        for scale in 1..=MAX_SCALE {
            let scale_factor = core::f64::consts::LOG2_E * (1u64 << scale) as f64;
            
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index_lg(value, scale, scale_factor);
                assert_eq!(
                    actual, expected,
                    "lg power of two mismatch at scale={}, exp={}: got {}, expected {}",
                    scale, exp, actual, expected
                );
            }
        }
    }

    /// Get the next representable f64 value greater than v.
    #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
    fn next_up(v: f64) -> f64 {
        let bits = v.to_bits();
        f64::from_bits(bits + 1)
    }

    /// Get the next representable f64 value less than v.
    #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
    fn next_down(v: f64) -> f64 {
        let bits = v.to_bits();
        f64::from_bits(bits - 1)
    }

    #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
    #[test]
    fn test_lookup_powers_of_two_boundary() {
        // Test that the lookup table correctly handles values at and near powers of two.
        // The lookup table has special handling for exact powers of two (significand == 0).
        // 
        // Key invariant: For exact powers of two, index = (exp << scale) - 1.
        // The lookup table correctly handles this via the significand == 0 check.
        
        let max_lookup_scale = crate::lookup::max_lookup_scale();
        
        for scale in 1..=max_lookup_scale {
            for exp in -100..=100 {
                let power_of_two = libm::ldexp(1.0, exp);
                let pow2_expected = (exp << scale) - 1;
                
                // Lookup must be correct for exact powers of two
                let pow2_lookup = crate::lookup::map_to_index_lookup(power_of_two, scale);
                assert_eq!(
                    pow2_lookup, pow2_expected,
                    "lookup error at power of two: scale={}, exp={}, value={:e}, got={}, expected={}",
                    scale, exp, power_of_two, pow2_lookup, pow2_expected
                );
            }
        }
    }

    #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
    #[test]
    fn test_lg_precision_errors_at_scale_boundaries() {
        // Demonstrate that the pure logarithm implementation can have floating-point 
        // precision errors at bucket boundaries that the lookup table avoids.
        //
        // The logarithm formula: floor(log(value) * scaleFactor)
        // When log(value) * scaleFactor is very close to an integer, floating-point
        // rounding can push it to the wrong side of the integer, causing off-by-one errors.
        //
        // This test compares lg against the Mapping (which uses lookup when available).
        // For scales within LOOKUP_SCALE, lookup is used and we see lg errors.
        // For scales beyond LOOKUP_SCALE, both use logarithm and should agree.
        
        extern crate std;
        
        let max_lookup_scale = crate::lookup::max_lookup_scale();
        
        // Test values near bucket boundaries
        const STEPS_PER_BOUNDARY: i32 = 50;
        
        std::println!("=== Precision Error Analysis ===");
        std::println!("Lookup table supports scales 1..={}", max_lookup_scale);
        std::println!("{:>5} {:>8} {:>12} {:>12} {:>10}", "Scale", "Method", "Tests", "Mismatches", "Rate");
        std::println!("{}", "-".repeat(55));
        
        let mut grand_total_tests = 0u64;
        let mut grand_total_mismatches = 0u64;
        
        for scale in 1..=MAX_SCALE {
            let scale_factor = core::f64::consts::LOG2_E * (1u64 << scale) as f64;
            let buckets_per_octave = 1i32 << scale;
            let uses_lookup = scale <= max_lookup_scale;
            
            let mut total_tests = 0u64;
            let mut lg_mapping_mismatches = 0u64;
            let mut example_mismatch: Option<(i32, f64, i32, i32)> = None;
            
            // For high scales, limit the number of boundaries tested to keep runtime reasonable
            let exp_range = if scale > 10 { -5..=5 } else { -10..=10 };
            
            // Test boundaries within each octave
            for exp in exp_range {
                // For high scales, sample fewer bucket offsets
                let step = if scale > 10 { 1 << (scale - 10) } else { 1 };
                let mut bucket_offset = 0;
                while bucket_offset < buckets_per_octave {
                    // The boundary value is 2^(exp + bucket_offset/2^scale)
                    let boundary_exp = (exp as f64) + (bucket_offset as f64) / (buckets_per_octave as f64);
                    let boundary_value = libm::exp2(boundary_exp);
                    
                    // Step through values near this boundary
                    let mut v = boundary_value;
                    for _ in 0..STEPS_PER_BOUNDARY {
                        v = next_up(v);
                        total_tests += 1;
                        
                        // Compare lg against Mapping (which uses lookup if available)
                        let m = Mapping::new(scale).unwrap();
                        let mapping_idx = m.map_to_index(v);
                        let lg_idx = map_to_index_lg(v, scale, scale_factor);
                        
                        if lg_idx != mapping_idx {
                            lg_mapping_mismatches += 1;
                            if example_mismatch.is_none() {
                                example_mismatch = Some((exp, v, lg_idx, mapping_idx));
                            }
                        }
                    }
                    
                    let mut v = boundary_value;
                    for _ in 0..STEPS_PER_BOUNDARY {
                        v = next_down(v);
                        total_tests += 1;
                        
                        let m = Mapping::new(scale).unwrap();
                        let mapping_idx = m.map_to_index(v);
                        let lg_idx = map_to_index_lg(v, scale, scale_factor);
                        
                        if lg_idx != mapping_idx {
                            lg_mapping_mismatches += 1;
                            if example_mismatch.is_none() {
                                example_mismatch = Some((exp, v, lg_idx, mapping_idx));
                            }
                        }
                    }
                    
                    bucket_offset += step;
                }
            }
            
            let mismatch_rate = (lg_mapping_mismatches as f64 / total_tests as f64) * 100.0;
            let method = if uses_lookup { "lookup" } else { "log" };
            std::println!("{:>5} {:>8} {:>12} {:>12} {:>9.4}%", scale, method, total_tests, lg_mapping_mismatches, mismatch_rate);
            
            if let Some((exp, value, lg_idx, mapping_idx)) = example_mismatch {
                std::println!("       Example: exp={}, value={:e}, lg={}, mapping={}", exp, value, lg_idx, mapping_idx);
            }
            
            grand_total_tests += total_tests;
            grand_total_mismatches += lg_mapping_mismatches;
        }
        
        std::println!("{}", "-".repeat(55));
        let grand_rate = (grand_total_mismatches as f64 / grand_total_tests as f64) * 100.0;
        std::println!("{:>5} {:>8} {:>12} {:>12} {:>9.4}%", "Total", "", grand_total_tests, grand_total_mismatches, grand_rate);
        std::println!("=======================================================");
        std::println!();
        std::println!("Summary:");
        std::println!("  - Scales 1..={} use lookup table (exact integer arithmetic)", max_lookup_scale);
        std::println!("  - Scales {}..={} use logarithm fallback", max_lookup_scale + 1, MAX_SCALE);
        std::println!("  - Mismatches show where lg() has floating-point precision errors");
        std::println!("  - 0% mismatch for 'log' method means lg agrees with itself (both have same errors)");
        std::println!("  - Larger tables (e.g., 1024 vs 64) catch more lg errors by covering more scales");
        
        // Assert that we tested enough values
        assert!(
            grand_total_tests > 1000,
            "Test should have checked many values, got only {}",
            grand_total_tests
        );
    }

    #[cfg(any(feature = "lookup-64", feature = "lookup-256", feature = "lookup-1024"))]
    #[test]
    fn test_lg_and_lookup_agree_at_all_scales() {
        // Test that lg and lookup agree at all supported scales.
        // The lookup table computes the full index at LOOKUP_SCALE, then shifts down,
        // which should give the same result as the pure logarithm formula.
        
        let max_lookup_scale = crate::lookup::max_lookup_scale();
        
        // Values chosen to be in the middle of buckets (not near boundaries)
        let test_values = [
            1.1, 1.3, 1.6, 1.8,
            2.3, 3.5, 5.0, 7.0,
            11.0, 23.0, 47.0, 97.0, 
            0.9, 0.7, 0.4, 0.15,
            1e5, 1e-5,
        ];
        
        for scale in 1..=max_lookup_scale {
            let scale_factor = core::f64::consts::LOG2_E * (1u64 << scale) as f64;
            
            for &v in &test_values {
                let lookup_idx = crate::lookup::map_to_index_lookup(v, scale);
                let lg_idx = map_to_index_lg(v, scale, scale_factor);
                
                assert_eq!(
                    lg_idx, lookup_idx,
                    "lg vs lookup mismatch: scale={}, value={:e}, lg={}, lookup={}",
                    scale, v, lg_idx, lookup_idx
                );
            }
        }
    }
}
