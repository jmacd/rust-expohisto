// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket boundary computation.
//!
//! Converts bucket indices back to f64 boundaries.  For scale ≤ 0 the
//! boundaries are exact powers of two.  For positive scales, they are
//! computed via `exp(index * ln(2) / 2^scale)` using a precomputed
//! inverse-factor table covering scales 1..=20.
//!
//! This entire module is gated behind `#[cfg(feature = "boundary")]`.

use crate::float64::{
    MAX_NORMAL_EXPONENT, MIN_NORMAL_EXPONENT, MIN_VALUE, pow2,
};
use crate::mapping::{Mapping, MappingError, MAX_SCALE};

/// `ln(2) / 2^scale` for scales 1..=20, indexed as `[scale - 1]`.
const INVERSE_FACTOR: [f64; MAX_SCALE as usize] = [
    0.34657359027997264,   // scale 1
    0.17328679513998632,   // scale 2
    0.08664339756999316,   // scale 3
    0.04332169878499658,   // scale 4
    0.02166084939249829,   // scale 5
    0.010830424696249145,  // scale 6
    0.0054152123481245725, // scale 7
    0.0027076061740622863, // scale 8
    0.0013538030870311431, // scale 9
    0.0006769015435155716, // scale 10
    0.0003384507717577858, // scale 11
    0.0001692253858788929, // scale 12
    8.461269293944645e-05, // scale 13
    4.230634646972322e-05, // scale 14
    2.115317323486161e-05, // scale 15
    1.0576586617430806e-05, // scale 16
    5.288293308715403e-06, // scale 17
    2.6441466543577014e-06, // scale 18
    1.3220733271788507e-06, // scale 19
    6.610366635894254e-07, // scale 20
];

/// Maximum valid bucket index for `lower_boundary` at a non-positive scale.
#[inline]
const fn max_normal_index_exp(scale: i32) -> i32 {
    let shift = (-scale) as u32;
    MAX_NORMAL_EXPONENT >> shift
}

/// Returns the lower boundary of a bucket at non-positive scale.
fn lower_boundary_exponent(index: i32, scale: i32) -> Result<f64, MappingError> {
    debug_assert!(scale <= 0);
    let shift = (-scale) as u32;

    if index < crate::exponent::min_normal_lower_boundary_index(scale) {
        return Err(MappingError::Underflow);
    }
    if index > max_normal_index_exp(scale) {
        return Err(MappingError::Overflow);
    }

    Ok(pow2(index << shift))
}

/// Minimum valid bucket index for `lower_boundary` at a positive scale.
#[inline]
const fn min_normal_index_log(scale: i32) -> i32 {
    MIN_NORMAL_EXPONENT << scale
}

/// Maximum valid bucket index for `lower_boundary` at a positive scale.
#[inline]
const fn max_normal_index_log(scale: i32) -> i32 {
    ((MAX_NORMAL_EXPONENT + 1) << scale) - 1
}

/// Returns the lower boundary of a bucket at positive scale.
fn lower_boundary_logarithm(index: i32, scale: i32) -> Result<f64, MappingError> {
    debug_assert!((1..=MAX_SCALE).contains(&scale));
    let inv = INVERSE_FACTOR[scale as usize - 1];
    let max_idx = max_normal_index_log(scale);
    let min_idx = min_normal_index_log(scale);

    if index >= max_idx {
        if index == max_idx {
            return Ok(2.0 * crate::float64::exp((index - (1 << scale)) as f64 * inv));
        }
        return Err(MappingError::Overflow);
    }

    if index <= min_idx {
        if index == min_idx {
            return Ok(MIN_VALUE);
        } else if index == min_idx - 1 {
            return Ok(crate::float64::exp((index + (1 << scale)) as f64 * inv) / 2.0);
        }
        return Err(MappingError::Underflow);
    }

    Ok(crate::float64::exp(index as f64 * inv))
}

impl Mapping {
    /// Returns the lower boundary of a bucket at the given index.
    ///
    /// For scale ≤ 0 this is an exact power of two.
    /// For positive scales, uses `exp()` with a precomputed
    /// inverse factor.
    #[inline]
    pub fn lower_boundary(&self, index: i32) -> Result<f64, MappingError> {
        if self.scale() <= 0 {
            lower_boundary_exponent(index, self.scale())
        } else {
            lower_boundary_logarithm(index, self.scale())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow2() {
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(1), 2.0);
        assert_eq!(pow2(-1), 0.5);
        assert_eq!(pow2(10), 1024.0);
        assert_eq!(pow2(MIN_NORMAL_EXPONENT), MIN_VALUE);
    }

    #[test]
    fn test_lower_boundary_scale_0() {
        let m = Mapping::new(0).unwrap();
        assert_eq!(m.lower_boundary(0).unwrap(), 1.0);
        assert_eq!(m.lower_boundary(1).unwrap(), 2.0);
        assert_eq!(m.lower_boundary(-1).unwrap(), 0.5);
        assert_eq!(m.lower_boundary(2).unwrap(), 4.0);
    }

    #[test]
    fn test_lower_boundary_min_scale() {
        let m = Mapping::new(crate::mapping::MIN_SCALE).unwrap();
        assert_eq!(m.lower_boundary(0).unwrap(), 1.0);
        assert!(m.lower_boundary(1).is_err());
    }

    #[test]
    fn test_inverse_factor_values() {
        for scale in 1..=MAX_SCALE {
            let expected = core::f64::consts::LN_2 / (1u64 << scale) as f64;
            assert_eq!(
                INVERSE_FACTOR[scale as usize - 1], expected,
                "inverse factor mismatch at scale {scale}"
            );
        }
    }
}
