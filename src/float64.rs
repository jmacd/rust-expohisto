// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! IEEE 754 double-precision floating-point constants and utilities.
//!
//! An optional math layer (`exp`, `ln`, `floor`) is compiled when the
//! `logarithm` or `boundary` features are active, using `std` f64 methods.

/// Size of an IEEE 754 double-precision floating-point significand.
pub const SIGNIFICAND_WIDTH: u32 = 52;

/// Size of an IEEE 754 double-precision floating-point exponent.
pub const EXPONENT_WIDTH: u32 = 11;

/// Mask for the significand of an IEEE 754 double-precision value: 0xFFFFFFFFFFFFF.
pub const SIGNIFICAND_MASK: u64 = (1 << SIGNIFICAND_WIDTH) - 1;

/// Exponent bias for IEEE 754 double-precision: 1023.
pub const EXPONENT_BIAS: i32 = f64::MAX_EXP - 1;

/// Mask for the exponent bits: 0x7FF0000000000000.
pub const EXPONENT_MASK: u64 = ((1u64 << EXPONENT_WIDTH) - 1) << SIGNIFICAND_WIDTH;

/// Minimum exponent of a normalized floating point: -1022.
pub const MIN_NORMAL_EXPONENT: i32 = -EXPONENT_BIAS + 1;

/// Maximum exponent of a normalized floating point: 1023.
#[cfg_attr(not(feature = "boundary"), allow(dead_code))]
pub const MAX_NORMAL_EXPONENT: i32 = EXPONENT_BIAS;

/// Smallest normal f64 value: 2^-1022 (same as `f64::MIN_POSITIVE`).
pub const MIN_VALUE: f64 = f64::MIN_POSITIVE;

/// Extracts the normalized base-2 exponent from an f64.
#[inline]
pub const fn get_normal_base2(value: f64) -> i32 {
    let raw_bits = value.to_bits();
    let raw_exponent = ((raw_bits & EXPONENT_MASK) >> SIGNIFICAND_WIDTH) as i32;
    raw_exponent - EXPONENT_BIAS
}

/// Returns the 52-bit significand as an unsigned value.
#[inline]
pub const fn get_significand(value: f64) -> u64 {
    value.to_bits() & SIGNIFICAND_MASK
}

/// Constructs 2^k as an f64 using direct IEEE 754 bit manipulation.
///
/// Valid for k in \[`MIN_NORMAL_EXPONENT`, `MAX_NORMAL_EXPONENT`\] (i.e. −1022..=1023).
/// Panics in debug mode if k is out of range.
#[cfg_attr(not(feature = "boundary"), allow(dead_code))]
#[inline]
pub const fn pow2(k: i32) -> f64 {
    debug_assert!(
        k >= MIN_NORMAL_EXPONENT && k <= MAX_NORMAL_EXPONENT,
        "pow2 out of range"
    );
    let biased = (k + EXPONENT_BIAS) as u64;
    f64::from_bits(biased << SIGNIFICAND_WIDTH)
}

// ── Math helpers (std only) ──────────────────────────────────────────
// Only compiled when `logarithm` or `boundary` features are active;
// both imply `std`. Excluded from `mapping-gen` (no features set).

#[cfg(any(feature = "logarithm", feature = "boundary"))]
mod math_imp {
    #[inline]
    pub fn exp(x: f64) -> f64 {
        x.exp()
    }
    #[inline]
    pub fn ln(x: f64) -> f64 {
        x.ln()
    }
    #[inline]
    pub fn floor(x: f64) -> f64 {
        x.floor()
    }
}

#[cfg(any(feature = "logarithm", feature = "boundary"))]
pub use math_imp::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_normal_base2() {
        assert_eq!(get_normal_base2(1.0), 0);
        assert_eq!(get_normal_base2(2.0), 1);
        assert_eq!(get_normal_base2(4.0), 2);
        assert_eq!(get_normal_base2(0.5), -1);
        assert_eq!(get_normal_base2(0.25), -2);
        assert_eq!(get_normal_base2(1.5), 0);
        assert_eq!(get_normal_base2(3.0), 1);
    }

    #[test]
    fn test_get_significand() {
        // 1.0 has significand 0 (implicit 1)
        assert_eq!(get_significand(1.0), 0);
        // 1.5 = 1 + 0.5, so significand is 2^51
        assert_eq!(get_significand(1.5), 1 << 51);
        // 2.0 has significand 0
        assert_eq!(get_significand(2.0), 0);
        // 1.25 = 1 + 0.25, so significand is 2^50
        assert_eq!(get_significand(1.25), 1 << 50);
    }

    #[test]
    fn test_constants() {
        assert_eq!(SIGNIFICAND_WIDTH, 52);
        assert_eq!(EXPONENT_WIDTH, 11);
        assert_eq!(EXPONENT_BIAS, 1023);
        assert_eq!(SIGNIFICAND_MASK, 0xFFFFFFFFFFFFF);
        assert_eq!(EXPONENT_MASK, 0x7FF0000000000000);
        assert_eq!(MIN_NORMAL_EXPONENT, -1022);
        assert_eq!(MAX_NORMAL_EXPONENT, 1023);
        assert_eq!(MIN_VALUE, f64::MIN_POSITIVE);
    }

    #[test]
    fn test_pow2() {
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(1), 2.0);
        assert_eq!(pow2(2), 4.0);
        assert_eq!(pow2(10), 1024.0);
        assert_eq!(pow2(-1), 0.5);
        assert_eq!(pow2(-2), 0.25);
        assert_eq!(pow2(-10), 1.0 / 1024.0);
        assert_eq!(pow2(MIN_NORMAL_EXPONENT), MIN_VALUE);
        assert_eq!(pow2(MAX_NORMAL_EXPONENT), 2.0_f64.powi(1023));
        // Verify bit-exact: every result has zero significand
        for k in -1022..=1023 {
            let v = pow2(k);
            assert_eq!(get_significand(v), 0, "pow2({k}) should have zero significand");
            assert_eq!(get_normal_base2(v), k, "pow2({k}) should have exponent {k}");
        }
    }
}
