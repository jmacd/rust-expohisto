// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! IEEE 754 double-precision floating-point constants and utilities.

/// Size of an IEEE 754 double-precision floating-point significand.
pub(crate) const SIGNIFICAND_WIDTH: u32 = 52;

/// Size of an IEEE 754 double-precision floating-point exponent.
pub(crate) const EXPONENT_WIDTH: u32 = 11;

/// Mask for the significand of an IEEE 754 double-precision value: 0xFFFFFFFFFFFFF.
pub(crate) const SIGNIFICAND_MASK: u64 = (1 << SIGNIFICAND_WIDTH) - 1;

/// Exponent bias for IEEE 754 double-precision: 1023.
pub(crate) const EXPONENT_BIAS: i32 = f64::MAX_EXP - 1;

/// Mask for the exponent bits: 0x7FF0000000000000.
pub(crate) const EXPONENT_MASK: u64 = ((1u64 << EXPONENT_WIDTH) - 1) << SIGNIFICAND_WIDTH;

/// Minimum exponent of a normalized floating point: -1022.
pub(crate) const MIN_NORMAL_EXPONENT: i32 = -EXPONENT_BIAS + 1;

/// Maximum exponent of a normalized floating point: 1023.
pub(crate) const MAX_NORMAL_EXPONENT: i32 = EXPONENT_BIAS;

/// Smallest normal f64 value: 2^-1022.
pub(crate) const MIN_VALUE: f64 = 2.2250738585072014e-308; // 0x1p-1022

/// Extracts the normalized base-2 exponent from an f64.
#[inline]
pub(crate) fn get_normal_base2(value: f64) -> i32 {
    let raw_bits = value.to_bits();
    let raw_exponent = ((raw_bits & EXPONENT_MASK) >> SIGNIFICAND_WIDTH) as i32;
    raw_exponent - EXPONENT_BIAS
}

/// Returns the 52-bit significand as an unsigned value.
#[inline]
pub(crate) fn get_significand(value: f64) -> u64 {
    value.to_bits() & SIGNIFICAND_MASK
}

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
    }

    #[test]
    fn test_get_significand() {
        // 1.0 has significand 0 (implicit 1)
        assert_eq!(get_significand(1.0), 0);
        // 1.5 = 1 + 0.5, so significand is 2^51
        assert_eq!(get_significand(1.5), 1 << 51);
        // 2.0 has significand 0
        assert_eq!(get_significand(2.0), 0);
    }
}
