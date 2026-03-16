// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Lookup table-based mapping for exponential histograms.
//!
//! This module provides compile-time generated boundary and index tables
//! at the highest compiled-in scale. The algorithm uses 2N linear buckets
//! per scale with one boundary correction.
//!
//! Mapping at any scale S ≤ TABLE_SCALE is computed by mapping at
//! TABLE_SCALE and right-shifting: `fine_index >> (TABLE_SCALE - S)`.
//! This eliminates all runtime table derivation and heap allocation.

include!(concat!(env!("OUT_DIR"), "/lookup_tables.rs"));

/// Maps a positive f64 value to a bucket index using a compiled lookup table.
///
/// Always computes at `TABLE_SCALE` using the full boundary and index
/// tables, then right-shifts to the requested scale.
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    use crate::float64::{get_normal_base2, get_significand};

    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    let linear_idx = (significand >> INDEX_SHIFT) as usize;
    let approx = INDEX_TABLE[linear_idx] as usize;

    let mut bucket = approx as i32;
    if significand >= BOUNDARIES[approx + 1] {
        bucket += 1;
    }

    let fine = (exponent << TABLE_SCALE) + bucket - 1;
    fine >> (TABLE_SCALE - scale)
}

/// Returns the native scale (resolution) of the lookup table.
#[inline]
pub const fn table_scale() -> i32 {
    TABLE_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powers_of_two() {
        for scale in 1..=TABLE_SCALE {
            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index(value, scale);
                assert_eq!(
                    actual, expected,
                    "power of two mismatch at scale={scale}, exp={exp}: got {actual}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn test_basic_values() {
        let scale = TABLE_SCALE.min(4);
        assert_eq!(map_to_index(1.0, scale), -1);
        let expected = (1 << scale) - 1;
        assert_eq!(map_to_index(2.0, scale), expected);
        let idx = map_to_index(1.5, scale);
        let max_idx = (1 << scale) - 1;
        assert!(
            idx >= 0 && idx < max_idx,
            "1.5 should be in [0, {max_idx}), got {idx}",
        );
    }

    #[test]
    fn test_table_scale() {
        // build.rs emits EXPECTED_TABLE_SCALE = highest enabled scale feature.
        let expected: i32 = env!("EXPECTED_TABLE_SCALE").parse().unwrap();
        assert_eq!(
            TABLE_SCALE, expected,
            "TABLE_SCALE ({}) != expected ({})", TABLE_SCALE, expected,
        );
    }

    #[test]
    fn test_all_scales_consistent() {
        let test_values: &[f64] =
            &[1.1, 1.5, 1.9, 2.5, 3.3, 7.7, 0.3, 0.7, 100.0, 1e-10, 1e10];
        for scale in 1..TABLE_SCALE {
            for &v in test_values {
                let direct = map_to_index(v, scale);
                let fine = map_to_index(v, TABLE_SCALE);
                let shifted = fine >> (TABLE_SCALE - scale);
                assert_eq!(
                    direct, shifted,
                    "scale {scale} mismatch for value {v}: direct={direct}, shifted={shifted}",
                );
            }
        }
    }
}
