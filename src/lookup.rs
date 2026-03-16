// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared lookup tables for exponential histogram mapping.
//!
//! This module provides compile-time generated boundary and index tables
//! at the highest compiled-in scale. Both NewRelic and Dynatrace algorithms
//! share the same `BOUNDARIES` array; they differ only in index table
//! size (2N vs N) and correction count (1 vs 2).
//!
//! Mapping at any scale S ≤ TABLE_SCALE is computed by mapping at
//! TABLE_SCALE and right-shifting: `fine_index >> (TABLE_SCALE - S)`.
//! This eliminates all runtime table derivation and heap allocation.

include!(concat!(env!("OUT_DIR"), "/lookup_tables.rs"));

/// Maps a positive f64 value to a bucket index using a compiled lookup table.
///
/// Always computes at `TABLE_SCALE` using the full boundary and index
/// tables, then right-shifts to the requested scale.
///
/// `index_table`: the compiled `{NR,DT}_INDEX` array for the selected algorithm.
/// `shift`: the compiled `{NR,DT}_SHIFT` constant.
/// `corrections`: number of boundary checks after the initial approximation
/// (NR=1, DT=2).
#[inline]
pub fn table_map_to_index(
    value: f64,
    scale: i32,
    index_table: &[u16],
    shift: u32,
    corrections: u32,
) -> i32 {
    use crate::float64::{get_normal_base2, get_significand};

    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    let linear_idx = (significand >> shift) as usize;
    let approx = index_table[linear_idx] as usize;

    let mut bucket = approx as i32;
    for c in 1..=corrections {
        if significand >= BOUNDARIES[approx + c as usize] {
            bucket += 1;
        }
    }

    let fine = (exponent << TABLE_SCALE) + bucket - 1;
    fine >> (TABLE_SCALE - scale)
}

/// Shared tests for lookup table-based mapping algorithms.
///
/// Tests powers-of-two invariant, basic values, table_scale bounds,
/// and cross-scale consistency. Used by both `newrelic` and `dynatrace`.
#[macro_export]
#[doc(hidden)]
macro_rules! lookup_tests {
    ($map_fn:path, $table_scale:expr) => {
        #[test]
        fn test_powers_of_two() {
            for scale in 1..=$table_scale {
                for exp in -10..=10 {
                    let value = 2.0_f64.powi(exp);
                    let expected = (exp << scale) - 1;
                    let actual = $map_fn(value, scale);
                    assert_eq!(
                        actual, expected,
                        "power of two mismatch at scale={scale}, exp={exp}: got {actual}, expected {expected}",
                    );
                }
            }
        }

        #[test]
        fn test_basic_values() {
            let scale = $table_scale.min(4);
            assert_eq!($map_fn(1.0, scale), -1);
            let expected = (1 << scale) - 1;
            assert_eq!($map_fn(2.0, scale), expected);
            let idx = $map_fn(1.5, scale);
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
                $table_scale, expected,
                "TABLE_SCALE ({}) != expected ({})", $table_scale, expected,
            );
        }

        #[test]
        fn test_all_scales_consistent() {
            let test_values: &[f64] =
                &[1.1, 1.5, 1.9, 2.5, 3.3, 7.7, 0.3, 0.7, 100.0, 1e-10, 1e10];
            for scale in 1..$table_scale {
                for &v in test_values {
                    let direct = $map_fn(v, scale);
                    let fine = $map_fn(v, $table_scale);
                    let shifted = fine >> ($table_scale - scale);
                    assert_eq!(
                        direct, shifted,
                        "scale {scale} mismatch for value {v}: direct={direct}, shifted={shifted}",
                    );
                }
            }
        }
    };
}

// Re-export for use in sibling modules.
pub use lookup_tests;

/// Generates the standard lookup module body (map_to_index + table_scale).
///
/// Used by both `newrelic` and `dynatrace` modules, which differ only in
/// the compiled index table and correction count.
#[macro_export]
#[doc(hidden)]
macro_rules! define_lookup_module {
    (index_table = $index:ident, shift = $shift:ident, corrections = $corrections:expr) => {
        // Re-export TABLE_SCALE for use by tests and downstream code.
        pub use $crate::lookup::TABLE_SCALE;

        /// Maps a positive f64 value to a bucket index using a lookup table.
        #[inline]
        pub fn map_to_index(value: f64, scale: i32) -> i32 {
            $crate::lookup::table_map_to_index(
                value,
                scale,
                &$crate::lookup::$index,
                $crate::lookup::$shift,
                $corrections,
            )
        }

        /// Returns the native scale (resolution) of the lookup table.
        #[inline]
        pub const fn table_scale() -> i32 {
            $crate::lookup::TABLE_SCALE
        }
    };
}

pub use define_lookup_module;
