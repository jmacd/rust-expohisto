// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared lookup tables for exponential histogram mapping.
//!
//! This module provides the compile-time generated boundary significand
//! table at the highest compiled-in scale, plus per-scale derived tables
//! computed lazily at runtime. Both NewRelic and Dynatrace algorithms
//! share the same per-scale boundaries; they differ only in index table
//! size (2N vs N) and correction count (1 vs 2).
//!
//! Per-scale boundaries are derived by striding through the full-scale
//! BOUNDARIES array, packed contiguously in a single allocation (~2×
//! the full-scale table, geometric series). Each scale's working set
//! fits tightly in cache.

include!(concat!(env!("OUT_DIR"), "/lookup_tables.rs"));

use std::sync::OnceLock;

/// Pre-computed per-scale boundary tables, packed contiguously.
struct PerScaleBoundaries {
    data: Box<[u64]>,
    /// `offsets[s]` = start index of scale-s boundaries in `data`.
    offsets: [u32; 16],
}

static PER_SCALE: OnceLock<PerScaleBoundaries> = OnceLock::new();

fn per_scale_boundaries() -> &'static PerScaleBoundaries {
    PER_SCALE.get_or_init(|| {
        let h = TABLE_SCALE as usize;
        // Total entries: sum_{s=1}^{H} (2^s + 3) = (2^{H+1} - 2) + 3*H
        let total: usize = (1..=h).map(|s| (1usize << s) + 3).sum();
        let mut data = Vec::with_capacity(total);
        let mut offsets = [0u32; 16];

    #[allow(clippy::needless_range_loop)] // `s` used both as index and for bit shifts
    for s in 1..=h {
            offsets[s] = data.len() as u32;
            let n_s = 1usize << s;
            let stride = 1usize << (h - s);

            data.push(0); // sentinel
            for k in 0..n_s {
                data.push(BOUNDARIES[1 + k * stride]);
            }
            data.push(1u64 << 52); // sentinel
            data.push(1u64 << 52); // sentinel
        }

        PerScaleBoundaries {
            data: data.into_boxed_slice(),
            offsets,
        }
    })
}

/// Returns the boundary table for the given scale (1..=TABLE_SCALE).
///
/// The returned slice has `2^scale + 3` entries:
/// `[sentinel=0, b[0], b[1], ..., b[2^scale - 1], sentinel=2^52, sentinel=2^52]`
///
/// Each entry is the significand threshold between adjacent log buckets,
/// derived by taking every `2^(H-S)`-th entry from the full-scale table.
#[inline]
pub fn boundaries(scale: i32) -> &'static [u64] {
    debug_assert!((1..=TABLE_SCALE).contains(&scale));
    let psb = per_scale_boundaries();
    let s = scale as usize;
    let start = psb.offsets[s] as usize;
    let len = (1usize << s) + 3;
    &psb.data[start..start + len]
}

/// Derives a linear-to-log index table from a boundaries slice.
///
/// For `count` equidistant linear buckets (each of width `1 << shift`
/// in significand space), stores the approximate log bucket containing
/// each linear bucket's lower bound.
pub fn derive_index_table(boundaries: &[u64], count: usize, shift: u32) -> Vec<u16> {
    let mut table = vec![0u16; count];
    let mut j: u16 = 0;
    #[allow(clippy::needless_range_loop)] // `i` used for both indexing and bit shift
    for i in 0..count {
        let lower_bound = (i as u64) << shift;
        while lower_bound >= boundaries[j as usize + 1] {
            j += 1;
        }
        table[i] = j;
    }
    table
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

/// Generates the standard lookup module body (OnceLock table + map_to_index + table_scale).
///
/// Used by both `newrelic` and `dynatrace` modules, which differ only in
/// `extra_bits` (NR=1 → 2N linear buckets, DT=0 → N) and `corrections` (NR=1, DT=2).
#[macro_export]
#[doc(hidden)]
macro_rules! define_lookup_module {
    (extra_bits = $extra_bits:expr, corrections = $corrections:expr) => {
        use std::sync::OnceLock;
        use $crate::lookup::{ScaleTables, TABLE_SCALE, table_map_to_index};

        static TABLES: OnceLock<ScaleTables> = OnceLock::new();

        fn tables() -> &'static ScaleTables {
            TABLES.get_or_init(|| ScaleTables::new($extra_bits))
        }

        /// Maps a positive f64 value to a bucket index using a lookup table.
        #[inline]
        pub fn map_to_index(value: f64, scale: i32) -> i32 {
            table_map_to_index(value, scale, tables(), $corrections)
        }

        /// Returns the native scale (resolution) of the lookup table.
        #[inline]
        pub const fn table_scale() -> i32 {
            TABLE_SCALE
        }
    };
}

pub use define_lookup_module;

///
/// Both NR and DT algorithms use this structure; they differ only in
/// `extra_bits` (NR=1 → 2N linear buckets, DT=0 → N linear buckets).
pub struct ScaleTables {
    data: Box<[u16]>,
    /// `offsets[s]` = start index of scale-s table in `data`.
    offsets: [u32; 16],
    /// Extra significand bits used for linear indexing (NR=1, DT=0).
    extra_bits: u32,
}

impl ScaleTables {
    /// Builds packed index tables for all scales 1..=TABLE_SCALE.
    ///
    /// `extra_bits`: 1 for NewRelic (2N linear buckets), 0 for Dynatrace (N).
    pub fn new(extra_bits: u32) -> Self {
        let h = TABLE_SCALE as usize;
        let total: usize = (1..=h).map(|s| 1usize << (s + extra_bits as usize)).sum();
        let mut data = Vec::with_capacity(total);
        let mut offsets = [0u32; 16];

        #[allow(clippy::needless_range_loop)]
        for s in 1..=h {
            offsets[s] = data.len() as u32;
            let count = 1usize << (s + extra_bits as usize);
            let shift = 52 - s as u32 - extra_bits;
            let b = boundaries(s as i32);
            let table = derive_index_table(b, count, shift);
            data.extend_from_slice(&table);
        }

        Self {
            data: data.into_boxed_slice(),
            offsets,
            extra_bits,
        }
    }

    /// Returns the index table for the given scale.
    #[inline]
    pub fn index_table(&self, scale: i32) -> &[u16] {
        let s = scale as usize;
        let start = self.offsets[s] as usize;
        let len = 1usize << (s + self.extra_bits as usize);
        &self.data[start..start + len]
    }

    /// Returns the significand shift for the given scale.
    #[inline]
    pub fn shift(&self, scale: i32) -> u32 {
        52 - scale as u32 - self.extra_bits
    }
}

/// Maps a positive f64 value to a bucket index using a lookup table.
///
/// `corrections`: number of boundary checks after the initial approximation
/// (NR=1, DT=2).
#[inline]
pub fn table_map_to_index(
    value: f64,
    scale: i32,
    tables: &ScaleTables,
    corrections: u32,
) -> i32 {
    use crate::float64::{get_normal_base2, get_significand};

    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);
    debug_assert!(value > 0.0);

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    let b = boundaries(scale);
    let index = tables.index_table(scale);
    let shift = tables.shift(scale);
    let linear_idx = (significand >> shift) as usize;
    let approx = index[linear_idx] as usize;

    let mut bucket = approx as i32;
    for c in 1..=corrections {
        if significand >= b[approx + c as usize] {
            bucket += 1;
        }
    }

    (exponent << scale) + bucket - 1
}
