// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! NewRelic lookup table-based mapping for exponential histograms.
//!
//! This algorithm uses pre-computed lookup tables for exact bucket mapping
//! without floating-point precision errors. Per-scale boundary and index
//! tables are derived at runtime from the full-scale BOUNDARIES array,
//! packed in single allocations. Each scale's working set fits tightly
//! in cache lines.

use std::sync::OnceLock;

use crate::float64::{get_normal_base2, get_significand};
use crate::lookup::{TABLE_SCALE, boundaries, derive_index_table};

/// Per-scale NR index tables, packed contiguously.
struct NrScaleTables {
    data: Box<[u16]>,
    /// `offsets[s]` = start index of scale-s table in `data`.
    offsets: [u32; 16],
}

static NR_TABLES: OnceLock<NrScaleTables> = OnceLock::new();

fn nr_tables() -> &'static NrScaleTables {
    NR_TABLES.get_or_init(|| {
        let h = TABLE_SCALE as usize;
        // NR: 2^(s+1) entries per scale, total ≈ 4N
        let total: usize = (1..=h).map(|s| 1usize << (s + 1)).sum();
        let mut data = Vec::with_capacity(total);
        let mut offsets = [0u32; 16];

        for s in 1..=h {
            offsets[s] = data.len() as u32;
            let count = 1usize << (s + 1);
            let shift = 52 - s as u32 - 1;
            let b = boundaries(s as i32);
            let table = derive_index_table(b, count, shift);
            data.extend_from_slice(&table);
        }

        NrScaleTables {
            data: data.into_boxed_slice(),
            offsets,
        }
    })
}

/// Returns the NR index table for the given scale.
#[inline]
fn index_table(scale: i32) -> &'static [u16] {
    let nrt = nr_tables();
    let s = scale as usize;
    let start = nrt.offsets[s] as usize;
    let len = 1usize << (s + 1);
    &nrt.data[start..start + len]
}

/// Maps a positive f64 value to a bucket index.
///
/// Uses per-scale boundaries and 2N linear buckets with one branch correction.
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    debug_assert!(scale > 0);
    debug_assert!(scale <= TABLE_SCALE);
    debug_assert!(value > 0.0);

    let significand = get_significand(value);
    let exponent = get_normal_base2(value);

    let b = boundaries(scale);
    let index = index_table(scale);
    let shift = 52 - scale as u32 - 1;
    let linear_idx = (significand >> shift) as usize;
    let approx = index[linear_idx] as usize;

    let mut bucket = approx as i32;
    if significand >= b[approx + 1] { bucket += 1; }

    (exponent << scale) + bucket - 1
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
                    "power of two mismatch at scale={}, exp={}: got {}, expected {}",
                    scale, exp, actual, expected
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
            "1.5 should be in [0, {}), got {}",
            max_idx,
            idx
        );
    }

    #[test]
    fn test_table_scale() {
        // TABLE_SCALE should be >= the feature selected (may be higher
        // when multiple table features are enabled)
        #[cfg(feature = "scale-4")]
        assert!(TABLE_SCALE >= 4);
        #[cfg(feature = "scale-6")]
        assert!(TABLE_SCALE >= 6);
        #[cfg(feature = "scale-8")]
        assert!(TABLE_SCALE >= 8);
        #[cfg(feature = "scale-10")]
        assert!(TABLE_SCALE >= 10);
        #[cfg(feature = "scale-12")]
        assert!(TABLE_SCALE >= 12);
        #[cfg(feature = "scale-14")]
        assert!(TABLE_SCALE >= 14);
    }

    #[test]
    fn test_all_scales_consistent() {
        // Verify that lower scales give the same result as
        // computing at TABLE_SCALE and right-shifting.
        let test_values: &[f64] = &[1.1, 1.5, 1.9, 2.5, 3.3, 7.7, 0.3, 0.7, 100.0, 1e-10, 1e10];

        for scale in 1..TABLE_SCALE {
            for &v in test_values {
                let direct = map_to_index(v, scale);
                let fine = map_to_index(v, TABLE_SCALE);
                let shifted = fine >> (TABLE_SCALE - scale);

                assert_eq!(
                    direct, shifted,
                    "scale {} mismatch for value {}: direct={}, shifted={}",
                    scale, v, direct, shifted
                );
            }
        }
    }
}
