// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Dynatrace lookup table generation for exponential histogram mapping.
//!
//! This generates tables for the Dynatrace algorithm which uses:
//! - N linear buckets (half of NewRelic's 2N) mapping to approximate log bucket indices
//! - N+2 boundaries (N exact boundaries + 2 sentinels) for two-branch correction
//!
//! The boundaries are identical to NewRelic's (computed via `compute_boundaries_exact`).
//! The difference is in the index table structure and the mapping function.

use crate::float64::{get_normal_base2, get_significand};
use crate::newrelic_table::compute_boundaries_exact;

/// Generated Dynatrace lookup tables for a specific scale.
#[derive(Debug, Clone)]
pub struct DynatraceTables {
    /// log_2 of table size (N = 2^index_bits buckets).
    pub index_bits: u32,
    /// Number of log buckets (N = 2^index_bits).
    pub n: usize,
    /// Maps each of N equidistant linear buckets to an approximate log bucket.
    /// Has N entries. The correction uses `rough as i32 - 1` as base, then
    /// two comparisons adjust upward — so `offset` ranges from -1 to rough+1.
    pub indices: Vec<u16>,
    /// Boundary significands with upper-inclusive sentinel at position 0.
    /// Layout: [sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]
    /// Has N+3 entries. The sentinel at position 0 and b[0]=1 ensure that
    /// significand=0 (exact powers of two) naturally maps one bucket lower,
    /// implementing OTel's upper-inclusive bucket semantics without a branch.
    pub boundaries: Vec<u64>,
    /// Shift to convert 52-bit significand to linear bucket index: 52 - scale.
    pub significand_shift: u32,
}

impl DynatraceTables {
    /// Generates Dynatrace lookup tables for a given number of index bits.
    pub fn generate(index_bits: u32) -> Self {
        let n = 1usize << index_bits;
        let raw_boundaries = compute_boundaries_exact(n, index_bits);
        Self::from_boundaries(index_bits, raw_boundaries)
    }

    /// Creates Dynatrace tables from pre-computed raw boundaries.
    ///
    /// `raw_boundaries` must have `2^index_bits` entries with `raw_boundaries[0] == 0`,
    /// as returned by `compute_boundaries_exact`.
    pub fn from_boundaries(index_bits: u32, raw_boundaries: Vec<u64>) -> Self {
        let n = 1usize << index_bits;
        debug_assert_eq!(raw_boundaries.len(), n);
        debug_assert_eq!(raw_boundaries[0], 0);

        // Build boundaries with upper-inclusive sentinel at position 0:
        // [sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]
        //
        // The sentinel at position 0 and b[0]=1 handle upper-inclusive semantics:
        // - For significand=0 (power of two), rough=0, offset starts at -1,
        //   neither correction fires → result = (exp << scale) - 1. Correct.
        // - For significand>=1, the b[0]=1 check fires, bringing offset to 0+.
        let mut boundaries = Vec::with_capacity(n + 3);
        boundaries.push(0); // sentinel at position 0
        boundaries.push(1); // upper-inclusive: b[0] = 1 instead of 0
        boundaries.extend_from_slice(&raw_boundaries[1..]);
        boundaries.push(1u64 << 52); // sentinel
        boundaries.push(1u64 << 52); // sentinel

        let indices = compute_dynatrace_indices(n, &boundaries, index_bits);
        let significand_shift = 52 - index_bits;

        Self {
            index_bits,
            n,
            indices,
            boundaries,
            significand_shift,
        }
    }
}

/// Compute the Dynatrace index table: N linear buckets, each mapping to
/// the largest boundary index c such that boundaries[c+1] <= lower_bound.
///
/// `boundaries` must have N+3 entries (sentinel + N+1 boundaries + sentinel).
/// Linear bucket i covers significands starting at i << (52 - scale).
/// The index c is chosen so that looking at boundaries[c+1] and boundaries[c+2]
/// is sufficient to find the exact log bucket (at most 2 corrections needed).
pub fn compute_dynatrace_indices(n: usize, boundaries: &[u64], scale: u32) -> Vec<u16> {
    let mut indices = vec![0u16; n];
    let mut c: u16 = 0;
    for i in 0..n {
        let mantissa_lower_bound = (i as u64) << (52 - scale);
        while boundaries[(c + 1) as usize] <= mantissa_lower_bound {
            c += 1;
        }
        indices[i] = c;
    }
    indices
}

/// Maps a positive f64 value to a bucket index using the Dynatrace algorithm.
///
/// This is the reference implementation used for testing the generated tables.
/// The production version lives in `src/dynatrace.rs` and uses pre-generated
/// static tables.
///
/// Upper-inclusive semantics are built into the boundary table: the sentinel
/// at position 0 and boundary[1]=1 ensure that significand=0 (exact powers
/// of two) naturally maps one bucket lower without a branch.
pub fn map_to_index_dynatrace(value: f64, tables: &DynatraceTables) -> i32 {
    let significand = get_significand(value);
    let exponent = get_normal_base2(value);
    let scale = tables.index_bits as i32;

    // Look up the rough bucket from N equidistant linear buckets
    let linear_idx = (significand >> tables.significand_shift) as usize;
    let rough = tables.indices[linear_idx] as usize;

    // Start at rough - 1 (may be -1 for significand=0 at rough=0).
    // Two corrections adjust upward based on boundary comparisons.
    let mut offset = rough as i32 - 1;
    if significand >= tables.boundaries[rough + 1] {
        offset += 1;
    }
    if significand >= tables.boundaries[rough + 2] {
        offset += 1;
    }

    (exponent << scale) + offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::newrelic_table::map_to_index_exact;

    #[test]
    fn test_generate_sizes() {
        for index_bits in [1, 2, 4, 6, 8, 10] {
            let tables = DynatraceTables::generate(index_bits);
            let n = 1usize << index_bits;
            assert_eq!(tables.n, n);
            assert_eq!(tables.indices.len(), n, "scale {}", index_bits);
            assert_eq!(tables.boundaries.len(), n + 3, "scale {}", index_bits);
            assert_eq!(tables.significand_shift, 52 - index_bits);
        }
    }

    #[test]
    fn test_boundaries_match_newrelic() {
        // Dynatrace and NewRelic use the same exact boundaries (with different layout).
        // Dynatrace: [sentinel=0, b[0]=1, b[1], ..., b[N-1], S, S]  (N+3 entries)
        // NewRelic:  [b[0]=1, b[1], ..., b[N-1], S]                  (N+1 entries)
        for index_bits in [4, 6, 8, 10] {
            let dt = DynatraceTables::generate(index_bits);
            let nr = crate::newrelic_table::LookupTables::generate(index_bits);

            // Dynatrace boundaries[1..N+1] should match NewRelic log_bucket_end[0..N]
            for k in 0..dt.n {
                assert_eq!(
                    dt.boundaries[k + 1], nr.log_bucket_end[k],
                    "boundary mismatch at scale={}, k={}",
                    index_bits, k
                );
            }
            // Sentinel at position 0
            assert_eq!(dt.boundaries[0], 0);
            // Sentinels at the end
            assert_eq!(dt.boundaries[dt.n + 1], 1u64 << 52);
            assert_eq!(dt.boundaries[dt.n + 2], 1u64 << 52);
        }
    }

    #[test]
    fn test_powers_of_two() {
        for index_bits in [1, 2, 4, 6, 8, 10] {
            let tables = DynatraceTables::generate(index_bits);
            let scale = index_bits as i32;

            for exp in -10..=10 {
                let value = 2.0_f64.powi(exp);
                let expected = (exp << scale) - 1;
                let actual = map_to_index_dynatrace(value, &tables);
                assert_eq!(
                    actual, expected,
                    "scale={}, 2^{}: expected {}, got {}",
                    index_bits, exp, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_vs_exact_comprehensive() {
        // Verify the Dynatrace lookup matches the exact bignum computation
        // for many values across several scales.
        // Note: scale 8+ takes too long for routine testing with bignum.
        for index_bits in [1, 2, 4, 6, 8] {
            let tables = DynatraceTables::generate(index_bits);
            let scale = index_bits as i32;
            let n = tables.n;

            let mut total = 0u64;
            let mut errors = 0u64;

            // Test values near every boundary in octaves 2^-5 .. 2^5
            for exp in -5..=5 {
                for bucket in 0..n {
                    let boundary_exp =
                        (exp as f64) + (bucket as f64) / (n as f64);
                    let boundary_value = 2.0_f64.powf(boundary_exp);

                    // Test values just above and below each boundary
                    let mut v = boundary_value;
                    for _ in 0..10 {
                        v = f64::from_bits(v.to_bits() + 1);
                        total += 1;
                        let dt_idx = map_to_index_dynatrace(v, &tables);
                        let exact_idx = map_to_index_exact(v, scale);
                        if dt_idx != exact_idx {
                            errors += 1;
                        }
                    }

                    let mut v = boundary_value;
                    for _ in 0..10 {
                        v = f64::from_bits(v.to_bits() - 1);
                        total += 1;
                        let dt_idx = map_to_index_dynatrace(v, &tables);
                        let exact_idx = map_to_index_exact(v, scale);
                        if dt_idx != exact_idx {
                            errors += 1;
                        }
                    }
                }
            }

            assert_eq!(
                errors, 0,
                "Dynatrace lookup had {} errors out of {} tests at scale {}",
                errors, total, index_bits
            );
        }
    }

    #[test]
    fn test_vs_newrelic() {
        // Verify Dynatrace and NewRelic give identical results.
        // Both now use upper-inclusive boundaries baked into the table.
        use crate::newrelic_table::LookupTables;

        for index_bits in [4, 6, 8, 10] {
            let dt = DynatraceTables::generate(index_bits);
            let nr = LookupTables::generate(index_bits);
            let scale = index_bits as i32;

            let test_values: &[f64] = &[
                1e-300, 1e-100, 1e-10, 0.001, 0.1, 0.5,
                1.0, 1.0000000000001, 1.1, 1.25, 1.41421356, 1.5, 1.9, 1.9999999999999,
                2.0, 3.0, 4.0, 10.0, 100.0,
                1e10, 1e100, 1e300,
            ];

            for &v in test_values {
                let dt_idx = map_to_index_dynatrace(v, &dt);

                // NewRelic mapping (upper-inclusive baked into tables, no sig==0 branch)
                let significand = get_significand(v);
                let exponent = get_normal_base2(v);
                let linear_idx = (significand >> nr.significand_shift) as usize;
                let approx = nr.log_bucket_index[linear_idx] as usize;
                let bucket = if significand >= nr.log_bucket_end[approx] {
                    approx + 1
                } else {
                    approx
                } as i32;
                let nr_idx = (exponent << scale) + bucket - 1;

                assert_eq!(
                    dt_idx, nr_idx,
                    "dt vs nr mismatch at scale={}, value={}: dt={}, nr={}",
                    index_bits, v, dt_idx, nr_idx
                );
            }
        }
    }

    #[test]
    fn test_index_table_half_size_of_newrelic() {
        // The whole point: Dynatrace uses N index entries, NewRelic uses 2N
        for index_bits in [4, 6, 8, 10] {
            let dt = DynatraceTables::generate(index_bits);
            let nr = crate::newrelic_table::LookupTables::generate(index_bits);
            assert_eq!(dt.indices.len() * 2, nr.log_bucket_index.len(),
                "scale {}", index_bits);
        }
    }
}
