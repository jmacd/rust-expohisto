// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared lookup tables for exponential histogram mapping.
//!
//! This module provides the compile-time generated boundary significand
//! table plus a shared index table derivation function. Both NewRelic
//! and Dynatrace algorithms reference the same BOUNDARIES array and
//! derive their linear index tables using the same function — they
//! differ only in table size (2N vs N) and correction count (1 vs 2).
//!
//! Lower scales are derived by right-shifting the fine-scale result:
//! `map_at_S(v) = map_at_H(v) >> (H - S)`.

include!(concat!(env!("OUT_DIR"), "/lookup_tables.rs"));

/// Derives a linear-to-log index table from BOUNDARIES.
///
/// For `count` equidistant linear buckets (each of width `1 << shift`
/// in significand space), stores the approximate log bucket containing
/// each linear bucket's lower bound.
///
/// Both NR and DT call this with the same logic — NR uses
/// `count = 2N, shift = 52 - TABLE_SCALE - 1` while DT uses
/// `count = N, shift = 52 - TABLE_SCALE`.
pub fn derive_index_table(count: usize, shift: u32) -> Vec<u16> {
    let mut table = vec![0u16; count];
    let mut j: u16 = 0;
    for i in 0..count {
        let lower_bound = (i as u64) << shift;
        while lower_bound >= BOUNDARIES[j as usize + 1] {
            j += 1;
        }
        table[i] = j;
    }
    table
}
