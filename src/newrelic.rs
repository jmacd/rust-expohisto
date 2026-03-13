// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! NewRelic lookup table-based mapping for exponential histograms.
//!
//! Uses 2N linear buckets per scale with one branch correction.
//! See [`crate::lookup::ScaleTables`] for shared infrastructure.

use std::sync::OnceLock;

use crate::lookup::{ScaleTables, TABLE_SCALE, table_map_to_index};

static NR_TABLES: OnceLock<ScaleTables> = OnceLock::new();

fn tables() -> &'static ScaleTables {
    NR_TABLES.get_or_init(|| ScaleTables::new(1)) // extra_bits=1 → 2N buckets
}

/// Maps a positive f64 value to a bucket index.
///
/// Uses per-scale boundaries and 2N linear buckets with one branch correction.
#[inline]
pub fn map_to_index(value: f64, scale: i32) -> i32 {
    table_map_to_index(value, scale, tables(), 1) // 1 correction
}

/// Returns the native scale (resolution) of the lookup table.
#[inline]
pub const fn table_scale() -> i32 {
    TABLE_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::lookup::lookup_tests!(map_to_index, TABLE_SCALE);
}
