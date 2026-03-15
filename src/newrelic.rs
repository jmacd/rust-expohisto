// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! NewRelic lookup table-based mapping for exponential histograms.
//!
//! Uses 2N linear buckets per scale with one branch correction.
//! See [`crate::lookup`] for shared infrastructure.

crate::lookup::define_lookup_module!(index_table = NR_INDEX, shift = NR_SHIFT, corrections = 1);

#[cfg(test)]
mod tests {
    use super::*;

    crate::lookup::lookup_tests!(map_to_index, TABLE_SCALE);
}
