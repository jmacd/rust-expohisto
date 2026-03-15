// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Dynatrace lookup table-based mapping for exponential histograms.
//!
//! Uses N linear buckets per scale with two branch corrections.
//! ~50% smaller index table than NewRelic at the cost of one extra comparison.
//! See [`crate::lookup::ScaleTables`] for shared infrastructure.

crate::lookup::define_lookup_module!(extra_bits = 0, corrections = 2);

#[cfg(test)]
mod tests {
    use super::*;

    crate::lookup::lookup_tests!(map_to_index, TABLE_SCALE);

    #[test]
    fn test_matches_newrelic() {
        #[cfg(feature = "newrelic")]
        {
            let test_values: &[f64] = &[
                1e-300, 1e-100, 1e-10, 0.001, 0.1, 0.5, 1.0, 1.5, 2.0,
                core::f64::consts::PI, 10.0, 100.0, 1e10, 1e100, 1e300,
                1.0000000000001, 1.9999999999999, 0.9999999999999,
            ];

            for scale in 1..=TABLE_SCALE {
                for &v in test_values {
                    let dt_idx = map_to_index(v, scale);
                    let nr_idx = crate::newrelic::map_to_index(v, scale);
                    assert_eq!(
                        dt_idx, nr_idx,
                        "dynatrace vs newrelic mismatch at scale={scale}, value={v}: dt={dt_idx}, nr={nr_idx}",
                    );
                }
            }
        }
    }
}
