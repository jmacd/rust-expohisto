// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate lookup tables for exponential histogram mapping.
//!
//! Computes exact boundary significands once (via bignum arithmetic in
//! `mapping-gen`), then writes `BOUNDARIES`, `TABLE_SCALE`, and the
//! index table to `lookup_tables.rs`.

use std::env;
use std::fs::File;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lookup_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    let scale = table_scale();
    println!("cargo:rustc-env=EXPECTED_TABLE_SCALE={scale}");

    // Compute exact boundaries once (expensive bignum arithmetic).
    let boundaries = expohisto_mapping_gen::generate_boundaries(scale);

    // Write the shared BOUNDARIES array and TABLE_SCALE constant.
    expohisto_mapping_gen::write_boundaries(&mut file, scale, &boundaries).unwrap();

    // Derive and write the index table from the same boundaries.
    // The mapping function always computes at TABLE_SCALE and
    // right-shifts to the requested scale.
    expohisto_mapping_gen::write_index_table(&mut file, scale, &boundaries)
        .unwrap();

    // Generate inverse factor table for boundary computation.
    let inv_path = Path::new(&out_dir).join("inverse_factors.rs");
    let mut inv_file = File::create(&inv_path).unwrap();
    let factors = expohisto_mapping_gen::generate_inverse_factors();
    expohisto_mapping_gen::write_inverse_factors(&mut inv_file, &factors).unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}

/// Returns the highest enabled scale feature, defaulting to 8.
fn table_scale() -> u32 {
    // Check from highest to lowest; features are additive so the highest wins.
    // Cargo sets CARGO_FEATURE_SCALE_<N> for each enabled `scale-<N>` feature.
    (1..=20)
        .rev()
        .find(|&s| env::var(format!("CARGO_FEATURE_SCALE_{s}")).is_ok())
        .unwrap_or(8)
}
