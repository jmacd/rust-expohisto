// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate lookup tables for exponential histogram mapping.
//!
//! Computes exact boundary significands once (via bignum arithmetic in
//! `mapping-gen`), then writes `BOUNDARIES`, `TABLE_SCALE`, and per-algorithm
//! index tables (`NR_INDEX`/`DT_INDEX`) to `lookup_tables.rs`.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lookup_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    // Declare our custom cfgs so rustc doesn't warn about them.
    println!("cargo:rustc-check-cfg=cfg(has_lookup_table)");

    if let Some(scale) = table_scale() {
        println!("cargo:rustc-cfg=has_lookup_table");
        println!("cargo:rustc-env=EXPECTED_TABLE_SCALE={scale}");

        // Compute exact boundaries once (expensive bignum arithmetic).
        let boundaries = expohisto_mapping_gen::generate_boundaries(scale);

        // Write the shared BOUNDARIES array and TABLE_SCALE constant.
        expohisto_mapping_gen::write_boundaries(&mut file, scale, &boundaries).unwrap();

        // Derive and write algorithm-specific index tables from the same
        // boundaries. Each algorithm compiles in exactly one index table;
        // the mapping function always computes at TABLE_SCALE and
        // right-shifts to the requested scale.
        if env::var("CARGO_FEATURE_NEWRELIC").is_ok() {
            expohisto_mapping_gen::write_index_table(&mut file, scale, &boundaries, 1, "NR")
                .unwrap();
        }
        if env::var("CARGO_FEATURE_DYNATRACE").is_ok() {
            expohisto_mapping_gen::write_index_table(&mut file, scale, &boundaries, 0, "DT")
                .unwrap();
        }
    } else {
        writeln!(file, "// No table features enabled").unwrap();
        writeln!(file, "pub const TABLE_SCALE: i32 = 0;").unwrap();
    }

    println!("cargo:rerun-if-changed=build.rs");
}

/// Returns the highest enabled scale feature, or None.
fn table_scale() -> Option<u32> {
    // Check from highest to lowest; features are additive so the highest wins.
    // Cargo sets CARGO_FEATURE_SCALE_<N> for each enabled `scale-<N>` feature.
    (1..=20)
        .rev()
        .find(|&s| env::var(format!("CARGO_FEATURE_SCALE_{s}")).is_ok())
}
