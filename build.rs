// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate a single shared lookup table for exponential histogram mapping.
//!
//! Generates one boundary table at the highest compiled-in scale, shared by
//! both NewRelic and Dynatrace algorithms. Lower scales are derived at runtime
//! by right-shifting the result: `map_at_S(v) = map_at_H(v) >> (H - S)`.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lookup_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    // Declare our custom cfg so rustc doesn't warn about it.
    println!("cargo:rustc-check-cfg=cfg(has_lookup_table)");

    if let Some(scale) = table_scale() {
        println!("cargo:rustc-cfg=has_lookup_table");
        println!("cargo:rustc-env=EXPECTED_TABLE_SCALE={scale}");
        expohisto_mapping_gen::generate_shared_boundaries(&mut file, scale).unwrap();
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
