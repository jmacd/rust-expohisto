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
    (1..=20).rev().find(|&s| has_scale_feature(s))
}

#[allow(clippy::match_like_matches_macro)] // each arm evaluates a distinct cfg!()
fn has_scale_feature(s: u32) -> bool {
    match s {
        1 => cfg!(feature = "scale-1"),
        2 => cfg!(feature = "scale-2"),
        3 => cfg!(feature = "scale-3"),
        4 => cfg!(feature = "scale-4"),
        5 => cfg!(feature = "scale-5"),
        6 => cfg!(feature = "scale-6"),
        7 => cfg!(feature = "scale-7"),
        8 => cfg!(feature = "scale-8"),
        9 => cfg!(feature = "scale-9"),
        10 => cfg!(feature = "scale-10"),
        11 => cfg!(feature = "scale-11"),
        12 => cfg!(feature = "scale-12"),
        13 => cfg!(feature = "scale-13"),
        14 => cfg!(feature = "scale-14"),
        15 => cfg!(feature = "scale-15"),
        16 => cfg!(feature = "scale-16"),
        17 => cfg!(feature = "scale-17"),
        18 => cfg!(feature = "scale-18"),
        19 => cfg!(feature = "scale-19"),
        20 => cfg!(feature = "scale-20"),
        _ => false,
    }
}
