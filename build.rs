// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate lookup tables for exponential histogram mapping.

use expohisto_mapping_gen::LookupTables;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lookup_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    // Determine which table size to generate based on features
    let scale: Option<u32> = if cfg!(feature = "lookup-1024") {
        Some(10)
    } else if cfg!(feature = "lookup-256") {
        Some(8)
    } else if cfg!(feature = "lookup-64") {
        Some(6)
    } else {
        None
    };

    if let Some(scale) = scale {
        let tables = LookupTables::generate(scale);
        tables.write_rust_source(&mut file).unwrap();
    } else {
        // Generate empty module
        writeln!(file, "// No lookup table feature enabled").unwrap();
        writeln!(file, "pub const LOOKUP_SCALE: i32 = 0;").unwrap();
    }

    // Tell cargo to rerun if features change
    println!("cargo:rerun-if-changed=build.rs");
}
