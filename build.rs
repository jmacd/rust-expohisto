// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate a single shared lookup table for exponential histogram mapping.
//!
//! Generates one boundary table at the highest compiled-in scale, shared by
//! both NewRelic and Dynatrace algorithms. Lower scales are derived at runtime
//! by right-shifting the result: `map_at_S(v) = map_at_H(v) >> (H - S)`.

use expohisto_mapping_gen::LookupTables;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lookup_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    if let Some(scale) = table_scale() {
        generate_tables(&mut file, scale).unwrap();
    } else {
        writeln!(file, "// No table features enabled").unwrap();
        writeln!(file, "pub const TABLE_SCALE: i32 = 0;").unwrap();
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn table_scale() -> Option<u32> {
    if cfg!(feature = "newrelic-14") || cfg!(feature = "dynatrace-14") {
        Some(14)
    } else if cfg!(feature = "newrelic-12") || cfg!(feature = "dynatrace-12") {
        Some(12)
    } else if cfg!(feature = "newrelic-10") || cfg!(feature = "dynatrace-10") {
        Some(10)
    } else if cfg!(feature = "newrelic-8") || cfg!(feature = "dynatrace-8") {
        Some(8)
    } else if cfg!(feature = "newrelic-6") || cfg!(feature = "dynatrace-6") {
        Some(6)
    } else if cfg!(feature = "newrelic-4") || cfg!(feature = "dynatrace-4") {
        Some(4)
    } else {
        None
    }
}

fn generate_tables<W: Write>(w: &mut W, table_scale: u32) -> std::io::Result<()> {
    let tables = LookupTables::generate(table_scale);
    let n = tables.n;

    writeln!(
        w,
        "// Auto-generated lookup tables at scale {} ({} log buckets)",
        table_scale, n
    )?;
    writeln!(w)?;

    writeln!(
        w,
        "/// Maximum histogram scale supported by this lookup table."
    )?;
    writeln!(w, "pub const TABLE_SCALE: i32 = {};", table_scale)?;
    writeln!(w)?;

    let raw_boundaries = &tables.log_bucket_end[..n];

    let mut shared_boundaries = Vec::with_capacity(n + 3);
    shared_boundaries.push(0u64);
    shared_boundaries.extend_from_slice(raw_boundaries);
    shared_boundaries.push(1u64 << 52);
    shared_boundaries.push(1u64 << 52);

    writeln!(
        w,
        "/// Boundary significands for exponential histogram mapping."
    )?;
    writeln!(
        w,
        "/// Layout: [sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]"
    )?;
    writeln!(w, "/// where N = 2^TABLE_SCALE = {}.", n)?;
    writeln!(w, "pub static BOUNDARIES: [u64; {}] = [", n + 3)?;
    for (i, &b) in shared_boundaries.iter().enumerate() {
        if i % 4 == 0 {
            write!(w, "    ")?;
        }
        if i == 0 {
            writeln!(w, "0x{:013X}, // sentinel", b)?;
        } else if i > n {
            writeln!(w, "0x{:013X}, // sentinel = 2^52", b)?;
        } else {
            write!(w, "0x{:013X},", b)?;
            if i % 4 == 3 {
                writeln!(w)?;
            }
        }
    }
    writeln!(w, "];")?;

    Ok(())
}
