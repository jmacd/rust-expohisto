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

    // Generate NewRelic lookup tables if any newrelic feature is enabled
    generate_newrelic_tables(&out_dir);

    // Tell cargo to rerun if features change
    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_newrelic_tables(out_dir: &str) {
    let dest_path = Path::new(out_dir).join("newrelic_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    // Determine which table size to generate based on features
    let scale: Option<u32> = if cfg!(feature = "newrelic-14") {
        Some(14)
    } else if cfg!(feature = "newrelic-12") {
        Some(12)
    } else if cfg!(feature = "newrelic-10") {
        Some(10)
    } else if cfg!(feature = "newrelic-8") {
        Some(8)
    } else if cfg!(feature = "newrelic-6") {
        Some(6)
    } else if cfg!(feature = "newrelic-4") {
        Some(4)
    } else {
        None
    };

    if let Some(scale) = scale {
        let tables = LookupTables::generate(scale);
        write_newrelic_source(&mut file, &tables).unwrap();
    } else {
        // Generate stub - this file won't be included if no newrelic feature is enabled
        writeln!(file, "// No newrelic feature enabled").unwrap();
        writeln!(file, "pub const TABLE_SCALE: i32 = 0;").unwrap();
    }
}

fn write_newrelic_source<W: std::io::Write>(w: &mut W, tables: &LookupTables) -> std::io::Result<()> {
    writeln!(
        w,
        "// Auto-generated NewRelic lookup tables with {} index bits ({} buckets)",
        tables.index_bits,
        tables.n
    )?;
    writeln!(w)?;

    writeln!(w, "use crate::float64::SIGNIFICAND_WIDTH;")?;
    writeln!(w)?;

    writeln!(w, "/// Maximum histogram scale supported by this lookup table.")?;
    writeln!(w, "pub const TABLE_SCALE: i32 = {};", tables.index_bits)?;
    writeln!(w)?;

    writeln!(w, "/// Number of bits to index into 2*N linear buckets.")?;
    writeln!(
        w,
        "const LINEAR_BUCKET_BITS: u32 = TABLE_SCALE as u32 + 1;"
    )?;
    writeln!(w)?;

    writeln!(
        w,
        "/// Shift to convert 52-bit significand to linear bucket index."
    )?;
    writeln!(
        w,
        "/// significand >> SIGNIFICAND_SHIFT yields an index in 0..2*N."
    )?;
    writeln!(
        w,
        "pub const SIGNIFICAND_SHIFT: u32 = SIGNIFICAND_WIDTH - LINEAR_BUCKET_BITS;"
    )?;
    writeln!(w)?;

    writeln!(
        w,
        "/// Maps linear bucket index to approximate log bucket index."
    )?;
    writeln!(
        w,
        "/// Linear bucket i starts at significand (i * 2^52) / (2 * N)."
    )?;
    writeln!(
        w,
        "pub const LOG_BUCKET_INDEX: [u16; 1 << LINEAR_BUCKET_BITS] = ["
    )?;
    for (i, &idx) in tables.log_bucket_index.iter().enumerate() {
        if i % 16 == 0 {
            write!(w, "    ")?;
        }
        write!(w, "{:4},", idx)?;
        if i % 16 == 15 || i == tables.log_bucket_index.len() - 1 {
            writeln!(w)?;
        }
    }
    writeln!(w, "];")?;
    writeln!(w)?;

    writeln!(w, "/// End significand (52-bit) for each log bucket.")?;
    writeln!(
        w,
        "/// Bucket i contains values with significand in [boundary[i], boundary[i+1])."
    )?;
    writeln!(
        w,
        "/// Last entry is a sentinel (2^52) for boundary checks."
    )?;
    writeln!(
        w,
        "pub const LOG_BUCKET_END: [u64; (1 << TABLE_SCALE) + 1] = ["
    )?;
    for (i, &boundary) in tables.log_bucket_end.iter().enumerate() {
        if i % 4 == 0 {
            write!(w, "    ")?;
        }
        if i == tables.n {
            writeln!(w, "0x{:013X}, // sentinel = 2^52", boundary)?;
        } else {
            write!(w, "0x{:013X},", boundary)?;
            if i % 4 == 3 {
                writeln!(w)?;
            }
        }
    }
    writeln!(w, "];")?;

    Ok(())
}
