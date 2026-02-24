// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate a single shared lookup table for exponential histogram mapping.
//!
//! Generates one boundary table at the highest compiled-in scale, shared by
//! both NewRelic and Dynatrace algorithms. Lower scales are derived at runtime
//! by right-shifting the result: `map_at_S(v) = map_at_H(v) >> (H - S)`.

use expohisto_mapping_gen::{compute_dynatrace_indices, LookupTables};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let nr_scale = newrelic_scale();
    let dt_scale = dynatrace_scale();

    let table_scale = match (nr_scale, dt_scale) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let dest_path = Path::new(&out_dir).join("lookup_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    if let Some(scale) = table_scale {
        generate_tables(&mut file, scale, nr_scale.is_some(), dt_scale.is_some()).unwrap();
    } else {
        writeln!(file, "// No table features enabled").unwrap();
        writeln!(file, "pub const TABLE_SCALE: i32 = 0;").unwrap();
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn newrelic_scale() -> Option<u32> {
    if cfg!(feature = "newrelic-14") {
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
    }
}

fn dynatrace_scale() -> Option<u32> {
    if cfg!(feature = "dynatrace-14") {
        Some(14)
    } else if cfg!(feature = "dynatrace-12") {
        Some(12)
    } else if cfg!(feature = "dynatrace-10") {
        Some(10)
    } else if cfg!(feature = "dynatrace-8") {
        Some(8)
    } else if cfg!(feature = "dynatrace-6") {
        Some(6)
    } else if cfg!(feature = "dynatrace-4") {
        Some(4)
    } else {
        None
    }
}

fn generate_tables<W: Write>(
    w: &mut W,
    table_scale: u32,
    emit_nr: bool,
    emit_dt: bool,
) -> std::io::Result<()> {
    let tables = LookupTables::generate(table_scale);
    let n = tables.n; // 2^table_scale

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

    // Extract raw boundaries: [1, b[1], ..., b[N-1]] (N entries)
    let raw_boundaries = &tables.log_bucket_end[..n];

    // Build shared boundary array in DT layout:
    // [sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]
    let mut shared_boundaries = Vec::with_capacity(n + 3);
    shared_boundaries.push(0u64);
    shared_boundaries.extend_from_slice(raw_boundaries);
    shared_boundaries.push(1u64 << 52);
    shared_boundaries.push(1u64 << 52);

    // Emit shared boundaries
    writeln!(
        w,
        "/// Shared boundary significands for exponential histogram mapping."
    )?;
    writeln!(
        w,
        "/// Layout: [sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]"
    )?;
    writeln!(w, "/// where N = 2^TABLE_SCALE = {}.", n)?;
    writeln!(
        w,
        "/// Both NewRelic and Dynatrace algorithms reference this same array."
    )?;
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
    writeln!(w)?;

    // NewRelic index table
    if emit_nr {
        let nr_shift = 52 - (table_scale + 1);
        writeln!(
            w,
            "/// NewRelic significand shift: 52 - (TABLE_SCALE + 1) = {}",
            nr_shift
        )?;
        writeln!(w, "pub const NR_SIGNIFICAND_SHIFT: u32 = {};", nr_shift)?;
        writeln!(w)?;

        // NR index table is already computed by LookupTables::generate
        writeln!(
            w,
            "/// NewRelic linear-to-log index table (2N = {} entries).",
            2 * n
        )?;
        writeln!(w, "pub static NR_INDEX: [u16; {}] = [", 2 * n)?;
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
    }

    // Dynatrace index table
    if emit_dt {
        let dt_shift = 52 - table_scale;
        writeln!(
            w,
            "/// Dynatrace significand shift: 52 - TABLE_SCALE = {}",
            dt_shift
        )?;
        writeln!(w, "pub const DT_SIGNIFICAND_SHIFT: u32 = {};", dt_shift)?;
        writeln!(w)?;

        let dt_index = compute_dynatrace_indices(n, &shared_boundaries, table_scale);
        writeln!(
            w,
            "/// Dynatrace linear-to-log index table (N = {} entries).",
            n
        )?;
        writeln!(w, "pub static DT_INDEX: [u16; {}] = [", n)?;
        for (i, &idx) in dt_index.iter().enumerate() {
            if i % 16 == 0 {
                write!(w, "    ")?;
            }
            write!(w, "{:4},", idx)?;
            if i % 16 == 15 || i == dt_index.len() - 1 {
                writeln!(w)?;
            }
        }
        writeln!(w, "];")?;
        writeln!(w)?;
    }

    Ok(())
}
