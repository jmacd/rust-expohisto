// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Build script to generate lookup tables for exponential histogram mapping.

use expohisto_mapping_gen::{compute_dynatrace_indices, compute_linear_to_log_mapping, LookupTables};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // Generate NewRelic lookup tables if any newrelic feature is enabled
    generate_newrelic_tables(&out_dir);

    // Generate Dynatrace lookup tables if any dynatrace feature is enabled
    generate_dynatrace_tables(&out_dir);

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

/// Derive coarser-scale boundaries from the finest table.
/// For scale S < TABLE_SCALE, boundary[k] at scale S = boundary[k * 2^(TABLE_SCALE-S)] at TABLE_SCALE.
fn derive_boundaries(fine_boundaries: &[u64], table_scale: u32, target_scale: u32) -> Vec<u64> {
    let step = 1usize << (table_scale - target_scale);
    let n = 1usize << target_scale;
    let mut boundaries = Vec::with_capacity(n);
    for k in 0..n {
        boundaries.push(fine_boundaries[k * step]);
    }
    boundaries
}

fn generate_dynatrace_tables(out_dir: &str) {
    let dest_path = Path::new(out_dir).join("dynatrace_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    // Determine which table size to generate based on features
    let scale: Option<u32> = if cfg!(feature = "dynatrace-14") {
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
    };

    if let Some(scale) = scale {
        let tables = LookupTables::generate(scale);
        write_dynatrace_source(&mut file, &tables).unwrap();
    } else {
        // Generate stub
        writeln!(file, "// No dynatrace feature enabled").unwrap();
        writeln!(file, "pub const TABLE_SCALE: i32 = 0;").unwrap();
    }
}

fn write_dynatrace_source<W: std::io::Write>(w: &mut W, tables: &LookupTables) -> std::io::Result<()> {
    let table_scale = tables.index_bits;

    writeln!(
        w,
        "// Auto-generated Dynatrace lookup tables with {} index bits ({} buckets)",
        table_scale, tables.n
    )?;
    writeln!(w, "// Per-scale tables for scales 1..={}", table_scale)?;
    writeln!(w)?;

    writeln!(w, "/// Maximum histogram scale supported by this lookup table.")?;
    writeln!(w, "pub const TABLE_SCALE: i32 = {};", table_scale)?;
    writeln!(w)?;

    // Extract the fine-resolution boundaries (without sentinel) from log_bucket_end
    let fine_boundaries: Vec<u64> = tables.log_bucket_end[..tables.n].to_vec();

    // Generate per-scale tables
    for s in 1..=table_scale {
        let n_s = 1usize << s;
        let sig_shift = 52 - s;

        // Derive boundaries for this scale
        let mut boundaries = if s == table_scale {
            fine_boundaries.clone()
        } else {
            derive_boundaries(&fine_boundaries, table_scale, s)
        };

        // Add two sentinels (2^52) for safe two-branch correction access
        boundaries.push(1u64 << 52);
        boundaries.push(1u64 << 52);

        // Compute Dynatrace-style indices (N entries, not 2N)
        // boundaries must include sentinels for this to work
        let indices = compute_dynatrace_indices(n_s, &boundaries, s);

        // Emit INDICES for this scale (N entries)
        writeln!(w, "static DT_INDICES_{}: [u16; {}] = [", s, n_s)?;
        for (i, &idx) in indices.iter().enumerate() {
            if i % 16 == 0 {
                write!(w, "    ")?;
            }
            write!(w, "{:4},", idx)?;
            if i % 16 == 15 || i == indices.len() - 1 {
                writeln!(w)?;
            }
        }
        writeln!(w, "];")?;
        writeln!(w)?;

        // Emit BOUNDARIES for this scale (N + 2 entries: N boundaries + 2 sentinels)
        // boundaries already has sentinels appended
        writeln!(w, "static DT_BOUNDARIES_{}: [u64; {}] = [", s, n_s + 2)?;
        for (i, &boundary) in boundaries.iter().enumerate() {
            if i % 4 == 0 {
                write!(w, "    ")?;
            }
            if i >= n_s {
                writeln!(w, "0x{:013X}, // sentinel", boundary)?;
            } else {
                write!(w, "0x{:013X},", boundary)?;
                if i % 4 == 3 {
                    writeln!(w)?;
                }
            }
        }
        if boundaries.len() % 4 != 0 && boundaries.last().map(|_| boundaries.len() - 1 < n_s).unwrap_or(false) {
            writeln!(w)?;
        }
        writeln!(w, "];")?;
        writeln!(w)?;

        writeln!(w, "// Scale {}: {} log buckets, {} linear buckets, significand_shift={}",
            s, n_s, n_s, sig_shift)?;
        writeln!(w)?;
    }

    // Emit the SCALE_MAPPINGS array
    writeln!(w, "/// Per-scale lookup table mappings, indexed by (scale - 1).")?;
    writeln!(w, "pub static SCALE_MAPPINGS: [DynatraceScaleMapping; TABLE_SCALE as usize] = [")?;
    for s in 1..=table_scale {
        let sig_shift = 52 - s;
        writeln!(w, "    DynatraceScaleMapping {{")?;
        writeln!(w, "        significand_shift: {},", sig_shift)?;
        writeln!(w, "        indices: &DT_INDICES_{},", s)?;
        writeln!(w, "        boundaries: &DT_BOUNDARIES_{},", s)?;
        writeln!(w, "    }},")?;
    }
    writeln!(w, "];")?;

    Ok(())
}

fn write_newrelic_source<W: std::io::Write>(w: &mut W, tables: &LookupTables) -> std::io::Result<()> {
    let table_scale = tables.index_bits;

    writeln!(
        w,
        "// Auto-generated NewRelic lookup tables with {} index bits ({} buckets)",
        table_scale, tables.n
    )?;
    writeln!(w, "// Per-scale tables for scales 1..={}", table_scale)?;
    writeln!(w)?;

    writeln!(w, "/// Maximum histogram scale supported by this lookup table.")?;
    writeln!(w, "pub const TABLE_SCALE: i32 = {};", table_scale)?;
    writeln!(w)?;

    // Extract the fine-resolution boundaries (without sentinel) from log_bucket_end
    let fine_boundaries: Vec<u64> = tables.log_bucket_end[..tables.n].to_vec();

    // Generate per-scale tables
    for s in 1..=table_scale {
        let n_s = 1usize << s;
        let sig_shift = 52 - (s + 1);

        // Derive boundaries for this scale
        let boundaries = if s == table_scale {
            fine_boundaries.clone()
        } else {
            derive_boundaries(&fine_boundaries, table_scale, s)
        };

        // Compute linear-to-log mapping for this scale's resolution
        let log_bucket_index = compute_linear_to_log_mapping(n_s, &boundaries);

        // Emit LOG_BUCKET_INDEX for this scale
        writeln!(w, "static LOG_BUCKET_INDEX_{}: [u16; {}] = [", s, 2 * n_s)?;
        for (i, &idx) in log_bucket_index.iter().enumerate() {
            if i % 16 == 0 {
                write!(w, "    ")?;
            }
            write!(w, "{:4},", idx)?;
            if i % 16 == 15 || i == log_bucket_index.len() - 1 {
                writeln!(w)?;
            }
        }
        writeln!(w, "];")?;
        writeln!(w)?;

        // Emit LOG_BUCKET_END for this scale (boundaries + sentinel)
        writeln!(w, "static LOG_BUCKET_END_{}: [u64; {}] = [", s, n_s + 1)?;
        for (i, &boundary) in boundaries.iter().enumerate() {
            if i % 4 == 0 {
                write!(w, "    ")?;
            }
            write!(w, "0x{:013X},", boundary)?;
            if i % 4 == 3 || i == boundaries.len() - 1 {
                writeln!(w)?;
            }
        }
        // Sentinel
        writeln!(w, "    0x{:013X}, // sentinel = 2^52", 1u64 << 52)?;
        writeln!(w, "];")?;
        writeln!(w)?;

        writeln!(w, "// Scale {}: {} log buckets, {} linear buckets, significand_shift={}",
            s, n_s, 2 * n_s, sig_shift)?;
        writeln!(w)?;
    }

    // Emit the SCALE_MAPPINGS array
    writeln!(w, "/// Per-scale lookup table mappings, indexed by (scale - 1).")?;
    writeln!(w, "pub static SCALE_MAPPINGS: [NewrelicScaleMapping; TABLE_SCALE as usize] = [")?;
    for s in 1..=table_scale {
        let sig_shift = 52 - (s + 1);
        writeln!(w, "    NewrelicScaleMapping {{")?;
        writeln!(w, "        significand_shift: {},", sig_shift)?;
        writeln!(w, "        log_bucket_index: &LOG_BUCKET_INDEX_{},", s)?;
        writeln!(w, "        log_bucket_end: &LOG_BUCKET_END_{},", s)?;
        writeln!(w, "    }},")?;
    }
    writeln!(w, "];")?;

    Ok(())
}
