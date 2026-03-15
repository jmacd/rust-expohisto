// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Lookup table generation for exponential histogram mapping.
//!
//! This crate provides utilities for generating exact lookup tables
//! used to map f64 values to histogram bucket indices efficiently.

// Re-use the canonical float64 definitions from the main crate to avoid
// maintaining a duplicate copy. The `pub` visibility on items there is
// restricted to `pub(crate)` in the main crate via its lib.rs.
#[path = "../../src/float64.rs"]
mod float64;
mod dynatrace_table;
mod newrelic_table;

pub use dynatrace_table::*;
pub use float64::*;
pub use newrelic_table::*;

use std::io::Write;

/// Generates a shared `lookup_tables.rs` source file containing `TABLE_SCALE`
/// and the `BOUNDARIES` array for the given scale.
///
/// The output is intended for `include!()` in the main crate's `lookup` module.
/// The `BOUNDARIES` layout is:
/// `[sentinel=0, b[0]=1, b[1], …, b[N−1], sentinel=2^52, sentinel=2^52]`
/// where N = 2^table_scale.
pub fn generate_shared_boundaries<W: Write>(w: &mut W, table_scale: u32) -> std::io::Result<()> {
    let tables = LookupTables::generate(table_scale);
    let n = tables.n;

    // log_bucket_end already has the upper-inclusive adjustment (b[0]=1).
    // Take [..n] to exclude the trailing sentinel that LookupTables appends.
    let adjusted_boundaries = &tables.log_bucket_end[..n];

    let mut shared = Vec::with_capacity(n + 3);
    shared.push(0u64); // leading sentinel
    shared.extend_from_slice(adjusted_boundaries);
    shared.push(1u64 << 52); // trailing sentinel
    shared.push(1u64 << 52); // trailing sentinel

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

    writeln!(
        w,
        "/// Boundary significands for exponential histogram mapping."
    )?;
    writeln!(
        w,
        "/// Layout: \\[sentinel=0, b\\[0\\]=1, b\\[1\\], ..., b\\[N-1\\], sentinel=2^52, sentinel=2^52\\]"
    )?;
    writeln!(w, "/// where N = 2^TABLE_SCALE = {}.", n)?;
    writeln!(w, "pub static BOUNDARIES: [u64; {}] = [", n + 3)?;
    for (i, &b) in shared.iter().enumerate() {
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
