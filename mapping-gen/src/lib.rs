// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Lookup table generation for exponential histogram mapping.
//!
//! This crate computes exact bucket boundary significands and derives
//! linear-to-log index tables for the NewRelic and Dynatrace algorithms.
//!
//! Build scripts call [`generate_boundaries`] once, then pass the result
//! to [`write_boundaries`] and [`write_index_table`].

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

/// Computes the sentinel-wrapped boundary array for a given scale.
///
/// Returns `N + 3` entries with layout:
/// `[sentinel=0, b[0]=1, b[1], …, b[N−1], sentinel=2^52, sentinel=2^52]`
/// where N = 2^table_scale. Boundaries are exact 52-bit significands
/// computed via bignum arithmetic.
pub fn generate_boundaries(table_scale: u32) -> Vec<u64> {
    let n = 1usize << table_scale;
    let mut raw = compute_boundaries_exact(n, table_scale);

    // Upper-inclusive adjustment: boundary[0] = 1 instead of 0, so
    // significand == 0 (exact powers of two) falls below, matching
    // OTel's upper-inclusive bucket semantics.
    debug_assert_eq!(raw[0], 0);
    raw[0] = 1;

    let mut boundaries = Vec::with_capacity(n + 3);
    boundaries.push(0u64); // leading sentinel
    boundaries.extend_from_slice(&raw);
    boundaries.push(1u64 << 52); // trailing sentinel
    boundaries.push(1u64 << 52); // trailing sentinel
    boundaries
}

/// Writes `TABLE_SCALE` and the `BOUNDARIES` array as Rust source.
///
/// `boundaries` must be the sentinel-wrapped array returned by
/// [`generate_boundaries`].
pub fn write_boundaries<W: Write>(
    w: &mut W,
    table_scale: u32,
    boundaries: &[u64],
) -> std::io::Result<()> {
    let n = 1usize << table_scale;
    debug_assert_eq!(boundaries.len(), n + 3);

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
    for (i, &b) in boundaries.iter().enumerate() {
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

/// Derives a linear-to-log index table from a sentinel-wrapped boundaries slice.
///
/// For `count` equidistant linear buckets (each of width `1 << shift`
/// in significand space), stores the approximate log bucket containing
/// each linear bucket's lower bound.
fn derive_index_table(boundaries: &[u64], count: usize, shift: u32) -> Vec<u16> {
    let mut table = vec![0u16; count];
    let mut j: u16 = 0;
    for i in 0..count {
        let lower_bound = (i as u64) << shift;
        while lower_bound >= boundaries[j as usize + 1] {
            j += 1;
        }
        table[i] = j;
    }
    table
}

/// Writes an algorithm-specific index table derived from `boundaries`.
///
/// `boundaries` must be the sentinel-wrapped array returned by
/// [`generate_boundaries`].
/// `extra_bits`: 1 for NewRelic (2N linear buckets), 0 for Dynatrace (N).
/// `prefix`: name prefix for generated symbols (e.g. "NR" or "DT").
///
/// Emits `{PREFIX}_INDEX: [u16; _]` and `{PREFIX}_SHIFT: u32`.
pub fn write_index_table<W: Write>(
    w: &mut W,
    table_scale: u32,
    boundaries: &[u64],
    extra_bits: u32,
    prefix: &str,
) -> std::io::Result<()> {
    let count = 1usize << (table_scale + extra_bits);
    let shift = 52 - table_scale - extra_bits;
    let index_table = derive_index_table(boundaries, count, shift);

    writeln!(w)?;
    writeln!(
        w,
        "/// Significand shift for {} algorithm at scale {}.",
        prefix, table_scale
    )?;
    writeln!(w, "pub const {}_SHIFT: u32 = {};", prefix, shift)?;
    writeln!(w)?;

    writeln!(
        w,
        "/// Linear-to-log index table for {} algorithm ({} entries).",
        prefix, count
    )?;
    writeln!(
        w,
        "pub static {}_INDEX: [u16; {}] = [",
        prefix, count
    )?;
    for (i, &idx) in index_table.iter().enumerate() {
        if i % 16 == 0 {
            write!(w, "    ")?;
        }
        write!(w, "{},", idx)?;
        if i % 16 == 15 {
            writeln!(w)?;
        }
    }
    if count % 16 != 0 {
        writeln!(w)?;
    }
    writeln!(w, "];")?;

    Ok(())
}
