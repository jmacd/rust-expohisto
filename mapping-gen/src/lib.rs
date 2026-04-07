// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! Lookup table generation for exponential histogram mapping.
//!
//! This crate computes exact bucket boundary significands and derives
//! the linear-to-log index table used by the lookup mapping algorithm.
//!
//! Build scripts call [`generate_boundaries`] once, then pass the result
//! to [`write_boundaries`] and [`write_index_table`].

// Re-use the canonical float64 definitions from the main crate to avoid
// maintaining a duplicate copy. The `pub` visibility on items there is
// restricted to `pub(crate)` in the main crate via its lib.rs.
#[path = "../../src/float64.rs"]
#[allow(dead_code)]
mod float64;
mod table;

pub use table::{compute_boundaries_exact, map_to_index};

use std::io::Write;

/// Maximum scale for inverse factor table (matches `mapping::MAX_SCALE`).
pub const MAX_SCALE: u32 = 16;

/// Computes `ln(2) / 2^scale` for scales 1..=`MAX_SCALE` as correctly
/// rounded f64 values using 256-bit precision via `rug`.
pub fn generate_inverse_factors() -> Vec<f64> {
    use rug::Float;

    const PRECISION: u32 = 256;

    let ln2 = Float::with_val(PRECISION, rug::float::Constant::Log2);

    (1..=MAX_SCALE)
        .map(|scale| {
            let divisor = Float::with_val(PRECISION, Float::i_pow_u(2, scale));
            let exact = Float::with_val(PRECISION, &ln2 / &divisor);
            // to_f64() rounds to nearest (default rug rounding mode)
            exact.to_f64()
        })
        .collect()
}

/// Writes the `INVERSE_FACTOR` array as Rust source.
///
/// `factors` must have `MAX_SCALE` entries as returned by
/// [`generate_inverse_factors`].
pub fn write_inverse_factors<W: Write>(w: &mut W, factors: &[f64]) -> std::io::Result<()> {
    assert_eq!(
        factors.len(),
        MAX_SCALE as usize,
        "expected {} factors, got {}",
        MAX_SCALE,
        factors.len()
    );

    writeln!(w, "// Auto-generated inverse factor table: ln(2) / 2^scale")?;
    writeln!(
        w,
        "// for scales 1..={}. Indexed as INVERSE_FACTOR[scale - 1].",
        MAX_SCALE
    )?;
    writeln!(w)?;
    writeln!(
        w,
        "/// `ln(2) / 2^scale` for scales 1..={}, indexed as `[scale - 1]`.",
        MAX_SCALE
    )?;
    writeln!(
        w,
        "/// Generated from 256-bit precision arithmetic, correctly rounded to f64."
    )?;
    writeln!(w, "const INVERSE_FACTOR: [f64; {}] = [", MAX_SCALE)?;
    for (i, &f) in factors.iter().enumerate() {
        let scale = i + 1;
        // Use crate::float64::from_bits for MSRV-compatible const context
        writeln!(
            w,
            "    crate::float64::from_bits(0x{:016X}), // scale {} ≈ {:.6e}",
            f.to_bits(),
            scale,
            f
        )?;
    }
    writeln!(w, "];")?;

    Ok(())
}

/// Computes the sentinel-wrapped boundary array for a given scale.
///
/// Returns `N + 3` entries with layout:
/// `[sentinel=0, b[0]=1, b[1], …, b[N−1], sentinel=2^52, sentinel=2^52]`
/// where N = 2^table_scale. Boundaries are exact 52-bit significands
/// computed via bignum arithmetic.
pub fn generate_boundaries(table_scale: u32) -> Vec<u64> {
    let n = 1usize << table_scale;
    let mut raw = compute_boundaries_exact(table_scale);

    // Upper-inclusive adjustment: boundary[0] = 1 instead of 0, so
    // significand == 0 (exact powers of two) falls below, matching
    // OTel's upper-inclusive bucket semantics.
    assert_eq!(
        raw[0], 0,
        "boundary[0] must be 0 before upper-inclusive adjustment"
    );
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
    assert_eq!(
        boundaries.len(),
        n + 3,
        "expected {} sentinel-wrapped boundaries, got {}",
        n + 3,
        boundaries.len()
    );

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
///
/// Panics if any index value exceeds `u16::MAX` (i.e. `table_scale > 16`).
fn derive_index_table(boundaries: &[u64], count: usize, shift: u32) -> Vec<u16> {
    let mut table = vec![0u16; count];
    let mut j: u32 = 0;
    for (i, entry) in table.iter_mut().enumerate() {
        let lower_bound = (i as u64) << shift;
        while lower_bound >= boundaries[j as usize + 1] {
            j += 1;
        }
        *entry = u16::try_from(j).unwrap_or_else(|_| {
            panic!(
                "index table value {j} at position {i} exceeds u16::MAX; \
                 table_scale > 16 requires wider index type"
            )
        });
    }
    table
}

/// Writes the index table derived from `boundaries`.
///
/// `boundaries` must be the sentinel-wrapped array returned by
/// [`generate_boundaries`].
///
/// Uses 2N linear buckets with 1 boundary correction (the lookup
/// algorithm). Emits `INDEX_SHIFT: u32` and `INDEX_TABLE: [u16; 2N]`.
pub fn write_index_table<W: Write>(
    w: &mut W,
    table_scale: u32,
    boundaries: &[u64],
) -> std::io::Result<()> {
    let extra_bits = 1u32; // 2N linear buckets
    let count = 1usize << (table_scale + extra_bits);
    let shift = 52 - table_scale - extra_bits;
    let index_table = derive_index_table(boundaries, count, shift);

    writeln!(w)?;
    writeln!(
        w,
        "/// Significand shift for the lookup algorithm at scale {}.",
        table_scale
    )?;
    writeln!(w, "pub const INDEX_SHIFT: u32 = {};", shift)?;
    writeln!(w)?;

    writeln!(w, "/// Linear-to-log index table ({} entries).", count)?;
    writeln!(w, "pub static INDEX_TABLE: [u16; {}] = [", count)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Float;

    /// Verifies that every generated inverse factor is the correctly
    /// rounded f64 of `ln(2) / 2^scale`, by comparing the generated
    /// value against a 256-bit precision computation.
    #[test]
    fn test_inverse_factors_correctly_rounded() {
        let factors = generate_inverse_factors();
        assert_eq!(factors.len(), MAX_SCALE as usize);

        let ln2 = Float::with_val(256, rug::float::Constant::Log2);

        for (i, &f) in factors.iter().enumerate() {
            let scale = (i + 1) as u32;
            let divisor = Float::with_val(256, Float::i_pow_u(2, scale));
            let exact = Float::with_val(256, &ln2 / &divisor);
            let expected = exact.to_f64();

            assert_eq!(
                f.to_bits(),
                expected.to_bits(),
                "inverse factor at scale {} not correctly rounded",
                scale
            );
        }
    }

    /// Verifies that the significand bits of ln(2)/2^scale are all
    /// identical (only the exponent changes when dividing by 2).
    #[test]
    fn test_inverse_factors_share_significand() {
        let factors = generate_inverse_factors();
        let sig_mask = (1u64 << 52) - 1;
        let first_sig = factors[0].to_bits() & sig_mask;

        for (i, &f) in factors.iter().enumerate() {
            assert_eq!(
                f.to_bits() & sig_mask,
                first_sig,
                "significand differs at scale {}",
                i + 1
            );
        }
    }
}
