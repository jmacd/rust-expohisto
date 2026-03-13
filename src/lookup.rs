// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared lookup tables for exponential histogram mapping.
//!
//! This module provides the compile-time generated boundary significand
//! table at the highest compiled-in scale, plus per-scale derived tables
//! computed lazily at runtime. Both NewRelic and Dynatrace algorithms
//! share the same per-scale boundaries; they differ only in index table
//! size (2N vs N) and correction count (1 vs 2).
//!
//! Per-scale boundaries are derived by striding through the full-scale
//! BOUNDARIES array, packed contiguously in a single allocation (~2×
//! the full-scale table, geometric series). Each scale's working set
//! fits tightly in cache.

include!(concat!(env!("OUT_DIR"), "/lookup_tables.rs"));

use std::sync::OnceLock;

/// Pre-computed per-scale boundary tables, packed contiguously.
struct PerScaleBoundaries {
    data: Box<[u64]>,
    /// `offsets[s]` = start index of scale-s boundaries in `data`.
    offsets: [u32; 16],
}

static PER_SCALE: OnceLock<PerScaleBoundaries> = OnceLock::new();

fn per_scale_boundaries() -> &'static PerScaleBoundaries {
    PER_SCALE.get_or_init(|| {
        let h = TABLE_SCALE as usize;
        // Total entries: sum_{s=1}^{H} (2^s + 3) = (2^{H+1} - 2) + 3*H
        let total: usize = (1..=h).map(|s| (1usize << s) + 3).sum();
        let mut data = Vec::with_capacity(total);
        let mut offsets = [0u32; 16];

    #[allow(clippy::needless_range_loop)] // `s` used both as index and for bit shifts
    for s in 1..=h {
            offsets[s] = data.len() as u32;
            let n_s = 1usize << s;
            let stride = 1usize << (h - s);

            data.push(0); // sentinel
            for k in 0..n_s {
                data.push(BOUNDARIES[1 + k * stride]);
            }
            data.push(1u64 << 52); // sentinel
            data.push(1u64 << 52); // sentinel
        }

        PerScaleBoundaries {
            data: data.into_boxed_slice(),
            offsets,
        }
    })
}

/// Returns the boundary table for the given scale (1..=TABLE_SCALE).
///
/// The returned slice has `2^scale + 3` entries:
/// `[sentinel=0, b[0], b[1], ..., b[2^scale - 1], sentinel=2^52, sentinel=2^52]`
///
/// Each entry is the significand threshold between adjacent log buckets,
/// derived by taking every `2^(H-S)`-th entry from the full-scale table.
#[inline]
pub fn boundaries(scale: i32) -> &'static [u64] {
    debug_assert!((1..=TABLE_SCALE).contains(&scale));
    let psb = per_scale_boundaries();
    let s = scale as usize;
    let start = psb.offsets[s] as usize;
    let len = (1usize << s) + 3;
    &psb.data[start..start + len]
}

/// Derives a linear-to-log index table from a boundaries slice.
///
/// For `count` equidistant linear buckets (each of width `1 << shift`
/// in significand space), stores the approximate log bucket containing
/// each linear bucket's lower bound.
pub fn derive_index_table(boundaries: &[u64], count: usize, shift: u32) -> Vec<u16> {
    let mut table = vec![0u16; count];
    let mut j: u16 = 0;
    #[allow(clippy::needless_range_loop)] // `i` used for both indexing and bit shift
    for i in 0..count {
        let lower_bound = (i as u64) << shift;
        while lower_bound >= boundaries[j as usize + 1] {
            j += 1;
        }
        table[i] = j;
    }
    table
}
