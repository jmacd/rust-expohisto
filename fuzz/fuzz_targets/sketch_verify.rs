//! Shared brute-force + snapshot helpers for the `Sketch` fuzz oracles.
//!
//! Included (not a fuzz target) by `sketch_oracle` and
//! `sketch_pn_oracle` via `#[path = "sketch_verify.rs"] mod sketch_verify;`.

use std::collections::BTreeMap;

use otel_expohisto::{Scale, Sketch, Width};

/// A comparable snapshot of a sketch's externally-visible state. Equal
/// snapshots imply identical exported histograms.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Snapshot {
    pub scale: i32,
    pub collapsed: bool,
    pub width: Width,
    pub underflow: u64,
    pub offset: i32,
    pub counts: Vec<u64>,
}

pub fn snapshot<const N: usize>(s: &Sketch<N>) -> Snapshot {
    let v = s.view();
    let p = v.positive();
    Snapshot {
        scale: v.scale(),
        collapsed: v.collapsed(),
        width: p.width(),
        underflow: v.underflow_count(),
        offset: p.offset(),
        counts: p.iter().collect(),
    }
}

/// Maps a selector byte into an increment that exercises the width tiers
/// (B1 → … → U64). `mode` picks a qualitatively different strategy.
pub fn decode_increment(sel: u8, mode: u8) -> u64 {
    match mode % 4 {
        // Powers of two — leap across width boundaries.
        0 => 1u64 << ((sel as u32) % 20),
        // Sit at each width's maximum.
        1 => match sel % 6 {
            0 => 1,
            1 => 3,
            2 => 15,
            3 => 255,
            4 => 65535,
            _ => 1 << 24,
        },
        // Just past each threshold to trigger immediate widening.
        2 => match sel % 6 {
            0 => 2,
            1 => 4,
            2 => 16,
            3 => 256,
            4 => 65536,
            _ => 1,
        },
        // Mostly unit inserts with rare spikes.
        _ => {
            if sel < 220 {
                1
            } else {
                1u64 << (sel - 220)
            }
        }
    }
}

/// Brute-force oracle for a single sketch against the `(value, incr)`
/// pairs recorded into it. Every value must be `>= 0`; zeros are allowed
/// and excluded from buckets. Verifies count, zero count, min/max,
/// bucketed total, the (always-retained) top bucket index, and the
/// accurate region exactly, with mass at or below the underflow floor
/// folded into the lowest slot.
pub fn verify_sketch<const N: usize>(
    s: &Sketch<N>,
    recorded: &[(f64, u64)],
    scale: i32,
    label: &str,
) {
    let total: u64 = recorded.iter().map(|&(_, incr)| incr).sum();
    assert_eq!(s.count(), total, "{label} N={N} scale={scale}: count");

    let v = s.view();
    let stats = v.stats();

    let nz_total: u64 = recorded
        .iter()
        .filter(|&&(value, _)| value != 0.0)
        .map(|&(_, incr)| incr)
        .sum();
    assert_eq!(
        v.zero_count(),
        total - nz_total,
        "{label} N={N} scale={scale}: zero_count",
    );

    let bucket_total: u64 = v.positive().iter().sum();
    assert_eq!(
        bucket_total, nz_total,
        "{label} N={N} scale={scale}: bucketed total != non-zero total",
    );

    if nz_total == 0 {
        assert!(v.positive().is_empty(), "{label}: expected no buckets");
        return;
    }

    let expected_min = recorded
        .iter()
        .filter(|&&(value, _)| value != 0.0)
        .map(|&(value, _)| value)
        .fold(f64::INFINITY, f64::min);
    let expected_max = recorded
        .iter()
        .filter(|&&(value, _)| value != 0.0)
        .map(|&(value, _)| value)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(stats.min, expected_min, "{label} N={N} scale={scale}: min");
    assert_eq!(stats.max, expected_max, "{label} N={N} scale={scale}: max");

    // Sum is checked with tolerance: f64 accumulation order differs from a
    // brute-force re-sum, so it is not bit-exact (and is not asserted to be
    // order-independent at the bit level for the same reason). Skip when
    // the brute-force sum is not finite (overflow to ±inf), since the exact
    // accumulation order then governs whether/where inf appears.
    let expected_sum: f64 = recorded.iter().map(|&(v, i)| v * i as f64).sum();
    if expected_sum.is_finite() {
        let sum_tol = expected_sum.abs() * 1e-9 + 1e-9;
        assert!(
            (stats.sum - expected_sum).abs() <= sum_tol,
            "{label} N={N} scale={scale}: sum {} vs expected {expected_sum}",
            stats.sum,
        );
    }

    // Brute-force per-bucket counts at the fixed scale. record_incr rounds
    // subnormals to (biased_exp=1, significand=1); map the resulting f64
    // through the same path so the oracle agrees.
    const SUBNORMAL_ROUNDED: f64 = f64::from_bits((1u64 << 52) | 1);
    let smap = Scale::new(scale).expect("scale is valid");
    let mut expected: BTreeMap<i32, u64> = BTreeMap::new();
    for &(value, incr) in recorded {
        if value != 0.0 {
            let idx = if value.to_bits() >> 52 == 0 {
                smap.map_to_index(SUBNORMAL_ROUNDED)
            } else {
                smap.map_to_index(value)
            };
            *expected.entry(idx).or_insert(0) += incr;
        }
    }
    let exp_min_idx = *expected.keys().next().unwrap();
    let exp_max_idx = *expected.keys().last().unwrap();

    let p = v.positive();
    let offset = p.offset();
    let counts: Vec<u64> = p.iter().collect();
    let last_idx = offset + counts.len() as i32 - 1;

    // The top is always retained accurately, regardless of collapse.
    assert_eq!(
        last_idx, exp_max_idx,
        "{label} N={N} scale={scale}: top bucket index",
    );

    if v.collapsed() {
        // The underflow is a side counter reported at the floor (offset):
        //   counts[0]            == mass at-or-below the floor
        //   underflow_count()    == mass strictly below the floor
        //   counts[k>=1]         == exact accurate buckets above the floor
        let uf_at_or_below: u64 = expected
            .iter()
            .filter(|(&idx, _)| idx <= offset)
            .map(|(_, &c)| c)
            .sum();
        let uf_strict: u64 = expected
            .iter()
            .filter(|(&idx, _)| idx < offset)
            .map(|(_, &c)| c)
            .sum();
        assert_eq!(
            counts[0], uf_at_or_below,
            "{label} N={N} scale={scale}: floor bucket (underflow + accurate@floor)",
        );
        assert_eq!(
            v.underflow_count(),
            uf_strict,
            "{label} N={N} scale={scale}: underflow_count (strictly below floor)",
        );
        for (k, &c) in counts.iter().enumerate().skip(1) {
            let idx = offset + k as i32;
            let exp = expected.get(&idx).copied().unwrap_or(0);
            assert_eq!(c, exp, "{label} N={N} scale={scale}: accurate bucket {idx}");
        }
    } else {
        // No collapse: the whole distribution is exact.
        assert_eq!(
            offset, exp_min_idx,
            "{label} N={N} scale={scale}: offset (uncollapsed)",
        );
        assert_eq!(v.underflow_count(), 0, "{label}: uncollapsed underflow");
        for (k, &c) in counts.iter().enumerate() {
            let idx = offset + k as i32;
            let exp = expected.get(&idx).copied().unwrap_or(0);
            assert_eq!(c, exp, "{label} N={N} scale={scale}: bucket {idx}");
        }
    }
}
