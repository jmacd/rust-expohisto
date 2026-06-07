#![no_main]

//! Differential + brute-force oracle for the fixed-scale `Sketch`.
//!
//! Two complementary checks catch different bug classes:
//!
//! 1. **Brute force** ([`verify_sketch`]): count, min/max/sum, zero count,
//!    bucketed total, the top bucket index, and — crucially — the
//!    *accurate* region must match a `BTreeMap` recomputed from the raw
//!    ops, with everything at or below the underflow floor folded into the
//!    lowest slot.
//! 2. **Order independence** ([`snapshot`]): the final state is fully
//!    determined by the multiset, so the original, sorted, and reversed
//!    feeds must produce an identical view. This is what exposes a
//!    collapse that over-folds live data (a window anchored at a phantom
//!    trailing-empty slot), which `verify_sketch` alone cannot see because
//!    it measures the accurate region relative to the sketch's own floor.

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Scale, Sketch, Width, MIN_SCALE};

#[derive(Clone, Copy)]
struct Op {
    value: f64,
    incr: u64,
}

/// A comparable snapshot of a sketch's externally-visible state.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Snapshot {
    scale: i32,
    collapsed: bool,
    width: Width,
    underflow: u64,
    offset: i32,
    counts: Vec<u64>,
}

fn snapshot<const N: usize>(s: &Sketch<N>) -> Snapshot {
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

fn build<const N: usize>(scale: i32, ops: &[Op]) -> Sketch<N> {
    let mut s = Sketch::<N>::new().with_scale(scale).unwrap();
    for op in ops {
        // All ops are pre-filtered to record successfully.
        s.record_incr(op.value, op.incr).unwrap();
    }
    s
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let scale = MIN_SCALE + (data[0] as i32) % (otel_expohisto::table_scale() - MIN_SCALE + 1);
    let partition_byte = data[1];
    let mode = data[2];
    let echo_ctl = data[3];
    let rest = &data[4..];

    let mut ops: Vec<Op> = rest
        .chunks_exact(9)
        .filter_map(|c| {
            let v = f64::from_le_bytes(c[..8].try_into().unwrap());
            // Sketch accepts non-negative finite values (0.0 is the zero
            // bucket); everything else is rejected and excluded.
            if !v.is_finite() || v < 0.0 {
                return None;
            }
            Some(Op {
                value: v,
                incr: decode_increment(c[8], mode),
            })
        })
        .collect();

    if ops.is_empty() {
        return;
    }

    // Echo phase: replay with growing multipliers to force counter
    // widening (B1 → … → U64) and large underflow accumulation.
    let echo_passes = (echo_ctl % 4) as usize;
    let base_len = ops.len();
    for pass in 0..echo_passes {
        for i in 0..base_len {
            let Op { value, incr } = ops[i];
            let mul = 1u64 << ((pass * 5 + 3).min(40));
            let boosted = incr.saturating_mul(mul);
            if boosted > 0 && boosted <= (1u64 << 40) {
                ops.push(Op { value, incr: boosted });
            }
        }
    }

    // Drop ops that would overflow the u64 total, so build() can unwrap.
    ops = truncate_to_no_overflow(ops);
    if ops.is_empty() {
        return;
    }

    check_sketch::<2>(scale, &ops);
    check_sketch::<8>(scale, &ops);
    check_sketch::<16>(scale, &ops);

    // Merge: split, build independently, combine, and verify.
    let split = (partition_byte as usize) % (ops.len() + 1);
    let (left, right) = ops.split_at(split);
    check_merge_same::<8>(scale, left, right, &ops);
    check_merge_same::<16>(scale, left, right, &ops);
    check_merge_diff::<8, 16>(scale, left, right, &ops);
    check_merge_diff::<16, 8>(scale, left, right, &ops);
});

/// Keep a prefix of ops whose increments sum without overflowing u64.
fn truncate_to_no_overflow(ops: Vec<Op>) -> Vec<Op> {
    let mut total: u64 = 0;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match total.checked_add(op.incr) {
            Some(t) => {
                total = t;
                out.push(op);
            }
            None => break,
        }
    }
    out
}

fn decode_increment(sel: u8, mode: u8) -> u64 {
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

fn check_sketch<const N: usize>(scale: i32, ops: &[Op]) {
    let s = build::<N>(scale, ops);
    verify_sketch(&s, ops, scale, "record");

    // Order independence: the final state must not depend on insertion
    // order. Sorted and reversed feeds of the same multiset must agree.
    let original = snapshot(&s);

    let mut sorted: Vec<Op> = ops.to_vec();
    sorted.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
    let s_sorted = build::<N>(scale, &sorted);
    assert_eq!(
        snapshot(&s_sorted),
        original,
        "N={N} scale={scale}: sorted feed diverged from original",
    );

    let reversed: Vec<Op> = ops.iter().rev().copied().collect();
    let s_rev = build::<N>(scale, &reversed);
    assert_eq!(
        snapshot(&s_rev),
        original,
        "N={N} scale={scale}: reversed feed diverged from original",
    );
}

fn check_merge_same<const N: usize>(scale: i32, left: &[Op], right: &[Op], all: &[Op]) {
    let mut a = build::<N>(scale, left);
    let b = build::<N>(scale, right);
    a.merge_from(&b).unwrap();
    verify_sketch(&a, all, scale, "merge_same");

    // Merge is commutative for equal pool sizes.
    let mut b2 = build::<N>(scale, right);
    let a2 = build::<N>(scale, left);
    b2.merge_from(&a2).unwrap();
    assert_eq!(
        snapshot(&a),
        snapshot(&b2),
        "N={N} scale={scale}: merge is not commutative",
    );
}

fn check_merge_diff<const N: usize, const M: usize>(
    scale: i32,
    left: &[Op],
    right: &[Op],
    all: &[Op],
) {
    let mut a = build::<N>(scale, left);
    let b = build::<M>(scale, right);
    a.merge_from(&b).unwrap();
    verify_sketch(&a, all, scale, "merge_diff");
}

/// Brute-force oracle for a single sketch.
fn verify_sketch<const N: usize>(s: &Sketch<N>, ops: &[Op], scale: i32, label: &str) {
    let total: u64 = ops.iter().map(|o| o.incr).sum();
    assert_eq!(s.count(), total, "{label} N={N} scale={scale}: count");

    let v = s.view();
    let stats = v.stats();

    let nz_total: u64 = ops
        .iter()
        .filter(|o| o.value != 0.0)
        .map(|o| o.incr)
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

    let expected_min = ops
        .iter()
        .filter(|o| o.value != 0.0)
        .map(|o| o.value)
        .fold(f64::INFINITY, f64::min);
    let expected_max = ops
        .iter()
        .filter(|o| o.value != 0.0)
        .map(|o| o.value)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(stats.min, expected_min, "{label} N={N} scale={scale}: min");
    assert_eq!(stats.max, expected_max, "{label} N={N} scale={scale}: max");

    // Brute-force per-bucket counts at the fixed scale. record_incr rounds
    // subnormals to (biased_exp=1, significand=1); map the resulting f64
    // through the same path so the oracle agrees.
    const SUBNORMAL_ROUNDED: f64 = f64::from_bits((1u64 << 52) | 1);
    let smap = Scale::new(scale).expect("scale is valid");
    let mut expected: BTreeMap<i32, u64> = BTreeMap::new();
    for o in ops {
        if o.value != 0.0 {
            let idx = if o.value.to_bits() >> 52 == 0 {
                smap.map_to_index(SUBNORMAL_ROUNDED)
            } else {
                smap.map_to_index(o.value)
            };
            *expected.entry(idx).or_insert(0) += o.incr;
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
        // Lowest slot is the underflow placeholder: it holds every count
        // at or below the floor; slots above it are exact.
        let uf_expected: u64 = expected
            .iter()
            .filter(|(&idx, _)| idx <= offset)
            .map(|(_, &c)| c)
            .sum();
        assert_eq!(
            counts[0], uf_expected,
            "{label} N={N} scale={scale}: underflow fold",
        );
        assert_eq!(
            counts[0],
            v.underflow_count(),
            "{label} N={N} scale={scale}: underflow_count vs offset bucket",
        );
        for (k, &c) in counts.iter().enumerate().skip(1) {
            let idx = offset + k as i32;
            let exp = expected.get(&idx).copied().unwrap_or(0);
            assert_eq!(
                c, exp,
                "{label} N={N} scale={scale}: accurate bucket {idx}",
            );
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
            assert_eq!(
                c, exp,
                "{label} N={N} scale={scale}: bucket {idx}",
            );
        }
    }
}
