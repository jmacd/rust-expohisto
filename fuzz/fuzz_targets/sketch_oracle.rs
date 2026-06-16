#![no_main]

//! Differential + brute-force oracle for the fixed-scale `Sketch`.
//!
//! Two complementary checks catch different bug classes:
//!
//! 1. **Brute force** (`verify_sketch`): count, min/max, zero count,
//!    bucketed total, the top bucket index, and — crucially — the
//!    *accurate* region must match a `BTreeMap` recomputed from the raw
//!    ops, with everything at or below the underflow floor folded into the
//!    lowest slot.
//! 2. **Order independence** (`snapshot`): the final state is fully
//!    determined by the multiset, so the original, sorted, and reversed
//!    feeds must produce an identical view. This is what exposes a
//!    collapse that over-folds live data (a window anchored at a phantom
//!    trailing-empty slot), which `verify_sketch` alone cannot see because
//!    it measures the accurate region relative to the sketch's own floor.

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Sketch, MIN_SCALE};

#[path = "sketch_verify.rs"]
mod sketch_verify;
use sketch_verify::{decode_increment, snapshot, verify_sketch};

#[derive(Clone, Copy)]
struct Op {
    value: f64,
    incr: u64,
}

fn pairs(ops: &[Op]) -> Vec<(f64, u64)> {
    ops.iter().map(|o| (o.value, o.incr)).collect()
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

fn check_sketch<const N: usize>(scale: i32, ops: &[Op]) {
    let s = build::<N>(scale, ops);
    verify_sketch(&s, &pairs(ops), scale, "record");

    // Under the separated-underflow design the exact partition is
    // intentionally order-dependent, but the aggregates and the
    // always-accurate top of the distribution are not. (Each order is also
    // independently checked against the ops by verify_sketch above.)
    let original = agg(&s);

    let mut sorted: Vec<Op> = ops.to_vec();
    sorted.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
    let s_sorted = build::<N>(scale, &sorted);
    verify_sketch(&s_sorted, &pairs(&sorted), scale, "record/sorted");
    assert_eq!(
        agg(&s_sorted),
        original,
        "N={N} scale={scale}: sorted feed changed an order-independent aggregate",
    );

    let reversed: Vec<Op> = ops.iter().rev().copied().collect();
    let s_rev = build::<N>(scale, &reversed);
    verify_sketch(&s_rev, &pairs(&reversed), scale, "record/reversed");
    assert_eq!(
        agg(&s_rev),
        original,
        "N={N} scale={scale}: reversed feed changed an order-independent aggregate",
    );
}

/// Order-independent fingerprint: count, min, max, and the
/// always-accurate top bucket index. (Sum is excluded — f64 accumulation
/// order makes it differ by ULPs across insert orders; it is checked
/// per-order with tolerance in `verify_sketch`.)
fn agg<const N: usize>(s: &Sketch<N>) -> (u64, u64, u64, i32) {
    let v = s.view();
    let st = v.stats();
    let top = v.positive().offset() + v.positive().len() as i32 - 1;
    (st.count, st.min.to_bits(), st.max.to_bits(), top)
}

fn check_merge_same<const N: usize>(scale: i32, left: &[Op], right: &[Op], all: &[Op]) {
    let mut a = build::<N>(scale, left);
    let b = build::<N>(scale, right);
    a.merge_from(&b).unwrap();
    verify_sketch(&a, &pairs(all), scale, "merge_same");

    // A single merge rebuilds symmetrically from the union, so merge is
    // exactly commutative for equal pool sizes (unlike sequential inserts).
    let mut b2 = build::<N>(scale, right);
    let a2 = build::<N>(scale, left);
    b2.merge_from(&a2).unwrap();
    assert_eq!(
        snapshot(&a),
        snapshot(&b2),
        "N={N} scale={scale}: merge is not commutative",
    );
}

fn check_merge_diff<const N: usize, const M: usize>(scale: i32, left: &[Op], right: &[Op], all: &[Op]) {
    let mut a = build::<N>(scale, left);
    let b = build::<M>(scale, right);
    a.merge_from(&b).unwrap();
    verify_sketch(&a, &pairs(all), scale, "merge_diff");
}
