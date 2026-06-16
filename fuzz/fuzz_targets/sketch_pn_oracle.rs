#![no_main]

//! Differential + brute-force oracle for the signed `SketchPN`.
//!
//! Reuses the single-sketch checks from `sketch_verify` on each range:
//! positive values are verified against the positive sub-sketch and the
//! magnitudes of negative values against the negative sub-sketch. On top
//! of that it checks the aggregate count, zero count, and signed min/max,
//! and asserts order independence (original/sorted/reversed) and merge
//! commutativity for equal pool sizes.

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{SketchPN, MIN_SCALE};

#[path = "sketch_verify.rs"]
mod sketch_verify;
use sketch_verify::{decode_increment, snapshot, verify_sketch, Snapshot};

#[derive(Clone, Copy)]
struct Op {
    value: f64,
    incr: u64,
}

/// Order-independent fingerprint of a `SketchPN`'s exported state.
#[derive(PartialEq, Eq, Debug)]
struct PnSnapshot {
    pos: Snapshot,
    neg: Snapshot,
    zero: u64,
    count: u64,
}

fn pn_snapshot<const K: usize, const L: usize>(s: &SketchPN<K, L>) -> PnSnapshot {
    PnSnapshot {
        pos: snapshot(s.positive()),
        neg: snapshot(s.negative()),
        zero: s.view().zero_count(),
        count: s.count(),
    }
}

fn build<const K: usize, const L: usize>(scale: i32, ops: &[Op]) -> SketchPN<K, L> {
    let mut s = SketchPN::<K, L>::new().with_scale(scale).unwrap();
    for op in ops {
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

    // Signed values: any finite f64 (positive, negative, or zero).
    let mut ops: Vec<Op> = rest
        .chunks_exact(9)
        .filter_map(|c| {
            let v = f64::from_le_bytes(c[..8].try_into().unwrap());
            if !v.is_finite() {
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

    // Echo phase: grow counts to force widening in both ranges.
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

    ops = truncate_to_no_overflow(ops);
    if ops.is_empty() {
        return;
    }

    check_pn::<8, 8>(scale, &ops);
    check_pn::<4, 16>(scale, &ops);
    check_pn::<16, 4>(scale, &ops);

    // Merge: split, build independently, combine, verify.
    let split = (partition_byte as usize) % (ops.len() + 1);
    let (left, right) = ops.split_at(split);
    check_merge_same::<8, 8>(scale, left, right, &ops);
    check_merge_same::<4, 16>(scale, left, right, &ops);
    check_merge_diff::<8, 8, 16, 4>(scale, left, right, &ops);
});

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

/// Verify a `SketchPN` against signed `(value, incr)` ops: aggregate
/// count, zero count, signed min/max, and each range against brute force.
fn verify_pn<const K: usize, const L: usize>(
    s: &SketchPN<K, L>,
    ops: &[Op],
    scale: i32,
    label: &str,
) {
    let v = s.view();
    let total: u64 = ops.iter().map(|o| o.incr).sum();
    assert_eq!(s.count(), total, "{label}: count");

    let zeros: u64 = ops.iter().filter(|o| o.value == 0.0).map(|o| o.incr).sum();
    assert_eq!(v.zero_count(), zeros, "{label}: zero_count");

    if total > 0 {
        // Zero observations contribute the value 0.0 to min/max.
        let mut emin = f64::INFINITY;
        let mut emax = f64::NEG_INFINITY;
        for o in ops {
            let x = if o.value == 0.0 { 0.0 } else { o.value };
            emin = emin.min(x);
            emax = emax.max(x);
        }
        assert_eq!(v.stats().min, emin, "{label}: min");
        assert_eq!(v.stats().max, emax, "{label}: max");
    }

    // Each range is an ordinary single-sketch over its sign's values
    // (negatives recorded by magnitude).
    let pos: Vec<(f64, u64)> = ops
        .iter()
        .filter(|o| o.value > 0.0)
        .map(|o| (o.value, o.incr))
        .collect();
    let neg: Vec<(f64, u64)> = ops
        .iter()
        .filter(|o| o.value < 0.0)
        .map(|o| (-o.value, o.incr))
        .collect();
    verify_sketch(s.positive(), &pos, scale, "pn.pos");
    verify_sketch(s.negative(), &neg, scale, "pn.neg");
}

fn check_pn<const K: usize, const L: usize>(scale: i32, ops: &[Op]) {
    let s = build::<K, L>(scale, ops);
    verify_pn(&s, ops, scale, "record");

    // The exact partition is order-dependent (separated underflow), but the
    // aggregates are not. verify_pn pins each order against the ops.
    let original = pn_agg(&s);

    let mut sorted: Vec<Op> = ops.to_vec();
    sorted.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
    let s_sorted = build::<K, L>(scale, &sorted);
    verify_pn(&s_sorted, &sorted, scale, "record/sorted");
    assert_eq!(
        pn_agg(&s_sorted),
        original,
        "K={K} L={L} scale={scale}: sorted feed changed an aggregate",
    );

    let reversed: Vec<Op> = ops.iter().rev().copied().collect();
    let s_rev = build::<K, L>(scale, &reversed);
    verify_pn(&s_rev, &reversed, scale, "record/reversed");
    assert_eq!(
        pn_agg(&s_rev),
        original,
        "K={K} L={L} scale={scale}: reversed feed changed an aggregate",
    );
}

/// Order-independent fingerprint of a `SketchPN` (sum excluded — it
/// differs by ULPs across orders; `verify_pn`/`verify_sketch` check it
/// per-order with tolerance).
fn pn_agg<const K: usize, const L: usize>(s: &SketchPN<K, L>) -> (u64, u64, u64, u64) {
    let v = s.view();
    let st = v.stats();
    (st.count, st.min.to_bits(), st.max.to_bits(), v.zero_count())
}

fn check_merge_same<const K: usize, const L: usize>(
    scale: i32,
    left: &[Op],
    right: &[Op],
    all: &[Op],
) {
    let mut a = build::<K, L>(scale, left);
    let b = build::<K, L>(scale, right);
    a.merge_from(&b).unwrap();
    verify_pn(&a, all, scale, "merge_same");

    // Each sub-range rebuilds symmetrically from the union, so a single PN
    // merge is exactly commutative for equal pool sizes.
    let mut b2 = build::<K, L>(scale, right);
    let a2 = build::<K, L>(scale, left);
    b2.merge_from(&a2).unwrap();
    assert_eq!(
        pn_snapshot(&a),
        pn_snapshot(&b2),
        "K={K} L={L} scale={scale}: merge is not commutative",
    );
}

fn check_merge_diff<const K: usize, const L: usize, const K2: usize, const L2: usize>(
    scale: i32,
    left: &[Op],
    right: &[Op],
    all: &[Op],
) {
    let mut a = build::<K, L>(scale, left);
    let b = build::<K2, L2>(scale, right);
    a.merge_from(&b).unwrap();
    verify_pn(&a, all, scale, "merge_diff");
}
