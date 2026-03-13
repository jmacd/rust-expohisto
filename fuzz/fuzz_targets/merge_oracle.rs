#![no_main]

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Histogram, Mapping};
use otel_expohisto::{P32, Precision};
use std::collections::BTreeMap;

/// A weighted insert operation: record `value` with multiplicity `incr`.
#[derive(Clone, Copy)]
struct Op {
    value: f64,
    incr: u64,
}

fuzz_target!(|data: &[u8]| {
    // Layout: [partition, mode, echo_ctl, ...(f64 value, u8 incr_sel) × N...]
    if data.len() < 12 {
        return;
    }

    let partition_byte = data[0];
    let mode = data[1];
    let echo_ctl = data[2];
    let rest = &data[3..];

    // Parse 9-byte chunks: 8 bytes f64 value + 1 byte increment selector.
    let mut ops: Vec<Op> = rest
        .chunks_exact(9)
        .filter_map(|c| {
            let v = f64::from_le_bytes(c[..8].try_into().unwrap());
            if !v.is_finite() || v < 0.0 {
                return None;
            }
            let incr = decode_increment(c[8], mode);
            Some(Op { value: v, incr })
        })
        .collect();

    if ops.is_empty() {
        return;
    }

    // Echo phase: replay earlier values with exponentially growing
    // multipliers. This forces counter widening (B1→B2→…→U64) and,
    // for large enough totals, stat widening (S32→S64).
    let echo_passes = (echo_ctl % 4) as usize;
    let base_len = ops.len();
    for pass in 0..echo_passes {
        for i in 0..base_len {
            let Op { value, incr } = ops[i];
            let scale = 1u64 << ((pass * 4 + 2).min(30));
            let boosted = incr.saturating_mul(scale);
            // Cap at 2^40 to keep the fuzzer responsive.
            if boosted > 0 && boosted <= (1u64 << 40) {
                ops.push(Op { value, incr: boosted });
            }
        }
    }

    let split = (partition_byte as usize) % (ops.len() + 1);
    let (left, right) = ops.split_at(split);

    check_merge_same::<8>(left, right, true);
    check_merge_same::<16>(left, right, true);
    check_merge_different::<8, 16>(left, right, true);
    check_merge_different::<16, 8>(left, right, true);

    // Also test with literal mode disabled.
    check_merge_same::<8>(left, right, false);
    check_merge_same::<16>(left, right, false);
    check_merge_different::<8, 16>(left, right, false);
    check_merge_different::<16, 8>(left, right, false);
});

/// Map a selector byte into an increment that exercises different
/// bucket-width tiers. The `mode` byte picks a strategy so the fuzzer
/// can explore qualitatively different counter distributions.
fn decode_increment(sel: u8, mode: u8) -> u64 {
    match mode % 5 {
        // Powers of 2 — leap across width boundaries.
        0 => 1u64 << ((sel as u32) % 24),

        // Sit right at each width's maximum value.
        1 => match sel {
            0..=63   => 1,                // B1 max
            64..=95  => 3,                // B2 max
            96..=127 => 15,               // B4 max
            128..=159 => 255,             // U8 max
            160..=191 => 65535,           // U16 max
            192..=223 => (1 << 24) - 1,   // large U32
            224..=255 => 1 << 28,         // toward U32 overflow
        },

        // Ramp: selector scaled by mode, covering a smooth range.
        2 => (sel as u64 + 1) * ((mode as u64 / 5) + 1),

        // Just past each threshold — immediate widening trigger.
        3 => match sel % 6 {
            0 => 2,      // exceeds B1
            1 => 4,      // exceeds B2
            2 => 16,     // exceeds B4
            3 => 256,    // exceeds U8
            4 => 65536,  // exceeds U16
            _ => 1,
        },

        // Mostly unit inserts with rare large spikes.
        _ => if sel < 200 { 1 } else { 1u64 << (sel - 200) },
    }
}

// ---------------------------------------------------------------------------
// Merge checks
// ---------------------------------------------------------------------------

fn check_merge_same<const N: usize>(left: &[Op], right: &[Op], literal_mode: bool) {
    let mut h1 = Histogram::<N, P32>::new().with_literal_mode(literal_mode);
    let mut ok_left: Vec<Op> = Vec::new();
    for &op in left {
        if h1.update_by_incr(op.value, op.incr).is_ok() {
            ok_left.push(op);
        }
    }

    let mut h2 = Histogram::<N, P32>::new().with_literal_mode(literal_mode);
    let mut ok_right: Vec<Op> = Vec::new();
    for &op in right {
        if h2.update_by_incr(op.value, op.incr).is_ok() {
            ok_right.push(op);
        }
    }

    if h1.merge_from(&h2).is_err() {
        return;
    }

    ok_left.extend_from_slice(&ok_right);
    verify_histogram(&mut h1, &ok_left);
}

fn check_merge_different<const N: usize, const M: usize>(left: &[Op], right: &[Op], literal_mode: bool) {
    let mut h1 = Histogram::<N, P32>::new().with_literal_mode(literal_mode);
    let mut ok_left: Vec<Op> = Vec::new();
    for &op in left {
        if h1.update_by_incr(op.value, op.incr).is_ok() {
            ok_left.push(op);
        }
    }

    let mut h2 = Histogram::<M, P32>::new().with_literal_mode(literal_mode);
    let mut ok_right: Vec<Op> = Vec::new();
    for &op in right {
        if h2.update_by_incr(op.value, op.incr).is_ok() {
            ok_right.push(op);
        }
    }

    if h1.merge_from_other(&h2).is_err() {
        return;
    }

    ok_left.extend_from_slice(&ok_right);
    verify_histogram(&mut h1, &ok_left);
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

fn verify_histogram<const N: usize, P: Precision>(hist: &mut Histogram<N, P>, inserted: &[Op]) {
    // ── 1. count ──────────────────────────────────────────────────────
    let total_count: u64 = inserted.iter().map(|op| op.incr).sum();
    if total_count == 0 {
        assert_eq!(hist.count(), 0, "count should be 0 for empty input");
        return;
    }

    assert_eq!(
        hist.count(),
        total_count,
        "count mismatch: hist={} expected={}",
        hist.count(),
        total_count,
    );

    // ── 2. min / max ──────────────────────────────────────────────────
    // Histograms start at S32 (f32 precision for min/max). Even after
    // widening to S64, values stored during the S32 phase retain only
    // f32 precision. Compare at f32 granularity unconditionally.
    let expected_min = inserted.iter().map(|op| op.value).fold(f64::INFINITY, f64::min);
    let expected_max = inserted.iter().map(|op| op.value).fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(
        hist.min() as f32,
        expected_min as f32,
        "min mismatch",
    );
    assert_eq!(
        hist.max() as f32,
        expected_max as f32,
        "max mismatch",
    );

    // ── 3. zero count ─────────────────────────────────────────────────
    let non_zero_total: u64 = inserted
        .iter()
        .filter(|op| op.value != 0.0)
        .map(|op| op.incr)
        .sum();
    let expected_zero_count = total_count - non_zero_total;

    let count = hist.count();
    let scale = hist.scale();
    let buckets = hist.positive();
    let bucket_total: u64 = (0..buckets.len()).map(|i| buckets.at(i)).sum();

    assert!(
        bucket_total <= count,
        "bucket total ({}) exceeds count ({})",
        bucket_total,
        count,
    );
    let actual_zero_count = count - bucket_total;

    assert_eq!(
        actual_zero_count, expected_zero_count,
        "zero count mismatch",
    );

    // ── 4. bucket distribution at the final scale ─────────────────────
    if non_zero_total == 0 {
        assert_eq!(buckets.len(), 0, "expected no buckets for all-zero input");
        return;
    }

    let mapping = Mapping::new(scale).expect("reported scale should be valid");

    let mut expected: BTreeMap<i32, u64> = BTreeMap::new();
    for op in inserted {
        if op.value != 0.0 {
            let idx = mapping.map_to_index(op.value);
            *expected.entry(idx).or_insert(0) += op.incr;
        }
    }

    let exp_min_idx = *expected.keys().next().unwrap();
    let exp_max_idx = *expected.keys().last().unwrap();
    let exp_len = (exp_max_idx - exp_min_idx + 1) as u32;

    assert_eq!(
        buckets.offset(),
        exp_min_idx,
        "offset mismatch: hist={} expected={} (scale={})",
        buckets.offset(),
        exp_min_idx,
        scale,
    );

    assert_eq!(
        buckets.len(),
        exp_len,
        "bucket len mismatch: hist={} expected={} (scale={})",
        buckets.len(),
        exp_len,
        scale,
    );

    for pos in 0..buckets.len() {
        let idx = exp_min_idx + pos as i32;
        let exp_count = expected.get(&idx).copied().unwrap_or(0);
        let act_count = buckets.at(pos);
        assert_eq!(
            act_count, exp_count,
            "bucket[{}] (idx {}): hist={} expected={} (scale={}, width={:?})",
            pos, idx, act_count, exp_count, scale, buckets.width(),
        );
    }
}
