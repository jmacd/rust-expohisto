#![no_main]

use libfuzzer_sys::fuzz_target;
use rust_expohisto::{Histogram, Mapping, max_scale};
use rust_expohisto::{P32, P64, Precision};
use std::collections::BTreeMap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let values: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();

    if values.is_empty() {
        return;
    }

    // Test with literal mode (default).
    check_histogram::<8, P32>(&values, true);
    check_histogram::<16, P32>(&values, true);
    check_histogram::<8, P64>(&values, true);
    check_histogram::<16, P64>(&values, true);

    // Test with literal mode disabled (bucket mode from start).
    check_histogram::<8, P32>(&values, false);
    check_histogram::<16, P32>(&values, false);
    check_histogram::<8, P64>(&values, false);
    check_histogram::<16, P64>(&values, false);
});

/// Reference-oracle test: insert every value, then verify the histogram
/// state matches an independently-computed expectation.
fn check_histogram<const N: usize, P: Precision>(values: &[f64], literal_mode: bool) {
    let mut hist = Histogram::<N, P>::new().with_literal_mode(literal_mode);
    let mut inserted: Vec<f64> = Vec::new();

    for &v in values {
        if hist.update(v).is_ok() {
            inserted.push(v);
        }
    }

    if inserted.is_empty() {
        return;
    }

    // ── 1. count ──────────────────────────────────────────────────────
    assert_eq!(
        hist.count(),
        inserted.len() as u64,
        "count mismatch: hist={} expected={}",
        hist.count(),
        inserted.len(),
    );

    // ── 2. min / max ──────────────────────────────────────────────────
    let expected_min = inserted.iter().copied().fold(f64::INFINITY, f64::min);
    let expected_max = inserted.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if P::STAT_WORDS == 2 {
        assert_eq!(
            hist.min() as f32,
            expected_min as f32,
            "min mismatch (P32)",
        );
        assert_eq!(
            hist.max() as f32,
            expected_max as f32,
            "max mismatch (P32)",
        );
    } else {
        assert_eq!(hist.min(), expected_min, "min mismatch");
        assert_eq!(hist.max(), expected_max, "max mismatch");
    }

    // ── 3. zero count ─────────────────────────────────────────────────
    let non_zero: Vec<f64> = inserted.iter().copied().filter(|&v| v != 0.0).collect();
    let expected_zero_count = (inserted.len() - non_zero.len()) as u64;

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
    if non_zero.is_empty() {
        assert_eq!(buckets.len(), 0, "expected no buckets for all-zero input");
        return;
    }

    let mapping = Mapping::new(scale).expect("reported scale should be valid");

    // Build expected index → count map.
    let mut expected: BTreeMap<i32, u64> = BTreeMap::new();
    for &v in &non_zero {
        let idx = mapping.map_to_index(v);
        *expected.entry(idx).or_insert(0) += 1;
    }

    let exp_min_idx = *expected.keys().next().unwrap();
    let exp_max_idx = *expected.keys().last().unwrap();
    let exp_len = (exp_max_idx - exp_min_idx + 1) as u32;

    // Compare offset.
    assert_eq!(
        buckets.offset(),
        exp_min_idx,
        "offset mismatch: hist={} expected={} (scale={})",
        buckets.offset(),
        exp_min_idx,
        scale,
    );

    // Compare bucket count.
    assert_eq!(
        buckets.len(),
        exp_len,
        "bucket len mismatch: hist={} expected={} (scale={})",
        buckets.len(),
        exp_len,
        scale,
    );

    // Compare each bucket value.
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

    // ── 5. scale optimality ───────────────────────────────────────────
    // Verify that the scale is the highest one where the index span
    // fits in capacity. We use the *initial* B1 capacity (before any
    // widening) because the histogram only lowers scale when span
    // exceeds the capacity at whatever width it currently has.
    //
    // Counter overflow can force widen-steps (each costs 1 scale),
    // so we only assert scale <= span-optimal-scale.
    let stat_words = P::STAT_WORDS;
    let b1_cap = ((N - stat_words) * 64) as i32;

    // Find the highest scale where span fits at B1 capacity.
    let mut optimal = max_scale();
    for s in (rust_expohisto::MIN_SCALE..=max_scale()).rev() {
        if let Ok(m) = Mapping::new(s) {
            let indices: Vec<i32> = non_zero.iter().map(|&v| m.map_to_index(v)).collect();
            let lo = *indices.iter().min().unwrap();
            let hi = *indices.iter().max().unwrap();
            if hi - lo + 1 <= b1_cap {
                optimal = s;
                break;
            }
        }
    }

    assert!(
        scale <= optimal,
        "scale too high: hist={} but optimal span-fit is {} (b1_cap={})",
        scale,
        optimal,
        b1_cap,
    );
}
