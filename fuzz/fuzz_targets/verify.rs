use std::collections::BTreeMap;

use otel_expohisto::{Histogram, Scale};

/// Verifies histogram state against expected (value, incr) operations.
///
/// Checks count, min, max, zero count, bucket distribution, and
/// per-bucket counts at the histogram's current scale.
pub fn verify_histogram<const N: usize>(hist: &mut Histogram<N>, ops: &[(f64, u64)], label: &str) {
    let total_count: u64 = ops.iter().map(|&(_, incr)| incr).sum();

    let v = hist.view();
    let stats = v.stats();
    assert_eq!(stats.count, total_count, "{label}: count mismatch");

    if total_count == 0 {
        assert_eq!(v.positive().len(), 0, "{label}: should have no buckets");
        return;
    }

    // min / max — only covers non-zero (bucketed) values
    let non_zero_ops = ops.iter().filter(|&&(v, _)| v != 0.0);
    let expected_min = non_zero_ops.clone().map(|&(v, _)| v).fold(f64::INFINITY, f64::min);
    let expected_max = non_zero_ops.clone().map(|&(v, _)| v).fold(f64::NEG_INFINITY, f64::max);
    if expected_min != f64::INFINITY {
        assert_eq!(stats.min, expected_min, "{label}: min mismatch");
        assert_eq!(stats.max, expected_max, "{label}: max mismatch");
    }

    // zero count
    let non_zero_total: u64 = ops
        .iter()
        .filter(|&&(v, _)| v != 0.0)
        .map(|&(_, incr)| incr)
        .sum();
    let expected_zero_count = total_count - non_zero_total;

    let count = stats.count;
    let scale = v.scale();
    let buckets = v.positive();
    let bucket_total: u64 = buckets.iter().sum();

    assert!(
        bucket_total <= count,
        "{label}: bucket total ({bucket_total}) exceeds count ({count})"
    );
    let actual_zero_count = count - bucket_total;
    assert_eq!(
        actual_zero_count, expected_zero_count,
        "{label}: zero count mismatch"
    );

    // bucket distribution at final scale
    if non_zero_total == 0 {
        assert_eq!(buckets.len(), 0, "{label}: expected no buckets");
        return;
    }

    let scale = Scale::new(scale).expect("reported scale should be valid");
    let mut expected: BTreeMap<i32, u64> = BTreeMap::new();

    // record_incr rounds subnormals to (biased_exp=1, significand=1).
    // Construct the f64 that results from this rounding so the oracle
    // maps it through the same code path as the histogram.
    // 52 is the IEEE 754 double-precision significand width.
    const SUBNORMAL_ROUNDED: f64 = f64::from_bits((1u64 << 52) | 1);

    for &(value, incr) in ops {
        if value != 0.0 {
            let mapped = if value.to_bits() >> 52 == 0 {
                scale.map_to_index(SUBNORMAL_ROUNDED)
            } else {
                scale.map_to_index(value)
            };
            *expected.entry(mapped).or_insert(0) += incr;
        }
    }

    let exp_min_idx = *expected.keys().next().unwrap();
    let exp_max_idx = *expected.keys().last().unwrap();
    let exp_len = (exp_max_idx - exp_min_idx + 1) as u32;

    assert_eq!(
        buckets.offset(),
        exp_min_idx,
        "{label}: offset mismatch (scale={scale})"
    );
    assert_eq!(
        buckets.len(),
        exp_len,
        "{label}: bucket len mismatch (scale={scale})"
    );

    for (pos, act_count) in buckets.iter().enumerate() {
        let idx = exp_min_idx + pos as i32;
        let exp_count = expected.get(&idx).copied().unwrap_or(0);
        assert_eq!(
            act_count,
            exp_count,
            "{label}: bucket[{pos}] (idx {idx}): hist={act_count} expected={exp_count} (scale={scale}, width={:?})",
            buckets.width()
        );
    }
}
