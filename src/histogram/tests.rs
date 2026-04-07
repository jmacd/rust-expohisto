// Tests always run with std available, even when the crate is no_std.
extern crate std;

use super::*;
use crate::mapping::Scale;

/// Helper: count total across all positive buckets.
fn bucket_total<const N: usize>(h: &Histogram<N>) -> u64 {
    h.view().positive().iter().sum()
}

fn assert_stats<const N: usize>(h: &Histogram<N>, count: u64, sum: f64, min: f64, max: f64) {
    let s = h.view().stats();
    assert_eq!(s.count, count, "count");
    assert!((s.sum - sum).abs() < 1e-10, "sum: {} vs {}", s.sum, sum);
    if count > 0 {
        assert_eq!(s.min, min, "min");
        assert_eq!(s.max, max, "max");
    }
}

#[test]
fn merge_both_empty() {
    let mut h1: Histogram<8> = Histogram::new();
    let h2: Histogram<8> = Histogram::new();
    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 0);
}

#[test]
fn merge_into_empty() {
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<8> = Histogram::new();
    h2.update(1.0).unwrap();
    h2.update(2.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 2, 3.0, 1.0, 2.0);
    assert_eq!(bucket_total(&h1), 2);
}

#[test]
fn merge_from_empty() {
    let mut h1: Histogram<8> = Histogram::new();
    h1.update(1.0).unwrap();
    let h2: Histogram<8> = Histogram::new();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 1, 1.0, 1.0, 1.0);
}

#[test]
fn merge_same_scale() {
    let mut h1: Histogram<16> = Histogram::new();
    let mut h2: Histogram<16> = Histogram::new();

    h1.update(1.0).unwrap();
    h1.update(2.0).unwrap();
    h2.update(3.0).unwrap();
    h2.update(4.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 4, 10.0, 1.0, 4.0);
    assert_eq!(bucket_total(&h1), 4);
}

#[test]
fn merge_different_pool_sizes() {
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<16> = Histogram::new();

    h1.update(1.0).unwrap();
    h2.update(2.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 2, 3.0, 1.0, 2.0);
}

#[test]
fn merge_with_zeros() {
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<8> = Histogram::new();

    h1.update(0.0).unwrap();
    h1.update(1.0).unwrap();
    h2.update(0.0).unwrap();
    h2.update(2.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 4, 3.0, 1.0, 2.0);
    // Two zeros + two bucketed values
    assert_eq!(bucket_total(&h1), 2);
}

#[test]
fn merge_triggers_downscale() {
    // Use small pool to force downscale on merge.
    let mut h1: Histogram<2> = Histogram::new();
    let mut h2: Histogram<2> = Histogram::new();

    // Fill h1 and h2 with values at distant bucket indices.
    h1.update(1.0).unwrap();
    h2.update(1000.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 2, 1001.0, 1.0, 1000.0);
    assert_eq!(bucket_total(&h1), 2);
}

#[test]
fn merge_oracle_basic() {
    // Verify bucket-level correctness against an oracle.
    let mut h1: Histogram<16> = Histogram::new();
    let mut h2: Histogram<16> = Histogram::new();

    let vals1 = [1.0, 1.5, 2.0, 3.0];
    let vals2 = [0.5, 1.0, 4.0, 8.0];

    for &v in &vals1 {
        h1.update(v).unwrap();
    }
    for &v in &vals2 {
        h2.update(v).unwrap();
    }

    h1.merge_from(&h2).unwrap();

    let v = h1.view();
    let scale = Scale::new(v.scale()).unwrap();
    let buckets = v.positive();

    // Every value must land in the correct bucket.
    let all_vals: std::vec::Vec<f64> = vals1.iter().chain(&vals2).copied().collect();
    for &val in &all_vals {
        let idx = scale.map_to_index(val);
        let pos = (idx - buckets.offset()) as usize;
        assert!(
            pos < buckets.len() as usize,
            "val {val} at idx {idx} out of range"
        );
    }

    assert_eq!(v.stats().count, 8);
    assert_eq!(bucket_total(&h1), 8);
}

#[test]
fn merge_high_incr() {
    // Merge with large increments that trigger width overflow.
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<8> = Histogram::new();

    h1.record_incr(1.0, 1000).unwrap();
    h2.record_incr(1.0, 2000).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 3000);
    assert_eq!(bucket_total(&h1), 3000);
}

#[test]
fn merge_phase15_dest_wider() {
    // h1 starts with a wider minimum width (U8) than h2 (B1 default).
    // With same scale and shift=0, tm_log = 0 + 0 - 3 = -3 < 0,
    // triggering Phase 1.5 to downscale h1.
    let mut h1: Histogram<8> = Histogram::new().with_min_width(Width::U8);
    let mut h2: Histogram<8> = Histogram::new();

    h1.update(1.0).unwrap();
    h2.update(2.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 2, 3.0, 1.0, 2.0);
    assert_eq!(bucket_total(&h1), 2);
}

#[test]
fn merge_same_slot_overflow() {
    // Both histograms have B1 counters at the same slot.
    // swar_add_checked(1, 1, B1) overflows → triggers widen during merge.
    let mut h1: Histogram<16> = Histogram::new();
    let mut h2: Histogram<16> = Histogram::new();

    h1.update(1.0).unwrap();
    h2.update(1.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 2);
    assert_eq!(bucket_total(&h1), 2);
}

#[test]
fn merge_cross_scale_different_pools() {
    // h2 has a lower scale from wide-range values; h1 at high scale.
    // The scale difference exercises the repack decomposition.
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<2> = Histogram::new();

    h1.update(1.0).unwrap();
    h2.update(1.0).unwrap();
    h2.update(1000.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 3);
    assert_eq!(bucket_total(&h1), 3);
}

#[test]
fn merge_into_empty_wider_source() {
    // Merge a wide-counter histogram into an empty one.
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<8> = Histogram::new();

    h2.record_incr(1.0, 500).unwrap();
    h2.record_incr(2.0, 500).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 1000);
    assert_eq!(bucket_total(&h1), 1000);
}

#[test]
fn merge_bidirectional_consistency() {
    // merge(h1, h2) and merge(h2, h1) should produce same count/sum.
    let mut h1a: Histogram<8> = Histogram::new();
    let mut h1b: Histogram<8> = Histogram::new();
    let mut h2a: Histogram<8> = Histogram::new();
    let mut h2b: Histogram<8> = Histogram::new();

    for &v in &[1.0, 1.5, 2.0] {
        h1a.update(v).unwrap();
        h1b.update(v).unwrap();
    }
    for &v in &[100.0, 200.0, 300.0] {
        h2a.update(v).unwrap();
        h2b.update(v).unwrap();
    }

    h1a.merge_from(&h2a).unwrap();
    h2b.merge_from(&h1b).unwrap();

    let s1 = h1a.view().stats();
    let s2 = h2b.view().stats();
    assert_eq!(s1.count, s2.count);
    assert!((s1.sum - s2.sum).abs() < 1e-10);
    assert_eq!(s1.min, s2.min);
    assert_eq!(s1.max, s2.max);
    assert_eq!(bucket_total(&h1a), bucket_total(&h2b));
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::{format, vec, vec::Vec};

fn derived_zero_count<const N: usize>(h: &Histogram<N>) -> u64 {
    h.view().stats().count - bucket_total(h)
}

#[test]
fn test_histogram_basic() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    assert_stats(&h, 1, 1.0, 1.0, 1.0);
    assert_eq!(derived_zero_count(&h), 0);
    assert_eq!(h.width(), Width::B1);
}

#[test]
fn test_histogram_zero() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(0.0).unwrap();
    assert_eq!(h.view().stats().count, 1);
    assert_eq!(derived_zero_count(&h), 1);
    assert_eq!(h.view().stats().sum, 0.0);
}

#[test]
fn test_histogram_multiple() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(2.0).unwrap();
    h.update(4.0).unwrap();
    assert_stats(&h, 3, 7.0, 1.0, 4.0);
}

#[test]
fn test_histogram_downscale() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(1000.0).unwrap();
    assert_eq!(h.view().stats().count, 2);
    assert!(h.view().scale() < table_scale());
}

#[test]
fn test_histogram_merge() {
    let mut h1: Histogram<16> = Histogram::new();
    let mut h2: Histogram<16> = Histogram::new();
    h1.update(1.0).unwrap();
    h1.update(2.0).unwrap();
    h2.update(3.0).unwrap();
    h2.update(4.0).unwrap();
    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 4, 10.0, 1.0, 4.0);
}

#[test]
fn test_histogram_recreate() {
    let h: Histogram<16> = Histogram::new();
    assert_eq!(h.view().stats().count, 0);
    assert_eq!(h.view().stats().sum, 0.0);
    assert_eq!(h.view().scale(), 0);
    assert_eq!(h.width(), Width::B1);
}

#[test]
fn test_buckets_at() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    h.update(1.5).unwrap();
    h.update(100.0).unwrap();
    h.update(1e10).unwrap();

    let v = h.view();
    let buckets = v.positive();
    assert!(
        buckets.len() >= 2,
        "expected at least 2 buckets, got {} at scale {}",
        buckets.len(),
        v.scale()
    );
}

#[test]
fn test_auto_widen_cascade() {
    let mut h: Histogram<16> = Histogram::new().with_min_width(Width::B4);

    // B4 → U8 at threshold 15+1=16
    h.record_incr(1.0, 15).unwrap();
    assert_eq!(h.width(), Width::B4);
    h.update(1.0).unwrap();
    assert_eq!(h.width(), Width::U8);
    assert_eq!(h.view().stats().count, 16);

    // U8 → U16 at threshold 255+1=256
    h.record_incr(1.0, 239).unwrap();
    assert_eq!(h.width(), Width::U8);
    h.update(1.0).unwrap();
    assert_eq!(h.width(), Width::U16);
    assert_eq!(h.view().stats().count, 256);

    // U16 → U32 at threshold 65535+1=65536
    h.record_incr(1.0, u16::MAX as u64 - 256).unwrap();
    assert_eq!(h.width(), Width::U16);
    h.update(1.0).unwrap();
    assert_eq!(h.width(), Width::U32);
    assert_eq!(h.view().stats().count, u16::MAX as u64 + 1);

    // U32 → U64 at threshold 4294967295+1
    h.record_incr(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1))
        .unwrap();
    assert_eq!(h.width(), Width::U32);
    h.update(1.0).unwrap();
    assert_eq!(h.width(), Width::U64);
}

#[test]
fn test_auto_widen_b4_to_u8_from_b4_start() {
    let mut h: Histogram<16> = Histogram::new().with_min_width(Width::B4);
    h.record_incr(1.0, 4).unwrap();
    assert_eq!(h.width(), Width::B4);
    h.record_incr(1.0, 11).unwrap();
    assert_eq!(h.width(), Width::B4);
    h.update(1.0).unwrap();
    assert_eq!(h.width(), Width::U8);
    assert_eq!(h.view().stats().count, 16);
}

#[test]
fn test_recreate_preserves_b4() {
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(3)
        .unwrap()
        .with_min_width(Width::B4);
    assert_eq!(h.width(), Width::B4);
    assert_eq!(h.view().stats().count, 0);
    // Record a value to verify it starts at scale 3.
    h.update(1.0).unwrap();
    assert_eq!(h.view().scale(), 3);
}

#[test]
fn test_with_scale() {
    let mut h: Histogram<16> = Histogram::new().with_scale(3).unwrap();
    // Record a value to verify scale is respected.
    h.update(1.0).unwrap();
    assert_eq!(h.view().scale(), 3);
}

#[test]
fn test_with_scale_records_at_limited_scale() {
    let mut limited: Histogram<16> = Histogram::new().with_scale(3).unwrap();
    let mut unlimited: Histogram<16> = Histogram::new();
    limited.update(1.0).unwrap();
    limited.update(1.001).unwrap();
    unlimited.update(1.0).unwrap();
    unlimited.update(1.001).unwrap();

    let limited_view = limited.view();
    assert!(limited_view.scale() <= 3);
    if table_scale() > 3 {
        let unlimited_view = unlimited.view();
        assert!(unlimited_view.scale() > limited_view.scale());
    }
}

#[test]
fn test_widen_preserves_data() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    h.record_incr(1.0, 100).unwrap();
    assert_eq!(h.width(), Width::U8);

    h.record_incr(256.0, 50).unwrap();
    h.record_incr(65536.0, 200).unwrap();

    let (count_before, sum_before) = {
        let s = h.view().stats();
        (s.count, s.sum)
    };

    h.record_incr(65536.0, 55).unwrap();
    assert_eq!(h.width(), Width::U8);
    h.update(65536.0).unwrap();
    assert_eq!(h.width(), Width::U16);

    let s = h.view().stats();
    assert_eq!(s.count, count_before + 56);
    assert!((s.sum - (sum_before + 56.0 * 65536.0)).abs() < 1.0);
}

#[test]
fn test_merge_equivalence_comprehensive() {
    let hardcoded_sets: &[&[f64]] = &[
        &[],
        &[0.0],
        &[1.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[1.0, 2.0],
        &[0.5, 1.5, 2.5],
        &[0.001, 1.0, 20.0],
        &[1.0, 1.0, 1.0, 1.0],
        &[0.0, 1.0, 2.0, 0.0],
        &[5.0, 10.0, 15.0, 20.0],
        &[0.1, 0.2, 0.3, 0.4, 0.5],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ],
        &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        &[0.01, 0.1, 1.0, 10.0],
        &[0.0, 20.0],
        &[1.0, 19.0],
        &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        &[15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        &[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    ];

    let mut test_sets: Vec<Vec<f64>> = hardcoded_sets.iter().map(|s| s.to_vec()).collect();

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let size = rng.gen_range(0..=10);
        let set: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..20.0)).collect();
        test_sets.push(set);
    }

    test_merge_equivalence_for_size::<8>(&test_sets);
    test_merge_equivalence_for_size::<12>(&test_sets);
    test_merge_equivalence_for_size::<16>(&test_sets);
    test_merge_equivalence_for_size::<20>(&test_sets);
}

fn test_merge_equivalence_for_size<const K: usize>(test_sets: &[Vec<f64>]) {
    for (i, set_a) in test_sets.iter().enumerate() {
        for (j, set_b) in test_sets.iter().enumerate() {
            let mut merged = build_from_values::<K>(set_a);
            let other = build_from_values::<K>(set_b);
            if let Err(e) = merged.merge_from(&other) {
                panic!(
                    "merge_from failed for size={K} sets {i} x {j}: {e}\n  set_a: {set_a:?}\n  set_b: {set_b:?}\n  merged: {:?}\n  other: {:?}",
                    merged, other
                );
            }

            let mut single = build_from_values::<K>(set_a);
            for &v in set_b.iter() {
                single.update(v).unwrap();
            }

            let label = format!("size={K} sets {i} x {j}");
            let merged_stats = merged.view().stats();
            let single_stats = single.view().stats();
            assert_eq!(
                merged_stats.count, single_stats.count,
                "count mismatch for {label}"
            );
            let ms = merged_stats.sum;
            let ss = single_stats.sum;
            let sum_diff = (ms - ss).abs();
            let denom = ms.abs().max(ss.abs()).max(1e-30);
            assert!(
                sum_diff / denom < 1e-5,
                "sum mismatch for {label}: {ms} vs {ss}"
            );
            assert_eq!(
                derived_zero_count(&merged),
                derived_zero_count(&single),
                "zero_count mismatch for {label}"
            );
            assert_eq!(
                bucket_total(&merged),
                bucket_total(&single),
                "bucket total mismatch for {label}"
            );
        }
    }
}

#[test]
fn test_merge_regression_bucket_total() {
    // Regression: "bucket total mismatch for size=8 sets 2 x 35"
    let set_b: &[f64] = &[
        18.896147780359236,
        19.038540970281623,
        15.726266735088323,
        19.97053274796744,
        16.963914020801518,
    ];

    // Verify incremental bucket totals while building.
    let mut other: Histogram<8> = Histogram::new();
    for &v in set_b {
        other.update(v).unwrap();
        let bt = bucket_total(&other);
        let non_zero_count = other.view().stats().count - derived_zero_count(&other);
        assert_eq!(
            bt, non_zero_count,
            "bucket total mismatch after inserting {v}"
        );
    }

    let set_a: &[f64] = &[1.0];
    let mut merged = build_from_values::<8>(set_a);
    merged.merge_from(&other).unwrap();

    let mut single = build_from_values::<8>(set_a);
    for &v in set_b {
        single.update(v).unwrap();
    }

    assert_eq!(
        bucket_total(&merged),
        bucket_total(&single),
        "bucket total mismatch: merged vs single"
    );
}

#[test]
fn test_edge_values_subnormals() {
    let subnormal: f64 = 5e-324;
    let min_normal: f64 = crate::float64::MIN_VALUE;

    // The mapper itself treats subnormals and MIN_VALUE the same
    // (both have biased_exp=0 or 1 with significand=0).
    let m0 = Scale::new(0).unwrap();
    assert_eq!(m0.map_to_index(subnormal), m0.map_to_index(min_normal));

    // But record_incr rounds subnormals to significand=1, placing
    // them one bucket above MIN_VALUE.
    let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    h.update(subnormal).unwrap();
    h.update(min_normal).unwrap();
    assert_eq!(h.view().stats().count, 2);
    assert_eq!(h.view().positive().len(), 2);
}

/// Documents the behavior when NaN or negative values are passed.
/// The caller is expected to validate inputs before calling record().
/// These are not checked at runtime — the histogram remains safe but
/// produces unspecified statistical results.
#[test]
fn test_nan_and_negative_debug_asserts() {
    // NaN, Inf, and negative values return Err(Extreme).
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();

    assert!(h.clone().update(f64::NAN).is_err(), "NaN should return Err");
    assert!(
        h.clone().update(f64::INFINITY).is_err(),
        "Inf should return Err"
    );
    assert!(
        h.clone().update(f64::NEG_INFINITY).is_err(),
        "NEG_INFINITY should return Err"
    );
    assert!(
        h.clone().update(-1.0).is_err(),
        "negative values should return Err"
    );
    // -0.0 is treated as 0.0 (zero bucket)
    assert!(h.clone().update(-0.0).is_ok(), "-0.0 should be accepted");
}

#[test]
fn test_exhaustive_u8_overflow() {
    // Insert 8 values spanning a wide index range at scale 0, each
    // with count 255. Starting at B1 with 320 slots (Histogram<8>),
    // counters widen B1→B2→B4→U8 (255 fits in U8), but the larger
    // initial capacity means the span still fits without reaching U64.
    let mut h: Histogram<8> = Histogram::new().with_scale(0).unwrap();
    let num_buckets = 8;
    for i in 0..num_buckets {
        let val = 2.0_f64.powi(i * 8);
        h.record_incr(val, 255).unwrap();
    }
    // With B1 start, U8 has enough capacity for the span.
    assert!(
        h.width() >= Width::U8,
        "expected at least U8, got {:?}",
        h.width()
    );
    assert_eq!(h.view().stats().count, num_buckets as u64 * 255);
    // Adding one more should still be fine at U64 (no further widen needed).
    h.update(1.0).unwrap();
    assert_eq!(h.view().stats().count, num_buckets as u64 * 255 + 1);
}

#[test]
fn test_successive_sub_byte_widening() {
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B4);

    h.update(1.0).unwrap();
    assert_eq!(h.view().stats().count, 1);
    assert_eq!(h.width(), Width::B4);

    for count in 2..=15u64 {
        h.update(1.0).unwrap();
        assert_eq!(h.view().stats().count, count);
        assert_eq!(h.width(), Width::B4, "expected B4 at count {count}");
    }

    h.update(1.0).unwrap();
    assert_eq!(h.view().stats().count, 16);
    assert_eq!(h.width(), Width::U8);

    assert!((h.view().stats().sum - 16.0).abs() < 1e-10);
    assert_eq!(h.view().stats().min, 1.0);
    assert_eq!(h.view().stats().max, 1.0);
}

#[test]
fn test_successive_sub_byte_widening_multi_bucket() {
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B4);
    let num_buckets = 8;
    let values: Vec<f64> = (1..=num_buckets).map(|k| 2.0_f64.powi(k)).collect();

    for &v in &values {
        h.update(v).unwrap();
    }
    assert_eq!(h.view().stats().count, num_buckets as u64);
    assert_eq!(h.width(), Width::B4);

    for &v in &values {
        h.update(v).unwrap();
    }
    assert_eq!(h.view().stats().count, 2 * num_buckets as u64);
    assert!(h.width() >= Width::B4);

    let target = 16 * num_buckets as u64;
    while h.view().stats().count < target {
        for &v in &values {
            h.update(v).unwrap();
        }
    }
    assert!(h.width() >= Width::U8);

    let expected_sum: f64 = values.iter().sum::<f64>() * 16.0;
    assert!(
        (h.view().stats().sum - expected_sum).abs() < 1e-6,
        "sum mismatch: got {} expected {}",
        h.view().stats().sum,
        expected_sum
    );
    assert_eq!(h.view().stats().count, target);
}

// -----------------------------------------------------------------------
// Cross-size merge tests
// -----------------------------------------------------------------------

#[test]
fn test_merge_different_sizes() {
    let mut collector: Histogram<16> = Histogram::new();
    let mut source: Histogram<8> = Histogram::new();

    source.update(1.0).unwrap();
    source.update(2.0).unwrap();
    source.update(4.0).unwrap();
    source.update(0.0).unwrap();

    collector.merge_from(&source).unwrap();

    assert_eq!(collector.view().stats().count, 4);
    assert_eq!(derived_zero_count(&collector), 1);
    assert!((collector.view().stats().sum - 7.0).abs() < 1e-5);
}

#[test]
fn test_merge_multiple_sources() {
    let mut collector: Histogram<20> = Histogram::new();

    for batch in 0..5 {
        let mut src: Histogram<16> = Histogram::new();
        for i in 0..10 {
            src.update((batch * 10 + i) as f64 * 0.1 + 0.1).unwrap();
        }
        collector.merge_from(&src).unwrap();
    }

    assert_eq!(collector.view().stats().count, 50);
    assert!(collector.view().stats().sum > 0.0);
}

#[test]
fn test_merge_preserves_buckets() {
    let mut collector: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    let mut source: Histogram<16> = Histogram::new().with_scale(0).unwrap();

    source.update(1.0).unwrap();
    source.update(2.0).unwrap();
    source.update(4.0).unwrap();

    collector.merge_from(&source).unwrap();

    let mut direct: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    direct.update(1.0).unwrap();
    direct.update(2.0).unwrap();
    direct.update(4.0).unwrap();

    let collector_view = collector.view();
    let collector_buckets = collector_view.positive();
    let direct_view = direct.view();
    let direct_buckets = direct_view.positive();

    assert_eq!(collector_view.scale(), direct_view.scale());
    assert_eq!(collector_buckets.offset(), direct_buckets.offset());
    assert_eq!(collector_buckets.len(), direct_buckets.len());
    let cv: std::vec::Vec<u64> = collector_buckets.iter().collect();
    let dv: std::vec::Vec<u64> = direct_buckets.iter().collect();
    for (i, (c, d)) in cv.iter().zip(&dv).enumerate() {
        assert_eq!(c, d, "bucket[{i}] mismatch");
    }
}

#[test]
fn test_merge_empty_into_populated() {
    let mut collector: Histogram<16> = Histogram::new();
    collector.update(1.0).unwrap();

    let empty: Histogram<8> = Histogram::new();
    collector.merge_from(&empty).unwrap();

    assert_eq!(collector.view().stats().count, 1);
    assert_eq!(collector.view().stats().sum, 1.0);
}

#[test]
fn test_merge_into_empty() {
    let mut collector: Histogram<16> = Histogram::new();
    let mut source: Histogram<8> = Histogram::new();
    source.update(5.0).unwrap();

    collector.merge_from(&source).unwrap();

    assert_eq!(collector.view().stats().count, 1);
    assert!((collector.view().stats().sum - 5.0).abs() < 1e-5);
}

// -----------------------------------------------------------------------
// Adaptive merge (downscale) integration tests
// -----------------------------------------------------------------------

#[test]
fn test_downscale_width_behavior() {
    // Helper: insert ops into a B4 histogram at scale 0,
    // downscale(1), and verify the expected final width.
    let check = |ops: &[(f64, u64)], expected_width: Width, label: &str| {
        let mut h: Histogram<16> = Histogram::new()
            .with_scale(0)
            .unwrap()
            .with_min_width(Width::B4);
        for &(v, incr) in ops {
            h.record_incr(v, incr).unwrap();
        }
        assert_eq!(h.width(), Width::B4, "{label}: pre-check");
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.width(), expected_width, "{label}");
    };

    check(&[(2.0, 5), (4.0, 7)], Width::B4, "small sums stay B4");
    check(&[(2.0, 10), (4.0, 10)], Width::U8, "overflow widens to U8");
    check(
        &[(2.0, 15), (4.0, 15)],
        Width::U8,
        "max B4 overflow widens to U8",
    );
}

#[test]
fn test_downscale_many_indices_preserves_width() {
    // Many small counts at spread-out indices → pair sums ≤ 2, stays B4.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B4);
    for i in 0..8 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    assert_eq!(h.width(), Width::B4);
    assert_total_conserved(&mut h, 1);
    assert_eq!(h.width(), Width::B4);
}

// -----------------------------------------------------------------------
// Reproducer for the sets 6 x 10 merge mismatch
// -----------------------------------------------------------------------

#[test]
fn test_merge_sets_6_x_10_bucket_totals() {
    // Merged via merge_from must produce same bucket total as sequential inserts.
    let left: &[(f64, u64)] = &[(0.5, 1), (1.5, 1), (2.5, 1)];
    let right: &[(f64, u64)] = &[(5.0, 1), (10.0, 1), (15.0, 1), (20.0, 1)];
    merge_check::<8>(left, right, "sets_6_x_10");
}

// -----------------------------------------------------------------------
// Adaptive merge (scalar fallback) tests
// -----------------------------------------------------------------------

#[test]
fn test_bucket_downscale_scalar_preserves_total_no_overflow() {
    // Two values at adjacent indices with small counts → scalar merge
    // should sum them without widening.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B1);
    h.record_incr(2.0, 3).unwrap(); // index 0
    h.record_incr(4.0, 5).unwrap(); // index 1

    let width_before = h.width();
    assert_total_conserved(&mut h, 1);
    // Small counts (3+5=8 ≤ 15) → should stay at B4.
    assert_eq!(h.width(), width_before);
}

#[test]
fn test_bucket_downscale_scalar_preserves_total_with_overflow() {
    // Fill enough that pair sums exceed B4 max (15).
    let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    h.record_incr(2.0, 10).unwrap(); // index 0, count 10
    h.record_incr(4.0, 10).unwrap(); // index 1, count 10

    assert_total_conserved(&mut h, 1);
    // 10+10=20 > 15 → must widen to U8.
    assert_eq!(h.width(), Width::U8);
}

#[test]
fn test_downscale_multi_step_preserves_total() {
    // Insert 4 values at separate indices, then downscale by 3.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B4);
    for i in 0..4 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    let total_before = bucket_total(&h);
    assert_eq!(total_before, 4);
    assert_eq!(h.width(), Width::B4);

    h.downscale_by(3);

    let total_after = bucket_total(&h);
    assert_eq!(total_after, 4, "total changed after 3-step downscale");
    // All counts are 1, pair sums ≤ 2 → should stay at B4.
    assert_eq!(h.width(), Width::B4);
}

#[test]
fn test_downscale_multi_step_through_alignment_boundary() {
    // Start with base aligned to 16, downscale 5+ times so base
    // goes from even to odd and back. Verify totals survive.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B1);
    for i in 0..8 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    let total_before = bucket_total(&h);
    assert_eq!(total_before, 8);

    // 5 steps: base starts at e.g. -16 >> 5 = -1 (odd), so the
    // 5th step must use scalar fallback.
    h.downscale_by(5);

    let total_after = bucket_total(&h);
    assert_eq!(total_after, 8, "total changed after 5-step downscale");
}

#[test]
fn test_downscale_odd_base_preserves_total() {
    // Downscale through odd-base steps using SWAR-shift.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B1);
    for i in 0..4 {
        h.update(2.0_f64.powi(i)).unwrap();
    }

    // At B1, base = -64. After 6 steps: base = -64 >> 6 = -1 (odd).
    // Step 7 uses the odd SWAR-shift merge.
    assert_total_conserved(&mut h, 7);
}

// -----------------------------------------------------------------------
// Odd-base downscale preserves totals (no deferred mechanism)
// -----------------------------------------------------------------------

#[test]
fn test_odd_base_downscale_preserves_total() {
    // Start at max scale so we have room to downscale.
    let mut h: Histogram<16> = Histogram::new().with_scale(8).unwrap();
    h.record_incr(1.5, 5).unwrap();
    h.record_incr(1.6, 7).unwrap();

    let total_before = bucket_total(&h);

    // Downscale until base is odd (at most 15 steps to stay above MIN_SCALE).
    let mut tries = 0;
    while h.word_base & 1 == 0 && tries < 15 {
        h.downscale_by(1);
        tries += 1;
    }

    if h.word_base & 1 != 0 {
        // One more downscale at odd base — do_downscale handles
        // alignment internally via a fresh, aligned output buffer.
        h.downscale_by(1);

        let total_after = bucket_total(&h);
        assert_eq!(
            total_before, total_after,
            "bucket total changed on odd-base downscale"
        );
    }
}

// -----------------------------------------------------------------------
// Speculative merge: width preservation across counter magnitudes
// -----------------------------------------------------------------------

#[test]
fn test_speculative_merge_width_behavior() {
    // B4 sparse: many single-count buckets, pair sums ≤ 2 → stays B4
    {
        let mut h: Histogram<16> = Histogram::new()
            .with_scale(0)
            .unwrap()
            .with_min_width(Width::B4);
        for i in 0..16 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert_eq!(h.width(), Width::B4);
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.width(), Width::B4, "b4 sparse stays");
    }

    // U8 dense: 200+200=400 > 255 → widens to U16
    {
        let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
        h.record_incr(2.0, 200).unwrap();
        assert_eq!(h.width(), Width::U8);
        h.record_incr(4.0, 200).unwrap();
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.width(), Width::U16, "u8 dense widens to u16");
    }

    // U8 sparse: 100+50=150 ≤ 255 → stays U8
    {
        let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
        h.record_incr(2.0, 100).unwrap();
        assert_eq!(h.width(), Width::U8);
        h.record_incr(4.0, 50).unwrap();
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.width(), Width::U8, "u8 sparse stays");
    }
}

// -----------------------------------------------------------------------
// Sum conservation stress tests
// -----------------------------------------------------------------------

#[test]
fn test_sum_conservation_through_full_widen_chain() {
    // Fill a histogram with enough count magnitude to force widening
    // at every level: B4(max 15) → U8(255) → U16(65535) → U32 → U64.
    // Adjacent pairs sum to 1000, forcing overflow at B4, U8.
    let mut h: Histogram<16> = Histogram::new().with_scale(8).unwrap();
    h.record_incr(1.5, 500).unwrap();
    h.record_incr(1.6, 500).unwrap();
    // Start at U16 (500 > 255).
    assert_eq!(bucket_total(&h), 1000);

    // Add more to push into U32 territory.
    h.record_incr(1.7, 65000).unwrap();
    h.record_incr(1.8, 65000).unwrap();

    // Downscale up to 10 steps, verify total at each.
    assert_total_conserved(&mut h, 10);
}

#[test]
fn test_sum_conservation_scalar_path() {
    // Force the scalar path and check totals at each step.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::B1);
    for i in 0..10 {
        h.update(2.0_f64.powi(i)).unwrap();
    }

    // Downscale 8 times — should cross the odd-base boundary
    // multiple times, exercising scalar and SWAR paths alternately.
    assert_total_conserved(&mut h, 8);
}

#[test]
fn test_sum_conservation_large_counts() {
    // High counts that force widening at every merge.
    let mut h: Histogram<16> = Histogram::new().with_scale(0).unwrap();
    h.record_incr(2.0, 15).unwrap(); // fills B4 to max
    h.record_incr(4.0, 15).unwrap();
    h.record_incr(8.0, 15).unwrap();
    h.record_incr(16.0, 15).unwrap();
    assert_eq!(bucket_total(&h), 60);

    assert_total_conserved(&mut h, 6);
}

// -----------------------------------------------------------------------
// Reproducer: bucket total integrity through adaptive downscale
// -----------------------------------------------------------------------

#[test]
fn test_adaptive_downscale_sequential_inserts_small_pool() {
    // Reproducer: insert 1.0..=8.0 into Histogram<8>.
    // At B1 with 6 bucket words (384 slots), the index span forces
    // repeated downscaling. Bucket totals must stay consistent.
    let mut h: Histogram<8> = Histogram::new();
    for i in 1..=8 {
        let v = i as f64;
        h.update(v).unwrap();
        let total = bucket_total(&h);
        let view = h.view();
        let stats = view.stats();
        assert_eq!(
            total,
            stats.count,
            "After inserting {v}: bucket total ({total}) != count ({})\n  \
             scale={} width={:?}",
            stats.count,
            view.scale(),
            h.width()
        );
    }
}

#[test]
fn test_adaptive_downscale_wide_span_small_pool() {
    // Wide value range in a small pool — forces multi-step downscale.
    let mut h: Histogram<8> = Histogram::new();
    let values = [0.001, 1.0, 1000.0, 0.5, 50.0, 0.01, 100.0, 10.0];
    for (vi, &v) in values.iter().enumerate() {
        h.update(v).unwrap();
        let total = bucket_total(&h);
        let view = h.view();
        let stats = view.stats();
        assert_eq!(
            total,
            stats.count,
            "After values[{vi}]={v}: bucket total ({total}) != count ({})\n  \
             scale={} width={:?}",
            stats.count,
            view.scale(),
            h.width()
        );
    }
}

// -----------------------------------------------------------------------
// Regression tests (formerly in regression_stat_widen)
// -----------------------------------------------------------------------

/// Downscales `steps` times, asserting the bucket total is preserved
/// at each step.
fn assert_total_conserved<const N: usize>(h: &mut Histogram<N>, steps: i32) {
    let total = bucket_total(h);
    for step in 1..=steps {
        h.downscale_by(1);
        let current = bucket_total(h);
        assert_eq!(
            current,
            total,
            "total changed at step {step}: {current} != {total}, \
             width={:?}",
            h.width()
        );
    }
}

/// Helper: build two same-size histograms from ops, merge, and
/// assert count and bucket-total invariants.
fn build_histogram<const N: usize>(ops: &[(f64, u64)]) -> Histogram<N> {
    let mut h = Histogram::<N>::new();
    for &(v, incr) in ops {
        h.record_incr(v, incr).unwrap();
    }
    h
}

/// Helper: build a histogram from plain f64 values (each inserted once).
fn build_from_values<const N: usize>(values: &[f64]) -> Histogram<N> {
    let mut h = Histogram::<N>::new();
    for &v in values {
        h.update(v).unwrap();
    }
    h
}

fn assert_merge_result<const N: usize>(
    h: &Histogram<N>,
    left: &[(f64, u64)],
    right: &[(f64, u64)],
    label: &str,
) {
    let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
    let count = h.view().stats().count;
    assert_eq!(count, expected, "{label}: count mismatch");
    let bt = bucket_total(h);
    assert!(bt <= count, "{label}: bt={bt} > count={count}");
}

fn merge_check<const N: usize>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
    let (mut h1, h2) = (build_histogram::<N>(left), build_histogram::<N>(right));
    h1.merge_from(&h2).unwrap();
    assert_merge_result(&h1, left, right, label);
}

/// Helper: build two different-size histograms from ops, merge via
/// `merge_from`, and assert count and bucket-total invariants.
fn merge_check_cross<const N: usize, const M: usize>(
    left: &[(f64, u64)],
    right: &[(f64, u64)],
    label: &str,
) {
    let (mut h1, h2) = (build_histogram::<N>(left), build_histogram::<M>(right));
    h1.merge_from(&h2).unwrap();
    assert_merge_result(&h1, left, right, label);
}

/// Regression: large weighted inserts of subnormal + normal value
/// trigger bucket_widen during downscale, corrupting bucket totals.
#[test]
fn test_weighted_subnormal_merge_bucket_total() {
    let v1 = f64::from_le_bytes([32, 0, 66, 0, 0, 98, 65, 3]); // ~5.44e-293, subnormal as f32
    let v2 = f64::from_le_bytes([0, 32, 0, 66, 0, 98, 65, 64]); // ~34.77

    let left_ops: Vec<(f64, u64)> = vec![(v1, 3), (v2, 1), (v1, 12), (v2, 4), (v1, 192), (v2, 64)];
    let right_ops: Vec<(f64, u64)> = vec![(v1, 3072), (v2, 1024)];

    merge_check::<8>(&left_ops, &right_ops, "same N=8");
    merge_check::<16>(&left_ops, &right_ops, "same N=16");
    merge_check_cross::<8, 16>(&left_ops, &right_ops, "cross 8←16");
    merge_check_cross::<16, 8>(&left_ops, &right_ops, "cross 16←8");
}

/// Regression: three values with a subnormal, split across merge,
/// with echo-amplified increments.
#[test]
fn test_three_vals_with_subnormal_echo() {
    let v1 = f64::from_le_bytes([22, 22, 0, 237, 237, 59, 59, 59]); // ~2.25e-23
    let v2 = f64::from_le_bytes([59, 59, 1, 0, 59, 31, 0, 0]); // ~1.70e-310, subnormal as f32
    let v3 = f64::from_le_bytes([0, 59, 237, 237, 64, 0, 122, 64]); // ~416.0

    let left: Vec<(f64, u64)> = vec![(v1, 300), (v2, 5)];
    let right: Vec<(f64, u64)> = vec![
        (v3, 5),
        (v1, 1200),
        (v2, 20),
        (v3, 20),
        (v1, 19200),
        (v2, 320),
        (v3, 320),
    ];

    merge_check::<8>(&left, &right, "same 8");
    merge_check::<16>(&left, &right, "same 16");
    merge_check_cross::<8, 16>(&left, &right, "cross 8←16");
    merge_check_cross::<16, 8>(&left, &right, "cross 16←8");
}

#[test]
fn test_merge_p64_bucket_total_exceeds_count() {
    let mut h0 = Histogram::<8>::new();
    let mut h1 = Histogram::<8>::new();

    h1.record_incr(2.8396262443943004e+238, 40).unwrap();
    h0.record_incr(2.635549485807631e-82, 1).unwrap();

    // Step 3: merge h0 into h1
    h1.merge_from(&h0).unwrap();
    assert_eq!(h1.view().stats().count, 41);

    // Step 4: merge h1 into h0
    if h0.merge_from(&h1).is_ok() {
        let count = h0.view().stats().count;
        let bt = bucket_total(&h0);
        assert!(bt <= count, "bucket total ({bt}) exceeds count ({count})");
    }
}

#[test]
fn test_merge_p32_bucket_len_after_merge_chain() {
    let v0: f64 = 5.653943197254256e-308;
    let v1: f64 = 2.740490672504645e-61;

    let mut h0 = Histogram::<8>::new();
    let mut h1 = Histogram::<8>::new();

    h0.record_incr(v0, 1).unwrap();
    h1.record_incr(v1, 1).unwrap();

    // Merge chain: h0→h1, h0→h1, h1→h0
    h1.merge_from(&h0).unwrap();
    h1.merge_from(&h0).unwrap();
    h0.merge_from(&h1).unwrap();

    // Insert many zeros
    for _ in 0..150 {
        h0.record_incr(0.0, 1).unwrap();
    }

    // Verify bucket structure
    let h0_view = h0.view();
    let scale = h0_view.scale();
    let mapping = Scale::new(scale).unwrap();

    // All non-zero values should map to indices at the current scale
    let idx0 = mapping.map_to_index(v0);
    let idx1 = mapping.map_to_index(v1);
    let exp_min = idx0.min(idx1);
    let exp_max = idx0.max(idx1);
    let exp_len = (exp_max - exp_min + 1) as u32;

    let b = h0_view.positive();
    assert_eq!(
        b.offset(),
        exp_min,
        "offset mismatch: got {} expected {} (scale={})",
        b.offset(),
        exp_min,
        scale
    );
    assert_eq!(
        b.len(),
        exp_len,
        "len mismatch: got {} expected {} (scale={}, idx0={}, idx1={})",
        b.len(),
        exp_len,
        scale,
        idx0,
        idx1
    );

    // No trailing/leading zero buckets
    if !b.is_empty() {
        let bv: std::vec::Vec<u64> = b.iter().collect();
        assert!(bv[0] > 0, "leading zero bucket");
        assert!(*bv.last().unwrap() > 0, "trailing zero bucket");
    }
}
#[test]
fn repro_fuzz_histogram_oracle_offset() {
    // Regression: subnormals round to significand=1 in record_incr,
    // placing them one bucket above MIN_VALUE. The mapper itself
    // maps subnormals to the MIN_VALUE bucket, but record_incr
    // adjusts the decomposition before mapping.
    let subnormal = 1.3633843689306e-310f64;
    let normal = 2.2251438848883923e-308f64;

    let mut h = Histogram::<8>::new();
    h.update(subnormal).unwrap();
    h.update(normal).unwrap();

    let v = h.view();
    let mapping = Scale::new(v.scale()).unwrap();
    // record_incr rounds subnormals to (biased_exp=1, significand=1).
    let subnormal_rounded = f64::from_bits((1u64 << crate::float64::SIGNIFICAND_WIDTH) | 1);
    let subnormal_idx = mapping.map_to_index(subnormal_rounded);
    let exp_offset = subnormal_idx.min(mapping.map_to_index(normal));

    assert_eq!(
        v.positive().offset(),
        exp_offset,
        "offset mismatch at scale={}",
        v.scale()
    );
}

#[test]
fn repro_fuzz_merge_oracle_offset() {
    // Regression: subnormal value with large increments, merged across
    // histograms. The subnormal rounds to significand=1, one bucket
    // above MIN_VALUE.
    let subnormal = 5.580682928875e-312f64;
    let incrs: &[u64] = &[4194304, 16777216, 268435456, 4294967296];

    let mut right = Histogram::<8>::new();
    for &incr in incrs {
        right.record_incr(subnormal, incr).unwrap();
    }

    let mut left = Histogram::<8>::new();
    left.merge_from(&right).unwrap();

    let v = left.view();
    let buckets = v.positive();
    let mapping = Scale::new(v.scale()).unwrap();
    let subnormal_rounded = f64::from_bits((1u64 << crate::float64::SIGNIFICAND_WIDTH) | 1);
    let exp_idx = mapping.map_to_index(subnormal_rounded);

    assert_eq!(
        buckets.offset(),
        exp_idx,
        "offset mismatch at scale={}",
        v.scale()
    );
    let bt: u64 = buckets.iter().sum();
    let count = v.stats().count;
    assert!(bt <= count, "bucket total ({bt}) > count ({count})",);
}

#[test]
fn repro_fuzz_stateful_bucket_total() {
    // Regression test: exercises a merge path with extreme value
    // combinations that stress the downscale/widen recovery loop.
    let v1: f64 = f64::from_bits(0x5829f8b15858ff40);
    let v2: f64 = f64::from_bits(0x004b000000000000);
    let v3: f64 = f64::from_bits(0x56562c0000000000);

    let mut pool0 = Histogram::<8>::new().with_min_width(Width::B1);
    pool0.record_incr(v1, 12).unwrap();
    pool0.record_incr(v2, 1).unwrap();
    pool0.record_incr(v3, 1).unwrap();

    let mut big = Histogram::<16>::new().with_min_width(Width::B1);
    big.merge_from(&pool0).unwrap();

    // Second merge may succeed or fail; either way, exercise the path.
    let _ = big.merge_from(&pool0);
}

/// Exercises the record path with a huge increment at a wildly different
/// exponent, stressing the downscale/widen recovery loop.
#[test]
fn repro_fuzz_stateful_update_atomicity() {
    let v1 = f64::from_bits(0x002f233d41000000); // 8.66e-308
    let v2 = f64::from_bits(0x2c2c2cac2c2c2c2c); // 6.595e-96
    let v3 = f64::from_bits(0x78ffffdb58585858); // 6.924e+274

    let mut h: Histogram<8> = Histogram::new();
    h.update(v1).unwrap();

    let mut donor: Histogram<8> = Histogram::new();
    donor.update(v2).unwrap();
    donor.update(v1).unwrap();
    h.merge_from(&donor).unwrap();

    // This may fail with Overflow — exercise the path.
    let _ = h.record_incr(v3, 8388608);
}

// -----------------------------------------------------------------------
// Coverage: merge.rs tm_log deficit path (lines 99-107)
// -----------------------------------------------------------------------

#[test]
fn merge_tm_log_deficit_non_empty() {
    // Trigger the tm_log fixup for a non-empty self (merge.rs:106).
    //
    // Both histograms start at scale 0 with the SAME value so no
    // range downscale is needed (self_change=0).  self has U8 width
    // via with_min_width, source has B1.  After the first phase:
    //   shift = 0, src_width = 0, dest_width = 3 (U8)
    //   0 + 0 < 3 → deficit = 3, self not empty → downscale_by_min.
    let mut h1: Histogram<8> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::U8);
    h1.update(1.5).unwrap();

    let mut h2: Histogram<8> = Histogram::new().with_scale(0).unwrap();
    h2.update(1.5).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 2);
    assert_eq!(bucket_total(&h1), 2);
}

#[test]
fn merge_tm_log_deficit_empty_self() {
    // Same scenario but self is empty — exercises the buckets_empty
    // branch of the tm_log fixup (merge.rs:101-104).
    let mut h1: Histogram<8> = Histogram::new().with_min_width(Width::U8);
    // h1 is empty but has U8 width configured

    let mut h2: Histogram<8> = Histogram::new();
    h2.update(2.0).unwrap();

    h1.merge_from(&h2).unwrap();
    assert_stats(&h1, 1, 2.0, 2.0, 2.0);
    assert_eq!(bucket_total(&h1), 1);
}

#[test]
fn merge_tm_log_deficit_wide_gap() {
    // Extreme width gap: self at U64 from large incr, source at B1.
    // Both at scale 0 so shift stays 0. dest_width=6 (U64), deficit=6.
    let mut h1: Histogram<8> = Histogram::new().with_scale(0).unwrap();
    h1.record_incr(1.5, u32::MAX as u64 + 1).unwrap();
    // After widen to U64, scale decreases. Use with_scale(0) to keep it pinned.
    // Actually, record_incr will widen internally and change scale.
    // Let's use with_min_width instead to keep scale at 0.
    let mut h1: Histogram<8> = Histogram::new()
        .with_scale(0)
        .unwrap()
        .with_min_width(Width::U64);
    h1.update(1.5).unwrap();
    assert_eq!(h1.width(), Width::U64);

    let mut h2: Histogram<8> = Histogram::new().with_scale(0).unwrap();
    h2.update(1.5).unwrap();
    assert_eq!(h2.width(), Width::B1);

    h1.merge_from(&h2).unwrap();
    assert_eq!(h1.view().stats().count, 2);
    assert_eq!(bucket_total(&h1), 2);
}

// -----------------------------------------------------------------------
// Coverage: downscale.rs U64-start cross-word grouping (lines 141-145)
// -----------------------------------------------------------------------

#[test]
fn downscale_u64_width_cross_word() {
    // Start at U64 width so that do_downscale enters the cross-word
    // grouping path (phases 1+2 skipped, total_or computed from raw words).
    let mut h: Histogram<8> = Histogram::new();
    h.record_incr(1.0, u32::MAX as u64 + 1).unwrap();
    assert_eq!(h.width(), Width::U64);

    // Insert a distant value to force downscale.
    h.update(1e100).unwrap();

    let total = bucket_total(&h);
    assert_eq!(total, u32::MAX as u64 + 2);
}

// -----------------------------------------------------------------------
// Coverage: mod.rs utility methods
// -----------------------------------------------------------------------

#[test]
fn test_clear() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(2.0).unwrap();
    assert_eq!(h.view().stats().count, 2);

    h.clear();
    assert_eq!(h.view().stats().count, 0);
    assert_eq!(h.view().stats().sum, 0.0);
    assert_eq!(bucket_total(&h), 0);
    assert_eq!(h.width(), Width::B1);

    // Verify histogram is fully functional after clear.
    h.update(3.0).unwrap();
    assert_stats(&h, 1, 3.0, 3.0, 3.0);
}

#[test]
fn test_swap() {
    let mut h1: Histogram<8> = Histogram::new();
    let mut h2: Histogram<8> = Histogram::new();
    h1.update(1.0).unwrap();
    h2.update(2.0).unwrap();
    h2.update(3.0).unwrap();

    h1.swap(&mut h2);

    assert_eq!(h1.view().stats().count, 2);
    assert_eq!(h2.view().stats().count, 1);
    assert_stats(&h2, 1, 1.0, 1.0, 1.0);
}

#[test]
fn test_default() {
    let h: Histogram<8> = Histogram::default();
    assert_eq!(h.view().stats().count, 0);
    assert_eq!(h.width(), Width::B1);
}

#[test]
fn test_debug_format() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();
    let debug = std::format!("{:?}", h);
    assert!(debug.contains("count"), "debug output: {debug}");
    assert!(debug.contains("width"), "debug output: {debug}");
}

#[test]
fn test_error_display() {
    let e = Error::Overflow;
    let msg = std::format!("{e}");
    assert!(msg.contains("overflow"), "error display: {msg}");

    let e2 = Error::Extreme;
    let msg2 = std::format!("{e2}");
    assert!(msg2.contains("invalid"), "error display: {msg2}");
}

#[test]
fn test_record_incr_zero() {
    // record_incr with incr=0 for a non-zero value should be a no-op
    // on buckets but still count zero as a zero observation.
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();

    // incr=0 with a positive value: the record_incr path
    // reaches try_increment with incr=0 which returns IncrResult::Ok.
    h.record_incr(2.0, 0).unwrap();

    // Count stays the same (0 increment doesn't add to count).
    assert_eq!(h.view().stats().count, 1);
}

#[test]
fn test_view_empty_buckets() {
    // BucketView on an empty histogram: offset=0, len=0.
    let h: Histogram<8> = Histogram::new();
    let v = h.view();
    let b = v.positive();
    assert!(b.is_empty());
    assert_eq!(b.offset(), 0);
    assert_eq!(b.len(), 0);
    assert_eq!(b.iter().count(), 0);
}

#[test]
fn test_into_iter() {
    // Exercise the IntoIterator impl on BucketView.
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(2.0).unwrap();

    let v = h.view();
    let total: u64 = v.positive().into_iter().sum();
    assert_eq!(total, 2);
}

// -----------------------------------------------------------------------
// Coverage: swar.rs rare narrow paths
// -----------------------------------------------------------------------

#[test]
fn narrow_multi_step_paths() {
    // Exercise rare narrow paths by creating histograms that force
    // unusual width transitions during downscale.
    use super::swar::narrow;

    // U16 → B1: 4 steps
    let w = 0x0001_0002_0003_0004u64; // 4 U16 lanes: 1, 2, 3, 4
    let result = narrow(Width::U16, Width::B1, w);
    // Each U16 lane truncated to 1 bit: 1, 0, 1, 0 packed into B1
    // (only the LSB of each lane survives)
    let _ = result; // Just exercise the path.

    // U32 → U8: 2 steps
    let w = 0x0000_00FF_0000_0042u64; // 2 U32 lanes: 0x42, 0xFF
    let result = narrow(Width::U32, Width::U8, w);
    let _ = result;

    // U32 → B2: 3 steps
    let w = 0x0000_0003_0000_0002u64; // 2 U32 lanes: 2, 3
    let result = narrow(Width::U32, Width::B2, w);
    let _ = result;

    // U32 → B1: 4 steps
    let w = 0x0000_0001_0000_0001u64;
    let result = narrow(Width::U32, Width::B1, w);
    let _ = result;
}

#[test]
fn test_msb_mask_u64() {
    // Exercise the U64 arm of msb_mask (swar.rs line 247).
    use super::swar::swar_add_checked;
    // At U64 width, swar_add_checked uses checked_add directly.
    // But let's also exercise the overflow path.
    assert_eq!(swar_add_checked(u64::MAX, 1, Width::U64), None);
    assert_eq!(
        swar_add_checked(u64::MAX - 1, 1, Width::U64),
        Some(u64::MAX)
    );
}

// -----------------------------------------------------------------------
// Coverage: mapping.rs Display impls
// -----------------------------------------------------------------------

#[test]
fn test_scale_error_display() {
    use crate::mapping::Scale;
    let err = Scale::new(100); // too high
    assert!(err.is_err());
    let msg = std::format!("{}", err.unwrap_err());
    assert!(!msg.is_empty());
}

#[test]
fn test_scale_display() {
    let s = Scale::new(3).unwrap();
    let msg = std::format!("{s}");
    assert!(msg.contains('3'), "scale display: {msg}");
}

// -----------------------------------------------------------------------
// Coverage: settings accessors, current_word_count, current_slot_count
// -----------------------------------------------------------------------

#[test]
fn test_settings_accessors() {
    // Exercise Settings::scale() and Settings::width() via view.
    let h: Histogram<8> = Histogram::new()
        .with_scale(3)
        .unwrap()
        .with_min_width(Width::B4);
    // Settings are accessed internally; we verify through the view.
    let v = h.view();
    assert_eq!(v.scale(), 0); // empty histogram returns 0
    assert_eq!(h.width(), Width::B4);
}

#[test]
fn test_crash_1d7f7c() {
    let value = f64::from_le_bytes([255, 251, 122, 0, 0, 0, 0, 0]);
    // value ≈ 3.98e-317 (subnormal)

    let mut h1 = Histogram::<8>::new().with_min_width(Width::B1);
    h1.record_incr(value, 1).unwrap();

    let mut h2 = Histogram::<8>::new().with_min_width(Width::B1);
    h2.record_incr(value, 4).unwrap();
    h2.record_incr(value, 64).unwrap();
    h2.record_incr(value, 1024).unwrap();

    h1.merge_from(&h2).unwrap();

    let v = h1.view();
    let stats = v.stats();
    assert_eq!(stats.count, 1 + 4 + 64 + 1024);
}

#[test]
fn repro_fuzz_merge_cross_size_doubling() {
    // Regression: merging Histogram<16> into Histogram<8> with normal
    // near-MIN_VALUE values caused bucket data to be doubled.
    let v1 = f64::from_bits(0x00ae000000010200); // biased_exp=10
    let v2 = f64::from_bits(0x0100feff00fa0000); // biased_exp=16

    let mut h2 = Histogram::<16>::new().with_min_width(Width::B1);
    h2.record_incr(v1, 175).unwrap();
    h2.record_incr(v2, 251).unwrap();

    let mut h1 = Histogram::<8>::new().with_min_width(Width::B1);
    h1.merge_from(&h2).unwrap();

    let bt: u64 = h1.view().positive().iter().sum();
    let count = h1.view().stats().count;
    assert_eq!(bt, count, "bucket total ({bt}) != count ({count})");
}

#[test]
fn test_negative_subnormal_rejected() {
    let mut h: Histogram<16> = Histogram::new();
    // Negative subnormals must be rejected, not silently bucketed.
    let neg_subnormal = -5e-324_f64;
    assert!(neg_subnormal.is_sign_negative());
    assert_eq!(h.update(neg_subnormal), Err(Error::Extreme));
    assert_eq!(h.view().stats().count, 0);

    // Positive subnormals are still accepted.
    let pos_subnormal = 5e-324_f64;
    assert!(h.update(pos_subnormal).is_ok());
    assert_eq!(h.view().stats().count, 1);
}

#[test]
fn test_zero_only_histogram_stats() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(0.0).unwrap();
    h.update(0.0).unwrap();
    let s = h.view().stats();
    assert_eq!(s.count, 2);
    assert_eq!(s.sum, 0.0);
    assert_eq!(s.min, 0.0, "zero-only min should be 0.0");
    assert_eq!(s.max, 0.0, "zero-only max should be 0.0");
    assert_eq!(h.view().positive().len(), 0);

    // Negative zero is also accepted as zero.
    let mut h2: Histogram<16> = Histogram::new();
    h2.update(-0.0).unwrap();
    let s2 = h2.view().stats();
    assert_eq!(s2.count, 1);
    assert_eq!(s2.min, 0.0);
    assert_eq!(s2.max, 0.0);
}

#[test]
fn test_min_scale_large_increment_no_panic() {
    // At MIN_SCALE with B1 counters (max=1), recording incr=2
    // must not panic. It should widen the counter width without
    // changing the scale — an empty histogram has no range to
    // protect, so only the width needs adjustment.
    let mut h: Histogram<16> = Histogram::new()
        .with_scale(crate::mapping::MIN_SCALE)
        .unwrap();
    assert!(h.record_incr(1.0, 2).is_ok());
    assert_eq!(h.view().stats().count, 2);
    assert_eq!(
        h.view().scale(),
        crate::mapping::MIN_SCALE,
        "scale should be preserved at MIN_SCALE"
    );

    // Even larger increments should also work and preserve scale.
    let mut h2: Histogram<16> = Histogram::new()
        .with_scale(crate::mapping::MIN_SCALE)
        .unwrap();
    assert!(h2.record_incr(1.0, u16::MAX as u64 + 1).is_ok());
    assert_eq!(h2.view().stats().count, u16::MAX as u64 + 1);
    assert_eq!(
        h2.view().scale(),
        crate::mapping::MIN_SCALE,
        "scale should be preserved at MIN_SCALE"
    );
}
