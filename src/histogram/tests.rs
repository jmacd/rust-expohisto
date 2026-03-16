// Tests always run with std available, even when the crate is no_std.
extern crate std;
use std::{format, vec, vec::Vec};

use super::*;
use super::swar::{
    narrow_word, swar_has_overflow, swar_narrow_compact, swar_step,
};

/// Helper: count total across all positive buckets.
fn bucket_total<const N: usize>(h: &mut Histogram<N>) -> u64 {
    h.view().positive().iter().sum()
}

fn derived_zero_count<const N: usize>(h: &mut Histogram<N>) -> u64 {
    h.view().count() - bucket_total(h)
}

#[test]
fn test_histogram_basic() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    assert_stats(&mut h, 1, 1.0, 1.0, 1.0);
    assert_eq!(derived_zero_count(&mut h), 0);
    assert_eq!(h.bucket_width(), BucketWidth::B1);
}

#[test]
fn test_histogram_zero() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(0.0).unwrap();
    assert_eq!(h.view().count(), 1);
    assert_eq!(derived_zero_count(&mut h), 1);
    assert_eq!(h.view().sum(), 0.0);
}

#[test]
fn test_histogram_multiple() {
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(2.0).unwrap();
    h.update(4.0).unwrap();
    assert_stats(&mut h, 3, 7.0, 1.0, 4.0);
}

#[cfg(has_lookup_table)]
#[test]
fn test_histogram_downscale() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(1000.0).unwrap();
    assert_eq!(h.view().count(), 2);
    assert!(h.view().scale() < max_scale());
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
    assert_stats(&mut h1, 4, 10.0, 1.0, 4.0);
}

#[test]
fn test_histogram_recreate() {
    let mut h: Histogram<16> = Histogram::new();
    assert_eq!(h.view().count(), 0);
    assert_eq!(h.view().sum(), 0.0);
    assert_eq!(h.view().scale(), 0);
    assert_eq!(h.bucket_width(), BucketWidth::B1);
}

#[test]
fn test_buckets_at() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0);
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
    let mut h: Histogram<16> = Histogram::new()
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);

    // B4 → U8 at threshold 15+1=16
    h.record(1.0, 15).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::B4);
    h.update(1.0).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U8);
    assert_eq!(h.view().count(), 16);

    // U8 → U16 at threshold 255+1=256
    h.record(1.0, 239).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U8);
    h.update(1.0).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U16);
    assert_eq!(h.view().count(), 256);

    // U16 → U32 at threshold 65535+1=65536
    h.record(1.0, u16::MAX as u64 - 256).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U16);
    h.update(1.0).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U32);
    assert_eq!(h.view().count(), u16::MAX as u64 + 1);

    // U32 → U64 at threshold 4294967295+1
    h.record(1.0, u32::MAX as u64 - (u16::MAX as u64 + 1))
        .unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U32);
    h.update(1.0).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U64);
}

#[test]
fn test_auto_widen_b4_to_u8_from_b4_start() {
    let mut h: Histogram<16> = Histogram::new()
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);
    h.record(1.0, 4).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::B4);
    h.record(1.0, 11).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::B4);
    h.update(1.0).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U8);
    assert_eq!(h.view().count(), 16);
}

#[test]
fn test_bucket_count_halves_on_widen() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0)
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);
    let initial_cap = h.bucket_capacity();
    assert_eq!(initial_cap, 16 * 16); // 256

    h.record(1.0, 16).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U8);
    assert_eq!(h.bucket_capacity(), 16 * 8); // 128
}

#[cfg(has_lookup_table)]
#[test]
fn test_recreate_preserves_b4() {
    let mut h: Histogram<16> = Histogram::new().with_scale(3)
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);
    assert_eq!(h.bucket_width(), BucketWidth::B4);
    assert_eq!(h.view().count(), 0);
    // Record a value to verify it starts at scale 3.
    h.update(1.0).unwrap();
    assert_eq!(h.view().scale(), 3);
}

#[cfg(has_lookup_table)]
#[test]
fn test_with_scale() {
    let mut h: Histogram<16> = Histogram::new().with_scale(3);
    // Record a value to verify scale is respected.
    h.update(1.0).unwrap();
    assert_eq!(h.view().scale(), 3);
}

#[cfg(has_lookup_table)]
#[test]
fn test_with_scale_records_at_limited_scale() {
    let mut limited: Histogram<16> = Histogram::new().with_scale(3);
    let mut unlimited: Histogram<16> = Histogram::new();
    limited.update(1.0).unwrap();
    limited.update(1.001).unwrap();
    unlimited.update(1.0).unwrap();
    unlimited.update(1.001).unwrap();

    let limited_view = limited.view();
    assert!(limited_view.scale() <= 3);
    if max_scale() > 3 {
        let unlimited_view = unlimited.view();
        assert!(unlimited_view.scale() > limited_view.scale());
    }
}

#[test]
fn test_widen_preserves_data() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0);
    h.record(1.0, 100).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U8);

    h.record(256.0, 50).unwrap();
    h.record(65536.0, 200).unwrap();

    let (count_before, sum_before) = {
        let v = h.view();
        (v.count(), v.sum())
    };

    h.record(65536.0, 55).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U8);
    h.update(65536.0).unwrap();
    assert_eq!(h.bucket_width(), BucketWidth::U16);

    let v = h.view();
    assert_eq!(v.count(), count_before + 56);
    assert!((v.sum() - (sum_before + 56.0 * 65536.0)).abs() < 1.0);
}

#[test]
fn test_merge_equivalence_comprehensive() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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
                panic!("merge_from failed for size={K} sets {i} x {j}: {e}\n  set_a: {set_a:?}\n  set_b: {set_b:?}\n  merged: {:?}\n  other: {:?}", merged, other);
            }

            let mut single = build_from_values::<K>(set_a);
            for &v in set_b.iter() {
                single.update(v).unwrap();
            }

            let label = format!("size={K} sets {i} x {j}");
            let merged_view = merged.view();
            let single_view = single.view();
            assert_eq!(merged_view.count(), single_view.count(), "count mismatch for {label}");
            let ms = merged_view.sum();
            let ss = single_view.sum();
            let sum_diff = (ms - ss).abs();
            let denom = ms.abs().max(ss.abs()).max(1e-30);
            assert!(
                sum_diff / denom < 1e-5,
                "sum mismatch for {label}: {ms} vs {ss}"
            );
            assert_eq!(
                derived_zero_count(&mut merged),
                derived_zero_count(&mut single),
                "zero_count mismatch for {label}"
            );
            assert_eq!(
                bucket_total(&mut merged),
                bucket_total(&mut single),
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
        let bt = bucket_total(&mut other);
        let non_zero_count = {
            let v = other.view();
            v.count()
        } - derived_zero_count(&mut other);
        assert_eq!(bt, non_zero_count, "bucket total mismatch after inserting {v}");
    }

    let set_a: &[f64] = &[1.0];
    let mut merged = build_from_values::<8>(set_a);
    merged.merge_from(&other).unwrap();

    let mut single = build_from_values::<8>(set_a);
    for &v in set_b {
        single.update(v).unwrap();
    }

    assert_eq!(
        bucket_total(&mut merged),
        bucket_total(&mut single),
        "bucket total mismatch: merged vs single"
    );
}

#[test]
fn test_edge_values_inf() {
    use crate::mapping::Mapping;

    let max_f64: f64 = f64::MAX;
    let inf: f64 = f64::INFINITY;

    let m0 = Mapping::new(0).unwrap();
    let idx_max = m0.map_to_index(max_f64);
    let idx_inf = m0.map_to_index(inf);
    assert_eq!(idx_max, idx_inf);

    let mut h: Histogram<16> = Histogram::new().with_scale(0);
    h.update(1.0).unwrap();
    h.update(max_f64).unwrap();
    // f64::MAX fits in f64 sum, but after adding infinity the sum
    // is infinite.
    h.update(inf).unwrap();
    assert_eq!(h.view().count(), 3);
    assert_eq!(h.view().max(), f64::INFINITY);
    assert!(h.view().sum().is_infinite());
    assert_eq!(h.view().min(), 1.0);
}

#[test]
fn test_edge_values_subnormals() {
    use crate::mapping::Mapping;

    let subnormal: f64 = 5e-324;
    let min_normal: f64 = crate::float64::MIN_VALUE;

    let m0 = Mapping::new(0).unwrap();
    assert_eq!(m0.map_to_index(subnormal), m0.map_to_index(min_normal));

    let mut h: Histogram<16> = Histogram::new().with_scale(0);
    h.update(subnormal).unwrap();
    h.update(min_normal).unwrap();
    assert_eq!(h.view().count(), 2);
    assert_eq!(h.view().positive().len(), 1);
}

/// Documents the behavior when NaN or negative values are passed.
/// The caller is expected to validate inputs before calling record().
/// These are not checked at runtime — the histogram remains safe but
/// produces unspecified statistical results.
#[test]
fn test_nan_and_negative_are_unchecked() {
    // NaN: treated as non-zero (enters bucket path), sum becomes NaN,
    // min/max become NaN. Count still increments normally.
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    assert!(h.update(f64::NAN).is_ok());
    assert_eq!(h.view().count(), 2);
    assert!(h.view().sum().is_nan());

    // Negative: treated as a normal positive value by the mapping
    // (bit pattern has same exponent structure). Count increments,
    // sum/min/max reflect the negative value.
    let mut h: Histogram<16> = Histogram::new();
    h.update(1.0).unwrap();
    assert!(h.update(-1.0).is_ok());
    assert_eq!(h.view().count(), 2);
    assert_eq!(h.view().sum(), 0.0);
    assert_eq!(h.view().min(), -1.0);
}

#[test]
fn test_exhaustive_u8_overflow() {
    // Insert 8 values spanning a wide index range at scale 0, each
    // with count 255. Starting at B1 with 320 slots (Histogram<8>),
    // counters widen B1→B2→B4→U8 (255 fits in U8), but the larger
    // initial capacity means the span still fits without reaching U64.
    let mut h: Histogram<8> = Histogram::new().with_scale(0);
    let num_buckets = 8;
    for i in 0..num_buckets {
        let val = 2.0_f64.powi(i * 8);
        h.record(val, 255).unwrap();
    }
    // With B1 start, U8 has enough capacity for the span.
    assert!(
        h.bucket_width() >= BucketWidth::U8,
        "expected at least U8, got {:?}",
        h.bucket_width()
    );
    assert_eq!(h.view().count(), num_buckets as u64 * 255);
    // Adding one more should still be fine at U64 (no further widen needed).
    h.update(1.0).unwrap();
    assert_eq!(h.view().count(), num_buckets as u64 * 255 + 1);
}

#[test]
fn test_successive_sub_byte_widening() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0)
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);

    h.update(1.0).unwrap();
    assert_eq!(h.view().count(), 1);
    assert_eq!(h.bucket_width(), BucketWidth::B4);

    for count in 2..=15u64 {
        h.update(1.0).unwrap();
        assert_eq!(h.view().count(), count);
        assert_eq!(
            h.bucket_width(),
            BucketWidth::B4,
            "expected B4 at count {count}"
        );
    }

    h.update(1.0).unwrap();
    assert_eq!(h.view().count(), 16);
    assert_eq!(h.bucket_width(), BucketWidth::U8);

    assert!((h.view().sum() - 16.0).abs() < 1e-10);
    assert_eq!(h.view().min(), 1.0);
    assert_eq!(h.view().max(), 1.0);
}

#[test]
fn test_successive_sub_byte_widening_multi_bucket() {
    let mut h: Histogram<16> = Histogram::new().with_scale(0).with_min_bucket_width(BucketWidth::B4);
    let num_buckets = 8;
    let values: Vec<f64> = (1..=num_buckets).map(|k| 2.0_f64.powi(k)).collect();

    for &v in &values {
        h.update(v).unwrap();
    }
    assert_eq!(h.view().count(), num_buckets as u64);
    assert_eq!(h.bucket_width(), BucketWidth::B4);

    for &v in &values {
        h.update(v).unwrap();
    }
    assert_eq!(h.view().count(), 2 * num_buckets as u64);
    assert!(h.bucket_width() >= BucketWidth::B4);

    let target = 16 * num_buckets as u64;
    while h.view().count() < target {
        for &v in &values {
            h.update(v).unwrap();
        }
    }
    assert!(h.bucket_width() >= BucketWidth::U8);

    let expected_sum: f64 = values.iter().sum::<f64>() * 16.0;
    assert!(
        (h.view().sum() - expected_sum).abs() < 1e-6,
        "sum mismatch: got {} expected {}",
        h.view().sum(),
        expected_sum
    );
    assert_eq!(h.view().count(), target);
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

    collector.merge_from_other(&source).unwrap();

    assert_eq!(collector.view().count(), 4);
    assert_eq!(derived_zero_count(&mut collector), 1);
    assert!((collector.view().sum() - 7.0).abs() < 1e-5);
}

#[test]
fn test_merge_multiple_sources() {
    let mut collector: Histogram<20> = Histogram::new();

    for batch in 0..5 {
        let mut src: Histogram<16> = Histogram::new();
        for i in 0..10 {
            src.update((batch * 10 + i) as f64 * 0.1 + 0.1).unwrap();
        }
        collector.merge_from_other(&src).unwrap();
    }

    assert_eq!(collector.view().count(), 50);
    assert!(collector.view().sum() > 0.0);
}

#[test]
fn test_merge_preserves_buckets() {
    let mut collector: Histogram<16> = Histogram::new().with_scale(0);
    let mut source: Histogram<16> = Histogram::new().with_scale(0);

    source.update(1.0).unwrap();
    source.update(2.0).unwrap();
    source.update(4.0).unwrap();

    collector.merge_from_other(&source).unwrap();

    let mut direct: Histogram<16> = Histogram::new().with_scale(0);
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
    for i in 0..collector_buckets.len() {
        assert_eq!(
            collector_buckets.at(i),
            direct_buckets.at(i),
            "bucket[{i}] mismatch"
        );
    }
}

#[test]
fn test_merge_empty_into_populated() {
    let mut collector: Histogram<16> = Histogram::new();
    collector.update(1.0).unwrap();

    let empty: Histogram<8> = Histogram::new();
    collector.merge_from_other(&empty).unwrap();

    assert_eq!(collector.view().count(), 1);
    assert_eq!(collector.view().sum(), 1.0);
}

#[test]
fn test_merge_into_empty() {
    let mut collector: Histogram<16> = Histogram::new();
    let mut source: Histogram<8> = Histogram::new();
    source.update(5.0).unwrap();

    collector.merge_from_other(&source).unwrap();

    assert_eq!(collector.view().count(), 1);
    assert!((collector.view().sum() - 5.0).abs() < 1e-5);
}

// -----------------------------------------------------------------------
// Flat layout capacity tests
// -----------------------------------------------------------------------

mod flat_layout {
    use super::*;

    #[test]
    fn test_capacity() {
        let h: Histogram<16> = Histogram::new();
        assert_eq!(h.bucket_capacity(), 1024); // 16 * 64 at B1
    }

    #[test]
    fn test_minimum_n() {
        let h: Histogram<8> = Histogram::new();
        assert_eq!(h.bucket_capacity(), 512); // 8 * 64 at B1
    }

    #[test]
    fn test_struct_size() {
        use core::mem;
        let size = mem::size_of::<Histogram<16>>();
        // 16 u64 words (128 bytes) + Stats (32 bytes) + fixed metadata.
        // Data pool + stats dominate; struct should not exceed pool + 80 bytes overhead.
        assert!(
            size <= 128 + 80,
            "Histogram<16> unexpectedly large: {} bytes",
            size
        );
    }
}

// -----------------------------------------------------------------------
// Narrow function unit tests
// -----------------------------------------------------------------------

/// Helper: pack 8 bytes into one u64, byte0 in the LSB.
fn pack_u8x8(b: [u8; 8]) -> u64 {
    u64::from_le_bytes(b)
}

/// Helper: pack 16 nibbles into one u64, nibble0 in the low 4 bits.
fn pack_b4x16(n: [u8; 16]) -> u64 {
    let mut w = 0u64;
    for (i, &nibble) in n.iter().enumerate() {
        w |= (nibble as u64 & 0xF) << (i * 4);
    }
    w
}

/// Helper: pack 4 u16s into one u64, short0 in the low 16 bits.
fn pack_u16x4(s: [u16; 4]) -> u64 {
    (s[0] as u64) | ((s[1] as u64) << 16) | ((s[2] as u64) << 32) | ((s[3] as u64) << 48)
}

/// Helper: pack 2 u32s into one u64, int0 in the low 32 bits.
fn pack_u32x2(lo: u32, hi: u32) -> u64 {
    (lo as u64) | ((hi as u64) << 32)
}

/// Asserts `swar_narrow_compact` produces `expected` prefix words
/// and zeroes all freed tail words.
fn assert_compact(width: BucketWidth, input: &[u64], expected: &[u64]) {
    let mut data = [0u64; 8];
    data[..input.len()].copy_from_slice(input);
    swar_narrow_compact(&mut data[..input.len()], width);
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(
            data[i], exp,
            "word {i}: got {:#018x}, expected {:#018x}",
            data[i], exp
        );
    }
    for (i, word) in data[expected.len()..input.len()].iter().enumerate() {
        assert_eq!(*word, 0, "word {} should be zeroed", expected.len() + i);
    }
}

/// Runs the full SWAR pipeline (step → overflow check → optional compact)
/// and asserts the result.
fn assert_swar_roundtrip(
    width: BucketWidth,
    input: &[u64],
    expect_overflow: bool,
    expected_after_step: &[u64],
    expected_after_compact: Option<&[u64]>,
) {
    let mut data = [0u64; 8];
    data[..input.len()].copy_from_slice(input);
    let slice = &mut data[..input.len()];

    swar_step(slice, width);
    for (i, &exp) in expected_after_step.iter().enumerate() {
        assert_eq!(
            slice[i], exp,
            "swar_step word {i}: got {:#018x}, expected {:#018x}",
            slice[i], exp
        );
    }

    assert_eq!(
        swar_has_overflow(slice, width),
        expect_overflow,
        "overflow mismatch"
    );

    if let Some(expected) = expected_after_compact {
        swar_narrow_compact(slice, width);
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                slice[i], exp,
                "compact word {i}: got {:#018x}, expected {:#018x}",
                slice[i], exp
            );
        }
        for (i, word) in slice[expected.len()..input.len()].iter().enumerate() {
            assert_eq!(*word, 0, "word {} should be zeroed after compact", expected.len() + i);
        }
    }
}

/// Helper for asserting `Stats` fields.
fn assert_stats<const N: usize>(
    h: &mut Histogram<N>,
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
) {
    let v = h.view();
    assert_eq!(v.count(), count, "count");
    assert_eq!(v.sum(), sum, "sum");
    assert_eq!(v.min(), min, "min");
    assert_eq!(v.max(), max, "max");
}

/// Builds a literal-mode and a bucket-mode histogram from the same
/// values and asserts they produce equivalent views.
fn assert_literal_matches_bucket(values: &[f64]) {
    let mut lit: Histogram<8> = Histogram::new();
    let mut bkt: Histogram<8> = Histogram::new().with_literal_mode(false);
    for &v in values {
        lit.update(v).unwrap();
        bkt.update(v).unwrap();
    }

    let lit_view = lit.view();
    let lit_buckets = lit_view.positive();
    let bkt_view = bkt.view();
    let bkt_buckets = bkt_view.positive();

    assert_eq!(lit_view.scale(), bkt_view.scale(), "scale mismatch");
    assert_eq!(lit_view.count(), bkt_view.count(), "count mismatch");
    assert_eq!(lit_view.sum(), bkt_view.sum(), "sum mismatch");
    assert_eq!(lit_buckets.offset(), bkt_buckets.offset(), "offset");
    assert_eq!(lit_buckets.len(), bkt_buckets.len(), "len");
    for i in 0..lit_buckets.len() {
        assert_eq!(
            lit_buckets.at(i),
            bkt_buckets.at(i),
            "bucket[{i}] mismatch"
        );
    }
}

#[test]
fn test_narrow_u8_to_b4_zeroes() {
    assert_eq!(narrow_word(0, BucketWidth::B4), 0);
}

#[test]
fn test_narrow_u8_to_b4() {
    let cases: &[([u8; 8], [u8; 16])] = &[
        (
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [15, 15, 15, 15, 15, 15, 15, 15],
            [15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [15, 0, 8, 0, 3, 0, 1, 0],
            [15, 0, 8, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ];
    for (i, (input_bytes, expected_nibbles)) in cases.iter().enumerate() {
        let result = narrow_word(pack_u8x8(*input_bytes), BucketWidth::B4);
        let expected = pack_b4x16(*expected_nibbles);
        assert_eq!(
            result, expected,
            "case {i}: got {result:#018x}, expected {expected:#018x}"
        );
    }
}

#[test]
fn test_narrow_u16_to_u8() {
    assert_eq!(narrow_word(0, BucketWidth::U8), 0);
    let cases: &[([u16; 4], [u8; 8])] = &[
        ([10, 20, 30, 40], [10, 20, 30, 40, 0, 0, 0, 0]),
        ([255, 255, 255, 255], [255, 255, 255, 255, 0, 0, 0, 0]),
    ];
    for (i, (input_shorts, expected_bytes)) in cases.iter().enumerate() {
        let result = narrow_word(pack_u16x4(*input_shorts), BucketWidth::U8);
        let expected = pack_u8x8(*expected_bytes) & 0xFFFF_FFFF;
        assert_eq!(
            result, expected,
            "case {i}: got {result:#018x}, expected {expected:#018x}"
        );
    }
}

#[test]
fn test_narrow_u32_to_u16() {
    assert_eq!(narrow_word(0, BucketWidth::U16), 0);
    let cases: &[(u32, u32, u64)] = &[
        (1000, 2000, 1000 | (2000 << 16)),
        (65535, 65535, 65535 | (65535 << 16)),
    ];
    for (i, &(a, b, expected)) in cases.iter().enumerate() {
        let result = narrow_word(pack_u32x2(a, b), BucketWidth::U16);
        assert_eq!(
            result, expected,
            "case {i}: got {result:#018x}, expected {expected:#018x}"
        );
    }
}

// -----------------------------------------------------------------------
// swar_narrow_compact end-to-end tests
// -----------------------------------------------------------------------

#[test]
fn test_swar_narrow_compact_two_words() {
    // B4: 2 words of U8 → 1 word of B4
    assert_compact(
        BucketWidth::B4,
        &[
            pack_u8x8([1, 2, 3, 4, 5, 6, 7, 8]),
            pack_u8x8([9, 10, 11, 12, 13, 14, 15, 0]),
        ],
        &[pack_b4x16([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
        ])],
    );
    // U8: 2 words of U16 → 1 word of U8
    assert_compact(
        BucketWidth::U8,
        &[pack_u16x4([10, 20, 30, 40]), pack_u16x4([50, 60, 70, 80])],
        &[pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80])],
    );
    // U16: 2 words of U32 → 1 word of U16
    assert_compact(
        BucketWidth::U16,
        &[pack_u32x2(100, 200), pack_u32x2(300, 400)],
        &[pack_u16x4([100, 200, 300, 400])],
    );
    // U32: 2 words of U64 → 1 word of U32
    assert_compact(
        BucketWidth::U32,
        &[1000u64, 2000u64],
        &[pack_u32x2(1000, 2000)],
    );
}

#[test]
fn test_swar_narrow_compact_four_words() {
    // B4: 4 words of U8 → 2 words of B4
    assert_compact(
        BucketWidth::B4,
        &[
            pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]),
            pack_u8x8([3, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 4]),
        ],
        &[
            pack_b4x16([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
            pack_b4x16([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]),
        ],
    );
}

#[test]
fn test_swar_narrow_compact_odd_word_counts() {
    // B4: 1 word → in-place narrow
    let input = [pack_u8x8([3, 7, 0, 15, 0, 0, 5, 6])];
    let expected = narrow_word(input[0], BucketWidth::B4);
    assert_compact(BucketWidth::B4, &input, &[expected]);

    // B4: 3 words → 2 compacted words
    let w0 = pack_u8x8([1, 0, 0, 0, 0, 0, 0, 0]);
    let w1 = pack_u8x8([0, 0, 0, 0, 0, 0, 0, 2]);
    let w2 = pack_u8x8([3, 0, 0, 0, 0, 0, 0, 4]);
    let lo0 = narrow_word(w0, BucketWidth::B4);
    let hi0 = narrow_word(w1, BucketWidth::B4);
    let lo1 = narrow_word(w2, BucketWidth::B4);
    assert_compact(
        BucketWidth::B4,
        &[w0, w1, w2],
        &[lo0 | (hi0 << 32), lo1],
    );

    // U8: 3 words → 2 compacted words
    let expected1 = narrow_word(pack_u16x4([255, 0, 128, 1]), BucketWidth::U8);
    assert_compact(
        BucketWidth::U8,
        &[
            pack_u16x4([10, 20, 30, 40]),
            pack_u16x4([50, 60, 70, 80]),
            pack_u16x4([255, 0, 128, 1]),
        ],
        &[pack_u8x8([10, 20, 30, 40, 50, 60, 70, 80]), expected1],
    );
}

// -----------------------------------------------------------------------
// swar_has_overflow tests
// -----------------------------------------------------------------------

#[test]
fn test_swar_has_overflow() {
    let cases: &[(&[u64], BucketWidth, bool)] = &[
        // B4: at-max → no overflow
        (
            &[pack_u8x8([15, 0, 8, 3, 1, 14, 7, 0])],
            BucketWidth::B4,
            false,
        ),
        // B4: one slot at 16 → overflow
        (
            &[pack_u8x8([15, 0, 16, 0, 0, 0, 0, 0])],
            BucketWidth::B4,
            true,
        ),
        // B4 boundary: all at max
        (
            &[pack_u8x8([15, 15, 15, 15, 15, 15, 15, 15])],
            BucketWidth::B4,
            false,
        ),
        // B4 boundary: one over
        (
            &[pack_u8x8([15, 15, 15, 16, 15, 15, 15, 15])],
            BucketWidth::B4,
            true,
        ),
        // U8: at-max
        (&[pack_u16x4([255, 0, 128, 1])], BucketWidth::U8, false),
        // U8: overflow
        (&[pack_u16x4([256, 0, 0, 0])], BucketWidth::U8, true),
        // U8 boundary: all at max
        (&[pack_u16x4([255, 255, 255, 255])], BucketWidth::U8, false),
        // U8 boundary: one over
        (&[pack_u16x4([255, 255, 256, 255])], BucketWidth::U8, true),
        // U16: at-max
        (&[pack_u32x2(65535, 0)], BucketWidth::U16, false),
        // U16: overflow
        (&[pack_u32x2(65536, 0)], BucketWidth::U16, true),
        // U16 boundary: all at max
        (&[pack_u32x2(65535, 65535)], BucketWidth::U16, false),
        // U32: at-max
        (&[u32::MAX as u64], BucketWidth::U32, false),
        // U32: overflow
        (&[u32::MAX as u64 + 1], BucketWidth::U32, true),
    ];
    for (i, &(data, width, expected)) in cases.iter().enumerate() {
        assert_eq!(
            swar_has_overflow(data, width),
            expected,
            "case {i}: width={width:?} expected={expected}"
        );
    }
}

// -----------------------------------------------------------------------
// Full SWAR pipeline: swar_step → overflow check → narrow_compact
// -----------------------------------------------------------------------

#[test]
fn test_swar_step_then_narrow_compact_roundtrip() {
    // B4: pair sums ≤ 15 → compact back to B4
    assert_swar_roundtrip(
        BucketWidth::B4,
        &[
            pack_b4x16([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pack_b4x16([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 6, 0]),
        ],
        false,
        &[
            pack_u8x8([3, 7, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 5, 6]),
        ],
        Some(&[pack_b4x16([
            3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6,
        ])]),
    );

    // U8: pair sums ≤ 255 → compact back to U8
    assert_swar_roundtrip(
        BucketWidth::U8,
        &[
            pack_u8x8([100, 50, 30, 20, 10, 5, 3, 1]),
            pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
        ],
        false,
        &[pack_u16x4([150, 50, 15, 4]), pack_u16x4([0, 0, 0, 0])],
        Some(&[pack_u8x8([150, 50, 15, 4, 0, 0, 0, 0])]),
    );

    // U16: pair sums ≤ 65535 → compact back to U16
    assert_swar_roundtrip(
        BucketWidth::U16,
        &[pack_u16x4([1000, 2000, 3000, 4000]), pack_u16x4([0; 4])],
        false,
        &[pack_u32x2(3000, 7000), pack_u32x2(0, 0)],
        Some(&[pack_u16x4([3000, 7000, 0, 0])]),
    );

    // U32: pair sum fits → compact back to U32
    assert_swar_roundtrip(
        BucketWidth::U32,
        &[pack_u32x2(100_000, 200_000), pack_u32x2(0, 0)],
        false,
        &[300_000u64, 0],
        Some(&[pack_u32x2(300_000, 0)]),
    );
}

#[test]
fn test_swar_step_then_narrow_compact_overflow() {
    // B4 pair sums > 15 → overflow, keep widened result
    let mut data = [
        pack_b4x16([8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        pack_b4x16([0; 16]),
    ];
    swar_step(&mut data, BucketWidth::B4);
    assert!(swar_has_overflow(&data, BucketWidth::B4));
    assert_eq!(data[0] & 0xFF, 17, "first byte sum should be 17");
}

// -----------------------------------------------------------------------
// Adaptive merge (downscale) integration tests
// -----------------------------------------------------------------------

#[test]
fn test_downscale_width_behavior() {
    // Helper: insert ops into a B4 histogram at scale 0,
    // downscale(1), and verify the expected final width.
    let check = |ops: &[(f64, u64)], expected_width: BucketWidth, label: &str| {
        let mut h: Histogram<16> =
            Histogram::new().with_scale(0).with_min_bucket_width(BucketWidth::B4);
        for &(v, incr) in ops {
            h.record(v, incr).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::B4, "{label}: pre-check");
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.bucket_width(), expected_width, "{label}");
    };

    check(&[(2.0, 5), (4.0, 7)], BucketWidth::B4, "small sums stay B4");
    check(&[(2.0, 10), (4.0, 10)], BucketWidth::U8, "overflow widens to U8");
    check(&[(2.0, 15), (4.0, 15)], BucketWidth::U8, "max B4 overflow widens to U8");
}

#[test]
fn test_downscale_many_indices_preserves_width() {
    // Many small counts at spread-out indices → pair sums ≤ 2, stays B4.
    let mut h: Histogram<16> = Histogram::new().with_scale(0)
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);
    for i in 0..8 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    assert_eq!(h.bucket_width(), BucketWidth::B4);
    assert_total_conserved(&mut h, 1);
    assert_eq!(h.bucket_width(), BucketWidth::B4);
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
// swar_step isolation tests
// -----------------------------------------------------------------------

#[test]
fn test_swar_step_single_word() {
    let cases: &[(BucketWidth, u64, u64, &str)] = &[
        // B4: 16 nibbles → 8 byte pair sums
        (
            BucketWidth::B4,
            pack_b4x16([1, 2, 3, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pack_u8x8([3, 3, 0, 15, 0, 0, 0, 0]),
            "b4",
        ),
        // U8: 8 bytes → 4 short pair sums
        (
            BucketWidth::U8,
            pack_u8x8([100, 200, 50, 50, 0, 0, 0, 0]),
            pack_u16x4([300, 100, 0, 0]),
            "u8",
        ),
        // U16: 4 shorts → 2 int pair sums
        (
            BucketWidth::U16,
            pack_u16x4([1000, 2000, 3000, 4000]),
            pack_u32x2(3000, 7000),
            "u16",
        ),
        // U32: 2 ints → 1 u64 sum
        (
            BucketWidth::U32,
            pack_u32x2(100000, 200000),
            300000,
            "u32",
        ),
    ];
    for &(width, input, expected, label) in cases {
        let mut data = [input];
        swar_step(&mut data, width);
        assert_eq!(data[0], expected, "{label}: got {:#018x}", data[0]);
    }
}

#[test]
fn test_swar_step_b4_max_pair_sum() {
    // Two 15s: sum = 30, which fits in U8 (max 255).
    let mut data = [pack_b4x16([
        15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ])];
    swar_step(&mut data, BucketWidth::B4);
    assert_eq!(data[0] & 0xFF, 30);
}

// -----------------------------------------------------------------------
// scale_reduction tests
// -----------------------------------------------------------------------

#[test]
fn test_scale_reduction() {
    let cases: &[(i32, i32, i32, i32, &str)] = &[
        (0, 4, 10, 0, "fits"),
        (0, 10, 10, 1, "exact boundary"),
        (0, 39, 10, 2, "double"),
        (-10, 10, 10, 2, "negative indices"),
        (5, 5, 10, 0, "zero span"),
    ];
    for &(low, high, cap, expected, label) in cases {
        assert_eq!(
            scale_reduction(HighLow { low, high }, cap),
            expected,
            "{label}"
        );
    }
}

// -----------------------------------------------------------------------
// Adaptive merge (scalar fallback) tests
// -----------------------------------------------------------------------

#[test]
fn test_bucket_downscale_scalar_preserves_total_no_overflow() {
    // Two values at adjacent indices with small counts → scalar merge
    // should sum them without widening.
    let mut h: Histogram<16> = Histogram::new().with_scale(0).with_literal_mode(false);
    h.record(2.0, 3).unwrap(); // index 0
    h.record(4.0, 5).unwrap(); // index 1

    let width_before = h.bucket_width();
    assert_total_conserved(&mut h, 1);
    // Small counts (3+5=8 ≤ 15) → should stay at B4.
    assert_eq!(h.bucket_width(), width_before);
}

#[test]
fn test_bucket_downscale_scalar_preserves_total_with_overflow() {
    // Fill enough that pair sums exceed B4 max (15).
    let mut h: Histogram<16> = Histogram::new().with_scale(0);
    h.record(2.0, 10).unwrap(); // index 0, count 10
    h.record(4.0, 10).unwrap(); // index 1, count 10

    assert_total_conserved(&mut h, 1);
    // 10+10=20 > 15 → must widen to U8.
    assert_eq!(h.bucket_width(), BucketWidth::U8);
}

#[test]
fn test_downscale_multi_step_preserves_total() {
    // Insert 4 values at separate indices, then downscale by 3.
    let mut h: Histogram<16> = Histogram::new().with_scale(0)
        .with_min_bucket_width(BucketWidth::B4)
        .with_literal_mode(false);
    for i in 0..4 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    let total_before = bucket_total(&mut h);
    assert_eq!(total_before, 4);
    assert_eq!(h.bucket_width(), BucketWidth::B4);

    h.downscale(3).unwrap();

    let total_after = bucket_total(&mut h);
    assert_eq!(total_after, 4, "total changed after 3-step downscale");
    // All counts are 1, pair sums ≤ 2 → should stay at B4.
    assert_eq!(h.bucket_width(), BucketWidth::B4);
}

#[test]
fn test_downscale_multi_step_through_alignment_boundary() {
    // Start with base aligned to 16, downscale 5+ times so base
    // goes from even to odd and back. Verify totals survive.
    let mut h: Histogram<16> = Histogram::new().with_scale(0).with_literal_mode(false);
    for i in 0..8 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    let total_before = bucket_total(&mut h);
    assert_eq!(total_before, 8);

    // 5 steps: base starts at e.g. -16 >> 5 = -1 (odd), so the
    // 5th step must use scalar fallback.
    h.downscale(5).unwrap();

    let total_after = bucket_total(&mut h);
    assert_eq!(total_after, 8, "total changed after 5-step downscale");
}

#[test]
fn test_downscale_odd_base_preserves_total() {
    // Downscale through odd-base steps using SWAR-shift.
    let mut h: Histogram<16> = Histogram::new().with_scale(0).with_literal_mode(false);
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

#[cfg(has_lookup_table)]
#[test]
fn test_odd_base_downscale_preserves_total() {
    // Start at max scale so we have room to downscale.
    let mut h: Histogram<16> = Histogram::new().with_scale(8);
    h.record(1.5, 5).unwrap();
    h.record(1.6, 7).unwrap();

    let total_before = bucket_total(&mut h);

    // Downscale until base is odd (at most 15 steps to stay above MIN_SCALE).
    let mut tries = 0;
    while h.index_base & 1 == 0 && tries < 15 {
        h.downscale(1).unwrap();
        tries += 1;
    }

    if h.index_base & 1 != 0 {
        // One more downscale at odd base — the saved-value fix-up
        // is handled internally by pairwise_merge.
        h.downscale(1).unwrap();

        let total_after = bucket_total(&mut h);
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
        let mut h: Histogram<16> = Histogram::new().with_scale(0)
            .with_min_bucket_width(BucketWidth::B4)
            .with_literal_mode(false);
        for i in 0..16 {
            h.update(2.0_f64.powi(i)).unwrap();
        }
        assert_eq!(h.bucket_width(), BucketWidth::B4);
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.bucket_width(), BucketWidth::B4, "b4 sparse stays");
    }

    // U8 dense: 200+200=400 > 255 → widens to U16
    {
        let mut h: Histogram<16> = Histogram::new().with_scale(0);
        h.record(2.0, 200).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.record(4.0, 200).unwrap();
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.bucket_width(), BucketWidth::U16, "u8 dense widens to u16");
    }

    // U8 sparse: 100+50=150 ≤ 255 → stays U8
    {
        let mut h: Histogram<16> = Histogram::new().with_scale(0);
        h.record(2.0, 100).unwrap();
        assert_eq!(h.bucket_width(), BucketWidth::U8);
        h.record(4.0, 50).unwrap();
        assert_total_conserved(&mut h, 1);
        assert_eq!(h.bucket_width(), BucketWidth::U8, "u8 sparse stays");
    }
}

// -----------------------------------------------------------------------
// Sum conservation stress tests
// -----------------------------------------------------------------------

#[cfg(has_lookup_table)]
#[test]
fn test_sum_conservation_through_full_widen_chain() {
    // Fill a histogram with enough count magnitude to force widening
    // at every level: B4(max 15) → U8(255) → U16(65535) → U32 → U64.
    // Adjacent pairs sum to 1000, forcing overflow at B4, U8.
    let mut h: Histogram<16> = Histogram::new().with_scale(8);
    h.record(1.5, 500).unwrap();
    h.record(1.6, 500).unwrap();
    // Start at U16 (500 > 255).
    assert_eq!(bucket_total(&mut h), 1000);

    // Add more to push into U32 territory.
    h.record(1.7, 65000).unwrap();
    h.record(1.8, 65000).unwrap();

    // Downscale up to 10 steps, verify total at each.
    assert_total_conserved(&mut h, 10);
}

#[test]
fn test_sum_conservation_scalar_path() {
    // Force the scalar path and check totals at each step.
    let mut h: Histogram<16> = Histogram::new().with_scale(0).with_literal_mode(false);
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
    let mut h: Histogram<16> = Histogram::new().with_scale(0);
    h.record(2.0, 15).unwrap(); // fills B4 to max
    h.record(4.0, 15).unwrap();
    h.record(8.0, 15).unwrap();
    h.record(16.0, 15).unwrap();
    assert_eq!(bucket_total(&mut h), 60);

    assert_total_conserved(&mut h, 6);
}

// -----------------------------------------------------------------------
// Narrow function: slot ordering correctness
// -----------------------------------------------------------------------

#[test]
fn test_narrow_u8_to_b4_preserves_slot_order() {
    // Verify that byte[i] maps to nibble[i], not a permuted position.
    for i in 0..8u8 {
        let mut bytes = [0u8; 8];
        bytes[i as usize] = (i + 1).min(15);
        let input = pack_u8x8(bytes);
        let result = narrow_word(input, BucketWidth::B4);

        // Extract nibble i from the result (low 32 bits).
        let nibble = (result >> (i as u64 * 4)) & 0xF;
        assert_eq!(
            nibble,
            (i + 1).min(15) as u64,
            "nibble {i}: expected {}, got {nibble}",
            (i + 1).min(15)
        );

        // All other nibbles should be zero.
        for j in 0..8u8 {
            if j != i {
                let other = (result >> (j as u64 * 4)) & 0xF;
                assert_eq!(
                    other, 0,
                    "nibble {j} should be 0 when only nibble {i} is set, got {other}"
                );
            }
        }
    }
}

#[test]
fn test_narrow_u16_to_u8_preserves_slot_order() {
    for i in 0..4u16 {
        let mut shorts = [0u16; 4];
        shorts[i as usize] = (i + 1).min(255);
        let input = pack_u16x4(shorts);
        let result = narrow_word(input, BucketWidth::U8);

        let byte = (result >> (i as u64 * 8)) & 0xFF;
        assert_eq!(
            byte,
            (i + 1).min(255) as u64,
            "byte {i}: expected {}, got {byte}",
            (i + 1).min(255)
        );
    }
}

#[test]
fn test_narrow_u32_to_u16_preserves_slot_order() {
    for i in 0..2u32 {
        let lo = if i == 0 { 42 } else { 0 };
        let hi = if i == 1 { 42 } else { 0 };
        let input = pack_u32x2(lo, hi);
        let result = narrow_word(input, BucketWidth::U16);

        let short = (result >> (i as u64 * 16)) & 0xFFFF;
        assert_eq!(short, 42, "short {i}: expected 42, got {short}");
    }
}

// -----------------------------------------------------------------------
// swar_has_overflow: boundary values
// -----------------------------------------------------------------------

#[test]
fn test_swar_has_overflow_multi_word() {
    // Overflow only in the last word — must still be detected.
    let data = [
        pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
        pack_u8x8([0, 0, 0, 0, 0, 0, 0, 0]),
        pack_u8x8([0, 0, 0, 0, 0, 0, 0, 16]),
    ];
    assert!(swar_has_overflow(&data, BucketWidth::B4));
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
        let total = bucket_total(&mut h);
        let view = h.view();
        assert_eq!(
            total,
            view.count(),
            "After inserting {v}: bucket total ({total}) != count ({})\n  \
             scale={} width={:?}",
            view.count(),
            view.scale(),
            h.bucket_width()
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
        let total = bucket_total(&mut h);
        let view = h.view();
        assert_eq!(
            total,
            view.count(),
            "After values[{vi}]={v}: bucket total ({total}) != count ({})\n  \
             scale={} width={:?}",
            view.count(),
            view.scale(),
            h.bucket_width()
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
        h.downscale(1).unwrap();
        let current = bucket_total(h);
        assert_eq!(
            current,
            total,
            "total changed at step {step}: {current} != {total}, \
             width={:?}",
            h.bucket_width()
        );
    }
}

/// Helper: build two same-size histograms from ops, merge, and
/// assert count and bucket-total invariants.
fn build_histogram<const N: usize>(ops: &[(f64, u64)]) -> Histogram<N> {
    let mut h = Histogram::<N>::new();
    for &(v, incr) in ops {
        h.record(v, incr).unwrap();
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

/// Helper: build a bucket-mode (non-literal) histogram from plain f64 values.
fn build_bucket<const N: usize>(values: &[f64]) -> Histogram<N> {
    let mut h = Histogram::<N>::new().with_literal_mode(false);
    for &v in values {
        h.update(v).unwrap();
    }
    h
}

fn assert_merge_result<const N: usize>(
    h: &mut Histogram<N>,
    left: &[(f64, u64)],
    right: &[(f64, u64)],
    label: &str,
) {
    let expected: u64 = left.iter().chain(right).map(|&(_, i)| i).sum();
    let count = h.view().count();
    assert_eq!(count, expected, "{label}: count mismatch");
    let bt = bucket_total(h);
    assert!(bt <= count, "{label}: bt={bt} > count={count}");
}

fn merge_check<const N: usize>(left: &[(f64, u64)], right: &[(f64, u64)], label: &str) {
    let (mut h1, h2) = (build_histogram::<N>(left), build_histogram::<N>(right));
    h1.merge_from(&h2).unwrap();
    assert_merge_result(&mut h1, left, right, label);
}

/// Helper: build two different-size histograms from ops, merge via
/// `merge_from_other`, and assert count and bucket-total invariants.
fn merge_check_cross<const N: usize, const M: usize>(
    left: &[(f64, u64)],
    right: &[(f64, u64)],
    label: &str,
) {
    let (mut h1, h2) = (build_histogram::<N>(left), build_histogram::<M>(right));
    h1.merge_from_other(&h2).unwrap();
    assert_merge_result(&mut h1, left, right, label);
}

#[test]
fn test_merge_needs_downscale_in_raw() {
    let mut h1 = Histogram::<8>::new();
    h1.update(1.0).unwrap();

    let mut h2 = Histogram::<8>::new();
    h2.update(1e30).unwrap();
    h2.update(1e-30).unwrap();

    let h2_view = h2.view();
    let h2_count = h2_view.count();
    let h2_sum = h2_view.sum();
    let h2_min = h2_view.min();
    let h2_max = h2_view.max();
    let h2_scale = h2_view.scale();
    let b2 = h2_view.positive();
    h1.merge_from_raw(
        &Stats {
            count: h2_count,
            sum: h2_sum,
            min: h2_min,
            max: h2_max,
        },
        &BucketDescriptor {
            scale: h2_scale,
            offset: b2.offset(),
            len: b2.len(),
        },
        |i| b2.at(i),
    )
    .unwrap();
    assert_eq!(h1.view().count(), 3);
    assert_eq!(bucket_total(&mut h1), 3);
}

/// Regression: large weighted inserts of subnormal + normal value
/// trigger bucket_widen during downscale, corrupting bucket totals.
#[test]
fn test_weighted_subnormal_merge_bucket_total() {
    let v1 = f64::from_le_bytes([32, 0, 66, 0, 0, 98, 65, 3]); // ~5.44e-293, subnormal as f32
    let v2 = f64::from_le_bytes([0, 32, 0, 66, 0, 98, 65, 64]); // ~34.77

    let left_ops: Vec<(f64, u64)> =
        vec![(v1, 3), (v2, 1), (v1, 12), (v2, 4), (v1, 192), (v2, 64)];
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

    h1.record(2.8396262443943004e+238, 40).unwrap();
    h0.record(2.635549485807631e-82, 1).unwrap();

    // Step 3: merge h0 into h1
    h1.merge_from(&h0).unwrap();
    assert_eq!(h1.view().count(), 41);

    // Step 4: merge h1 into h0
    if h0.merge_from(&h1).is_ok() {
        let count = h0.view().count();
        let bt = bucket_total(&mut h0);
        assert!(bt <= count, "bucket total ({bt}) exceeds count ({count})");
    }
}

#[test]
fn test_merge_p32_bucket_len_after_merge_chain() {
    use crate::Mapping;

    let v0: f64 = 5.653943197254256e-308;
    let v1: f64 = 2.740490672504645e-61;

    let mut h0 = Histogram::<8>::new();
    let mut h1 = Histogram::<8>::new();

    h0.record(v0, 1).unwrap();
    h1.record(v1, 1).unwrap();

    // Merge chain: h0→h1, h0→h1, h1→h0
    h1.merge_from(&h0).unwrap();
    h1.merge_from(&h0).unwrap();
    h0.merge_from(&h1).unwrap();

    // Insert many zeros
    for _ in 0..150 {
        h0.record(0.0, 1).unwrap();
    }

    // Verify bucket structure
    let h0_view = h0.view();
    let scale = h0_view.scale();
    let mapping = Mapping::new(scale).unwrap();

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
        assert!(b.at(0) > 0, "leading zero bucket");
        assert!(b.at(b.len() - 1) > 0, "trailing zero bucket");
    }
}

// -----------------------------------------------------------------------
// Literal mode tests
// -----------------------------------------------------------------------

#[test]
fn test_literal_mode_default() {
    let mut h: Histogram<8> = Histogram::new();
    assert!(h.is_literal());
    let v = h.view();
    assert_eq!(v.count(), 0);
    assert_eq!(v.sum(), 0.0);
}

#[test]
fn test_literal_mode_stores_values() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();
    h.update(2.0).unwrap();
    h.update(4.0).unwrap();
    assert!(h.is_literal());
    let v = h.view();
    assert_eq!(v.count(), 3);
    assert_eq!(v.sum(), 7.0);
    assert_eq!(v.min(), 1.0);
    assert_eq!(v.max(), 4.0);
}

#[test]
fn test_literal_mode_capacity() {
    // Histogram<8>: all 8 words available for literals.
    let mut h: Histogram<8> = Histogram::new();
    for i in 0..8 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    assert!(h.is_literal(), "should still be literal with 8 values");
    assert_eq!(h.view().count(), 8);

    // 9th value should trigger promotion.
    h.update(256.0).unwrap();
    assert!(!h.is_literal(), "should promote on 9th value");
    assert_eq!(h.view().count(), 9);
}

#[test]
fn test_literal_mode_opt_out() {
    let mut h: Histogram<8> = Histogram::new().with_literal_mode(false);
    assert!(!h.is_literal());
    h.update(1.0).unwrap();
    assert!(!h.is_literal());
}

#[test]
fn test_literal_mode_recreate_resets() {
    let mut h: Histogram<8> = Histogram::new();
    // Fill beyond literal capacity (8 slots) to trigger promotion.
    for i in 0..9 {
        h.update(2.0_f64.powi(i)).unwrap();
    }
    assert!(!h.is_literal());
    // Re-creating the histogram resets to literal mode.
    h = Histogram::new();
    assert!(h.is_literal());
    assert_eq!(h.view().count(), 0);
}

#[test]
fn test_literal_mode_zero_values() {
    // Zero values don't consume literal slots.
    let mut h: Histogram<8> = Histogram::new();
    h.update(0.0).unwrap();
    h.update(0.0).unwrap();
    h.update(0.0).unwrap();
    assert!(h.is_literal());
    let v = h.view();
    assert_eq!(v.count(), 3);
    assert_eq!(v.sum(), 0.0);
    // Bucket view should be empty (zeros are tracked in MMSC only).
    assert!(v.positive().is_empty());
}

#[test]
fn test_literal_mode_identical_values() {
    let mut h: Histogram<8> = Histogram::new();
    for _ in 0..6 {
        h.update(42.0).unwrap();
    }
    assert!(h.is_literal());
    let v = h.view();
    let buckets = v.positive();
    assert_eq!(v.count(), 6);
    assert_eq!(v.sum(), 252.0);
    assert_eq!(v.min(), 42.0);
    assert_eq!(v.max(), 42.0);
    // All map to the same bucket → len should be 1.
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets.at(0), 6);
}

#[test]
fn test_literal_mode_bucket_view() {
    // Compare literal-mode virtual view to bucket-mode actual view.
    assert_literal_matches_bucket(&[1.0, 2.0, 4.0]);
}

#[test]
fn test_literal_promotion_optimal_scale() {
    // Verify that promotion picks the optimal scale (matching what
    // bucket mode would choose given the same values).
    let values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
    assert_literal_matches_bucket(&values);
}

#[test]
fn test_literal_record() {
    let mut h: Histogram<8> = Histogram::new();
    // 3 copies of the same value.
    h.record(5.0, 3).unwrap();
    assert!(h.is_literal());
    assert_eq!(h.view().count(), 3);
    assert_eq!(h.view().sum(), 15.0);
}

#[test]
fn test_literal_record_overflow() {
    // Histogram<8>: 8 literal slots. incr=9 should promote.
    let mut h: Histogram<8> = Histogram::new();
    h.record(3.25, 9).unwrap();
    assert!(!h.is_literal());
    assert_eq!(h.view().count(), 9);
}

#[test]
fn test_literal_merge_literal_into_literal() {
    let mut a = build_from_values::<8>(&[1.0, 2.0]);
    let b = build_from_values::<8>(&[4.0, 8.0]);
    a.merge_from(&b).unwrap();
    assert_stats(&mut a, 4, 15.0, 1.0, 8.0);
}

#[test]
fn test_literal_merge_literal_into_bucket() {
    let mut collector = build_bucket::<16>(&[1.0, 2.0]);
    let source = build_from_values::<16>(&[4.0, 8.0]);
    assert!(source.is_literal());
    collector.merge_from(&source).unwrap();
    assert_stats(&mut collector, 4, 15.0, 1.0, 8.0);
}

#[test]
fn test_literal_merge_bucket_into_literal() {
    let mut collector = build_from_values::<16>(&[1.0]);
    assert!(collector.is_literal());
    let source = build_bucket::<16>(&[4.0, 8.0]);
    collector.merge_from(&source).unwrap();
    assert!(!collector.is_literal());
    assert_stats(&mut collector, 3, 13.0, 1.0, 8.0);
}

#[test]
fn test_literal_merge_preserves_source() {
    let mut collector = build_bucket::<16>(&[1.0]);
    let mut source = build_from_values::<16>(&[4.0]);
    assert!(source.is_literal());
    collector.merge_from(&source).unwrap();
    assert!(source.is_literal());
    assert_stats(&mut source, 1, 4.0, 4.0, 4.0);
}

#[test]
fn test_literal_merge_cross_size() {
    let mut collector = build_bucket::<16>(&[1.0]);
    let source = build_from_values::<8>(&[4.0, 8.0]);
    assert!(source.is_literal());
    collector.merge_from_other(&source).unwrap();
    assert_stats(&mut collector, 3, 13.0, 1.0, 8.0);
}

#[test]
fn test_merge_literal_source_not_promoted() {
    // Verify that merging a literal source into a bucket destination
    // inserts literal values one by one without promoting the source.
    // The source must remain in literal mode after merge.

    // Same-size merge: literal source into bucket dest.
    let mut dest = build_bucket::<16>(&[1.0, 2.0, 3.0]);
    let source = build_from_values::<16>(&[10.0, 20.0, 30.0]);
    assert!(source.is_literal());
    dest.merge_from(&source).unwrap();
    assert!(source.is_literal(), "same-size merge must not promote source");
    assert_eq!(dest.view().count(), 6);

    // Cross-size merge: small literal source into large bucket dest.
    let mut big = build_bucket::<16>(&[1.0, 2.0, 3.0]);
    let small = build_from_values::<8>(&[100.0, 200.0]);
    assert!(small.is_literal());
    big.merge_from_other(&small).unwrap();
    assert!(small.is_literal(), "cross-size merge must not promote source");
    assert_eq!(big.view().count(), 5);

    // Wide-range literal values: ensure even with values spanning
    // many scales, the source stays literal and dest absorbs them
    // correctly through incremental insertion.
    let mut dest2 = build_bucket::<16>(&[1.0]);
    let source2 = build_from_values::<16>(&[1e-200, 1e200]);
    assert!(source2.is_literal());
    dest2.merge_from(&source2).unwrap();
    assert!(source2.is_literal(), "wide-range merge must not promote source");
    assert_eq!(dest2.view().count(), 3);
    assert_eq!(bucket_total(&mut dest2), 3, "all three non-zero values should be in buckets");
}

#[test]
fn test_literal_equivalence_with_bucket_mode() {
    // Verify that a promoted literal histogram and a bucket-mode
    // histogram produce the same bucket totals for the same inputs.
    let values = [1.5, 2.7, 0.3, 100.0, 42.0, 7.7, 13.0, 55.5, 999.0];
    assert_literal_matches_bucket(&values);
}

#[test]
fn test_literal_subnormal_values() {
    let subnormal = 5.0e-324_f64; // smallest subnormal
    let mut h: Histogram<8> = Histogram::new();
    h.update(subnormal).unwrap();
    h.update(subnormal).unwrap();
    assert!(h.is_literal());
    assert_eq!(h.view().count(), 2);
    // With f64 stats, subnormals are preserved exactly.
    // Check bucket view to verify data integrity.
    assert_eq!(h.view().positive().at(0), 2);
}

#[test]
fn test_literal_empty_bucket_view() {
    let mut h: Histogram<8> = Histogram::new();
    assert!(h.is_literal());
    let v = h.view();
    let buckets = v.positive();
    assert!(buckets.is_empty());
    assert_eq!(buckets.len(), 0);
}

#[test]
fn test_literal_scale_matches_bucket() {
    assert_literal_matches_bucket(&[1.0, 1024.0]);
}

#[test]
fn test_literal_debug_format() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(1.0).unwrap();
    let debug = format!("{:?}", h);
    assert!(
        debug.contains("literal"),
        "Debug should mention literal mode"
    );
}

// -- Quantile estimation tests (require `boundary` feature) ----------------

#[cfg(feature = "boundary")]
mod quantile_tests {
    use super::*;
    use std::eprintln;

    #[test]
    fn test_quantile_empty_histogram() {
    let mut h: Histogram<8> = Histogram::new();
    let qs = [0.0, 0.5, 1.0];
    let v = h.view();
    let vals: Vec<_> = v.quantiles(&qs).collect();
    assert_eq!(vals.len(), 3);
    for v in &vals {
        assert!(v.value.is_nan(), "empty histogram should yield NaN");
    }
}

#[test]
fn test_quantile_single_value() {
    let mut h: Histogram<8> = Histogram::new();
    h.update(42.0).unwrap();
    let qs = [0.0, 0.5, 1.0];
    let v = h.view();
    let vals: Vec<_> = v.quantiles(&qs).collect();
    assert_eq!(vals[0].value, 42.0); // p0 = min
    assert_eq!(vals[2].value, 42.0); // p100 = max
    assert!(
        (vals[1].value - 42.0).abs() < 1.0,
        "p50 = {} should be near 42.0",
        vals[1].value
    );
}

#[test]
fn test_quantile_with_zeros() {
    let mut h: Histogram<8> = Histogram::new();
    for _ in 0..90 {
        h.update(0.0).unwrap();
    }
    for _ in 0..10 {
        h.update(100.0).unwrap();
    }

    let qs = [0.0, 0.5, 0.89, 0.95, 1.0];
    let v = h.view();
    let vals: Vec<_> = v.quantiles(&qs).collect();
    assert_eq!(vals[0].value, 0.0, "p0 = min = 0");
    assert_eq!(vals[1].value, 0.0, "p50 should be 0 (90% are zeros)");
    assert_eq!(vals[2].value, 0.0, "p89 should still be 0");
    assert!(vals[3].value > 0.0, "p95 should be > 0");
    assert_eq!(vals[4].value, 100.0, "p100 = max");
}

/// Tests monotonicity, p0=min, p100=max, clamping, and ExactSizeIterator.
#[test]
fn test_quantile_properties() {
    let mut h: Histogram<8> = Histogram::new();
    for v in 1..=1000 {
        h.update(v as f64).unwrap();
    }

    let qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
    let view = h.view();
    let iter = view.quantiles(&qs);
    assert_eq!(iter.len(), qs.len(), "ExactSizeIterator");
    let vals: Vec<_> = iter.collect();

    // p0 = min, p100 = max.
    assert_eq!(vals[0].value, view.min());
    assert_eq!(vals[qs.len() - 1].value, view.max());

    // All values clamped to [min, max].
    for v in &vals {
        assert!(
            v.value >= view.min() && v.value <= view.max(),
            "q={} value {} outside [{}, {}]",
            v.quantile,
            v.value,
            view.min(),
            view.max()
        );
    }

    // Monotonically non-decreasing.
    for w in vals.windows(2) {
        assert!(
            w[0].value <= w[1].value,
            "not monotonic: q{}={} > q{}={}",
            w[0].quantile,
            w[0].value,
            w[1].quantile,
            w[1].value,
        );
    }

    // Rough sanity: p50 should be near 500.
    assert!(
        (vals[3].value - 500.0).abs() < 100.0,
        "p50 = {} should be near 500",
        vals[3].value
    );
}

/// All-same-value histogram: every quantile should return that value.
#[test]
fn test_quantile_all_same_value() {
    let mut h: Histogram<8> = Histogram::new();
    for _ in 0..100 {
        h.update(7.0).unwrap();
    }
    let qs = [0.0, 0.25, 0.5, 0.75, 1.0];
    let view = h.view();
    let vals: Vec<_> = view.quantiles(&qs).collect();
    for v in &vals {
        assert_eq!(v.value, 7.0, "all-same histogram: q{}={}", v.quantile, v.value);
    }
}

/// Monotonicity with many closely-spaced quantiles.
#[test]
fn test_quantile_fine_grained_monotonicity() {
    let mut h: Histogram<16> = Histogram::new();
    for v in 1..=500 {
        h.update(v as f64).unwrap();
    }
    let qs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let view = h.view();
    let vals: Vec<_> = view.quantiles(&qs).collect();

    assert_eq!(vals[0].value, view.min());
    assert_eq!(vals[100].value, view.max());

    for w in vals.windows(2) {
        assert!(
            w[0].value <= w[1].value,
            "not monotonic at q={}: {} > {}",
            w[1].quantile, w[0].value, w[1].value,
        );
    }
}

// -- Distribution-based goodness-of-fit test ------------------------------

/// Error function via Horner form of the Abramowitz & Stegun
/// approximation (max error ~1.5 × 10⁻⁷).
fn erf(x: f64) -> f64 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-a * a).exp();
    if x < 0.0 { -result } else { result }
}

/// Computes reduced χ²/df of histogram bucket counts vs a theoretical
/// CDF. Bins with expected count < 5 are merged with neighbours.
fn reduced_chi_squared<const N: usize>(
    h: &mut Histogram<N>,
    cdf: fn(f64) -> f64,
) -> f64 {
    let histogram_view = h.view();
    let scale = histogram_view.scale();
    let mapping = Mapping::new(scale).unwrap();
    let total = histogram_view.count() as f64;
    let view = histogram_view.positive();

    // Collect (observed, expected) per bucket, merging on the fly.
    let mut merged: Vec<(f64, f64)> = Vec::new();
    let (mut acc_o, mut acc_e) = (0.0, 0.0);
    for pos in 0..view.len() {
        let index = view.offset() + pos as i32;
        let lower = mapping.lower_boundary(index).unwrap_or(0.0);
        let upper = mapping.lower_boundary(index + 1).unwrap_or(f64::INFINITY);
        acc_o += view.at(pos) as f64;
        acc_e += total * (cdf(upper) - cdf(lower));
        if acc_e >= 5.0 {
            merged.push((acc_o, acc_e));
            acc_o = 0.0;
            acc_e = 0.0;
        }
    }
    if acc_e > 0.0 {
        if let Some(last) = merged.last_mut() {
            last.0 += acc_o;
            last.1 += acc_e;
        }
    }

    let df = merged.len().saturating_sub(1).max(1);
    let chi2: f64 = merged.iter().map(|(o, e)| (o - e).powi(2) / e).sum();
    chi2 / df as f64
}

/// Chi-squared goodness-of-fit: validates histogram bucket counts
/// against three theoretical CDFs and spot-checks the p50 quantile.
#[test]
fn test_goodness_of_fit() {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Exp, LogNormal, Uniform};

    struct Case {
        name: &'static str,
        seed: u64,
        sample: fn(&mut rand::rngs::StdRng) -> f64,
        cdf: fn(f64) -> f64,
        p50_expected: f64,
    }

    let cases = [
        Case {
            name: "Exponential(1.5)",
            seed: 42,
            sample: |rng| Exp::new(1.5).unwrap().sample(rng),
            cdf: |x| 1.0 - (-1.5 * x).exp(),
            p50_expected: core::f64::consts::LN_2 / 1.5,
        },
        Case {
            name: "LogNormal(2, 0.5)",
            seed: 123,
            sample: |rng| LogNormal::new(2.0, 0.5).unwrap().sample(rng),
            cdf: |x| {
                0.5 * (1.0 + erf((x.ln() - 2.0) / (0.5 * std::f64::consts::SQRT_2)))
            },
            p50_expected: (2.0_f64).exp(),
        },
        Case {
            name: "Uniform(10, 500)",
            seed: 999,
            sample: |rng| Uniform::new(10.0, 500.0).sample(rng),
            cdf: |x| ((x - 10.0) / 490.0).clamp(0.0, 1.0),
            p50_expected: 255.0,
        },
    ];

    for case in &cases {
        let mut rng = rand::rngs::StdRng::seed_from_u64(case.seed);
        let mut h: Histogram<160> = Histogram::new();
        for _ in 0..1_000_000 {
            h.update((case.sample)(&mut rng)).unwrap();
        }

        let reduced = reduced_chi_squared(&mut h, case.cdf);
        eprintln!("{}: χ²/df={reduced:.4}", case.name);
        assert!(
            reduced < 2.0,
            "{}: reduced χ²={reduced:.4} exceeds 2.0",
            case.name,
        );

        // Spot-check p0, p50, p100.
        let qs = [0.0, 0.5, 1.0];
        let view = h.view();
        let vals: Vec<_> = view.quantiles(&qs).collect();
        assert_eq!(vals[0].value, view.min(), "{}: p0 must equal min", case.name);
        assert_eq!(vals[2].value, view.max(), "{}: p100 must equal max", case.name);

        let p50_err = ((vals[1].value - case.p50_expected) / case.p50_expected).abs();
        assert!(
            p50_err < 0.05,
            "{}: p50={:.4} expected={:.4} err={:.2}%",
            case.name,
            vals[1].value,
            case.p50_expected,
            p50_err * 100.0,
        );
    }
}
} // mod quantile_tests

#[test]
fn repro_fuzz_histogram_oracle_offset() {
    // Regression: subnormals must map to the same bucket as MIN_VALUE
    // at all positive scales. Previously, logarithm and lookup-table
    // mappers treated subnormals as distinct values, producing wrong
    // bucket indices that disagreed across scales.
    let subnormal = 1.3633843689306e-310f64;
    let normal = 2.2251438848883923e-308f64;
    let min_value = crate::float64::MIN_VALUE;

    // At every scale, the subnormal must have the same index as MIN_VALUE.
    for s in 0..=max_scale() {
        let m = Mapping::new(s).unwrap();
        assert_eq!(
            m.map_to_index(subnormal),
            m.map_to_index(min_value),
            "subnormal must map to MIN_VALUE bucket at scale={s}"
        );
    }

    // Bucket mode and literal mode must agree.
    for literal in [true, false] {
        let mut h = Histogram::<8>::new().with_literal_mode(literal);
        h.update(subnormal).unwrap();
        h.update(normal).unwrap();

        let v = h.view();
        let mapping = Mapping::new(v.scale()).unwrap();
        let exp_offset = mapping.map_to_index(min_value)
            .min(mapping.map_to_index(normal));

        assert_eq!(
            v.positive().offset(),
            exp_offset,
            "literal={literal}: offset mismatch at scale={}",
            v.scale()
        );
    }
}

#[test]
fn repro_fuzz_merge_oracle_offset() {
    // Regression: subnormal value with large increments, merged across
    // histograms. The subnormal must map to MIN_VALUE's bucket.
    let subnormal = 5.580682928875e-312f64;
    let incrs: &[u64] = &[4194304, 16777216, 268435456, 4294967296];

    for literal in [true, false] {
        let mut right = Histogram::<8>::new().with_literal_mode(literal);
        for &incr in incrs {
            right.record(subnormal, incr).unwrap();
        }

        let mut left = Histogram::<8>::new().with_literal_mode(literal);
        left.merge_from(&right).unwrap();

        let v = left.view();
        let buckets = v.positive();
        let mapping = Mapping::new(v.scale()).unwrap();
        let exp_idx = mapping.map_to_index(crate::float64::MIN_VALUE);

        assert_eq!(
            buckets.offset(), exp_idx,
            "literal={literal}: offset mismatch at scale={}", v.scale()
        );
        let bt: u64 = buckets.iter().sum();
        assert!(bt <= v.count(),
            "literal={literal}: bucket total ({bt}) > count ({})", v.count());
    }
}

#[test]
fn repro_fuzz_stateful_bucket_total() {
    // Regression: merge atomicity. When a cross-size merge fails
    // (Overflow at MIN_SCALE), the destination must be unchanged.
    let v1: f64 = f64::from_bits(0x5829f8b15858ff40);
    let v2: f64 = f64::from_bits(0x004b000000000000);
    let v3: f64 = f64::from_bits(0x56562c0000000000);

    let mut pool0 = Histogram::<8>::new().with_literal_mode(false);
    pool0.record(v1, 12).unwrap();
    pool0.record(v2, 1).unwrap();
    pool0.record(v3, 1).unwrap();

    let mut big = Histogram::<16>::new().with_literal_mode(false);
    big.merge_from_other(&pool0).unwrap();

    // Snapshot big before the second (failing) merge.
    let big_before = big.clone();
    let merge2 = big.merge_from_other(&pool0);

    let vb = big.view();
    let bt: u64 = vb.positive().iter().sum();
    assert!(bt <= vb.count(),
        "bucket total ({bt}) exceeds count ({})", vb.count());

    if merge2.is_err() {
        // On failure, histogram must be unchanged.
        let mut vbefore = big_before.clone();
        assert_eq!(vb.count(), vbefore.view().count(),
            "failed merge must not change count");
    }
}

/// Reproducer for stateful_oracle crash: bucket len mismatch at scale=-10
/// after a failed record (Hammer with huge increment).
///
/// Root cause: record lacked snapshot/rollback, so a failed
/// widen_one_step (Overflow at MIN_SCALE) left buckets corrupted.
#[test]
fn repro_fuzz_stateful_update_atomicity() {
    let v1 = f64::from_bits(0x002f233d41000000); // 8.66e-308
    let v2 = f64::from_bits(0x2c2c2cac2c2c2c2c); // 6.595e-96
    let v3 = f64::from_bits(0x78ffffdb58585858); // 6.924e+274

    // Build pool[0]: two small values via insert + merge
    let mut h: Histogram<8> = Histogram::new();
    h.update(v1).unwrap();

    let mut donor: Histogram<8> = Histogram::new();
    donor.update(v2).unwrap();
    donor.update(v1).unwrap();
    h.merge_from(&donor).unwrap();

    // Snapshot before hammer
    let before = h.clone();

    // Hammer: huge increment at a wildly different exponent
    let result = h.record(v3, 8388608);

    if result.is_err() {
        // On failure the histogram MUST be unchanged.
        let hv = h.view();
        let mut bv = before.clone();
        let bvv = bv.view();
        assert_eq!(
            hv.positive().len(),
            bvv.positive().len(),
            "failed update must not change bucket len (scale={})",
            hv.scale()
        );
        assert_eq!(
            hv.count(),
            bvv.count(),
            "failed update must not change count"
        );
    }
}
