// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for exponential histogram mapping functions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use otel_expohisto::{table_scale, Scale};

/// 100 test values with significands roughly uniformly distributed across [1.0, 2.0).
/// 10 base significands × 10 magnitude groups = 100 values.
const TEST_VALUES: &[f64] = &[
    // Base significands (approx binary): 1.03, 1.13, 1.24, 1.35, 1.47, 1.58, 1.69, 1.78, 1.87, 1.96
    // Group 1: magnitude ~2^-996 to ~2^-664
    1.03e-300, 1.13e-280, 1.24e-260, 1.35e-240, 1.47e-220, 1.58e-210, 1.69e-205, 1.78e-201,
    1.87e-200, 1.96e-200, // Group 2: magnitude ~2^-332 to ~2^-166
    1.03e-100, 1.13e-90, 1.24e-80, 1.35e-70, 1.47e-65, 1.58e-60, 1.69e-55, 1.78e-52, 1.87e-51,
    1.96e-50, // Group 3: magnitude ~2^-33 to ~2^-7
    1.03e-10, 1.13e-8, 1.24e-7, 1.35e-6, 1.47e-5, 1.58e-4, 1.69e-3, 1.78e-3, 1.87e-2, 1.96e-2,
    // Group 4: magnitude ~2^-3 to ~2^0
    0.129, 0.226, 0.311, 0.423, 0.587, 0.632, 0.743, 0.891, 0.937, 0.981,
    // Group 5: magnitude ~2^0 (near 1)
    1.03, 1.13, 1.24, 1.35, 1.47, 1.58, 1.69, 1.78, 1.87, 1.96,
    // Group 6: magnitude ~2^1 to ~2^6
    2.06, 2.83, 3.72, 5.41, 7.35, 9.48, 12.4, 21.3, 33.7, 56.8,
    // Group 7: magnitude ~2^7 to ~2^13
    103.0, 226.0, 496.0, 778.0, 1350.0, 2470.0, 3690.0, 5580.0, 7140.0, 8920.0,
    // Group 8: magnitude ~2^16 to ~2^33
    1.03e5, 1.13e6, 1.24e7, 1.35e8, 1.47e8, 1.58e9, 1.69e9, 1.78e9, 1.87e10, 1.96e10,
    // Group 9: magnitude ~2^50 to ~2^166
    1.03e15, 1.13e20, 1.24e25, 1.35e30, 1.47e35, 1.58e40, 1.69e42, 1.78e45, 1.87e48, 1.96e50,
    // Group 10: magnitude ~2^200 to ~2^1000
    1.03e60, 1.13e100, 1.24e140, 1.35e170, 1.47e200, 1.58e220, 1.69e250, 1.78e275, 1.87e290,
    1.96e307,
];

fn bench_map_to_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_to_index_100x");

    // Non-positive scales (exponent mapping)
    for scale in [-10, -5, -1, 0] {
        let scale = Scale::new(scale).unwrap();
        group.bench_function(BenchmarkId::new("exponent", scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(scale.map_to_index(black_box(v)));
                }
            })
        });
    }

    // Positive scales - benchmark the lookup table algorithm via Scale
    let max = table_scale();
    let scales: Vec<i32> = [1, 4, 6, 8, 10, 12, 14, 16]
        .into_iter()
        .filter(|&s| s <= max)
        .collect();

    for &scale in &scales {
        let scale = Scale::new(scale).unwrap();
        group.bench_function(BenchmarkId::new("lookup", scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(scale.map_to_index(black_box(v)));
                }
            })
        });
    }

    // When bench-all is enabled, also benchmark the logarithm algorithm directly
    #[cfg(feature = "bench-all")]
    {
        let log_scales: Vec<i32> = [1, 4, 6, 8, 10, 12, 14, 16].to_vec();

        for &scale in &log_scales {
            group.bench_function(BenchmarkId::new("logarithm", scale), |b| {
                b.iter(|| {
                    for &v in TEST_VALUES {
                        black_box(otel_expohisto::logarithm::map_to_index(black_box(v), scale));
                    }
                })
            });
        }
    }

    group.finish();
}

fn bench_lower_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_boundary");

    // Non-positive scales (exponent mapping)
    for scale in [-10, -5, -1, 0] {
        let scale = Scale::new(scale).unwrap();
        // Use indices that are valid for this scale
        let indices: Vec<i32> = (-10..=10).collect();
        group.bench_function(BenchmarkId::new("exponent", scale), |b| {
            b.iter(|| {
                for &idx in &indices {
                    let _ = black_box(scale.lower_boundary(black_box(idx)));
                }
            })
        });
    }

    // Positive scales - benchmark up to table_scale()
    let max = table_scale();
    let scales: Vec<i32> = [1, 4, 8, 10, 12, 14, 16]
        .into_iter()
        .filter(|&s| s <= max)
        .collect();

    for scale in scales {
        let scale = Scale::new(scale).unwrap();
        // Use indices that are representative for this scale
        let indices: Vec<i32> = (-100..=100).collect();
        group.bench_function(BenchmarkId::new("logarithm", scale), |b| {
            b.iter(|| {
                for &idx in &indices {
                    let _ = black_box(scale.lower_boundary(black_box(idx)));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_map_to_index, bench_lower_boundary);
criterion_main!(benches);
