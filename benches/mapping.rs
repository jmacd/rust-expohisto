// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for exponential histogram mapping functions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_expohisto::{Mapping, map_to_index_lg};

/// Test values spanning the full range of normal f64 values.
const TEST_VALUES: &[f64] = &[
    1e-300, 1e-100, 1e-10, 0.001, 0.1, 0.5, 1.0, 1.5, 2.0, core::f64::consts::PI, 10.0, 100.0, 1e10, 1e100,
    1e300,
];

fn bench_map_to_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_to_index");

    // Non-positive scales (exponent mapping)
    for scale in [-10, -5, -1, 0] {
        let mapping = Mapping::new(scale).unwrap();
        group.bench_function(BenchmarkId::new("exponent", scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(mapping.map_to_index(black_box(v)));
                }
            })
        });
    }

    // Positive scales - these use lookup tables if enabled
    for scale in [1, 4, 6, 8, 10, 12, 14, 20] {
        let mapping = Mapping::new(scale).unwrap();
        let label = if cfg!(any(
            feature = "lookup-4",
            feature = "lookup-6",
            feature = "lookup-8",
            feature = "lookup-10",
            feature = "lookup-12",
            feature = "lookup-14"
        )) {
            "lookup_or_log"
        } else {
            "logarithm"
        };
        group.bench_function(BenchmarkId::new(label, scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(mapping.map_to_index(black_box(v)));
                }
            })
        });

        // Reference lg implementation for comparison
        let scale_factor = core::f64::consts::LOG2_E * (1u64 << scale) as f64;
        group.bench_function(BenchmarkId::new("lg_reference", scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(map_to_index_lg(black_box(v), scale, scale_factor));
                }
            })
        });
    }

    group.finish();
}

fn bench_lower_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_boundary");

    // Non-positive scales (exponent mapping)
    for scale in [-10, -5, -1, 0] {
        let mapping = Mapping::new(scale).unwrap();
        // Use indices that are valid for this scale
        let indices: Vec<i32> = (-10..=10).collect();
        group.bench_function(BenchmarkId::new("exponent", scale), |b| {
            b.iter(|| {
                for &idx in &indices {
                    let _ = black_box(mapping.lower_boundary(black_box(idx)));
                }
            })
        });
    }

    // Positive scales (logarithm mapping)
    for scale in [1, 4, 8, 10, 12, 14, 20] {
        let mapping = Mapping::new(scale).unwrap();
        // Use indices that are representative for this scale
        let indices: Vec<i32> = (-100..=100).collect();
        group.bench_function(BenchmarkId::new("logarithm", scale), |b| {
            b.iter(|| {
                for &idx in &indices {
                    let _ = black_box(mapping.lower_boundary(black_box(idx)));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_map_to_index, bench_lower_boundary);
criterion_main!(benches);
