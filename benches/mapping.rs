// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for exponential histogram mapping functions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_expohisto::{Mapping, max_scale};

/// Test values spanning the full range of normal f64 values.
const TEST_VALUES: &[f64] = &[
    1e-300, 1e-100, 1e-10, 0.001, 0.1, 0.5, 1.0, 1.5, 2.0, core::f64::consts::PI, 10.0, 100.0, 1e10, 1e100,
    1e300,
];

fn bench_map_to_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_to_index");

    // Determine the algorithm label
    let algo_label = if cfg!(any(
        feature = "newrelic-4",
        feature = "newrelic-6",
        feature = "newrelic-8",
        feature = "newrelic-10",
        feature = "newrelic-12",
        feature = "newrelic-14"
    )) {
        "newrelic"
    } else {
        "logarithm"
    };

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

    // Positive scales - benchmark up to max_scale()
    let max = max_scale();
    let scales: Vec<i32> = [1, 4, 6, 8, 10, 12, 14, 20]
        .into_iter()
        .filter(|&s| s <= max)
        .collect();

    for scale in scales {
        let mapping = Mapping::new(scale).unwrap();
        group.bench_function(BenchmarkId::new(algo_label, scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(mapping.map_to_index(black_box(v)));
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

    // Positive scales - benchmark up to max_scale()
    let max = max_scale();
    let scales: Vec<i32> = [1, 4, 8, 10, 12, 14, 20]
        .into_iter()
        .filter(|&s| s <= max)
        .collect();

    for scale in scales {
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
