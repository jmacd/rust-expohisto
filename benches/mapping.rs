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

    // Determine the algorithm label for the primary (Mapping-dispatched) algorithm
    let algo_label = if cfg!(any(
        feature = "newrelic-4",
        feature = "newrelic-6",
        feature = "newrelic-8",
        feature = "newrelic-10",
        feature = "newrelic-12",
        feature = "newrelic-14"
    )) {
        "newrelic"
    } else if cfg!(any(
        feature = "dynatrace-4",
        feature = "dynatrace-6",
        feature = "dynatrace-8",
        feature = "dynatrace-10",
        feature = "dynatrace-12",
        feature = "dynatrace-14"
    )) {
        "dynatrace"
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

    // Positive scales - benchmark the primary algorithm via Mapping
    let max = max_scale();
    let scales: Vec<i32> = [1, 4, 6, 8, 10, 12, 14, 20]
        .into_iter()
        .filter(|&s| s <= max)
        .collect();

    for &scale in &scales {
        let mapping = Mapping::new(scale).unwrap();
        group.bench_function(BenchmarkId::new(algo_label, scale), |b| {
            b.iter(|| {
                for &v in TEST_VALUES {
                    black_box(mapping.map_to_index(black_box(v)));
                }
            })
        });
    }

    // When bench-all is enabled, also benchmark the non-primary algorithms directly
    #[cfg(feature = "bench-all")]
    {
        // Dynatrace direct (when newrelic is primary via Mapping)
        #[cfg(any(
            feature = "dynatrace-4",
            feature = "dynatrace-6",
            feature = "dynatrace-8",
            feature = "dynatrace-10",
            feature = "dynatrace-12",
            feature = "dynatrace-14"
        ))]
        {
            let dt_max = rust_expohisto::dynatrace::table_scale();
            let dt_scales: Vec<i32> = [1, 4, 6, 8, 10, 12, 14]
                .into_iter()
                .filter(|&s| s <= dt_max)
                .collect();

            for &scale in &dt_scales {
                let sm = rust_expohisto::dynatrace::get_scale_mapping(scale);
                group.bench_function(BenchmarkId::new("dynatrace", scale), |b| {
                    b.iter(|| {
                        for &v in TEST_VALUES {
                            black_box(rust_expohisto::dynatrace::map_to_index(
                                black_box(v),
                                scale,
                                sm,
                            ));
                        }
                    })
                });
            }
        }

        // Logarithm direct
        #[cfg(feature = "logarithm")]
        {
            let log_scales: Vec<i32> = [1, 4, 6, 8, 10, 12, 14, 20].to_vec();

            for &scale in &log_scales {
                let sf = rust_expohisto::logarithm::scale_factor(scale);
                group.bench_function(BenchmarkId::new("logarithm", scale), |b| {
                    b.iter(|| {
                        for &v in TEST_VALUES {
                            black_box(rust_expohisto::logarithm::map_to_index(
                                black_box(v),
                                scale,
                                sf,
                            ));
                        }
                    })
                });
            }
        }
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
