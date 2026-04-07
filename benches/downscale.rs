// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks measuring the cost of downscale operations,
//! including the SWAR-shift path for odd `index_base`.
//!
//! Three scenarios:
//!
//! 1. **fill_past_capacity**: Insert enough unique values at a high scale
//!    to force multiple auto-downscale rounds, exercising the even and
//!    odd SWAR merge paths.
//!
//! 2. **merge_cross_scale**: Merge a high-scale histogram into a
//!    low-scale one, forcing the source to downscale.
//!
//! 3. **multi_step_downscale**: Directly measure `downscale(N)` on a
//!    pre-built histogram for various step counts.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use otel_expohisto::Histogram;

mod common;

use common::unique_values;

fn bench_downscale(c: &mut Criterion) {
    // -----------------------------------------------------------------------
    // Scenario 1: fill past capacity — forces auto-downscale rounds.
    //
    // At B1 with Histogram<16> (1024 capacity), inserting more unique values
    // than the capacity forces repeated downscale, eventually hitting the
    // odd-base SWAR-shift path.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("downscale/fill_past_cap");

        for &(label, n_vals, scale) in &[
            ("H16_s8_1800", 1800usize, 8i32),
            ("H16_s4_400", 400, 4),
            ("H16_s8_800", 800, 8),
        ] {
            let values = unique_values(scale, n_vals);

            group.bench_function(BenchmarkId::new("fill", label), |b| {
                b.iter(|| {
                    let mut h: Histogram<16> = Histogram::new().with_scale(scale).unwrap();
                    for &v in &values {
                        let _ = h.update(black_box(v));
                    }
                    black_box(&h);
                })
            });
        }

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 2: merge cross-scale — merge a high-scale histogram into
    // a low-scale target, forcing the target to downscale.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("downscale/merge_cross_scale");

        for &(label, hi_scale, lo_scale, n_vals) in &[
            ("s8_to_s2", 8i32, 2i32, 200usize),
            ("s6_to_s0", 6, 0, 100),
            ("s8_to_s0", 8, 0, 100),
        ] {
            let hi_vals = unique_values(hi_scale, n_vals);
            let lo_vals = unique_values(lo_scale, n_vals.min(50));

            group.bench_function(BenchmarkId::new("merge", label), |b| {
                // Pre-build the source histogram (high scale).
                let mut src: Histogram<16> = Histogram::new().with_scale(hi_scale).unwrap();
                for &v in &hi_vals {
                    let _ = src.update(v);
                }

                b.iter(|| {
                    let mut dst: Histogram<16> = Histogram::new().with_scale(lo_scale).unwrap();
                    for &v in &lo_vals {
                        let _ = dst.update(v);
                    }
                    let _ = dst.merge_from(black_box(&src));
                    black_box(&dst);
                })
            });
        }

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 3: multi-step downscale on a pre-built histogram.
    //
    // We insert sparse values at a high scale so B1 merge steps
    // succeed (no overflow, 1 hit per bucket), then measure the
    // cost of N successive downscale steps.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("downscale/multi_step");

        let scale = 8;
        let n_vals = 100;
        let values = unique_values(scale, n_vals);

        for &steps in &[1, 2, 4, 6, 8] {
            group.bench_function(BenchmarkId::new("steps", steps), |b| {
                // Build a fresh histogram each iteration.
                b.iter(|| {
                    let mut h: Histogram<16> = Histogram::new().with_scale(scale).unwrap();
                    for &v in &values {
                        let _ = h.update(v);
                    }
                    h.downscale(black_box(steps));
                    black_box(&h);
                })
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_downscale);
criterion_main!(benches);
