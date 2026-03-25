// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks measuring the performance cost of sub-byte bit-level bucket
//! indexing (B4) compared with starting at byte-aligned widths (U8+).
//!
//! Three scenarios:
//!
//! 1. **unique**: Each value lands in its own bucket — B4 is ideal (1 hit per
//!    bucket, no widening). Measures the steady-state cost of sub-byte get/set.
//!
//! 2. **dup_N**: Each value is recorded N times, forcing widening through the
//!    sub-byte chain. Measures the SWAR widening cost under pressure.
//!
//! 3. **reset_loop**: Repeated clear+fill cycles, showing amortized cost
//!    including the benefit of starting wider (skip sub-byte entirely).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use otel_expohisto::{Histogram, Width};

mod common;

use common::unique_values;

const WIDTHS: &[(&str, Width)] = &[
    ("B1", Width::B1),
    ("B2", Width::B2),
    ("B4", Width::B4),
    ("U8", Width::U8),
    ("U16", Width::U16),
];

fn bench_sub_byte(c: &mut Criterion) {
    // Use scale 0 so bucket indices are small and we focus on counter ops.
    let scale = 0;
    // 64 distinct values — fits B1 in one u64 word, B4 in four.
    let n_unique = 64;

    let values = unique_values(scale, n_unique);

    // -----------------------------------------------------------------------
    // Scenario 1: unique fill — one hit per bucket, no widening
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("sub_byte/unique_fill");

        for &(label, min_w) in WIDTHS {
            group.bench_function(BenchmarkId::new("start", label), |b| {
                b.iter(|| {
                    let mut h: Histogram<16> = Histogram::new()
                        .with_scale(scale)
                        .unwrap()
                        .with_min_width(min_w);
                    for &v in &values {
                        h.update(black_box(v)).unwrap();
                    }
                    black_box(&h);
                })
            });
        }

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 2: repeated hits — force widening through sub-byte chain
    //
    // dup_2:  2 hits per bucket — B1 overflows, B2 handles easily
    // dup_4:  4 hits per bucket — B2 overflows, B4 handles easily
    // dup_16: 16 hits per bucket — B4 overflows, U8 handles it
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("sub_byte/dup_pressure");

        for &(dup_label, reps) in &[("dup_2", 2u64), ("dup_4", 4), ("dup_16", 16)] {
            for &(width_label, min_w) in WIDTHS {
                let id = format!("{dup_label}/{width_label}");
                group.bench_function(BenchmarkId::new("start", &id), |b| {
                    b.iter(|| {
                        let mut h: Histogram<16> = Histogram::new()
                            .with_scale(scale)
                            .unwrap()
                            .with_min_width(min_w);
                        for &v in &values {
                            h.record_incr(black_box(v), reps).unwrap();
                        }
                        black_box(&h);
                    })
                });
            }
        }

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 3: reset loop — amortized clear+fill across 10 cycles
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("sub_byte/reset_loop");

        for &(label, min_w) in WIDTHS {
            group.bench_function(BenchmarkId::new("start", label), |b| {
                b.iter(|| {
                    let mut h: Histogram<16> = Histogram::new()
                        .with_scale(scale)
                        .unwrap()
                        .with_min_width(min_w);
                    for _cycle in 0..10 {
                        for &v in &values {
                            h.update(black_box(v)).unwrap();
                        }
                        h = Histogram::new()
                            .with_scale(scale)
                            .unwrap()
                            .with_min_width(min_w);
                    }
                    black_box(&h);
                })
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_sub_byte);
criterion_main!(benches);
