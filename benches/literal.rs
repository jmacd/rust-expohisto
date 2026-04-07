// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks measuring the performance of literal mode — the cold-start
//! optimization that stores raw f64 values before promoting to buckets.
//!
//! Six scenarios:
//!
//! 1. **insert_literal**: Per-insert cost while in literal mode, compared
//!    with bucket mode at the same insert count.
//!
//! 2. **promotion**: Cost of the (capacity+1)th insert that triggers
//!    promotion from literal to bucket mode.
//!
//! 3. **amortized**: Total cost of inserting M values (M >> literal capacity)
//!    comparing literal-start vs bucket-start. Measures the amortized benefit
//!    of skipping early downscale work.
//!
//! 4. **read_buckets**: Cost of iterating BucketView in literal mode vs
//!    bucket mode.
//!
//! 5. **merge_literal_src**: Cost of merging a literal-mode source into a
//!    bucket-mode destination.
//!
//! 6. **reset_loop**: Cost of repeated clear-and-fill cycles, comparing
//!    literal-start vs bucket-start.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use otel_expohisto::{Histogram, Width};

mod common;

use common::unique_values;

/// Generate values spanning a wide range (forces downscale in bucket mode).
fn wide_range_values(count: usize) -> Vec<f64> {
    let mut vals = Vec::with_capacity(count);
    for i in 0..count {
        // Spread across many decades: 1e-6 to 1e6
        let t = i as f64 / count as f64;
        vals.push(10.0f64.powf(t * 12.0 - 6.0));
    }
    vals
}

fn bench_literal(c: &mut Criterion) {
    // -----------------------------------------------------------------------
    // Scenario 1: per-insert cost — literal vs bucket mode.
    //
    // Insert 1..capacity values into a small histogram. Compare the cost
    // of each insert in literal mode vs bucket mode.
    // Histogram<8>: literal capacity = 8.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("literal/insert");

        // Use scale 0 and values that land in adjacent buckets (no downscale pressure).
        let values = unique_values(0, 8);

        for &count in &[1, 2, 4, 8] {
            let v = &values[..count];

            group.bench_function(BenchmarkId::new("literal", count), |b| {
                b.iter(|| {
                    let mut h: Histogram<8> = Histogram::new();
                    for &val in v {
                        h.update(black_box(val)).unwrap();
                    }
                    black_box(&h);
                })
            });

            group.bench_function(BenchmarkId::new("bucket", count), |b| {
                b.iter(|| {
                    let mut h: Histogram<8> = Histogram::new().with_min_width(Width::B1);
                    for &val in v {
                        h.update(black_box(val)).unwrap();
                    }
                    black_box(&h);
                })
            });
        }

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 2: promotion — the step function.
    //
    // Measure the cost of the insert that triggers promotion. We pre-fill
    // to capacity, then time the one extra insert.
    // Two sub-scenarios: narrow range (no downscale needed at promotion)
    // and wide range (promotion must pick a lower scale).
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("literal/promotion");

        // Narrow range: all values close together, no downscale at promotion.
        let narrow = unique_values(0, 9);

        group.bench_function("narrow_H8", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new();
                // Fill to capacity (8 literals for H8).
                for &v in &narrow[..8] {
                    h.update(v).unwrap();
                }
                // The 9th insert triggers promotion.
                h.update(black_box(narrow[8])).unwrap();
                black_box(&h);
            })
        });

        // Wide range: values span many decades, promotion must downscale.
        let wide = wide_range_values(9);

        group.bench_function("wide_H8", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new();
                for &v in &wide[..8] {
                    h.update(v).unwrap();
                }
                h.update(black_box(wide[8])).unwrap();
                black_box(&h);
            })
        });

        // Compare: bucket mode inserting 9 values (no promotion overhead).
        group.bench_function("bucket_narrow_H8", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new().with_min_width(Width::B1);
                for &v in &narrow[..9] {
                    h.update(black_box(v)).unwrap();
                }
                black_box(&h);
            })
        });

        group.bench_function("bucket_wide_H8", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new().with_min_width(Width::B1);
                for &v in &wide[..9] {
                    h.update(black_box(v)).unwrap();
                }
                black_box(&h);
            })
        });

        // Larger histogram: Histogram<16> has 16 literal slots.
        let narrow16 = unique_values(0, 17);
        let wide16 = wide_range_values(17);

        group.bench_function("narrow_H16", |b| {
            b.iter(|| {
                let mut h: Histogram<16> = Histogram::new();
                for &v in &narrow16[..16] {
                    h.update(v).unwrap();
                }
                h.update(black_box(narrow16[16])).unwrap();
                black_box(&h);
            })
        });

        group.bench_function("wide_H16", |b| {
            b.iter(|| {
                let mut h: Histogram<16> = Histogram::new();
                for &v in &wide16[..16] {
                    h.update(v).unwrap();
                }
                h.update(black_box(wide16[16])).unwrap();
                black_box(&h);
            })
        });

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 3: amortized cost — total insert cost for M >> capacity.
    //
    // Insert 50 or 200 values, comparing literal-start vs bucket-start.
    // The narrow case shows literal wins (skip early downscale work);
    // the wide case shows literal picks optimal scale in one shot.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("literal/amortized");

        for &(label, n) in &[("50v", 50usize), ("200v", 200)] {
            // Narrow: adjacent bucket indices at scale 0.
            let narrow = unique_values(0, n);
            // Wide: values spanning many decades.
            let wide = wide_range_values(n);

            for &(range_label, vals) in [("narrow", &narrow), ("wide", &wide)].iter() {
                let tag = format!("{label}/{range_label}");

                group.bench_function(BenchmarkId::new("literal_H8", &tag), |b| {
                    b.iter(|| {
                        let mut h: Histogram<8> = Histogram::new();
                        for &v in vals.iter() {
                            let _ = h.update(black_box(v));
                        }
                        black_box(&h);
                    })
                });

                group.bench_function(BenchmarkId::new("bucket_H8", &tag), |b| {
                    b.iter(|| {
                        let mut h: Histogram<8> = Histogram::new().with_min_width(Width::B1);
                        for &v in vals.iter() {
                            let _ = h.update(black_box(v));
                        }
                        black_box(&h);
                    })
                });

                group.bench_function(BenchmarkId::new("literal_H16", &tag), |b| {
                    b.iter(|| {
                        let mut h: Histogram<16> = Histogram::new();
                        for &v in vals.iter() {
                            let _ = h.update(black_box(v));
                        }
                        black_box(&h);
                    })
                });

                group.bench_function(BenchmarkId::new("bucket_H16", &tag), |b| {
                    b.iter(|| {
                        let mut h: Histogram<16> = Histogram::new().with_min_width(Width::B1);
                        for &v in vals.iter() {
                            let _ = h.update(black_box(v));
                        }
                        black_box(&h);
                    })
                });
            }
        }

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 4: read cost — iterate BucketView in literal vs bucket mode.
    //
    // Pre-fill a histogram, then measure the cost of reading all buckets
    // via the BucketView API.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("literal/read_buckets");

        let values = unique_values(0, 8);

        // Literal mode: 8 values stored as literals.
        let mut h_lit: Histogram<8> = Histogram::new();
        for &v in &values {
            h_lit.update(v).unwrap();
        }

        // Bucket mode: same 8 values in bucket counters.
        let mut h_bkt: Histogram<8> = Histogram::new().with_min_width(Width::B1);
        for &v in &values {
            h_bkt.update(v).unwrap();
        }

        group.bench_function("literal_8v", |b| {
            b.iter(|| {
                let v = h_lit.view();
                let bv = v.positive();
                let mut sum = 0u64;
                for count in bv.iter() {
                    sum += black_box(count);
                }
                black_box(sum);
            })
        });

        group.bench_function("bucket_8v", |b| {
            b.iter(|| {
                let v = h_bkt.view();
                let bv = v.positive();
                let mut sum = 0u64;
                for count in bv.iter() {
                    sum += black_box(count);
                }
                black_box(sum);
            })
        });

        // Also measure scale() in literal vs bucket mode.
        group.bench_function("literal_scale", |b| {
            b.iter(|| {
                black_box(h_lit.view().scale());
            })
        });

        group.bench_function("bucket_scale", |b| {
            b.iter(|| {
                black_box(h_bkt.view().scale());
            })
        });

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 5: merge — literal source into bucket destination.
    //
    // Pre-fill a destination in bucket mode, then merge a small
    // literal-mode source. Compare with merging a bucket-mode source.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("literal/merge");

        let dst_values = unique_values(0, 100);
        let src_values = unique_values(0, 8);

        // Build a bucket-mode destination template.
        let mut dst_template: Histogram<16> = Histogram::new().with_min_width(Width::B1);
        for &v in &dst_values {
            dst_template.update(v).unwrap();
        }

        // Literal source.
        let mut src_lit: Histogram<8> = Histogram::new();
        for &v in &src_values {
            src_lit.update(v).unwrap();
        }

        // Bucket source (same values).
        let mut src_bkt: Histogram<8> = Histogram::new().with_min_width(Width::B1);
        for &v in &src_values {
            src_bkt.update(v).unwrap();
        }

        group.bench_function("literal_src_8v", |b| {
            b.iter(|| {
                let mut dst = dst_template.clone();
                dst.merge_from(black_box(&src_lit)).unwrap();
                black_box(&dst);
            })
        });

        group.bench_function("bucket_src_8v", |b| {
            b.iter(|| {
                let mut dst = dst_template.clone();
                dst.merge_from(black_box(&src_bkt)).unwrap();
                black_box(&dst);
            })
        });

        // Also test literal-to-literal merge.
        let mut dst_lit: Histogram<8> = Histogram::new();
        dst_lit.update(42.0).unwrap();
        dst_lit.update(43.0).unwrap();

        let mut src_lit_small: Histogram<8> = Histogram::new();
        src_lit_small.update(44.0).unwrap();
        src_lit_small.update(45.0).unwrap();

        group.bench_function("literal_to_literal_2v", |b| {
            b.iter(|| {
                let mut dst = dst_lit.clone();
                dst.merge_from(black_box(&src_lit_small)).unwrap();
                black_box(&dst);
            })
        });

        group.finish();
    }

    // -----------------------------------------------------------------------
    // Scenario 6: reset loop — clear resets to literal mode.
    //
    // Repeated clear+fill cycles with few values. Literal mode avoids
    // bucket setup each cycle.
    // -----------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("literal/reset_loop");

        let values = unique_values(0, 5);

        group.bench_function("literal_5v_x20", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new();
                for _cycle in 0..20 {
                    for &v in &values {
                        h.update(black_box(v)).unwrap();
                    }
                    h = Histogram::new();
                }
                black_box(&h);
            })
        });

        group.bench_function("bucket_5v_x20", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new().with_min_width(Width::B1);
                for _cycle in 0..20 {
                    for &v in &values {
                        h.update(black_box(v)).unwrap();
                    }
                    h = Histogram::new().with_min_width(Width::B1);
                }
                black_box(&h);
            })
        });

        // Same but with enough values to force promotion each cycle.
        let more_values = unique_values(0, 10);

        group.bench_function("literal_10v_x20", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new();
                for _cycle in 0..20 {
                    for &v in &more_values {
                        h.update(black_box(v)).unwrap();
                    }
                    h = Histogram::new();
                }
                black_box(&h);
            })
        });

        group.bench_function("bucket_10v_x20", |b| {
            b.iter(|| {
                let mut h: Histogram<8> = Histogram::new().with_min_width(Width::B1);
                for _cycle in 0..20 {
                    for &v in &more_values {
                        h.update(black_box(v)).unwrap();
                    }
                    h = Histogram::new().with_min_width(Width::B1);
                }
                black_box(&h);
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_literal);
criterion_main!(benches);
