// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for boundary table generation.
//!
//! This measures the cost of computing exact boundaries at various scales,
//! which is relevant for build-time table generation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use expohisto_mapping_gen::compute_boundaries_exact;

fn bench_boundary_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary_generation");

    // index_bits=10 takes ~61ms, 12 takes 2s, 14 takes 45s, etc.
    for index_bits in [4, 6, 8, 10] {
        group.bench_function(BenchmarkId::new("compute_boundaries", index_bits), |b| {
            b.iter(|| {
                black_box(compute_boundaries_exact(black_box(index_bits)));
            })
        });
    }

    group.finish();
}

fn bench_table_sizes(c: &mut Criterion) {
    // Report the sizes of the generated tables
    println!("\n=== Generated Table Sizes ===");
    for index_bits in [4u32, 6, 8, 10, 12, 14] {
        let n = 1usize << index_bits;
        let index_bytes = 2 * n * 2; // 2N entries × 2 bytes (u16)
        let boundary_bytes = (n + 3) * 8; // (N+3) entries × 8 bytes (u64), sentinel-wrapped
        let total_bytes = index_bytes + boundary_bytes;
        println!(
            "scale {:2}: {:5} buckets | {:6} bytes index + {:6} bytes boundaries = {:6} bytes ({:.1} KB)",
            index_bits,
            n,
            index_bytes,
            boundary_bytes,
            total_bytes,
            total_bytes as f64 / 1024.0
        );
    }
    println!();

    // Dummy benchmark to satisfy criterion
    let mut group = c.benchmark_group("table_sizes");
    group.bench_function("info", |b| b.iter(|| black_box(1 + 1)));
    group.finish();
}

criterion_group!(benches, bench_boundary_generation, bench_table_sizes);
criterion_main!(benches);
