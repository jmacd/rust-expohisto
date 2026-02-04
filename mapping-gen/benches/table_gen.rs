// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for lookup table generation.
//!
//! This measures the cost of generating tables at runtime,
//! which is relevant for lazy initialization strategies.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use expohisto_mapping_gen::LookupTables;

fn bench_table_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("table_generation");

    // Scale 12 takes ~2s, scale 14 takes ~45s due to BigUint verification
    for scale in [4, 6, 8, 10, 12, 14] {
        group.bench_function(BenchmarkId::new("generate", scale), |b| {
            b.iter(|| {
                black_box(LookupTables::generate(black_box(scale)));
            })
        });
    }

    group.finish();
}

fn bench_table_sizes(c: &mut Criterion) {
    // Just report the sizes, not really a benchmark
    println!("\n=== Table Sizes ===");
    for scale in [4u32, 6, 8, 10, 12, 14] {
        let n = 1usize << scale;
        let index_bytes = 2 * n * 2; // 2N entries × 2 bytes (u16)
        let boundary_bytes = (n + 1) * 8; // (N+1) entries × 8 bytes (u64)
        let total_bytes = index_bytes + boundary_bytes;
        println!(
            "Scale {:2}: {:5} buckets | {:6} bytes index + {:6} bytes boundaries = {:6} bytes ({:.1} KB)",
            scale, n, index_bytes, boundary_bytes, total_bytes, total_bytes as f64 / 1024.0
        );
    }
    println!();

    // Dummy benchmark to satisfy criterion
    let mut group = c.benchmark_group("table_sizes");
    group.bench_function("info", |b| b.iter(|| black_box(1 + 1)));
    group.finish();
}

criterion_group!(benches, bench_table_generation, bench_table_sizes);
criterion_main!(benches);
