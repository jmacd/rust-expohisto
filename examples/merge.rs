// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates merging two histograms and cross-size merging.
//!
//! Run with: `cargo run --example merge`

use otel_expohisto::Histogram;

fn print_histogram<const N: usize>(label: &str, h: &mut Histogram<N>) {
    let v = h.view();
    let stats = v.stats();
    println!("{label}:");
    println!(
        "  count={}, sum={:.1}, min={:.1}, max={:.1}, scale={}",
        stats.count,
        stats.sum,
        stats.min,
        stats.max,
        v.scale()
    );
    let b = v.positive();
    println!(
        "  buckets: {} slots at {:?}, offset={}",
        b.len(),
        b.width(),
        b.offset()
    );
}

fn main() {
    // --- Same-size merge ---
    println!("=== Same-Size Merge ===\n");

    let mut server_a: Histogram<16> = Histogram::new();
    let mut server_b: Histogram<16> = Histogram::new();

    // Simulate latencies from two servers
    for &v in &[1.2, 1.5, 2.0, 2.3, 3.1] {
        server_a.update(v).unwrap();
    }
    for &v in &[0.8, 1.1, 1.9, 5.0, 8.2] {
        server_b.update(v).unwrap();
    }

    print_histogram("Server A", &mut server_a);
    print_histogram("Server B", &mut server_b);

    // Merge B into A — A now contains the combined distribution
    server_a.merge_from(&server_b).unwrap();
    print_histogram("\nMerged (A + B)", &mut server_a);

    // --- Cross-size merge ---
    println!("\n=== Cross-Size Merge ===\n");

    // A small edge histogram and a larger aggregator
    let mut edge: Histogram<8> = Histogram::new();
    let mut aggregator: Histogram<32> = Histogram::new();

    for &v in &[1.0, 2.0, 4.0, 8.0] {
        edge.update(v).unwrap();
    }
    for &v in &[0.5, 1.5, 3.0] {
        aggregator.update(v).unwrap();
    }

    print_histogram("Edge (H8)", &mut edge);
    print_histogram("Aggregator (H32)", &mut aggregator);

    // merge_from allows different N values
    aggregator.merge_from(&edge).unwrap();
    print_histogram("\nAggregator after merge", &mut aggregator);
}
