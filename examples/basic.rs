// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Basic usage of the exponential histogram.
//!
//! Run with: `cargo run --example basic`

use otel_expohisto::Histogram;

fn main() {
    // Create a histogram with 16 u64 words (128 bytes) of bucket/literal data.
    // In bucket mode that gives 16 bucket words: up to 1024 one-bit buckets
    // at the default B1 width.
    let mut hist: Histogram<16> = Histogram::new();

    // Record some latency observations (in milliseconds)
    let latencies = [1.2, 2.5, 1.8, 3.1, 2.0, 1.5, 4.7, 2.3, 1.9, 2.8];
    for &ms in &latencies {
        hist.update(ms).unwrap();
    }

    // Access aggregate statistics and bucket data through a view
    let v = hist.view();
    let stats = v.stats();
    println!("=== Histogram Statistics ===");
    println!("  count: {}", stats.count);
    println!("  sum:   {:.1}", stats.sum);
    println!("  min:   {:.1}", stats.min);
    println!("  max:   {:.1}", stats.max);
    println!("  scale: {}", v.scale());

    // Iterate over non-empty buckets
    let buckets = v.positive();
    println!("\n=== Bucket Data ===");
    println!("  offset: {}", buckets.offset());
    println!("  width:  {:?}", buckets.width());
    println!("  count:  {}", buckets.len());
    for (i, count) in buckets.iter().enumerate() {
        if count > 0 {
            println!("  bucket[{}]: {}", buckets.offset() as usize + i, count);
        }
    }
}
