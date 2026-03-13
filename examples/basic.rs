// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Basic usage of the exponential histogram.
//!
//! Run with: `cargo run --example basic`

use rust_expohisto::{Histogram, P32};

fn main() {
    // Create a histogram with 16 u64 words (128 bytes) of data pool.
    // P32 uses 2 words for MMSC stats (f32/u32), leaving 14 words
    // for bucket data: up to 896 one-bit buckets at the default B1 width.
    let mut hist: Histogram<16, P32> = Histogram::new();

    // Record some latency observations (in milliseconds)
    let latencies = [1.2, 2.5, 1.8, 3.1, 2.0, 1.5, 4.7, 2.3, 1.9, 2.8];
    for &ms in &latencies {
        hist.update(ms).unwrap();
    }

    // Access aggregate statistics
    println!("=== Histogram Statistics ===");
    println!("  count: {}", hist.count());
    println!("  sum:   {:.1}", hist.sum());
    println!("  min:   {:.1}", hist.min());
    println!("  max:   {:.1}", hist.max());
    println!("  scale: {}", hist.scale());

    // Iterate over non-empty buckets
    let buckets = hist.positive();
    println!("\n=== Bucket Data ===");
    println!("  offset: {}", buckets.offset());
    println!("  width:  {:?}", buckets.width());
    println!("  count:  {}", buckets.len());
    for i in 0..buckets.len() {
        if buckets.at(i) > 0 {
            println!("  bucket[{}]: {}", buckets.offset() as u32 + i, buckets.at(i));
        }
    }
}
