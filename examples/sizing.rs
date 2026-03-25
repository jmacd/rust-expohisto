// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Demonstrates how to choose N (pool size) and minimum bucket width.
//!
//! Run with: `cargo run --example sizing`

use otel_expohisto::{Histogram, Width};

fn show_capacity<const N: usize>(label: &str) {
    let bucket_words = N;
    let bucket_bits = bucket_words * 64;

    println!(
        "{label} ({N} words = {} bytes, {bucket_words} bucket words):",
        N * 8
    );
    println!(
        "  {:>4} {:>10} {:>12} {:>12}",
        "Width", "Bits/ctr", "Max buckets", "Max count"
    );
    println!(
        "  {:>4} {:>10} {:>12} {:>12}",
        "----", "--------", "-----------", "---------"
    );

    let widths = [
        ("B1", 1u32, 1u64),
        ("B2", 2, 3),
        ("B4", 4, 15),
        ("U8", 8, 255),
        ("U16", 16, 65535),
        ("U32", 32, u32::MAX as u64),
        ("U64", 64, u64::MAX),
    ];

    for (name, bits, max_count) in widths {
        let buckets = bucket_bits / bits as usize;
        let max_str = if max_count > 1_000_000_000 {
            format!("~{:.1e}", max_count as f64)
        } else {
            format!("{max_count}")
        };
        println!("  {:>4} {:>10} {:>12} {:>12}", name, bits, buckets, max_str);
    }
    println!();
}

fn main() {
    println!("=== Choosing Your Parameters ===\n");

    // --- Pool size (N) ---
    println!("--- Effect of N (pool size in u64 words) ---\n");
    show_capacity::<8>("Histogram<8>");
    show_capacity::<16>("Histogram<16>");
    show_capacity::<32>("Histogram<32>");

    // --- Practical example ---
    println!("--- Practical Sizing ---\n");

    // For a typical latency histogram: values from 0.1ms to 10s
    // At scale 4 (16 buckets per octave, ~2.2% error):
    //   10s / 0.1ms = 100,000x range = ~17 octaves
    //   17 octaves × 16 buckets = ~272 buckets needed
    //
    // Histogram<16> gives 1024 B1 buckets → plenty.
    // Even after widening to B4: 256 buckets → still covers the range.

    let mut h: Histogram<16> = Histogram::new().with_scale(4).unwrap();
    let values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 10000.0];
    for &v in &values {
        h.update(v).unwrap();
    }

    let v = h.view();
    let stats = v.stats();
    println!(
        "Histogram<16> with scale=4 after {} values spanning {:.0}x range:",
        stats.count,
        stats.max / stats.min
    );
    let bw = v.positive().width();
    println!(
        "  scale: {}, width: {:?}, capacity: {}",
        v.scale(),
        bw,
        v.positive().bucket_count()
    );
    let b = v.positive();
    println!(
        "  using {} of {} buckets ({:.0}% utilization)",
        b.len(),
        b.bucket_count(),
        b.len() as f64 / b.bucket_count() as f64 * 100.0
    );

    println!("\n--- Minimum Bucket Width ---\n");

    // Skip sub-byte overhead by starting at U8 (fewer buckets, faster ops)
    let mut fast: Histogram<16> = Histogram::new().with_min_width(Width::U8);
    for &v in &values {
        fast.update(v).unwrap();
    }

    println!(
        "with_min_width(U8): {} buckets at {:?}",
        fast.view().positive().bucket_count(),
        fast.view().positive().width()
    );

    let mut dense: Histogram<16> = Histogram::new();
    for &v in &values {
        dense.update(v).unwrap();
    }
    println!(
        "default (B1 start):        {} buckets at {:?}",
        dense.view().positive().bucket_count(),
        dense.view().positive().width()
    );
}
