// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Relative-error mode: `Sketch` (collapse-left) and `SketchPN` (signed).
//!
//! Run with: `cargo run --example sketch`

use otel_expohisto::{Sketch, SketchPN};

fn alpha(scale: i32) -> f64 {
    let b = 2f64.powf(2f64.powi(-scale));
    (b - 1.0) / (b + 1.0)
}

fn main() {
    // A small pool with a guaranteed relative error of at most ~1%.
    // `with_relative_error` picks the coarsest scale whose worst-case
    // relative error meets the target (maximizing range before collapse).
    let mut s: Sketch<16> = Sketch::new().with_relative_error(0.01).unwrap();
    let scale = s.scale();
    println!("=== Sketch: relative-error mode ===");
    println!("  scale: {}  (worst-case relative error a = {:.3}%)", scale, alpha(scale) * 100.0);

    // A wide, latency-tail-like spread: many small values, a long high tail.
    // The window cannot hold ~50 octaves, so the small end collapses while
    // the top keeps full resolution.
    for e in -40..10 {
        s.update(2f64.powi(e)).unwrap();
    }
    for &v in &[900.0, 950.0, 980.0, 1000.0] {
        s.update(v).unwrap();
    }

    let v = s.view();
    let stats = v.stats();
    println!("\n=== Aggregate (exact) ===");
    println!("  count: {}  min: {:e}  max: {:.0}", stats.count, stats.min, stats.max);
    println!("  collapsed: {}", v.collapsed());
    println!(
        "  underflow_count: {}  (inaccurate low mass, below the floor)",
        v.underflow_count()
    );

    // The top of the distribution stays accurate: the max lands in a bucket
    // whose boundaries bracket it within `a`.
    let buckets = v.positive();
    println!("\n=== OTel export shape ===");
    println!("  offset: {}  width: {:?}  bucket_counts.len: {}", buckets.offset(), buckets.width(), buckets.len());
    let counts: Vec<u64> = buckets.iter().collect();
    println!("  bucket_counts[0] (underflow + floor bucket): {}", counts[0]);
    println!("  bucket_counts[last] (top, brackets the max):  {}", counts[counts.len() - 1]);
    // Everything is accounted for: folded export sums to the bucketed total.
    let exported: u64 = counts.iter().sum();
    println!("  sum(bucket_counts) + zero_count == count: {}", exported + v.zero_count() == stats.count);

    // Merge is same-scale and commutative; pool sizes may differ.
    let mut a: Sketch<16> = Sketch::new().with_scale(scale).unwrap();
    let mut b: Sketch<8> = Sketch::new().with_scale(scale).unwrap();
    for i in 1..=100 {
        a.update(i as f64).unwrap();
        b.update((i as f64) * 10.0).unwrap();
    }
    a.merge_from(&b).unwrap();
    println!("\n=== Merge (same scale, different pool sizes) ===");
    println!("  merged count: {}  max: {:.0}", a.count(), a.view().stats().max);

    // Signed values: SketchPN pairs a positive and a negative Sketch.
    let mut pn: SketchPN<8, 8> = SketchPN::new().with_relative_error(0.02).unwrap();
    for x in [-1000.0, -3.0, 0.0, 2.5, 50.0, 4000.0] {
        pn.update(x).unwrap();
    }
    let pv = pn.view();
    println!("\n=== SketchPN: signed values ===");
    println!("  count: {}  zero_count: {}", pv.stats().count, pv.zero_count());
    println!("  min: {:.0}  max: {:.0}", pv.stats().min, pv.stats().max);
    println!("  positive buckets: {}  negative buckets: {}", pv.positive().len(), pv.negative().len());
}
