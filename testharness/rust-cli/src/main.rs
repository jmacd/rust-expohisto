// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! CLI tool that reads hex-encoded f64 values from stdin, feeds them into
//! an exponential histogram, and prints the result in a canonical text format
//! for cross-implementation comparison.
//!
//! Usage: expohisto-cli [--scale N] [--size N]

use std::io::{self, BufRead, Write};

use otel_expohisto::{Histogram, Width, table_scale};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut scale: i32 = table_scale();
    let mut size: u32 = 160;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scale" => {
                i += 1;
                scale = args.get(i).expect("--scale requires a value").parse().expect("invalid scale");
            }
            "--size" => {
                i += 1;
                size = args.get(i).expect("--size requires a value").parse().expect("invalid size");
            }
            _ => {
                eprintln!("unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // N=250 is the maximum allowed. Choose min_width so that
    // effective slot capacity >= requested size.
    let min_width = select_min_width(size);

    let mut hist: Histogram<250> = Histogram::new()
        .with_scale(scale)
        .expect("invalid scale")
        .with_min_width(min_width);

    let stdin = io::stdin();
    let mut zero_count: u64 = 0;

    for line in stdin.lock().lines() {
        let line = line.expect("stdin read error");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let bits = u64::from_str_radix(trimmed, 16).expect("invalid hex f64 bits");
        let mut value = f64::from_bits(bits);

        // Normalize subnormals to MIN_POSITIVE so both implementations
        // agree on the bucket assignment.
        if value > 0.0 && value < f64::MIN_POSITIVE {
            value = f64::MIN_POSITIVE;
        }

        match hist.update(value) {
            Ok(()) => {
                if value == 0.0 {
                    zero_count += 1;
                }
            }
            Err(_) => {
                // Skip NaN, Inf, negative non-zero — they are filtered
                // by the orchestrator, but handle gracefully here.
            }
        }
    }

    let v = hist.view();
    let stats = v.stats();
    let positive = v.positive();

    // Normalize min/max: Rust doesn't update min/max for zeros, but
    // Go does. Adjust so both produce identical output.
    let (min_val, max_val) = if stats.count == 0 {
        (0.0_f64, 0.0_f64)
    } else if zero_count > 0 {
        (0.0_f64.min(stats.min), 0.0_f64.max(stats.max))
    } else {
        (stats.min, stats.max)
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "scale={}", v.scale()).unwrap();
    writeln!(out, "count={}", stats.count).unwrap();
    writeln!(out, "sum={:016x}", normalize_zero(stats.sum)).unwrap();
    writeln!(out, "min={:016x}", normalize_zero(min_val)).unwrap();
    writeln!(out, "max={:016x}", normalize_zero(max_val)).unwrap();
    writeln!(out, "zero_count={}", zero_count).unwrap();
    writeln!(out, "positive_offset={}", positive.offset()).unwrap();

    let counts: Vec<String> = positive.iter().map(|c| c.to_string()).collect();
    writeln!(out, "positive_counts=[{}]", counts.join(",")).unwrap();
}

/// Canonicalize ±0.0 to +0.0 to avoid sign-of-zero mismatches.
fn normalize_zero(v: f64) -> u64 {
    if v == 0.0 {
        0u64
    } else {
        v.to_bits()
    }
}

/// Choose the minimum counter width so that Histogram<250> has at
/// least `size` bucket slots.
///
/// N=250 words × slots_per_word:
///   B1  → 16000    B2 → 8000    B4 → 4000
///   U8  → 2000     U16 → 1000   U32 → 500   U64 → 250
fn select_min_width(size: u32) -> Width {
    const N: u32 = 250;
    if size <= N {
        Width::U64
    } else if size <= N * 2 {
        Width::U32
    } else if size <= N * 4 {
        Width::U16
    } else if size <= N * 8 {
        Width::U8
    } else if size <= N * 16 {
        Width::B4
    } else if size <= N * 32 {
        Width::B2
    } else {
        Width::B1
    }
}
