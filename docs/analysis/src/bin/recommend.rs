// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Generates a recommendation table: for each (contrast, count, target_error),
//! find the smallest Histogram<N> that achieves the target relative error.
//!
//! Sweeps N values from small to large, runs actual histograms with
//! lognormal data, and reports the first N whose median scale meets the
//! error target.
//!
//! Run with: `cd docs/analysis && cargo run --release --bin recommend`

use otel_expohisto::{Histogram, Width};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};

const NUM_SEEDS: usize = 51;
const INITIAL_SCALE: i32 = 10;

// ── Helpers ──

fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p < 0.5 { return -inv_normal_cdf(1.0 - p); }
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
        / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
}

fn approx_extreme(n: usize) -> f64 {
    if n <= 1 { return 0.56; }
    inv_normal_cdf(1.0 - 1.0 / (2.0 * n as f64))
}

fn lognormal_params(contrast: f64, n: usize) -> (f64, f64) {
    let d_n = approx_extreme(n);
    let ln_sigma = (contrast.ln() / (2.0 * d_n.max(0.5))).max(0.001);
    let ln_mu = (100.0_f64).ln();
    (ln_mu, ln_sigma)
}

fn relative_error(scale: i32) -> f64 {
    let base = 2.0_f64.powf(2.0_f64.powi(-scale));
    (base - 1.0) / (base + 1.0)
}

/// Minimum scale needed to achieve target relative error.
fn min_scale_for_error(target_err: f64) -> i32 {
    for s in -10..=20 {
        if relative_error(s) <= target_err {
            return s;
        }
    }
    20
}

fn width_name(w: Width) -> &'static str {
    match w {
        Width::B1 => "B1", Width::B2 => "B2", Width::B4 => "B4",
        Width::U8 => "U8", Width::U16 => "U16", Width::U32 => "U32",
        Width::U64 => "U64",
    }
}

// ── Simulation ──

struct RunResult {
    scale: i32,
    width: Width,
}

fn run_one<const N: usize>(ln_mu: f64, ln_sigma: f64, n_max: usize, seed: u64) -> RunResult {
    let mut hist: Histogram<N> = Histogram::new()
        .with_scale(INITIAL_SCALE)
        .expect("valid scale");
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = LogNormal::new(ln_mu, ln_sigma).expect("valid distribution");
    for _ in 0..n_max {
        let sample = dist.sample(&mut rng);
        hist.update(sample).unwrap();
    }
    let v = hist.view();
    RunResult { scale: v.scale(), width: v.positive().width() }
}

/// Run with a specific N, return median scale and most common width.
fn run_median_for_n(
    n_words: usize,
    ln_mu: f64,
    ln_sigma: f64,
    n_max: usize,
    seeds: &[u64],
) -> (i32, Width) {
    // We need a dispatch macro because N is a const generic.
    // Supported N values: 4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,
    //                     36,40,44,48,52,56,58,60,64
    macro_rules! dispatch {
        ($($v:literal),+) => {
            match n_words {
                $( $v => {
                    let mut results: Vec<RunResult> = seeds.iter()
                        .map(|&s| run_one::<$v>(ln_mu, ln_sigma, n_max, s))
                        .collect();
                    results.sort_by(|a, b| b.scale.cmp(&a.scale).then(a.width.cmp(&b.width)));
                    let med = &results[results.len() / 2];
                    (med.scale, med.width)
                }, )+
                _ => panic!("unsupported N={}", n_words),
            }
        }
    }
    dispatch!(4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,58,60,64,
             72,80,90,100,110,122,130,140,160,180,200,220,250)
}

/// For a given (contrast, count, target_error), find the smallest N
/// whose median scale achieves the target error.
/// Returns (n_words, scale, width) or None if no N suffices.
fn find_min_n(
    contrast: f64,
    n_max: usize,
    target_err: f64,
    seeds: &[u64],
) -> Option<(usize, i32, Width)> {
    let (ln_mu, ln_sigma) = lognormal_params(contrast, n_max);
    let min_scale = min_scale_for_error(target_err);

    // Sweep N from small to large
    let candidates: &[usize] = &[
        4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        36, 40, 44, 48, 52, 56, 58, 60, 64,
        72, 80, 90, 100, 110, 122, 130, 140, 160, 180, 200, 220, 250,
    ];

    for &n_words in candidates {
        let (scale, width) = run_median_for_n(n_words, ln_mu, ln_sigma, n_max, seeds);
        if scale >= min_scale {
            return Some((n_words, scale, width));
        }
    }
    None
}

fn main() {
    let seeds: Vec<u64> = (1..=NUM_SEEDS as u64).collect();

    let contrasts: &[(f64, &str)] = &[
        (10.0, "10×"),
        (100.0, "10²"),
        (1_000.0, "10³"),
        (10_000.0, "10⁴"),
        (100_000.0, "10⁵"),
        (1_000_000.0, "10⁶"),
    ];
    let counts: &[(usize, &str)] = &[
        (10, "10"),
        (100, "100"),
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];
    let targets: &[(f64, &str, &str)] = &[
        (0.01, "1%", "Fine"),
        (0.05, "5%", "Medium"),
        (0.10, "10%", "Coarse"),
    ];

    eprintln!("Sweeping {} N values × {} contrasts × {} counts × {} targets = {} searches",
        24, contrasts.len(), counts.len(), targets.len(),
        24 * contrasts.len() * counts.len() * targets.len());
    eprintln!("(early exit on first match, {} seeds each)", NUM_SEEDS);

    // ── Output ──
    println!("# Histogram Sizing Recommendations");
    println!();
    println!("For each (contrast, count) cell, shows the smallest `Histogram<N>`");
    println!("whose **empirical median** relative error meets the target.");
    println!("Based on {} seeds per configuration with lognormal data.", NUM_SEEDS);
    println!();
    println!("Three quality tiers:");
    println!("- **Fine** (≤1%): scale ≥ 6");
    println!("- **Medium** (≤5%): scale ≥ 4");
    println!("- **Coarse** (≤10%): scale ≥ 3");
    println!();

    for &(target_err, err_label, tier_name) in targets {
        let min_s = min_scale_for_error(target_err);
        println!("## {} — target ≤{} error (scale ≥ {})", tier_name, err_label, min_s);
        println!();
        println!("Each cell: `N` (bytes) scale·width — or `—` if no N ≤ 64 suffices.");
        println!();

        // Header
        print!("| {:>8} ", "C \\ n");
        for &(_, nlabel) in counts {
            print!("| {:^16} ", nlabel);
        }
        println!("|");
        print!("|{:-<10}", "");
        for _ in counts {
            print!("|{:-<18}", "");
        }
        println!("|");

        for &(contrast, clabel) in contrasts {
            print!("| {:>8} ", clabel);
            for &(n, _nlabel) in counts {
                eprint!("  {} err={} n={} ... ", clabel, err_label, _nlabel);
                match find_min_n(contrast, n, target_err, &seeds) {
                    Some((nw, scale, width)) => {
                        let bytes = nw * 8 + 48; // approximate struct overhead
                        let cell = format!(
                            "N={} ({}B) {}·{}",
                            nw, bytes, scale, width_name(width)
                        );
                        eprint!("{}\n", cell);
                        print!("| {:^16} ", cell);
                    }
                    None => {
                        eprintln!("—");
                        print!("| {:^16} ", "—");
                    }
                }
            }
            println!("|");
        }
        println!();
    }

    // ── Summary ──
    println!("## Quick Reference");
    println!();
    println!("| Workload | Contrast | Count | Recommended N | Bytes | Tier |");
    println!("|----------|----------|-------|---------------|-------|------|");

    let quick_ref = [
        ("Low-rate API", 100.0, 1_000, 0.05),
        ("Typical API", 1_000.0, 10_000, 0.05),
        ("High-rate API", 1_000.0, 100_000, 0.05),
        ("Wide-range API", 100_000.0, 10_000, 0.10),
        ("Embedded/IoT", 100.0, 100, 0.10),
    ];

    for &(workload, contrast, n, target) in &quick_ref {
        let clabel = match contrast as u64 {
            10 => "10×",
            100 => "10²",
            1_000 => "10³",
            10_000 => "10⁴",
            100_000 => "10⁵",
            1_000_000 => "10⁶",
            _ => "?",
        };
        let tier = if target <= 0.01 { "Fine" }
                   else if target <= 0.05 { "Medium" }
                   else { "Coarse" };
        match find_min_n(contrast, n, target, &seeds) {
            Some((nw, _scale, _width)) => {
                let bytes = nw * 8 + 48;
                println!(
                    "| {} | {} | {} | `Histogram<{}>` | {} | {} |",
                    workload, clabel, n, nw, bytes, tier
                );
            }
            None => {
                println!(
                    "| {} | {} | {} | >512B needed | — | {} |",
                    workload, clabel, n, tier
                );
            }
        }
    }
    println!();
    println!("Byte counts are approximate (N×8 + ~48 bytes struct overhead).");
}
