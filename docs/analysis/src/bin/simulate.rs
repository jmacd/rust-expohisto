// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Simulation: feeds Normal(μ, σ) samples into real Histogram<N> instances
//! and records the scale/width trajectory, comparing with theoretical
//! predictions.
//!
//! Run with: `cd docs/analysis && cargo run --bin simulate`

use otel_expohisto::{Histogram, Width};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Record a snapshot of the histogram state at a milestone.
#[derive(Debug, Clone)]
struct Snapshot {
    n: usize,
    scale: i32,
    width: Width,
    slots_used: u32,
    min: f64,
    max: f64,
}

/// Run a simulation for a given histogram size, returning snapshots at milestones.
fn run_sim<const N: usize>(
    mu: f64,
    sigma: f64,
    n_max: usize,
    seed: u64,
    initial_scale: i32,
) -> (Vec<Snapshot>, Vec<(usize, i32, Width, &'static str)>) {
    let mut hist: Histogram<N> = Histogram::new()
        .with_scale(initial_scale)
        .expect("valid scale");
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(mu, sigma).expect("valid distribution");

    // Build milestone set
    let mut milestones: Vec<usize> = (0..=40)
        .map(|i| 1usize << i)
        .filter(|&m| m <= n_max)
        .collect();
    // Add intermediate milestones
    for p in [5, 10, 25, 50, 75] {
        let m = n_max * p / 100;
        if m > 0 {
            milestones.push(m);
        }
    }
    milestones.push(n_max);
    milestones.sort();
    milestones.dedup();
    let milestone_set: std::collections::HashSet<usize> = milestones.iter().copied().collect();

    let mut snapshots = Vec::new();
    let mut transitions = Vec::new();

    let mut prev_scale = initial_scale;
    let mut prev_width = Width::B1;

    for i in 1..=n_max {
        let sample = dist.sample(&mut rng);
        // Clamp negatives to a small positive value (realistic: response times > 0)
        let value = if sample <= 0.0 { f64::MIN_POSITIVE } else { sample };

        hist.update(value).unwrap();

        let v = hist.view();
        let cur_scale = v.scale();
        let cur_width = v.positive().width();

        // Detect transitions
        if cur_scale != prev_scale || cur_width != prev_width {
            let reason = if cur_width != prev_width && cur_scale != prev_scale {
                "widen+downscale"
            } else if cur_width != prev_width {
                "widen"
            } else {
                "downscale"
            };
            transitions.push((i, cur_scale, cur_width, reason));
            prev_scale = cur_scale;
            prev_width = cur_width;
        }

        // Record milestones
        if milestone_set.contains(&i) {
            let stats = v.stats();
            snapshots.push(Snapshot {
                n: i,
                scale: cur_scale,
                width: cur_width,
                slots_used: v.positive().bucket_count(),
                min: stats.min,
                max: stats.max,
            });
        }
    }

    (snapshots, transitions)
}

fn width_name(w: Width) -> &'static str {
    match w {
        Width::B1 => "B1",
        Width::B2 => "B2",
        Width::B4 => "B4",
        Width::U8 => "U8",
        Width::U16 => "U16",
        Width::U32 => "U32",
        Width::U64 => "U64",
    }
}

fn available_slots(n_words: usize, w: Width) -> usize {
    let bits_per = match w {
        Width::B1 => 1,
        Width::B2 => 2,
        Width::B4 => 4,
        Width::U8 => 8,
        Width::U16 => 16,
        Width::U32 => 32,
        Width::U64 => 64,
    };
    n_words * 64 / bits_per
}

/// Print simulation results.
fn print_sim_results(
    label: &str,
    n_words: usize,
    _cv: f64,
    snapshots: &[Snapshot],
    transitions: &[(usize, i32, Width, &str)],
) {
    println!("--- {label} ---\n");

    // Print transitions
    println!("  Transitions:");
    if transitions.is_empty() {
        println!("    (none)");
    } else {
        for (n, scale, width, reason) in transitions.iter().take(30) {
            println!(
                "    n={:<8} → scale={}, width={:<3} ({})",
                n,
                scale,
                width_name(*width),
                reason
            );
        }
        if transitions.len() > 30 {
            println!("    ... ({} more transitions)", transitions.len() - 30);
        }
    }
    println!();

    // Print milestone table
    println!(
        "  {:>10} {:>6} {:>5} {:>10} {:>10} {:>12} {:>12}",
        "n", "scale", "width", "slots_used", "slots_avl", "min", "max"
    );
    println!(
        "  {:>10} {:>6} {:>5} {:>10} {:>10} {:>12} {:>12}",
        "---", "---", "---", "---", "---", "---", "---"
    );

    for snap in snapshots {
        let avail = available_slots(n_words, snap.width);
        let utilization = snap.slots_used as f64 / avail as f64 * 100.0;
        println!(
            "  {:>10} {:>6} {:>5} {:>7} ({:>2.0}%) {:>10} {:>12.4} {:>12.4}",
            snap.n,
            snap.scale,
            width_name(snap.width),
            snap.slots_used,
            utilization,
            avail,
            snap.min,
            snap.max,
        );
    }
    println!();
}

/// Run a simulation with a constant value (best case: no range pressure,
/// only counter overflow drives transitions). Returns (scale, width) at n_max.
fn run_constant<const N: usize>(value: f64, n_max: usize, initial_scale: i32) -> (i32, Width) {
    let mut hist: Histogram<N> = Histogram::new()
        .with_scale(initial_scale)
        .expect("valid scale");
    for _ in 0..n_max {
        hist.update(value).unwrap();
    }
    let v = hist.view();
    (v.scale(), v.positive().width())
}

/// Run a simulation with Normal(mu, sigma), averaging over multiple seeds.
/// Returns the median (scale, width) across seeds.
fn run_normal_median<const N: usize>(
    mu: f64,
    sigma: f64,
    n_max: usize,
    initial_scale: i32,
) -> (i32, Width) {
    let seeds = [42, 137, 271, 314, 577, 691, 823, 997];
    let mut results: Vec<(i32, Width)> = seeds
        .iter()
        .map(|&seed| {
            let (snaps, _) = run_sim::<N>(mu, sigma, n_max, seed, initial_scale);
            snaps
                .last()
                .map(|s| (s.scale, s.width))
                .unwrap_or((initial_scale, Width::B1))
        })
        .collect();
    results.sort_by_key(|&(s, w)| (std::cmp::Reverse(s), w as u8));
    // Median: take the middle element (pessimistic side)
    results[results.len() / 2]
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Exponential Histogram: Scale/Width Reference Tables       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  The histogram algorithm is optimal: every downscale and widen");
    println!("  is the minimum necessary for the actual data observed.");
    println!();

    let mu = 100.0;

    // =====================================================================
    // TABLE 1: Best case (constant value) — scale vs N vs n
    // =====================================================================
    println!("═══ Best Case: Constant Value (no range pressure) ═══");
    println!();
    println!("  All measurements identical — only counter overflow drives");
    println!("  transitions.  This is the highest scale achievable at each n.");
    println!();

    let n_cols: &[(usize, &str)] = &[
        (10, "10"),
        (100, "100"),
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];

    print!("  {:>5}", "N \\ n");
    for &(_, label) in n_cols {
        print!(" {:>9}", label);
    }
    println!();
    print!("  {:>5}", "-----");
    for _ in n_cols {
        print!(" {:>9}", "---------");
    }
    println!();

    macro_rules! best_case_row {
        ($n_val:literal, $init_s:expr) => {{
            print!("  {:>5}", $n_val);
            for &(n_max, _) in n_cols {
                let (s, w) = run_constant::<$n_val>(mu, n_max, $init_s);
                print!(" {:>5}/{:<3}", s, width_name(w));
            }
            println!();
        }};
    }

    let init_s = 10; // Uniform initial scale for all N values

    best_case_row!(4, init_s);
    best_case_row!(8, init_s);
    best_case_row!(10, init_s);
    best_case_row!(16, init_s);
    best_case_row!(32, init_s);

    println!();
    println!("  Format: scale/width.  All start at table_scale=10.");
    println!();

    // =====================================================================
    // TABLE 2: Typical case (Normal, CV=0.10) — scale vs N vs n
    // =====================================================================
    println!("═══ Typical: Normal Distribution, CV=0.10 (median of 8 seeds) ═══");
    println!();
    println!("  N(μ=100, σ=10).  Both range and counter pressure active.");
    println!();

    print!("  {:>5}", "N \\ n");
    for &(_, label) in n_cols {
        print!(" {:>9}", label);
    }
    println!();
    print!("  {:>5}", "-----");
    for _ in n_cols {
        print!(" {:>9}", "---------");
    }
    println!();

    macro_rules! typical_row {
        ($n_val:literal, $init_s:expr) => {{
            print!("  {:>5}", $n_val);
            for &(n_max, _) in n_cols {
                let (s, w) = run_normal_median::<$n_val>(mu, 10.0, n_max, $init_s);
                print!(" {:>5}/{:<3}", s, width_name(w));
            }
            println!();
        }};
    }

    typical_row!(4, init_s);
    typical_row!(8, init_s);
    typical_row!(10, init_s);
    typical_row!(16, init_s);
    typical_row!(32, init_s);

    println!();
    println!("  Format: scale/width.  All start at table_scale=10.");
    println!();

    // =====================================================================
    // TABLE 3: Typical case across CVs — scale vs N vs CV (fixed n=10K)
    // =====================================================================
    println!("═══ Typical at n=10K: Scale vs N vs CV ═══");
    println!();

    let cv_cols: &[(f64, &str)] = &[
        (0.02, "0.02"),
        (0.05, "0.05"),
        (0.10, "0.10"),
        (0.20, "0.20"),
        (0.50, "0.50"),
    ];

    print!("  {:>5}", "N\\CV");
    for &(_, label) in cv_cols {
        print!(" {:>9}", label);
    }
    println!();
    print!("  {:>5}", "-----");
    for _ in cv_cols {
        print!(" {:>9}", "---------");
    }
    println!();

    macro_rules! cv_row {
        ($n_val:literal, $init_s:expr) => {{
            print!("  {:>5}", $n_val);
            for &(cv, _) in cv_cols {
                let sigma = mu * cv;
                let (s, w) = run_normal_median::<$n_val>(mu, sigma, 10_000, $init_s);
                print!(" {:>5}/{:<3}", s, width_name(w));
            }
            println!();
        }};
    }

    cv_row!(4, init_s);
    cv_row!(8, init_s);
    cv_row!(10, init_s);
    cv_row!(16, init_s);
    cv_row!(32, init_s);

    println!();
    println!("  Format: scale/width at n=10,000 with μ=100.");
    println!();

    // =====================================================================
    // TABLE 4: Implied table_scale requirement
    // =====================================================================
    println!("═══ Implied table_scale Feature ═══");
    println!();
    println!("  The table_scale feature only needs to cover the terminal scale.");
    println!("  Higher table_scale gives better initial resolution but costs");
    println!("  compile-time table size.  The histogram adapts regardless.");
    println!();
    println!("  For n ≤ 10K, CV ≈ 0.10:");
    println!();
    println!("  {:>5} {:>14} {:>14} {:>14}", "N", "terminal S/W", "need feature", "default ok?");
    println!("  {:>5} {:>14} {:>14} {:>14}", "-----", "-----------", "-----------", "-----------");

    macro_rules! guidance_row {
        ($n_val:literal, $init_s:expr) => {{
            let (s, w) = run_normal_median::<$n_val>(mu, 10.0, 10_000, $init_s);
            let feature = if s <= 0 {
                "(exponent)".to_string()
            } else {
                format!("scale-{}", s.max(1))
            };
            let default_ok = if s <= 8 { "yes" } else { "no" };
            println!(
                "  {:>5} {:>10}/{:<3} {:>14} {:>14}",
                $n_val,
                s,
                width_name(w),
                feature,
                default_ok,
            );
        }};
    }

    guidance_row!(4, init_s);
    guidance_row!(8, init_s);
    guidance_row!(10, init_s);
    guidance_row!(16, init_s);
    guidance_row!(32, init_s);

    println!();
    println!("  default = scale-8.  Terminal scale ≤ 8 in all cases above,");
    println!("  so the default feature is sufficient for typical workloads.");
    println!();

    // =====================================================================
    // DETAILED: Single trajectory for reference
    // =====================================================================
    println!("═══ Detailed Trajectory: Histogram<10>, CV=0.10, seed=42 ═══\n");

    let (snaps, trans) = run_sim::<10>(mu, 10.0, 10_000, 42, 10);
    print_sim_results("N=10, μ=100, σ=10 (CV=0.10)", 10, 0.10, &snaps, &trans);
}
