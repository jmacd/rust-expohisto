// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Empirical validation of the reference table.
//!
//! For each (contrast, count, histogram_size) cell, this program:
//!   1. Runs 51 seeds with the actual histogram, recording scale, width,
//!      and observed contrast (max/min of samples).
//!   2. Computes the greedy theoretical prediction.
//!   3. Reports closeness metrics: scale error, contrast fidelity,
//!      and cross-seed consistency.
//!
//! Run with: `cd docs/analysis && cargo run --release --bin closeness`

use otel_expohisto::{Histogram, Width};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};
use std::f64::consts::PI;

const NUM_SEEDS: usize = 51; // odd for clean median
const INITIAL_SCALE: i32 = 10;

// ── Theoretical helpers ──

fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < 0.5 {
        return -inv_normal_cdf(1.0 - p);
    }
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
        / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
}

fn approx_extreme(n: usize) -> f64 {
    if n <= 1 {
        return 0.56;
    }
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

// ── Greedy theoretical model ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum WidthLevel {
    B1 = 0, B2 = 1, B4 = 2, U8 = 3, U16 = 4, U32 = 5, U64 = 6,
}

impl WidthLevel {
    const ALL: [WidthLevel; 7] = [
        Self::B1, Self::B2, Self::B4, Self::U8, Self::U16, Self::U32, Self::U64,
    ];

    fn counter_max(self) -> u64 {
        if self as u8 == 6 { return u64::MAX; }
        (1u64 << (1u32 << (self as u32))) - 1
    }

    fn available_slots(self, n_words: usize) -> usize {
        n_words * 64 / (1u32 << (self as u32)) as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::B1 => "B1", Self::B2 => "B2", Self::B4 => "B4",
            Self::U8 => "U8", Self::U16 => "U16", Self::U32 => "U32", Self::U64 => "U64",
        }
    }

    fn from_max_value(value: u64) -> Self {
        for w in Self::ALL {
            if w.counter_max() >= value { return w; }
        }
        Self::U64
    }
}

fn range_slots(ln_sigma: f64, d_n: f64, scale: i32) -> usize {
    // In log-space, range = 2 * d_n * ln_sigma
    // In octaves: range / ln(2)
    // In slots at scale S: range * 2^S / ln(2)
    let log_span = 2.0 * d_n * ln_sigma;
    let scale_factor = 2.0_f64.powi(scale);
    (log_span * scale_factor / (2.0_f64).ln()).ceil().max(1.0) as usize
}

fn lognormal_p_mode(ln_sigma: f64, scale: i32) -> f64 {
    let ln_base = (2.0_f64).ln() / (2.0_f64).powi(scale);
    ln_base / (ln_sigma * (2.0 * PI).sqrt())
}

fn greedy_trajectory(
    n_words: usize,
    initial_scale: i32,
    ln_sigma: f64,
    n_max: usize,
) -> (i32, WidthLevel) {
    let min_scale: i32 = -10;
    let mut scale = initial_scale;
    let mut width = WidthLevel::B1;

    let mut milestones: Vec<usize> = (0..=40)
        .map(|i| 1usize << i)
        .filter(|&m| m <= n_max)
        .collect();
    milestones.push(n_max);
    milestones.sort();
    milestones.dedup();

    for &n in &milestones {
        if n == 0 { continue; }
        let d_n = approx_extreme(n);

        for _ in 0..20 {
            let mut changed = false;

            let effective_mode_count = if width == WidthLevel::B1 {
                let k = range_slots(ln_sigma, d_n, scale) as f64;
                let birthday_n = (PI * k.max(1.0) / 2.0).sqrt();
                if n as f64 > birthday_n { 2.0 } else { 1.0 }
            } else {
                let p = lognormal_p_mode(ln_sigma, scale);
                n as f64 * p
            };

            let needed_width =
                WidthLevel::from_max_value(effective_mode_count.ceil().max(1.0) as u64);

            if needed_width > width {
                let next_idx = width as u8 + 1;
                width = WidthLevel::ALL[next_idx as usize];
                scale = (scale - 1).max(min_scale);
                changed = true;

                let r = range_slots(ln_sigma, d_n, scale);
                let avail = width.available_slots(n_words);
                if r > avail && scale > min_scale {
                    let mut ts = scale;
                    while range_slots(ln_sigma, d_n, ts) > avail && ts > min_scale {
                        ts -= 1;
                    }
                    scale = ts;
                }
                continue;
            }

            let r = range_slots(ln_sigma, d_n, scale);
            let avail = width.available_slots(n_words);
            if r > avail && scale > min_scale {
                let mut ts = scale;
                while range_slots(ln_sigma, d_n, ts) > avail && ts > min_scale {
                    ts -= 1;
                }
                scale = ts;
                changed = true;
            }

            if !changed { break; }
        }
    }

    (scale, width)
}

// ── Empirical run ──

struct RunResult {
    scale: i32,
    width: Width,
    observed_contrast: f64,
}

fn run_one<const N: usize>(
    ln_mu: f64,
    ln_sigma: f64,
    n_max: usize,
    seed: u64,
) -> RunResult {
    let mut hist: Histogram<N> = Histogram::new()
        .with_scale(INITIAL_SCALE)
        .expect("valid scale");
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = LogNormal::new(ln_mu, ln_sigma).expect("valid distribution");

    let mut obs_max: f64 = f64::NEG_INFINITY;
    let mut obs_min: f64 = f64::INFINITY;

    for _ in 0..n_max {
        let sample = dist.sample(&mut rng);
        hist.update(sample).unwrap();
        if sample > obs_max { obs_max = sample; }
        if sample < obs_min { obs_min = sample; }
    }

    let v = hist.view();
    RunResult {
        scale: v.scale(),
        width: v.positive().width(),
        observed_contrast: obs_max / obs_min,
    }
}

macro_rules! run_all {
    ($n_val:literal, $ln_mu:expr, $ln_sigma:expr, $n_max:expr, $seeds:expr) => {{
        $seeds.iter()
            .map(|&s| run_one::<$n_val>($ln_mu, $ln_sigma, $n_max, s))
            .collect::<Vec<_>>()
    }};
}

fn dispatch_run_all(
    n_words: usize,
    ln_mu: f64,
    ln_sigma: f64,
    n_max: usize,
    seeds: &[u64],
) -> Vec<RunResult> {
    match n_words {
        10 => run_all!(10, ln_mu, ln_sigma, n_max, seeds),
        26 => run_all!(26, ln_mu, ln_sigma, n_max, seeds),
        58 => run_all!(58, ln_mu, ln_sigma, n_max, seeds),
        _ => panic!("unsupported N={}", n_words),
    }
}

// ── Stats ──

fn median_i32(xs: &mut [i32]) -> i32 {
    xs.sort();
    xs[xs.len() / 2]
}

fn percentile_f64(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p * (xs.len() - 1) as f64).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn mean_f64(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
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
    let sizes: &[(usize, &str)] = &[
        (10, "S"),
        (26, "M"),
        (58, "L"),
    ];

    println!("# Empirical vs Theoretical Closeness Analysis");
    println!();
    println!("{} seeds per cell. Lognormal data with median ≈ 100.", NUM_SEEDS);
    println!();

    // ── Part 1: Contrast fidelity ──
    println!("## 1. Contrast Fidelity");
    println!();
    println!("Does our lognormal parameterization produce the target contrast?");
    println!("Shows median observed contrast / target contrast (should be ≈ 1.0).");
    println!();
    println!(
        "| {:>8} | {:>6} | {:>12} | {:>12} | {:>8} |",
        "Target C", "n", "Med obs C", "C_obs/C_tgt", "IQR ratio"
    );
    println!(
        "|{:-<10}|{:-<8}|{:-<14}|{:-<14}|{:-<10}|",
        "", "", "", "", ""
    );

    // Only need one histogram size for contrast validation
    for &(contrast, clabel) in contrasts {
        for &(n, nlabel) in counts {
            let (ln_mu, ln_sigma) = lognormal_params(contrast, n);
            let results = dispatch_run_all(10, ln_mu, ln_sigma, n, &seeds);

            let mut contrasts_obs: Vec<f64> =
                results.iter().map(|r| r.observed_contrast).collect();
            let med = percentile_f64(&mut contrasts_obs, 0.5);
            let p25 = percentile_f64(&mut contrasts_obs, 0.25);
            let p75 = percentile_f64(&mut contrasts_obs, 0.75);
            let ratio = med / contrast;
            let iqr_ratio = format!("{:.2}–{:.2}", p25 / contrast, p75 / contrast);

            println!(
                "| {:>8} | {:>6} | {:>12.1} | {:>12.3} | {:>8} |",
                clabel, nlabel, med, ratio, iqr_ratio
            );
        }
    }
    println!();

    // ── Part 2: Scale closeness per cell ──
    println!("## 2. Scale Prediction Accuracy");
    println!();
    println!("For each cell: empirical median scale vs greedy theoretical prediction.");
    println!("|Δ| = |theory − empirical|. Shows (empirical → theory = |Δ|).");
    println!();

    let mut all_deltas: Vec<f64> = Vec::new();
    let mut total_cells = 0;
    let mut within_0 = 0;
    let mut within_1 = 0;
    let mut within_2 = 0;

    println!(
        "| {:>5} | {:>4} | {:>4} | {:>4} | {:>4} | {:>3} |",
        "C", "n", "Size", "emp", "thy", "|Δ|"
    );
    println!(
        "|{:-<7}|{:-<6}|{:-<6}|{:-<6}|{:-<6}|{:-<5}|",
        "", "", "", "", "", ""
    );

    for &(contrast, clabel) in contrasts {
        for &(n, nlabel) in counts {
            let (ln_mu, ln_sigma) = lognormal_params(contrast, n);

            for &(n_words, slabel) in sizes {
                let results = dispatch_run_all(n_words, ln_mu, ln_sigma, n, &seeds);
                let mut scales: Vec<i32> = results.iter().map(|r| r.scale).collect();
                let emp_scale = median_i32(&mut scales);

                let (thy_scale, _thy_width) =
                    greedy_trajectory(n_words, INITIAL_SCALE, ln_sigma, n);

                let delta = (thy_scale - emp_scale).abs();
                all_deltas.push(delta as f64);
                total_cells += 1;
                if delta == 0 { within_0 += 1; }
                if delta <= 1 { within_1 += 1; }
                if delta <= 2 { within_2 += 1; }

                println!(
                    "| {:>5} | {:>4} | {:>4} | {:>4} | {:>4} | {:>3} |",
                    clabel, nlabel, slabel, emp_scale, thy_scale, delta
                );
            }
        }
    }

    println!();

    // ── Part 3: Cross-seed consistency ──
    println!("## 3. Cross-Seed Consistency");
    println!();
    println!("Shows the spread of terminal scale across {} seeds.", NUM_SEEDS);
    println!("IQR = interquartile range (p25–p75). Tight IQR means stable results.");
    println!();
    println!(
        "| {:>5} | {:>4} | {:>4} | {:>4} | {:>4} | {:>4} | {:>8} |",
        "C", "n", "Size", "min", "med", "max", "IQR"
    );
    println!(
        "|{:-<7}|{:-<6}|{:-<6}|{:-<6}|{:-<6}|{:-<6}|{:-<10}|",
        "", "", "", "", "", "", ""
    );

    for &(contrast, clabel) in contrasts {
        for &(n, nlabel) in counts {
            let (ln_mu, ln_sigma) = lognormal_params(contrast, n);

            for &(n_words, slabel) in sizes {
                let results = dispatch_run_all(n_words, ln_mu, ln_sigma, n, &seeds);
                let mut scales: Vec<i32> = results.iter().map(|r| r.scale).collect();
                scales.sort();
                let med = scales[scales.len() / 2];
                let p25 = scales[scales.len() / 4];
                let p75 = scales[3 * scales.len() / 4];
                let mn = scales[0];
                let mx = scales[scales.len() - 1];

                println!(
                    "| {:>5} | {:>4} | {:>4} | {:>4} | {:>4} | {:>4} | {:>4}–{:<3} |",
                    clabel, nlabel, slabel, mn, med, mx, p25, p75
                );
            }
        }
    }

    println!();

    // ── Part 4: Summary statistics ──
    println!("## 4. Summary");
    println!();
    let mean_delta = mean_f64(&all_deltas);
    let rmse = (all_deltas.iter().map(|d| d * d).sum::<f64>() / all_deltas.len() as f64).sqrt();
    let max_delta = all_deltas.iter().cloned().fold(0.0_f64, f64::max);

    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Total cells | {} |", total_cells);
    println!("| Mean |Δscale| | {:.2} |", mean_delta);
    println!("| RMSE(Δscale) | {:.2} |", rmse);
    println!("| Max |Δscale| | {:.0} |", max_delta);
    println!(
        "| Exact match (|Δ|=0) | {} ({:.0}%) |",
        within_0,
        100.0 * within_0 as f64 / total_cells as f64
    );
    println!(
        "| Within ±1 scale | {} ({:.0}%) |",
        within_1,
        100.0 * within_1 as f64 / total_cells as f64
    );
    println!(
        "| Within ±2 scale | {} ({:.0}%) |",
        within_2,
        100.0 * within_2 as f64 / total_cells as f64
    );
    println!();
    println!("The greedy theoretical model uses expected statistics (mean extreme");
    println!("values, expected mode bucket counts). The actual histogram transitions");
    println!("are triggered by realized tail events and Poisson count spikes, which");
    println!("occur earlier than expected values predict. This systematic bias");
    println!("causes the theoretical model to overestimate scale.");
}
