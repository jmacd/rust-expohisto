// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Empirical validation of the theoretical predictions for exponential
//! histograms fed normal distributions.
//!
//! Runs many seeds per configuration and compares observed statistics
//! (extreme values, terminal scale/width) against:
//!  - The simplified formula: S_t ≈ log₂(N) − log₂(CV) − 4
//!  - A refined greedy trajectory model that tracks both constraints
//!
//! The key finding: there are two regimes:
//!  - **Counter-limited**: small CV → mode bucket fills fast, widens drive
//!    all transitions. N (storage size) barely matters.
//!  - **Range-limited**: large CV → distribution spans many buckets,
//!    range pressure is the binding constraint. N matters a lot.
//!
//! Run with: `cd docs/analysis && cargo run --release --bin validate`

use otel_expohisto::{Histogram, Width};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::BTreeMap;
use std::f64::consts::PI;

const NUM_SEEDS: usize = 200;
const NUM_SEEDS_GUMBEL: usize = 2000;
const LN2: f64 = core::f64::consts::LN_2;

// ── Theoretical helpers ──

/// Rational approximation of the inverse normal CDF (Abramowitz & Stegun).
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
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

/// Approximate location parameter for the maximum of n standard normals.
fn approx_extreme(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    inv_normal_cdf(1.0 - 1.0 / (2.0 * n as f64))
}

/// Simplified terminal scale (range-limited, assumes U64 terminal width).
fn simplified_terminal_scale(n_words: usize, cv: f64) -> f64 {
    (n_words as f64).log2() - cv.log2() - 4.0
}

/// Mode bucket probability at a given scale.
///   p_mode = (base−1) / (CV × √(2π))
fn mode_bucket_probability(cv: f64, scale: i32) -> f64 {
    let base_minus_1 = 2.0_f64.powf(2.0_f64.powi(-scale)) - 1.0;
    base_minus_1 / (cv * (2.0 * PI).sqrt())
}

// ── Width arithmetic ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum WidthLevel {
    B1 = 0,
    B2 = 1,
    B4 = 2,
    U8 = 3,
    U16 = 4,
    U32 = 5,
    U64 = 6,
}

impl WidthLevel {
    const ALL: [WidthLevel; 7] = [
        Self::B1,
        Self::B2,
        Self::B4,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
    ];

    fn bits_per_counter(self) -> u32 {
        1 << (self as u32)
    }

    fn counter_max(self) -> u64 {
        if self as u8 == 6 {
            return u64::MAX;
        }
        (1u64 << self.bits_per_counter()) - 1
    }

    fn available_slots(self, n_words: usize) -> usize {
        n_words * 64 / self.bits_per_counter() as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::B1 => "B1",
            Self::B2 => "B2",
            Self::B4 => "B4",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
        }
    }

    /// Smallest width that can hold `value`.
    fn from_max_value(value: u64) -> Self {
        for w in Self::ALL {
            if w.counter_max() >= value {
                return w;
            }
        }
        Self::U64
    }
}

// ── Greedy trajectory simulator ──

#[derive(Debug, Clone, Copy)]
struct TheoreticalState {
    scale: i32,
    width: WidthLevel,
}

/// Compute the range (in slots) for a normal distribution at given scale.
fn range_slots(cv: f64, d_n: f64, scale: i32) -> usize {
    let r = cv * d_n;
    let log_span = if r >= 1.0 {
        // Distribution extends to zero; use floor
        5.0 + (1.0 + r).ln()
    } else {
        ((1.0 + r) / (1.0 - r)).ln()
    };
    let scale_factor = 2.0_f64.powi(scale);
    (log_span * scale_factor / LN2).ceil().max(1.0) as usize
}

/// Simulate the greedy trajectory for (N, CV, S₀) through milestones up to n.
///
/// Models the histogram's actual behavior: iterate through sample counts,
/// widen ONE step at a time (each widen halves available slots, which may
/// force an immediate range-driven downscale), and check range independently.
///
/// Returns the predicted terminal (scale, width), the range utilization
/// (range_slots / available_slots), and a regime classification.
fn greedy_trajectory(
    n_words: usize,
    initial_scale: i32,
    cv: f64,
    n_max: usize,
) -> (TheoreticalState, f64, &'static str) {
    let min_scale: i32 = -10;

    let mut state = TheoreticalState {
        scale: initial_scale,
        width: WidthLevel::B1,
    };

    // Build milestones: powers of 2 up to n_max
    let mut milestones: Vec<usize> = (0..=40)
        .map(|i| 1usize << i)
        .filter(|&m| m <= n_max)
        .collect();
    milestones.push(n_max);
    milestones.sort();
    milestones.dedup();

    for &n in &milestones {
        if n == 0 {
            continue;
        }
        let d_n = approx_extreme(n);

        // Stabilization loop for this milestone
        for _ in 0..20 {
            let mut changed = false;

            // Counter check at current scale
            let effective_mode_count = if state.width == WidthLevel::B1 {
                let k = range_slots(cv, d_n, state.scale) as f64;
                let birthday_n = (PI * k.max(1.0) / 2.0).sqrt();
                if n as f64 > birthday_n {
                    2.0
                } else {
                    1.0
                }
            } else {
                let p = mode_bucket_probability(cv, state.scale);
                n as f64 * p
            };

            let needed_width =
                WidthLevel::from_max_value(effective_mode_count.ceil().max(1.0) as u64);

            // Widen ONE step at a time so we can check range after each
            if needed_width > state.width {
                let next_width_idx = state.width as u8 + 1;
                state.width = WidthLevel::ALL[next_width_idx as usize];
                state.scale = (state.scale - 1).max(min_scale);
                changed = true;

                // After this single widen, check range immediately
                let r = range_slots(cv, d_n, state.scale);
                let avail = state.width.available_slots(n_words);
                if r > avail && state.scale > min_scale {
                    let mut test_s = state.scale;
                    while range_slots(cv, d_n, test_s) > avail && test_s > min_scale {
                        test_s -= 1;
                    }
                    state.scale = test_s;
                }
                // Continue loop to check if MORE widening is needed at new scale
                continue;
            }

            // Range check (no widen needed)
            let r = range_slots(cv, d_n, state.scale);
            let avail = state.width.available_slots(n_words);
            if r > avail && state.scale > min_scale {
                let mut test_s = state.scale;
                while range_slots(cv, d_n, test_s) > avail && test_s > min_scale {
                    test_s -= 1;
                }
                state.scale = test_s;
                changed = true;
            }

            if !changed {
                break;
            }
        }
    }

    // Classify regime based on final state
    let d_n = approx_extreme(n_max);
    let final_range = range_slots(cv, d_n, state.scale);
    let final_avail = state.width.available_slots(n_words);
    let utilization = final_range as f64 / final_avail as f64;
    let regime = if utilization > 0.6 {
        "range"
    } else if utilization < 0.15 {
        "counter"
    } else {
        "mixed"
    };

    (state, utilization, regime)
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

// ── Per-run result ──

#[derive(Debug, Clone)]
struct RunResult {
    terminal_scale: i32,
    terminal_width: Width,
    slots_used: u32,
    observed_d_max: f64,
    observed_d_min: f64,
}

/// Run one experiment: feed n samples from N(μ, σ) into Histogram<N>.
fn run_one<const N: usize>(
    mu: f64,
    sigma: f64,
    n_max: usize,
    seed: u64,
    initial_scale: i32,
) -> RunResult {
    let mut hist: Histogram<N> = Histogram::new()
        .with_scale(initial_scale)
        .expect("valid scale");
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(mu, sigma).expect("valid distribution");

    let mut obs_max: f64 = f64::NEG_INFINITY;
    let mut obs_min: f64 = f64::INFINITY;

    for _ in 0..n_max {
        let sample = dist.sample(&mut rng);
        let value = if sample <= 0.0 {
            f64::MIN_POSITIVE
        } else {
            sample
        };
        if sample > obs_max {
            obs_max = sample;
        }
        if sample < obs_min {
            obs_min = sample;
        }
        hist.update(value).unwrap();
    }

    let v = hist.view();
    RunResult {
        terminal_scale: v.scale(),
        terminal_width: v.positive().width(),
        slots_used: v.positive().bucket_count(),
        observed_d_max: (obs_max - mu) / sigma,
        observed_d_min: (mu - obs_min) / sigma,
    }
}

// ── Statistics helpers ──

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

fn std_error(xs: &[f64]) -> f64 {
    std_dev(xs) / (xs.len() as f64).sqrt()
}

fn median_i32(xs: &mut [i32]) -> f64 {
    xs.sort();
    let n = xs.len();
    if n % 2 == 0 {
        (xs[n / 2 - 1] + xs[n / 2]) as f64 / 2.0
    } else {
        xs[n / 2] as f64
    }
}

/// Dispatch to the correct const-generic histogram size.
macro_rules! run_experiment {
    ($n_words:expr, $mu:expr, $sigma:expr, $n_max:expr, $init_s:expr, $seeds:expr) => {{
        match $n_words {
            4 => $seeds
                .iter()
                .map(|&s| run_one::<4>($mu, $sigma, $n_max, s, $init_s))
                .collect::<Vec<_>>(),
            8 => $seeds
                .iter()
                .map(|&s| run_one::<8>($mu, $sigma, $n_max, s, $init_s))
                .collect::<Vec<_>>(),
            10 => $seeds
                .iter()
                .map(|&s| run_one::<10>($mu, $sigma, $n_max, s, $init_s))
                .collect::<Vec<_>>(),
            16 => $seeds
                .iter()
                .map(|&s| run_one::<16>($mu, $sigma, $n_max, s, $init_s))
                .collect::<Vec<_>>(),
            32 => $seeds
                .iter()
                .map(|&s| run_one::<32>($mu, $sigma, $n_max, s, $init_s))
                .collect::<Vec<_>>(),
            _ => panic!("unsupported N={}", $n_words),
        }
    }};
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Empirical Validation: {} seeds per configuration          ║", NUM_SEEDS);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mu = 100.0;
    let seeds: Vec<u64> = (1..=NUM_SEEDS as u64).collect();

    // ═══════════════════════════════════════════════════════════════════
    // TABLE 1: Extreme value validation (histogram-independent)
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Table 1: Extreme Value Approximation Validation ═══");
    println!();
    println!("  For N(μ=100, σ=CV×100), comparing observed standardised");
    println!("  extremes (max−μ)/σ and (μ−min)/σ against Φ⁻¹(1−1/(2n)).");
    println!("  Mean ± standard error over {} seeds.", NUM_SEEDS);
    println!("  Note: Φ⁻¹ is a slight OVERESTIMATE of E[max] — this is");
    println!("  expected, not a bug in the histogram.");
    println!();
    println!(
        "  {:>6} {:>6} {:>8} {:>14} {:>14} {:>7}",
        "CV", "n", "d(n)≈", "obs d_max", "obs d_min", "obs/thy"
    );
    println!(
        "  {:>6} {:>6} {:>8} {:>14} {:>14} {:>7}",
        "------", "------", "--------", "--------------", "--------------", "-------"
    );

    let cvs = [0.02, 0.05, 0.10, 0.20];
    let ns: &[usize] = &[100, 1_000, 10_000, 100_000];

    // Only need to run extreme value test once (CV doesn't affect standardised extremes)
    for &n in ns {
        let theoretical_d = approx_extreme(n);
        let mut d_maxes = Vec::with_capacity(NUM_SEEDS);
        let mut d_mins = Vec::with_capacity(NUM_SEEDS);

        // Use CV=0.10 as representative (results are CV-independent)
        let sigma = mu * 0.10;
        for &seed in &seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let dist = Normal::new(mu, sigma).expect("valid");
            let mut obs_max: f64 = f64::NEG_INFINITY;
            let mut obs_min: f64 = f64::INFINITY;
            for _ in 0..n {
                let s = dist.sample(&mut rng);
                if s > obs_max {
                    obs_max = s;
                }
                if s < obs_min {
                    obs_min = s;
                }
            }
            d_maxes.push((obs_max - mu) / sigma);
            d_mins.push((mu - obs_min) / sigma);
        }

        let mean_max = mean(&d_maxes);
        let se_max = std_error(&d_maxes);
        let mean_min = mean(&d_mins);
        let se_min = std_error(&d_mins);
        let ratio = mean_max / theoretical_d;

        println!(
            "  {:>6} {:>6} {:>8.4} {:>7.4}±{:<5.4} {:>7.4}±{:<5.4} {:>7.3}",
            "—", n, theoretical_d, mean_max, se_max, mean_min, se_min, ratio,
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // TABLE 1b: Full Gumbel distribution validation
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Table 1b: Gumbel Distribution of Extremes ═══");
    println!();
    println!("  The maximum of n i.i.d. standard normals converges to");
    println!("  Gumbel(aₙ, bₙ) where:");
    println!("    aₙ = Φ⁻¹(1 − 1/n)          (location)");
    println!("    bₙ = 1 / (n · φ(aₙ))        (scale)");
    println!("    E[max] = aₙ + γ·bₙ           (γ = 0.5772 Euler-Mascheroni)");
    println!("    Var[max] = π²·bₙ²/6");
    println!("    CDF: exp(−exp(−(x−aₙ)/bₙ))");
    println!();
    println!("  {} seeds per n value, standardised as (max−μ)/σ.", NUM_SEEDS_GUMBEL);
    println!();

    // Standard normal PDF
    let phi = |x: f64| -> f64 { (-x * x / 2.0).exp() / (2.0 * PI).sqrt() };
    // Euler-Mascheroni constant
    let euler_gamma: f64 = 0.5772156649015329;

    // Gumbel CDF: P(max ≤ x) = exp(-exp(-(x - a)/b))
    let gumbel_cdf = |x: f64, a: f64, b: f64| -> f64 { (-(-(x - a) / b).exp()).exp() };
    // Gumbel quantile: Q(p) = a - b·ln(-ln(p))
    let gumbel_quantile = |p: f64, a: f64, b: f64| -> f64 { a - b * (-p.ln()).ln() };

    // Part A: Moments comparison
    println!("  ── Moments: observed vs Gumbel theory ──");
    println!();
    println!(
        "  {:>7} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}",
        "n", "E[max]thy", "E[max]obs", "err/SE",
        "SD[max]t", "SD[max]o", "ratio",
    );
    println!(
        "  {:>7} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}",
        "-------", "----------", "----------", "-------",
        "----------", "----------", "-------",
    );

    let gumbel_seeds: Vec<u64> = (1..=NUM_SEEDS_GUMBEL as u64).collect();
    let gumbel_ns: &[usize] = &[50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000];
    let sigma_g = mu * 0.10; // representative

    // Collect d_max samples for each n (reuse for quantile analysis)
    let mut all_d_maxes: Vec<(usize, Vec<f64>)> = Vec::new();

    for &n in gumbel_ns {
        let a_n = inv_normal_cdf(1.0 - 1.0 / n as f64);
        let b_n = 1.0 / (n as f64 * phi(a_n));
        let e_max_theory = a_n + euler_gamma * b_n;
        let sd_max_theory = PI * b_n / (6.0_f64).sqrt();

        let mut d_maxes = Vec::with_capacity(NUM_SEEDS_GUMBEL);
        for &seed in &gumbel_seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let dist = Normal::new(mu, sigma_g).expect("valid");
            let mut obs_max: f64 = f64::NEG_INFINITY;
            for _ in 0..n {
                let s = dist.sample(&mut rng);
                if s > obs_max {
                    obs_max = s;
                }
            }
            d_maxes.push((obs_max - mu) / sigma_g);
        }

        let obs_mean = mean(&d_maxes);
        let obs_se = std_error(&d_maxes);
        let obs_sd = std_dev(&d_maxes);
        let mean_err_se = (obs_mean - e_max_theory) / obs_se;
        let sd_ratio = obs_sd / sd_max_theory;

        println!(
            "  {:>7} {:>10.5} {:>10.5} {:>+6.1}σ   {:>10.5} {:>10.5} {:>7.3}",
            n, e_max_theory, obs_mean, mean_err_se,
            sd_max_theory, obs_sd, sd_ratio,
        );

        all_d_maxes.push((n, d_maxes));
    }
    println!();

    // Part B: Quantile (QQ) comparison for selected n values
    println!("  ── Quantile comparison: observed vs Gumbel ──");
    println!();
    println!("  Shows observed quantiles against Gumbel(aₙ, bₙ) predictions.");
    println!("  Good fit ⟹ ratio ≈ 1.000 across all quantiles.");
    println!();

    let quantile_probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
    let detail_ns: &[usize] = &[100, 1_000, 10_000, 100_000];

    for &target_n in detail_ns {
        if let Some((_, ref d_maxes)) = all_d_maxes.iter().find(|(n, _)| *n == target_n) {
            let a_n = inv_normal_cdf(1.0 - 1.0 / target_n as f64);
            let b_n = 1.0 / (target_n as f64 * phi(a_n));

            let mut sorted = d_maxes.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            println!("  n = {} (aₙ={:.4}, bₙ={:.5}):", target_n, a_n, b_n);
            println!(
                "    {:>5}  {:>10} {:>10} {:>8}",
                "p", "Gumbel Q", "obs Q", "obs/thy"
            );
            for &p in &quantile_probs {
                let thy_q = gumbel_quantile(p, a_n, b_n);
                let idx_f = p * (sorted.len() - 1) as f64;
                let idx_lo = idx_f.floor() as usize;
                let idx_hi = (idx_lo + 1).min(sorted.len() - 1);
                let frac = idx_f - idx_lo as f64;
                let obs_q = sorted[idx_lo] * (1.0 - frac) + sorted[idx_hi] * frac;
                let ratio = obs_q / thy_q;
                println!(
                    "    {:>5.2}  {:>10.5} {:>10.5} {:>8.4}",
                    p, thy_q, obs_q, ratio,
                );
            }
            println!();
        }
    }

    // Part C: Kolmogorov-Smirnov test statistic
    println!("  ── Kolmogorov-Smirnov test vs Gumbel ──");
    println!();
    println!("  D_n = max|F_emp(x) − F_gumbel(x)|");
    println!("  Critical value at α=0.05: 1.36/√(num_seeds) = {:.4}",
        1.36 / (NUM_SEEDS_GUMBEL as f64).sqrt());
    println!();
    println!(
        "  {:>7} {:>8} {:>8} {:>8}",
        "n", "D_n", "crit", "pass?"
    );
    println!(
        "  {:>7} {:>8} {:>8} {:>8}",
        "-------", "--------", "--------", "--------"
    );

    let ks_crit = 1.36 / (NUM_SEEDS_GUMBEL as f64).sqrt();

    for (target_n, ref d_maxes) in &all_d_maxes {
        let a_n = inv_normal_cdf(1.0 - 1.0 / *target_n as f64);
        let b_n = 1.0 / (*target_n as f64 * phi(a_n));

        let mut sorted = d_maxes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let k = sorted.len();
        let mut d_stat: f64 = 0.0;
        for (i, &x) in sorted.iter().enumerate() {
            let f_emp = (i + 1) as f64 / k as f64;
            let f_thy = gumbel_cdf(x, a_n, b_n);
            let gap = (f_emp - f_thy).abs();
            if gap > d_stat {
                d_stat = gap;
            }
            let f_emp_left = i as f64 / k as f64;
            let gap_left = (f_emp_left - f_thy).abs();
            if gap_left > d_stat {
                d_stat = gap_left;
            }
        }

        let pass = if d_stat < ks_crit { "✓" } else { "✗" };
        println!(
            "  {:>7} {:>8.5} {:>8.4} {:>8}",
            target_n, d_stat, ks_crit, pass,
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // TABLE 2: Greedy trajectory vs empirical — the main comparison
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Table 2: Greedy Model vs Empirical Terminal State ═══");
    println!();
    println!("  Compares the refined greedy trajectory model against");
    println!("  empirical median and the simplified formula.");
    println!("  regime: 'counter' = counter overflow dominates,");
    println!("          'range'   = range/slot pressure dominates,");
    println!("          'mixed'   = both constraints active.");
    println!();
    println!(
        "  {:>3} {:>5} {:>5}  {:>6} {:>5}  {:>6} {:>5}  {:>6} {:>5}  {:>7} {:>7}",
        "N", "CV", "n",
        "emp_S", "emp_W",
        "gdy_S", "gdy_W",
        "sim_S", "sim_W",
        "regime", "util%",
    );
    println!(
        "  {:>3} {:>5} {:>5}  {:>6} {:>5}  {:>6} {:>5}  {:>6} {:>5}  {:>7} {:>7}",
        "---", "-----", "-----",
        "------", "-----",
        "------", "-----",
        "------", "-----",
        "-------", "-------",
    );

    let hist_sizes: &[(usize, i32)] = &[(4, 10), (8, 10), (10, 10), (16, 10), (32, 10)];
    let test_ns: &[usize] = &[1_000, 10_000, 100_000];

    for &(n_words, init_s) in hist_sizes {
        for &cv in &cvs {
            let sigma = mu * cv;
            for &n in test_ns {
                // Empirical
                let results = run_experiment!(n_words, mu, sigma, n, init_s, seeds);
                let mut scales: Vec<i32> = results.iter().map(|r| r.terminal_scale).collect();
                let emp_med_s = median_i32(&mut scales);

                // Most common (scale, width) pair
                let mut dist: BTreeMap<(i32, &str), usize> = BTreeMap::new();
                for r in &results {
                    *dist
                        .entry((r.terminal_scale, width_name(r.terminal_width)))
                        .or_insert(0) += 1;
                }
                let (emp_mode_key, _) = dist.iter().max_by_key(|(_, &v)| v).unwrap();
                let emp_mode_w = emp_mode_key.1;

                // Greedy model
                let (greedy_state, util, regime) =
                    greedy_trajectory(n_words, init_s, cv, n);

                // Simplified formula
                let simple_s = simplified_terminal_scale(n_words, cv);

                // Width from simplified (just estimate from expected mode count)
                let simple_p = mode_bucket_probability(cv, simple_s.floor() as i32);
                let simple_mode_ct = n as f64 * simple_p;
                let simple_w = WidthLevel::from_max_value(simple_mode_ct.ceil().max(1.0) as u64);

                let n_label = match n {
                    100 => "100",
                    1_000 => "1K",
                    10_000 => "10K",
                    100_000 => "100K",
                    1_000_000 => "1M",
                    _ => "?",
                };

                println!(
                    "  {:>3} {:>5.2} {:>5}  {:>5.0}  {:>5}  {:>5}  {:>5}  {:>5.0}  {:>5}  {:>7} {:>6.0}%",
                    n_words,
                    cv,
                    n_label,
                    emp_med_s,
                    emp_mode_w,
                    greedy_state.scale,
                    greedy_state.width.name(),
                    simple_s,
                    simple_w.name(),
                    regime,
                    util * 100.0,
                );
            }
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // TABLE 3: Regime classification matrix
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Table 3: Regime Classification (n=10K) ═══");
    println!();
    println!("  Counter-limited (C): mode bucket overflow is the binding");
    println!("    constraint; adding more storage (bigger N) doesn't help.");
    println!("  Range-limited (R): distribution span exceeds slot capacity;");
    println!("    bigger N directly improves terminal scale.");
    println!("  Mixed (M): both constraints are comparably tight.");
    println!();

    let regime_ns: &[usize] = &[1_000, 10_000, 100_000];
    for &n in regime_ns {
        let n_label = match n {
            1_000 => "1K",
            10_000 => "10K",
            100_000 => "100K",
            _ => "?",
        };
        print!("  n={:<6}", n_label);
        for &cv in &cvs {
            print!("  CV={:<4}", cv);
        }
        println!();
        for &(n_words, init_s) in hist_sizes {
            print!("  N={:<5}", n_words);
            for &cv in &cvs {
                let (state, _util, regime) = greedy_trajectory(n_words, init_s, cv, n);
                let tag = match regime {
                    "counter" => "C",
                    "range" => "R",
                    _ => "M",
                };
                print!(
                    "  {:>2}/{:<3}{:>1}",
                    state.scale,
                    state.width.name(),
                    tag,
                );
            }
            println!();
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // TABLE 4: Scale accuracy — greedy vs simplified vs empirical
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Table 4: Prediction Accuracy (n=10K) ═══");
    println!();
    println!("  |Δ| = |predicted_scale − empirical_median_scale|");
    println!("  Greedy model accounts for counter-limited regime;");
    println!("  simplified formula assumes range-limited + U64.");
    println!();
    println!(
        "  {:>3} {:>5}  {:>5} {:>5} {:>5} {:>6} {:>6} {:>6}",
        "N", "CV", "emp", "gdy", "sim", "|Δ|gdy", "|Δ|sim", "regime"
    );
    println!(
        "  {:>3} {:>5}  {:>5} {:>5} {:>5} {:>6} {:>6} {:>6}",
        "---", "-----", "-----", "-----", "-----", "------", "------", "------"
    );

    let n_accuracy = 10_000;
    for &(n_words, init_s) in hist_sizes {
        for &cv in &cvs {
            let sigma = mu * cv;
            let results = run_experiment!(n_words, mu, sigma, n_accuracy, init_s, seeds);
            let mut scales: Vec<i32> = results.iter().map(|r| r.terminal_scale).collect();
            let emp = median_i32(&mut scales);

            let (greedy_state, _, regime) = greedy_trajectory(n_words, init_s, cv, n_accuracy);
            let gdy = greedy_state.scale as f64;
            let sim = simplified_terminal_scale(n_words, cv);

            let delta_gdy = (gdy - emp).abs();
            let delta_sim = (sim - emp).abs();

            println!(
                "  {:>3} {:>5.2}  {:>5.1} {:>5.0} {:>5.1} {:>6.1} {:>6.1} {:>6}",
                n_words, cv, emp, gdy, sim, delta_gdy, delta_sim, regime,
            );
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // TABLE 5: Terminal Bucket Resolution (Δμ/σ)
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Table 5: Terminal Bucket Resolution (Δμ/σ) ═══");
    println!();
    println!("  Theory predicts Δμ/σ ≈ 11.1/N only in range-limited regime.");
    println!("  In counter-limited regime, Δμ/σ depends on CV and n, not N.");
    println!("  Measured from median terminal scale, n=100K.");
    println!();
    println!(
        "  {:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "N", "CV", "med_S", "Δμ/σ obs", "Δμ/σ thy", "ratio", "regime"
    );
    println!(
        "  {:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "----", "------", "--------", "----------", "----------", "--------", "--------"
    );

    let n_for_terminal = 100_000;
    for &(n_words, init_s) in hist_sizes {
        let theoretical_delta = 11.1 / n_words as f64;
        for &cv in &cvs {
            let sigma = mu * cv;
            let results =
                run_experiment!(n_words, mu, sigma, n_for_terminal, init_s, seeds);
            let mut scales: Vec<i32> = results.iter().map(|r| r.terminal_scale).collect();
            let med_s = median_i32(&mut scales);

            let (_, _, regime) = greedy_trajectory(n_words, init_s, cv, n_for_terminal);

            let base_minus_1 = 2.0_f64.powf(2.0_f64.powi(-med_s as i32)) - 1.0;
            let delta_over_sigma = base_minus_1 / cv;
            let ratio = delta_over_sigma / theoretical_delta;

            println!(
                "  {:>4} {:>6.2} {:>8.1} {:>10.4} {:>10.4} {:>8.2} {:>8}",
                n_words, cv, med_s, delta_over_sigma, theoretical_delta, ratio, regime,
            );
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Summary ═══");
    println!();
    println!("  1. Extreme value approximation Φ⁻¹(1−1/(2n)) overestimates E[max]");
    println!("     by ~1-4% (better at larger n). This is the known Gumbel bias.");
    println!();
    println!("  2. The simplified formula S_t ≈ log₂(N) − log₂(CV) − 4 is only");
    println!("     valid in the RANGE-LIMITED regime (large CV, large n, small N).");
    println!();
    println!("  3. For small CV (≤ 0.05), the histogram is COUNTER-LIMITED:");
    println!("     the mode bucket fills so fast that counter width drives all");
    println!("     transitions. N (storage size) has little effect on terminal scale.");
    println!();
    println!("  4. Both models overestimate terminal scale by 2-4 levels because");
    println!("     they use expected statistics. The real histogram's transitions");
    println!("     are triggered by tail realizations and Poisson spikes in bucket");
    println!("     counts, which occur earlier than the expected values predict.");
    println!("     The simplified formula is paradoxically more accurate (~1 level");
    println!("     off in range-limited cases) because its errors partially cancel.");
    println!();
    println!("  5. The 'universal ratio' Δμ/σ ≈ 11.1/N only holds in the range-");
    println!("     limited regime. In the counter-limited regime, Δμ/σ is governed");
    println!("     by the interplay of n and CV, independent of N.");
}

