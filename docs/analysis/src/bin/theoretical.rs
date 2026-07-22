// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Theoretical analysis of exponential histogram scale/width dynamics
//! for normal distributions.
//!
//! Models the expected state transitions of `Histogram<N>` when fed
//! samples from N(μ, σ) with CV = σ/μ.
//!
//! **Important**: These are optimistic upper bounds on the terminal
//! scale.  The real histogram (see `simulate`) never does unnecessary
//! work — every downscale and widen is the minimum needed for the
//! actual data.  This model underestimates the data requirements
//! because it uses expected statistics (mean order statistics, mean
//! mode-bucket count) instead of the worst-case realizations that
//! drive real transitions.  Use `simulate` for ground truth.
//!
//! Run with: `cd docs/analysis && cargo run --bin theoretical`

use std::f64::consts::{LN_2, PI};

/// Width levels matching otel-expohisto::Width variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Width {
    B1 = 0,
    B2 = 1,
    B4 = 2,
    U8 = 3,
    U16 = 4,
    U32 = 5,
    U64 = 6,
}

impl Width {
    fn name(self) -> &'static str {
        match self {
            Width::B1 => "B1",
            Width::B2 => "B2",
            Width::B4 => "B4",
            Width::U8 => "U8",
            Width::U16 => "U16",
            Width::U32 => "U32",
            Width::U64 => "U64",
        }
    }

    fn bits_per_counter(self) -> u32 {
        1 << (self as u32)
    }

    fn counter_max(self) -> u64 {
        u64::MAX >> (64 - self.bits_per_counter())
    }

    fn slots_per_word(self) -> u32 {
        64 / self.bits_per_counter()
    }

    fn available_slots(self, n_words: usize) -> usize {
        n_words * self.slots_per_word() as usize
    }

    /// Smallest width whose counter_max >= value.
    fn from_max_value(value: u64) -> Width {
        const ALL: [Width; 7] = [
            Width::B1,
            Width::B2,
            Width::B4,
            Width::U8,
            Width::U16,
            Width::U32,
            Width::U64,
        ];
        for w in ALL {
            if w.counter_max() >= value {
                return w;
            }
        }
        Width::U64
    }
}

/// Histogram state: scale + width.
#[derive(Debug, Clone, Copy)]
struct State {
    scale: i32,
    width: Width,
}

impl State {
    fn slots(&self, n_words: usize) -> usize {
        self.width.available_slots(n_words)
    }
}

/// Expected maximum of n standard normals (exact via Φ⁻¹).
fn expected_extreme(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // Use the rational approximation for Φ⁻¹(p) where p = 1 - 1/(2n)
    inv_normal_cdf(1.0 - 1.0 / (2.0 * n as f64))
}

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

/// Index span (in slots) needed for a normal distribution at given scale.
///
/// Computes the word-aligned span: how many u64 words the min..max
/// index range would occupy.
fn range_analysis(cv: f64, d_n: f64, scale: i32, width: Width) -> (usize, usize) {
    // Logarithmic span of the ±d·σ range around μ
    let r = cv * d_n;
    if r >= 1.0 {
        // Distribution extends to 0 or below; in practice, negative
        // values are clamped. Use v_min = μ·exp(-5) as a floor (covers
        // the realistic response-time case where values never reach 0).
        let log_span = 5.0 + (1.0 + r).ln();
        let scale_factor = 2.0_f64.powi(scale);
        let slot_span = (log_span * scale_factor / LN_2).ceil().max(1.0) as usize;
        let word_span = (slot_span + width.slots_per_word() as usize - 1)
            / width.slots_per_word() as usize;
        return (slot_span, word_span);
    }
    let log_span = ((1.0 + r) / (1.0 - r)).ln();
    let scale_factor = 2.0_f64.powi(scale);
    let slot_span = (log_span * scale_factor / LN_2).ceil().max(1.0) as usize;
    let word_span = (slot_span + width.slots_per_word() as usize - 1)
        / width.slots_per_word() as usize;
    (slot_span, word_span)
}

/// Expected count in the mode bucket.
fn mode_bucket_count(n: usize, cv: f64, scale: i32) -> f64 {
    // f(μ) = 1/(σ√(2π)), bucket width at μ = μ·(2^(2^(-S)) - 1) ≈ μ·ln(2)/2^S
    // p_mode = f(μ) × Δ(μ) = (μ/σ) × ln(2) / (√(2π) × 2^S)
    //        = ln(2) / (CV × √(2π) × 2^S)
    let base_minus_1 = 2.0_f64.powf(2.0_f64.powi(-scale)) - 1.0;
    let p_mode = base_minus_1 / (cv * (2.0 * PI).sqrt());
    n as f64 * p_mode
}

/// Simulate the expected state evolution.
fn simulate_theoretical(
    n_words: usize,
    initial_scale: i32,
    cv: f64,
    n_max: usize,
) -> Vec<(usize, State, usize, usize, f64)> {
    let mut state = State {
        scale: initial_scale,
        width: Width::B1,
    };
    let mut events: Vec<(usize, State, usize, usize, f64)> = Vec::new();
    let min_scale: i32 = -10;

    // Record initial state
    events.push((0, state, 0, state.slots(n_words), 0.0));

    // Milestones: every power of 2, and specific counts of interest
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

        // Iterate until state stabilizes — models the cascade where
        // downscale doubles mode-bucket counts, which may trigger
        // further widening, which halves slots and may require more
        // downscale.
        let d_n = expected_extreme(n);
        loop {
            let mut changed = false;

            // Check counter overflow at CURRENT scale.
            //
            // For B1 counters (max value 1), use the birthday-paradox
            // model: a collision is expected at ~√(π·K/2) insertions
            // where K is the number of occupied buckets (approximated
            // by the range span).  For wider counters, use the expected
            // mode-bucket count (the densest bucket in the peak).
            let effective_mode_count = if state.width == Width::B1 {
                let (slot_span, _) = range_analysis(cv, d_n, state.scale, state.width);
                let k = (slot_span as f64).max(1.0);
                // Birthday: expected first collision at √(π·K/2)
                let birthday_n = (PI * k / 2.0).sqrt();
                if n as f64 > birthday_n {
                    2.0 // forces B1→B2
                } else {
                    1.0
                }
            } else {
                mode_bucket_count(n, cv, state.scale)
            };
            let needed_width = Width::from_max_value(effective_mode_count.ceil().max(1.0) as u64);

            if needed_width > state.width {
                // Each widen step costs one scale level (halves slots,
                // implemented via do_downscale which merges pairs)
                let widen_steps = needed_width as u32 - state.width as u32;
                state.scale = (state.scale - widen_steps as i32).max(min_scale);
                state.width = needed_width;
                changed = true;
            }

            // Check range overflow at current (post-widen) scale
            let (_, word_span) = range_analysis(cv, d_n, state.scale, state.width);
            if word_span > n_words && state.scale > min_scale {
                let mut test_scale = state.scale;
                let mut test_span = word_span;
                while test_span > n_words && test_scale > min_scale {
                    test_scale -= 1;
                    let (_, ws) = range_analysis(cv, d_n, test_scale, state.width);
                    test_span = ws;
                }
                state.scale = test_scale;
                changed = true;
            }

            if !changed {
                break;
            }
        }

        let mode_count = mode_bucket_count(n, cv, state.scale);
        let (final_slot_span, _) = range_analysis(cv, d_n, state.scale, state.width);
        let avail = state.slots(n_words);

        events.push((n, state, final_slot_span, avail, mode_count));
    }

    events
}

/// Print a table header.
fn print_header() {
    println!(
        "{:>10} {:>6} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n", "scale", "width", "mode_count", "slots_used", "slots_avl", "b/σ", "Δ/σ"
    );
    println!(
        "{:>10} {:>6} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "---", "---", "---", "---", "---", "---", "---", "---"
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Theoretical Analysis: Exponential Histogram × Normal Dist  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let configs: Vec<(usize, i32, &str)> = vec![
        (4, 8, "Histogram<4>  (80B struct)"),
        (8, 8, "Histogram<8>  (112B struct)"),
        (10, 10, "Histogram<10> (128B struct)"),
        (16, 8, "Histogram<16> (176B struct)"),
        (32, 8, "Histogram<32> (304B struct)"),
    ];

    let cvs = [0.02, 0.05, 0.10, 0.20, 0.50];
    let n_max = 100_000;

    // Part 1: Terminal state summary
    println!("═══ Terminal State at n={n_max} (optimistic upper bounds) ═══\n");
    println!("  Note: The histogram algorithm is optimal — it never does");
    println!("  unnecessary work.  These bounds use expected statistics;");
    println!("  actual realizations drive transitions earlier.  See `simulate`.\n");
    println!(
        "{:<28} {:>5} {:>6} {:>5} {:>10} {:>10} {:>7}",
        "Config", "CV", "scale", "width", "slots_used", "slots_avl", "b/σ"
    );
    println!(
        "{:<28} {:>5} {:>6} {:>5} {:>10} {:>10} {:>7}",
        "---", "---", "---", "---", "---", "---", "---"
    );

    for &(n_words, init_scale, label) in &configs {
        for &cv in &cvs {
            let events = simulate_theoretical(n_words, init_scale, cv, n_max);
            if let Some(&(_, state, slot_span, avail, _)) = events.last() {
                let base_minus_1 = 2.0_f64.powf(2.0_f64.powi(-state.scale)) - 1.0;
                let buckets_per_sigma = base_minus_1 / cv;
                let b_per_sigma = 1.0 / buckets_per_sigma;
                println!(
                    "{:<28} {:>5.2} {:>6} {:>5} {:>10} {:>10} {:>7.1}",
                    label,
                    cv,
                    state.scale,
                    state.width.name(),
                    slot_span.min(99999),
                    avail,
                    b_per_sigma,
                );
            }
        }
        println!();
    }

    // Part 2: Detailed trajectory for selected configs
    println!("\n═══ State Trajectories ═══\n");

    let detail_configs: Vec<(usize, i32, f64, usize, &str)> = vec![
        (10, 10, 0.05, 10_000, "Histogram<10>, CV=0.05, n≤10K"),
        (10, 10, 0.10, 10_000, "Histogram<10>, CV=0.10, n≤10K"),
        (10, 10, 0.20, 10_000, "Histogram<10>, CV=0.20, n≤10K"),
        (16, 8, 0.10, 100_000, "Histogram<16>, CV=0.10, n≤100K"),
    ];

    for &(n_words, init_scale, cv, n_max, label) in &detail_configs {
        println!("--- {label} ---\n");
        print_header();

        let events = simulate_theoretical(n_words, init_scale, cv, n_max);
        let mut prev_scale = init_scale;
        let mut prev_width = Width::B1;

        for &(n, state, slot_span, avail, mode_count) in &events {
            let base_minus_1 = 2.0_f64.powf(2.0_f64.powi(-state.scale)) - 1.0;
            let delta_over_sigma = base_minus_1 / cv;
            let b_per_sigma = 1.0 / delta_over_sigma;

            // Show transition markers
            let marker = if state.scale != prev_scale || state.width != prev_width {
                " ◄"
            } else {
                ""
            };

            println!(
                "{:>10} {:>6} {:>5} {:>10.1} {:>10} {:>10} {:>10.1} {:>10.4}{}",
                n,
                state.scale,
                state.width.name(),
                mode_count,
                slot_span.min(99999),
                avail,
                b_per_sigma,
                delta_over_sigma,
                marker,
            );
            prev_scale = state.scale;
            prev_width = state.width;
        }
        println!();
    }

    // Part 3: Sizing guidance
    println!("═══ Sizing Guidance ═══\n");
    println!("For a target of n measurements per interval with known CV:\n");
    println!(
        "{:>8} {:>6} {:>5} {:>7} {:>10}",
        "N", "CV", "Scale", "Width", "Description"
    );
    println!(
        "{:>8} {:>6} {:>5} {:>7} {:>10}",
        "---", "---", "---", "---", "---"
    );

    let guidance = [
        (10, 0.05, 10_000, "Short intervals, tight dist"),
        (10, 0.10, 10_000, "Short intervals, moderate dist"),
        (10, 0.20, 10_000, "Short intervals, wide dist"),
        (16, 0.10, 100_000, "Long intervals, moderate dist"),
        (16, 0.20, 100_000, "Long intervals, wide dist"),
        (32, 0.10, 1_000_000, "Cumulative, moderate dist"),
    ];

    for &(n_words, cv, n_max, desc) in &guidance {
        let events = simulate_theoretical(n_words, 10.min(8), cv, n_max);
        if let Some(&(_, state, _, _, _)) = events.last() {
            println!(
                "{:>8} {:>6.2} {:>5} {:>7} {}",
                n_words,
                cv,
                state.scale,
                state.width.name(),
                desc,
            );
        }
    }
    println!();
    println!("Recommended: start with `with_scale(S)` and `with_min_width(W)` from this table");
    println!("to avoid early transition churn. The histogram will still adapt if needed.");
}
