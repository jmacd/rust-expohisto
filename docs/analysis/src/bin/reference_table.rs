// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Generates a reference table showing exponential histogram resolution
//! as a function of data contrast (max/min ratio) and measurement count.
//!
//! For each (contrast, count) pair, runs actual Histogram<N> instances
//! with lognormal-distributed data matching the specified contrast,
//! and reports the empirical terminal (scale, width, relative error).
//!
//! Output is markdown suitable for inclusion in docs/.
//!
//! Run with: `cd docs/analysis && cargo run --release --bin reference_table`

use otel_expohisto::{Histogram, Width};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};
use std::f64::consts::PI;

const NUM_SEEDS: usize = 441; // 21×21 for stable medians
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

/// Expected extreme of n standard normals (Gumbel location parameter).
fn approx_extreme(n: usize) -> f64 {
    if n <= 1 {
        return 0.56; // E[|Z|] for n=1
    }
    inv_normal_cdf(1.0 - 1.0 / (2.0 * n as f64))
}

/// Lognormal parameters that produce a given contrast over n measurements.
///
/// In log-space, the distribution is Normal(ln_mu, ln_sigma).
/// The observed range over n samples spans ~2·d(n)·ln_sigma in log-space,
/// giving contrast = exp(2·d(n)·ln_sigma).
///
/// Returns (ln_mu, ln_sigma).
fn lognormal_params(contrast: f64, n: usize) -> (f64, f64) {
    let d_n = approx_extreme(n);
    // ln_sigma = ln(contrast) / (2 * d(n))
    // Use a floor to avoid degenerate parameters for very small n
    let ln_sigma = (contrast.ln() / (2.0 * d_n.max(0.5))).max(0.001);
    let ln_mu = (100.0_f64).ln(); // median ≈ 100
    (ln_mu, ln_sigma)
}

/// Relative error at a given scale: (base-1)/(base+1).
fn relative_error(scale: i32) -> f64 {
    let base = 2.0_f64.powf(2.0_f64.powi(-scale));
    (base - 1.0) / (base + 1.0)
}

/// Theoretical mode bucket probability for a lognormal distribution.
///
/// In log-space, the distribution is N(ln_mu, ln_sigma).
/// The bucket width in log-space is ln(base) = ln(2)/2^S.
/// p_mode = ln(2) / (2^S × ln_sigma × √(2π))
#[allow(dead_code)]
fn lognormal_p_mode(ln_sigma: f64, scale: i32) -> f64 {
    let ln_base = (2.0_f64).ln() / (2.0_f64).powi(scale);
    ln_base / (ln_sigma * (2.0 * PI).sqrt())
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

#[allow(dead_code)]
fn width_bits(w: Width) -> u32 {
    match w {
        Width::B1 => 1,
        Width::B2 => 2,
        Width::B4 => 4,
        Width::U8 => 8,
        Width::U16 => 16,
        Width::U32 => 32,
        Width::U64 => 64,
    }
}

// ── Simulation ──

#[derive(Clone)]
struct RunResult {
    scale: i32,
    width: Width,
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

    for _ in 0..n_max {
        let sample = dist.sample(&mut rng);
        hist.update(sample).unwrap();
    }

    let v = hist.view();
    RunResult {
        scale: v.scale(),
        width: v.positive().width(),
    }
}

/// Run simulation for a given N, return median (scale, width).
macro_rules! run_median {
    ($n_val:literal, $ln_mu:expr, $ln_sigma:expr, $n_max:expr, $seeds:expr) => {{
        let mut results: Vec<RunResult> = $seeds
            .iter()
            .map(|&s| run_one::<$n_val>($ln_mu, $ln_sigma, $n_max, s))
            .collect();
        // Sort by (scale desc, width asc) — pick median
        results.sort_by(|a, b| {
            b.scale
                .cmp(&a.scale)
                .then(a.width.cmp(&b.width))
        });
        results[results.len() / 2].clone()
    }};
}

fn dispatch_run(
    n_words: usize,
    ln_mu: f64,
    ln_sigma: f64,
    n_max: usize,
    seeds: &[u64],
) -> RunResult {
    match n_words {
        10 => run_median!(10, ln_mu, ln_sigma, n_max, seeds),
        26 => run_median!(26, ln_mu, ln_sigma, n_max, seeds),
        58 => run_median!(58, ln_mu, ln_sigma, n_max, seeds),
        122 => run_median!(122, ln_mu, ln_sigma, n_max, seeds),
        250 => run_median!(250, ln_mu, ln_sigma, n_max, seeds),
        _ => panic!("unsupported N={}", n_words),
    }
}

// ── Table cell ──

struct Cell {
    scale: i32,
    width: Width,
    rel_err_pct: f64,
}

impl Cell {
    fn format(&self) -> String {
        format!(
            "{}·{} {:.1}%",
            self.scale,
            width_name(self.width),
            self.rel_err_pct
        )
    }

    #[allow(dead_code)]
    fn format_compact(&self) -> String {
        format!("{}/{:.1}%", self.scale, self.rel_err_pct)
    }
}

fn main() {
    let seeds: Vec<u64> = (1..=NUM_SEEDS as u64).collect();

    // Configuration
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

    let sizes: &[(usize, &str, usize)] = &[
        (10, "XS", 128),
        (26, "S", 256),
        (58, "M", 512),
        (122, "L", 1024),
        (250, "XL", 2048),
    ];

    // ── Print preamble ──
    eprintln!("Running {} seeds per configuration...", NUM_SEEDS);
    eprintln!(
        "{} contrasts × {} counts × {} sizes = {} configs",
        contrasts.len(),
        counts.len(),
        sizes.len(),
        contrasts.len() * counts.len() * sizes.len()
    );

    // ── Collect all results ──
    // results[contrast_idx][count_idx][size_idx]
    let mut results: Vec<Vec<Vec<Cell>>> = Vec::new();

    for (ci, &(_contrast, clabel)) in contrasts.iter().enumerate() {
        results.push(Vec::new());
        for (ni, &(n, nlabel)) in counts.iter().enumerate() {
            results[ci].push(Vec::new());
            let (ln_mu, ln_sigma) = lognormal_params(_contrast, n);

            for &(n_words, _slabel, _bytes) in sizes {
                eprint!("  C={:<8} n={:<6} N={:<3} ...", clabel, nlabel, n_words);
                let r = dispatch_run(n_words, ln_mu, ln_sigma, n, &seeds);
                let re = relative_error(r.scale) * 100.0;
                eprintln!(
                    " scale={} width={} err={:.1}%",
                    r.scale,
                    width_name(r.width),
                    re
                );
                results[ci][ni].push(Cell {
                    scale: r.scale,
                    width: r.width,
                    rel_err_pct: re,
                });
            }
        }
    }

    // ── Generate markdown ──
    println!("# Exponential Histogram: Resolution Reference Table");
    println!();
    println!("Terminal scale, counter width, and relative error for variable-width");
    println!("exponential histograms fed lognormal-distributed data, as a function");
    println!("of data contrast (max/min ratio) and measurement count per interval.");
    println!();
    println!("Five histogram sizes are compared:");
    println!();
    for &(n_words, label, bytes) in sizes {
        let total_bits = n_words * 64;
        println!(
            "- **{}** — `Histogram<{}>`: {} bytes, {} bits of counter storage",
            label, n_words, bytes, total_bits
        );
    }
    println!();
    println!("Results are the **median of {} independent runs** using the actual", NUM_SEEDS);
    println!("histogram implementation. Data is drawn from a lognormal distribution");
    println!("whose parameters are chosen so that the expected contrast (max/min of");
    println!("all samples) matches the specified contrast for the given count.");
    println!();

    // Column widths
    let _count_labels: Vec<&str> = counts.iter().map(|(_, l)| *l).collect();
    let col_w = 14;

    // ── Print the table ──
    println!("## Resolution Table");
    println!();
    println!("Each cell shows: `scale / relative_error%`");
    println!("Width abbreviations: B1(1-bit) B2(2) B4(4) U8(8) U16(16) U32(32) U64(64)");
    println!();

    // Header
    print!("| {:^18} | {:^4} ", "Contrast", "Size");
    for &(_, nl) in counts {
        print!("| {:^col_w$} ", nl);
    }
    println!("|");

    // Separator
    print!("|:{:-<18}:|:{:-<4}:", "", "");
    for _ in counts {
        print!("|:{:-<col_w$}:", "");
    }
    println!("|");

    // Data rows
    for (ci, &(_contrast, clabel)) in contrasts.iter().enumerate() {
        for (si, &(_n_words, slabel, bytes)) in sizes.iter().enumerate() {
            let contrast_cell = if si == 0 {
                format!("{}", clabel)
            } else {
                String::new()
            };

            let size_cell = format!("{} ({}B)", slabel, bytes);

            print!("| {:^18} | {:^4} ", contrast_cell, size_cell);
            for ni in 0..counts.len() {
                let cell = &results[ci][ni][si];
                let s = cell.format();
                print!("| {:^col_w$} ", s);
            }
            println!("|");
        }

        // Blank separator between contrast groups (except last)
        if ci < contrasts.len() - 1 {
            // Use a thin separator row
            print!("| {:^18} | {:^4} ", "", "");
            for _ in counts {
                print!("| {:^col_w$} ", "");
            }
            println!("|");
        }
    }

    println!();

    // ── Interpretation guide ──
    println!("## How to Read This Table");
    println!();
    println!("**Rows** are grouped by contrast — the ratio of the largest to");
    println!("smallest observed value in your data:");
    println!();
    println!("| Contrast | Octaves | Example range |");
    println!("|----------|---------|---------------|");
    println!("| 10×      | 3.3     | 10ms – 100ms  |");
    println!("| 10²      | 6.6     | 1ms – 100ms   |");
    println!("| 10³      | 10      | 1ms – 1s      |");
    println!("| 10⁴      | 13      | 1ms – 10s     |");
    println!("| 10⁵      | 17      | 1ms – 100s    |");
    println!("| 10⁶      | 20      | 1μs – 1s      |");
    println!();
    println!("**Columns** are the number of measurements per collection interval.");
    println!("A typical OTel SDK collecting every 15–60 seconds at 100–1000 RPS");
    println!("sees n ≈ 1,500–60,000 measurements per interval.");
    println!();
    println!("**Cell values** show `scale·width error%` where:");
    println!("- **Scale** determines bucket resolution (higher = finer)");
    println!("- **Width** is the counter bit-width (B4 = 4-bit, U16 = 16-bit, etc.)");
    println!("- **Error%** is the worst-case relative error = (base−1)/(base+1)");
    println!();

    // ── Theoretical context ──
    println!("## Relationship to OTel Specification Defaults");
    println!();
    println!("The OTel specification recommends `MaxSize = 160` fixed-width buckets.");
    println!("The following table (from the specification) shows the ideal scale for");
    println!("160 buckets as a function of input range:");
    println!();
    println!("| Input range | Contrast | Ideal Scale | Relative error |");
    println!("|-------------|----------|-------------|----------------|");
    println!("| 1ms – 4ms   | 4×       | 6           | 0.54%          |");
    println!("| 1ms – 100ms | 10²      | 4           | 2.2%           |");
    println!("| 1ms – 1s    | 10³      | 4           | 2.2%           |");
    println!("| 1ms – 100s  | 10⁵      | 3           | 4.3%           |");
    println!("| 1μs – 10s   | 10⁷      | 2           | 8.6%           |");
    println!();
    println!("With variable-width counters, the scale depends on both contrast");
    println!("**and** count, because counter overflow reduces the number of");
    println!("available slots. The reference table above shows this interaction.");
    println!();
    println!("Key observations:");
    println!();
    println!("- At low counts (n ≤ 100), variable-width histograms achieve");
    println!("  **higher** scale than the fixed-width 160-bucket default because");
    println!("  narrow counters (B1–B4) provide more slots than 160.");
    println!();
    println!("- At moderate counts (n ≈ 1K–10K), L (1024B) and XL (2048B)");
    println!("  match or exceed the 160-bucket default's resolution.");
    println!();
    println!("- At high counts (n ≈ 100K–1M), counter pressure reduces scale");
    println!("  below the range-only ideal. This is the price of compact storage.");
    println!();

    // ── Sizing guidance ──
    println!("## Sizing Guidance");
    println!();
    println!("| Size | Bytes | Best for |");
    println!("|------|-------|----------|");
    println!("| XS (`Histogram<10>`)  | 128   | Embedded, high-cardinality, `no_std` |");
    println!("| S  (`Histogram<26>`)  | 256   | Constrained environments, many histograms |");
    println!("| M  (`Histogram<58>`)  | 512   | General-purpose OTel metrics |");
    println!("| L  (`Histogram<122>`) | 1024  | High-resolution, long intervals |");
    println!("| XL (`Histogram<250>`) | 2048  | Maximum resolution, low-cardinality |");
    println!();
    println!("Choose the smallest size whose error% is acceptable for your");
    println!("contrast and count. For most OTel workloads (contrast 10³–10⁵,");
    println!("n ≈ 1K–100K), M (512B) or L (1024B) provides 2–9% error.");
}
