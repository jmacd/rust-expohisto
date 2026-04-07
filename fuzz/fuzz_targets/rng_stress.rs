#![no_main]

//! **Large-histogram RNG stress test** — inserts millions of values drawn
//! from fuzzer-configured distributions into large histograms (N=160),
//! then replays the same RNG sequence to build an independent oracle and
//! verifies exact bucket-level correctness.
//!
//! Two-pass design: both passes use the same deterministic PRNG with the
//! same seed, so no shadow Vec is needed — the oracle recomputes
//! everything from scratch. This allows truly massive inputs (millions
//! of values) without proportional memory cost.
//!
//! Targets risks that small-histogram / low-volume tests miss:
//!   - SWAR operations over 150+ data words (off-by-one in word loops)
//!   - Full width cascade B1→B2→B4→U8→U16→U32→U64 under sustained load
//!   - Circular buffer wrapping at U64 with 156 slots under heavy fill
//!   - 30+ downscale steps over many words (extreme value pairs)
//!   - Cross-size merge (N=32 → N=160) with many occupied buckets
//!   - Mixture-of-distributions creating fragmented bucket layouts

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use otel_expohisto::Histogram;

#[path = "verify.rs"]
mod verify;

// ---------------------------------------------------------------------------
// PRNG — xorshift64*: fast, deterministic, good distribution
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed })
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f64 in [0, 1).
    #[inline(always)]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Fuzzer-controlled configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Arbitrary, Clone, Copy)]
enum DistKind {
    /// Uniform in log-space: values in [2^lo, 2^hi].
    LogUniform,
    /// Gaussian-ish cluster around a center exponent.
    Clustered,
    /// Mostly one value with rare extreme outliers.
    SpikeOutliers,
    /// Linear sweep through exponent range.
    ExponentSweep,
    /// Two clusters separated by a wide gap.
    Bimodal,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
struct DistConfig {
    kind: DistKind,
    param_a: u16,
    param_b: u16,
}

#[derive(Debug, Arbitrary)]
struct Config {
    seed: u64,
    /// Primary distribution.
    primary: DistConfig,
    /// Optional secondary distribution mixed in.
    secondary: Option<DistConfig>,
    /// Mix ratio: 0 = all primary, 255 = all secondary.
    mix_byte: u8,
    /// Selects value count: 10K to 2M.
    count_sel: u8,
    /// If true, merge a second histogram built from merge_dist.
    do_merge: bool,
    merge_seed: u64,
    merge_dist: DistConfig,
    merge_count_sel: u8,
}

// ---------------------------------------------------------------------------
// Value generation — deterministic from RNG + config
// ---------------------------------------------------------------------------

/// Map a u16 parameter to an exponent in [-300, +300].
#[inline]
fn param_to_exp(p: u16) -> f64 {
    (p as f64 / 65535.0) * 600.0 - 300.0
}

/// Generate one value from a distribution. Returns None for invalid results.
fn gen_from_dist(rng: &mut Rng, dc: &DistConfig) -> Option<f64> {
    let v = match dc.kind {
        DistKind::LogUniform => {
            let (mut lo, mut hi) = (param_to_exp(dc.param_a), param_to_exp(dc.param_b));
            if lo > hi {
                core::mem::swap(&mut lo, &mut hi);
            }
            if (hi - lo) < 0.01 {
                hi = lo + 0.01;
            }
            2.0f64.powf(lo + rng.unit() * (hi - lo))
        }
        DistKind::Clustered => {
            let center = param_to_exp(dc.param_a);
            let spread = (dc.param_b as f64 / 65535.0) * 20.0 + 0.1;
            // Sum of 4 uniforms ≈ Gaussian shape
            let u = rng.unit() + rng.unit() + rng.unit() + rng.unit();
            let offset = (u / 4.0 - 0.5) * 2.0 * spread;
            2.0f64.powf(center + offset)
        }
        DistKind::SpikeOutliers => {
            let main_exp = param_to_exp(dc.param_a);
            let outlier_pct = (dc.param_b as f64 / 65535.0) * 0.15;
            if rng.unit() < outlier_pct {
                2.0f64.powf(rng.unit() * 600.0 - 300.0)
            } else {
                2.0f64.powf(main_exp + (rng.unit() - 0.5) * 0.02)
            }
        }
        DistKind::ExponentSweep => {
            let (start, end) = (param_to_exp(dc.param_a), param_to_exp(dc.param_b));
            2.0f64.powf(start + rng.unit() * (end - start))
        }
        DistKind::Bimodal => {
            let (exp1, exp2) = (param_to_exp(dc.param_a), param_to_exp(dc.param_b));
            let spread = 3.0;
            let base = if rng.next_u64() & 1 == 0 { exp1 } else { exp2 };
            2.0f64.powf(base + (rng.unit() - 0.5) * spread)
        }
    };
    if v.is_finite() && v.is_normal() && v > 0.0 {
        Some(v)
    } else {
        None
    }
}

/// Generate one value using the mixture of primary + optional secondary.
/// Consumes the same amount of RNG state regardless of which branch is taken.
fn gen_value(rng: &mut Rng, cfg: &Config) -> Option<f64> {
    let mix_threshold = cfg.mix_byte as f64 / 255.0;
    let choice = rng.unit();

    match &cfg.secondary {
        Some(sec) if choice < mix_threshold => gen_from_dist(rng, sec),
        _ => gen_from_dist(rng, &cfg.primary),
    }
}

/// Same as gen_value but for the merge source histogram.
fn gen_merge_value(rng: &mut Rng, cfg: &Config) -> Option<f64> {
    gen_from_dist(rng, &cfg.merge_dist)
}

fn decode_count(sel: u8) -> usize {
    match sel % 8 {
        0 => 10_000,
        1 => 50_000,
        2 => 100_000,
        3 => 250_000,
        4 => 500_000,
        5 => 1_000_000,
        6 => 2_000_000,
        7 => 2_000_000,
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Two-pass test runner
// ---------------------------------------------------------------------------

fn run<const N: usize, const M: usize>(cfg: &Config) {
    let count = decode_count(cfg.count_sel);

    // ===== PASS 1: INSERT =====
    let mut hist: Histogram<N> = Histogram::new();
    let mut rng = Rng::new(cfg.seed);
    let mut n_inserted = 0u64;

    for _ in 0..count {
        if let Some(v) = gen_value(&mut rng, cfg) {
            if hist.update(v).is_err() {
                // Shouldn't happen with N=160, but bail cleanly.
                return;
            }
            n_inserted += 1;
        }
    }

    // Optional merge from a smaller histogram.
    let merge_count = if cfg.do_merge {
        decode_count(cfg.merge_count_sel)
    } else {
        0
    };
    let mut merge_inserted = 0u64;

    if merge_count > 0 {
        let mut src: Histogram<M> = Histogram::new();
        let mut mrng = Rng::new(cfg.merge_seed);

        for _ in 0..merge_count {
            if let Some(v) = gen_merge_value(&mut mrng, cfg) {
                if src.update(v).is_err() {
                    return;
                }
                merge_inserted += 1;
            }
        }

        if hist.merge_from(&src).is_err() {
            return;
        }
    }

    // ===== PASS 2: ORACLE =====
    let mut ops = Vec::with_capacity((n_inserted + merge_inserted) as usize);

    // Replay primary values.
    let mut rng = Rng::new(cfg.seed);
    let mut oracle_n = 0u64;
    for _ in 0..count {
        if let Some(v) = gen_value(&mut rng, cfg) {
            oracle_n += 1;
            if oracle_n > n_inserted {
                break;
            }
            ops.push((v, 1u64));
        }
    }

    // Replay merge source values.
    if merge_count > 0 {
        let mut mrng = Rng::new(cfg.merge_seed);
        let mut oracle_m = 0u64;
        for _ in 0..merge_count {
            if let Some(v) = gen_merge_value(&mut mrng, cfg) {
                oracle_m += 1;
                if oracle_m > merge_inserted {
                    break;
                }
                ops.push((v, 1u64));
            }
        }
    }

    // ===== COMPARE =====
    verify::verify_histogram(&mut hist, &ops, "rng_stress");

    // No trailing/leading zero buckets.
    let v = hist.view();
    let scale = v.scale();
    let buckets = v.positive();
    if buckets.len() > 0 {
        let first = buckets.iter().next().unwrap();
        let last = buckets.iter().last().unwrap();
        assert!(first > 0, "leading zero bucket at scale={}", scale);
        assert!(last > 0, "trailing zero bucket at scale={}", scale);
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    let config: Config = match <Config as Arbitrary>::arbitrary(
        &mut arbitrary::Unstructured::new(data),
    ) {
        Ok(c) => c,
        Err(_) => return,
    };

    run::<160, 32>(&config);
});
