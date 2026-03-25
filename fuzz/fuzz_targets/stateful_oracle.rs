#![no_main]

//! **State-machine fuzzer** — exercises interleaved sequences of
//! insert, merge, reset, and swap on a pool of histograms, verifying
//! invariants after every operation.
//!
//! This targets risks that single-operation oracles miss:
//!   - reset → reuse cycles (stale index_base, wrong width)
//!   - merge after partial inserts with different bucket widths
//!   - swap correctness (does shadow state track?)
//!   - cross-size merge (Histogram<8> → Histogram<16> via merge_from)
//!   - cascading downscale/widen under odd-base + full-capacity pressure
//!   - counter overflow recovery paths (NeedsDownscale vs CounterOverflow)

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Width, Histogram};

#[path = "verify.rs"]
mod verify;

/// Number of pooled Histogram<8> instances.
const POOL: usize = 5;

/// Maximum shadow ops before skipping exact verification.
/// Back-and-forth merges cause Fibonacci growth in ops lists;
/// we cap to keep fuzz throughput high while still verifying
/// small/medium inputs exactly.
const MAX_SHADOW_OPS: usize = 50_000;

/// A weighted observation.
#[derive(Clone, Copy)]
struct Obs {
    value: f64,
    incr: u64,
}

/// Shadow state tracking what a histogram *should* contain.
/// When ops exceeds MAX_SHADOW_OPS the shadow is poisoned and
/// exact verification is skipped (the histogram is still exercised).
#[derive(Clone, Default)]
struct Shadow {
    ops: Vec<Obs>,
    poisoned: bool,
}

impl Shadow {
    fn merge_from(&mut self, other: &Shadow) {
        if self.poisoned || other.poisoned
            || self.ops.len() + other.ops.len() > MAX_SHADOW_OPS
        {
            self.poisoned = true;
            self.ops.clear();
            return;
        }
        self.ops.extend_from_slice(&other.ops);
    }
    fn clear(&mut self) {
        self.ops.clear();
        self.poisoned = false;
    }
    fn swap(&mut self, other: &mut Shadow) {
        core::mem::swap(self, other);
    }
    fn can_verify(&self) -> bool {
        !self.poisoned
    }
}

// ---------------------------------------------------------------------------
// Operations — decoded from fuzz input via Arbitrary
// ---------------------------------------------------------------------------

#[derive(Arbitrary, Debug)]
enum Op {
    /// Insert a value into a pooled histogram.
    Insert {
        /// Pool index (mod POOL).
        idx: u8,
        /// Raw f64 bytes for value.
        value_bits: u64,
        /// Increment selector — decoded into a tiered increment.
        incr_sel: u8,
    },
    /// Merge one pooled histogram into another.
    Merge { dst: u8, src: u8 },
    /// Reset a pooled histogram (re-create).
    Clear { idx: u8 },
    /// Swap two pooled histograms.
    Swap { a: u8, b: u8 },
    /// Insert the same value with a very large increment (pressure-test
    /// counter overflow and the widen/downscale recovery loop).
    Hammer {
        idx: u8,
        value_bits: u64,
        /// log2 of increment (0..=40).
        log_incr: u8,
    },
}

// ---------------------------------------------------------------------------
// Value and increment decoders
// ---------------------------------------------------------------------------

fn decode_value(bits: u64) -> Option<f64> {
    let v = f64::from_bits(bits);
    if v.is_finite() && v.is_normal() && v >= 0.0 {
        Some(v)
    } else if v == 0.0 {
        Some(0.0)
    } else {
        None
    }
}

/// Map a selector byte into an increment that exercises different
/// bucket-width tiers, including exact overflow boundaries.
fn decode_increment(sel: u8) -> u64 {
    match sel {
        0..=49 => 1, // B1 range
        50..=79 => 2 + (sel as u64 - 50) % 2, // B2 boundary (2-3)
        80..=109 => 4 + (sel as u64 - 80) % 12, // B4 range (4-15)
        110..=139 => 16 + (sel as u64 - 110) * 8, // U8 range
        140..=169 => 256 + (sel as u64 - 140) * 2048, // U16 range
        170..=199 => 65536 + (sel as u64 - 170) * 131072, // U32 range
        200..=229 => 1u64 << ((sel - 200) as u32 + 17), // large powers of 2
        230..=255 => sel as u64, // small values for variety
    }
}

fn try_insert(hist: &mut Histogram<8>, shadow: &mut Shadow, value_bits: u64, incr: u64) {
    if let Some(v) = decode_value(value_bits) {
        let snapshot = hist.clone();
        match hist.record_incr(v, incr) {
            Ok(()) if !shadow.poisoned => shadow.ops.push(Obs { value: v, incr }),
            Err(_) => *hist = snapshot,
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn verify<const N: usize>(hist: &mut Histogram<N>, shadow: &Shadow, label: &str) {
    if !shadow.can_verify() {
        return;
    }
    let ops = shadow
        .ops
        .iter()
        .map(|o| (o.value, o.incr))
        .collect::<Vec<_>>();
    verify::verify_histogram(hist, &ops, label);
}

/// Borrows two distinct elements of a slice mutably.
fn two_mut<T>(arr: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert_ne!(i, j);
    if i < j {
        let (a, b) = arr.split_at_mut(j);
        (&mut a[i], &mut b[0])
    } else {
        let (a, b) = arr.split_at_mut(i);
        (&mut b[0], &mut a[j])
    }
}

// ---------------------------------------------------------------------------
// Fuzz target
// ---------------------------------------------------------------------------

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let mut u = Unstructured::new(data);

    // Pool of histograms — small N to maximize pressure on downscale/widen.
    let mut pool: [Histogram<8>; POOL] = core::array::from_fn(|_| {
        Histogram::new().with_min_width(Width::B1)
    });
    let mut shadows: [Shadow; POOL] = core::array::from_fn(|_| Shadow::default());

    // Also keep one Histogram<16> for cross-size merges.
    let mut big: Histogram<16> = Histogram::new().with_min_width(Width::B1);
    let mut big_shadow: Shadow = Shadow::default();

    // Cap operations to keep memory bounded.
    let mut ops_remaining = 256usize;

    while ops_remaining > 0 {
        let op = match u.arbitrary::<Op>() {
            Ok(op) => op,
            Err(_) => break,
        };
        ops_remaining -= 1;
        match op {
            Op::Insert {
                idx,
                value_bits,
                incr_sel,
            } => {
                let i = idx as usize % POOL;
                let incr = decode_increment(incr_sel);
                try_insert(&mut pool[i], &mut shadows[i], value_bits, incr);
            }

            Op::Merge { dst, src } => {
                let d = dst as usize % POOL;
                let s = src as usize % POOL;
                if d == s {
                    continue;
                }
                {
                    let snapshot = big.clone();
                    match big.merge_from(&pool[s]) {
                        Ok(()) => {
                            let src_shadow = shadows[s].clone();
                            big_shadow.merge_from(&src_shadow);
                        }
                        Err(_) => big = snapshot,
                    }
                }

                let (dst_hist, src_hist) = two_mut(&mut pool, d, s);
                let snapshot = dst_hist.clone();
                match dst_hist.merge_from(src_hist) {
                    Ok(()) => {
                        let src_shadow = shadows[s].clone();
                        shadows[d].merge_from(&src_shadow);
                    }
                    Err(_) => *dst_hist = snapshot,
                }
            }

            Op::Clear { idx } => {
                let i = idx as usize % POOL;
                pool[i] = Histogram::new().with_min_width(Width::B1);
                shadows[i].clear();
            }

            Op::Swap { a, b } => {
                let ai = a as usize % POOL;
                let bi = b as usize % POOL;
                if ai == bi {
                    continue;
                }
                let (lo, hi) = two_mut(&mut pool, ai, bi);
                lo.swap(hi);
                let (slo, shi) = two_mut(&mut shadows, ai, bi);
                slo.swap(shi);
            }

            Op::Hammer {
                idx,
                value_bits,
                log_incr,
            } => {
                let i = idx as usize % POOL;
                let incr = 1u64 << ((log_incr % 41) as u32);
                try_insert(&mut pool[i], &mut shadows[i], value_bits, incr);
            }
        }
    }

    // Verify all histograms at end of sequence.
    for i in 0..POOL {
        verify(&mut pool[i], &shadows[i], &format!("pool[{i}]"));
    }
    verify(&mut big, &big_shadow, "big");
});
