#![no_main]

//! **State-machine fuzzer** — exercises interleaved sequences of
//! insert, merge, clear, and swap on a pool of histograms, verifying
//! invariants after every operation.
//!
//! This targets risks that single-operation oracles miss:
//!   - clear → reuse cycles (stale index_base, wrong bucket_width)
//!   - merge after partial inserts with different bucket widths
//!   - swap correctness (does shadow state track?)
//!   - cross-precision merge (P32 ↔ P64 via merge_from_other)
//!   - cascading downscale/widen under odd-base + full-capacity pressure
//!   - counter overflow recovery paths (NeedsDownscale vs CounterOverflow)

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rust_expohisto::{Histogram, Mapping, P32, P64};
use std::collections::BTreeMap;

/// Number of P32 histograms in the pool.
const P32_POOL: usize = 3;
/// Number of P64 histograms in the pool.
const P64_POOL: usize = 2;

/// A weighted observation.
#[derive(Clone, Copy)]
struct Obs {
    value: f64,
    incr: u64,
}

/// Shadow state tracking what a histogram *should* contain.
#[derive(Clone, Default)]
struct Shadow {
    ops: Vec<Obs>,
}

impl Shadow {
    fn count(&self) -> u64 {
        self.ops.iter().map(|o| o.incr).sum()
    }
    fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    fn merge_from(&mut self, other: &Shadow) {
        self.ops.extend_from_slice(&other.ops);
    }
    fn clear(&mut self) {
        self.ops.clear();
    }
    fn swap(&mut self, other: &mut Shadow) {
        core::mem::swap(self, other);
    }
}

// ---------------------------------------------------------------------------
// Operations — decoded from fuzz input via Arbitrary
// ---------------------------------------------------------------------------

#[derive(Arbitrary, Debug)]
enum Op {
    /// Insert a value into a P32 histogram.
    InsertP32 {
        /// Pool index (mod P32_POOL).
        idx: u8,
        /// Raw f64 bytes for value.
        value_bits: u64,
        /// Increment selector — decoded into a tiered increment.
        incr_sel: u8,
    },
    /// Insert a value into a P64 histogram.
    InsertP64 {
        idx: u8,
        value_bits: u64,
        incr_sel: u8,
    },
    /// Merge one P32 histogram into another (same type).
    MergeP32 { dst: u8, src: u8 },
    /// Merge one P64 histogram into another (same type).
    MergeP64 { dst: u8, src: u8 },
    /// Clear a P32 histogram.
    ClearP32 { idx: u8 },
    /// Clear a P64 histogram.
    ClearP64 { idx: u8 },
    /// Swap two P32 histograms.
    SwapP32 { a: u8, b: u8 },
    /// Insert the same value with a very large increment (pressure-test
    /// counter overflow and the widen/downscale recovery loop).
    HammerP32 {
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
        0..=49 => 1,                    // B1 range
        50..=79 => 2 + (sel as u64 - 50) % 2,   // B2 boundary (2-3)
        80..=109 => 4 + (sel as u64 - 80) % 12,  // B4 range (4-15)
        110..=139 => 16 + (sel as u64 - 110) * 8, // U8 range
        140..=169 => 256 + (sel as u64 - 140) * 2048, // U16 range
        170..=199 => 65536 + (sel as u64 - 170) * 131072, // U32 range
        200..=229 => 1u64 << ((sel - 200) as u32 + 17),   // large powers of 2
        230..=255 => sel as u64,        // small values for variety
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn verify_p32<const N: usize>(hist: &Histogram<N, P32>, shadow: &Shadow, label: &str) {
    verify_generic::<N, P32>(hist, shadow, label, true);
}

fn verify_p64<const N: usize>(hist: &Histogram<N, P64>, shadow: &Shadow, label: &str) {
    verify_generic::<N, P64>(hist, shadow, label, false);
}

fn verify_generic<const N: usize, P: rust_expohisto::Precision>(
    hist: &Histogram<N, P>,
    shadow: &Shadow,
    label: &str,
    is_p32: bool,
) {
    let expected_count = shadow.count();
    assert_eq!(
        hist.count(),
        expected_count,
        "{}: count mismatch: hist={} expected={}",
        label,
        hist.count(),
        expected_count,
    );

    if expected_count == 0 {
        assert_eq!(hist.positive().len(), 0, "{}: should have no buckets", label);
        return;
    }

    // -- min / max --
    let expected_min = shadow.ops.iter().map(|o| o.value).fold(f64::INFINITY, f64::min);
    let expected_max = shadow.ops.iter().map(|o| o.value).fold(f64::NEG_INFINITY, f64::max);

    if is_p32 {
        assert_eq!(
            hist.min() as f32,
            expected_min as f32,
            "{}: min mismatch (P32)",
            label,
        );
        assert_eq!(
            hist.max() as f32,
            expected_max as f32,
            "{}: max mismatch (P32)",
            label,
        );
    } else {
        assert_eq!(hist.min(), expected_min, "{}: min mismatch (P64)", label);
        assert_eq!(hist.max(), expected_max, "{}: max mismatch (P64)", label);
    }

    // -- zero count --
    let non_zero_total: u64 = shadow
        .ops
        .iter()
        .filter(|o| o.value != 0.0)
        .map(|o| o.incr)
        .sum();
    let expected_zero_count = expected_count - non_zero_total;

    let buckets = hist.positive();
    let bucket_total: u64 = (0..buckets.len()).map(|i| buckets.at(i)).sum();

    assert!(
        bucket_total <= hist.count(),
        "{}: bucket total ({}) exceeds count ({})",
        label,
        bucket_total,
        hist.count(),
    );
    let actual_zero_count = hist.count() - bucket_total;
    assert_eq!(
        actual_zero_count, expected_zero_count,
        "{}: zero count mismatch (actual={}, expected={})",
        label,
        actual_zero_count,
        expected_zero_count,
    );

    // -- bucket distribution at final scale --
    if non_zero_total == 0 {
        assert_eq!(buckets.len(), 0, "{}: expected no buckets", label);
        return;
    }

    let scale = hist.scale();
    let mapping = Mapping::new(scale).expect("reported scale should be valid");

    let mut expected_buckets: BTreeMap<i32, u64> = BTreeMap::new();
    for op in &shadow.ops {
        if op.value != 0.0 {
            let idx = mapping.map_to_index(op.value);
            *expected_buckets.entry(idx).or_insert(0) += op.incr;
        }
    }

    let exp_min_idx = *expected_buckets.keys().next().unwrap();
    let exp_max_idx = *expected_buckets.keys().last().unwrap();
    let exp_len = (exp_max_idx - exp_min_idx + 1) as u32;

    assert_eq!(
        buckets.offset(),
        exp_min_idx,
        "{}: offset mismatch: hist={} expected={} (scale={})",
        label,
        buckets.offset(),
        exp_min_idx,
        scale,
    );

    assert_eq!(
        buckets.len(),
        exp_len,
        "{}: bucket len mismatch: hist={} expected={} (scale={})",
        label,
        buckets.len(),
        exp_len,
        scale,
    );

    for pos in 0..buckets.len() {
        let idx = exp_min_idx + pos as i32;
        let exp_count = expected_buckets.get(&idx).copied().unwrap_or(0);
        let act_count = buckets.at(pos);
        assert_eq!(
            act_count, exp_count,
            "{}: bucket[{}] (idx {}): hist={} expected={} (scale={}, width={:?})",
            label,
            pos,
            idx,
            act_count,
            exp_count,
            scale,
            buckets.width(),
        );
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

    // Use first byte to decide literal mode for each pool member.
    let lit_ctl: u8 = u.arbitrary().unwrap_or(0xFF);

    // Pool of histograms — small N to maximize pressure on downscale/widen.
    let mut p32: [Histogram<8, P32>; P32_POOL] = core::array::from_fn(|i| {
        Histogram::new().with_literal_mode(lit_ctl & (1 << i) != 0)
    });
    let mut sp32: [Shadow; P32_POOL] = core::array::from_fn(|_| Shadow::default());

    let mut p64: [Histogram<8, P64>; P64_POOL] = core::array::from_fn(|i| {
        Histogram::new().with_literal_mode(lit_ctl & (1 << (P32_POOL + i)) != 0)
    });
    let mut sp64: [Shadow; P64_POOL] = core::array::from_fn(|_| Shadow::default());

    // Also keep one Histogram<16> for cross-size merges.
    let mut big: Histogram<16, P32> = Histogram::new().with_literal_mode(lit_ctl & 0x80 != 0);
    let mut sbig: Shadow = Shadow::default();

    // Cap operations to keep memory bounded.
    let mut ops_remaining = 256usize;

    while ops_remaining > 0 {
        let op = match u.arbitrary::<Op>() {
            Ok(op) => op,
            Err(_) => break,
        };
        ops_remaining -= 1;
        match op {
            Op::InsertP32 {
                idx,
                value_bits,
                incr_sel,
            } => {
                let i = idx as usize % P32_POOL;
                if let Some(v) = decode_value(value_bits) {
                    let incr = decode_increment(incr_sel);
                    if p32[i].update_by_incr(v, incr).is_ok() {
                        sp32[i].ops.push(Obs { value: v, incr });
                    }
                }
            }

            Op::InsertP64 {
                idx,
                value_bits,
                incr_sel,
            } => {
                let i = idx as usize % P64_POOL;
                if let Some(v) = decode_value(value_bits) {
                    let incr = decode_increment(incr_sel);
                    if p64[i].update_by_incr(v, incr).is_ok() {
                        sp64[i].ops.push(Obs { value: v, incr });
                    }
                }
            }

            Op::MergeP32 { dst, src } => {
                let d = dst as usize % P32_POOL;
                let s = src as usize % P32_POOL;
                if d == s {
                    continue;
                }
                let _ = big.merge_from_other(&p32[s]);
                if big.count() == sp32[s].count() + sbig.count() {
                    sbig.merge_from(&sp32[s]);
                }

                let (lo, hi) = if d < s {
                    let (a, b) = p32.split_at_mut(s);
                    (&mut a[d], &b[0])
                } else {
                    let (a, b) = p32.split_at_mut(d);
                    (&mut b[0], &a[s] as &Histogram<8, P32>)
                };
                if lo.merge_from(hi).is_ok() {
                    let src_shadow = sp32[s].clone();
                    sp32[d].merge_from(&src_shadow);
                }
            }

            Op::MergeP64 { dst, src } => {
                let d = dst as usize % P64_POOL;
                let s = src as usize % P64_POOL;
                if d == s {
                    continue;
                }
                let (lo, hi) = if d < s {
                    let (a, b) = p64.split_at_mut(s);
                    (&mut a[d], &b[0])
                } else {
                    let (a, b) = p64.split_at_mut(d);
                    (&mut b[0], &a[s] as &Histogram<8, P64>)
                };
                if lo.merge_from(hi).is_ok() {
                    let src_shadow = sp64[s].clone();
                    sp64[d].merge_from(&src_shadow);
                }
            }

            Op::ClearP32 { idx } => {
                let i = idx as usize % P32_POOL;
                p32[i].clear();
                sp32[i].clear();
            }

            Op::ClearP64 { idx } => {
                let i = idx as usize % P64_POOL;
                p64[i].clear();
                sp64[i].clear();
            }

            Op::SwapP32 { a, b } => {
                let ai = a as usize % P32_POOL;
                let bi = b as usize % P32_POOL;
                if ai == bi {
                    continue;
                }
                let (lo, hi) = if ai < bi {
                    let (a, b) = p32.split_at_mut(bi);
                    (&mut a[ai], &mut b[0])
                } else {
                    let (a, b) = p32.split_at_mut(ai);
                    (&mut b[0], &mut a[bi])
                };
                lo.swap(hi);
                let (slo, shi) = if ai < bi {
                    let (a, b) = sp32.split_at_mut(bi);
                    (&mut a[ai], &mut b[0])
                } else {
                    let (a, b) = sp32.split_at_mut(ai);
                    (&mut b[0], &mut a[bi])
                };
                slo.swap(shi);
            }

            Op::HammerP32 {
                idx,
                value_bits,
                log_incr,
            } => {
                let i = idx as usize % P32_POOL;
                if let Some(v) = decode_value(value_bits) {
                    let incr = 1u64 << ((log_incr % 41) as u32);
                    if p32[i].update_by_incr(v, incr).is_ok() {
                        sp32[i].ops.push(Obs { value: v, incr });
                    }
                }
            }
        }
    }

    // Verify all histograms at end of sequence.
    for i in 0..P32_POOL {
        verify_p32(&p32[i], &sp32[i], &format!("p32[{i}]"));
    }
    for i in 0..P64_POOL {
        verify_p64(&p64[i], &sp64[i], &format!("p64[{i}]"));
    }
    if !sbig.is_empty() {
        verify_p32(&big, &sbig, "big");
    }
});
