#![no_main]

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Width, Histogram};

#[path = "verify.rs"]
mod verify;

/// A weighted insert operation: record `value` with multiplicity `incr`.
#[derive(Clone, Copy)]
struct Op {
    value: f64,
    incr: u64,
}

fuzz_target!(|data: &[u8]| {
    // Layout: [partition, mode, echo_ctl, ...(f64 value, u8 incr_sel) × N...]
    if data.len() < 12 {
        return;
    }

    let partition_byte = data[0];
    let mode = data[1];
    let echo_ctl = data[2];
    let rest = &data[3..];

    // Parse 9-byte chunks: 8 bytes f64 value + 1 byte increment selector.
    let mut ops: Vec<Op> = rest
        .chunks_exact(9)
        .filter_map(|c| {
            let v = f64::from_le_bytes(c[..8].try_into().unwrap());
            if !v.is_finite() || v < 0.0 {
                return None;
            }
            let incr = decode_increment(c[8], mode);
            Some(Op { value: v, incr })
        })
        .collect();

    if ops.is_empty() {
        return;
    }

    // Echo phase: replay earlier values with exponentially growing
    // multipliers. This forces counter widening (B1→B2→…→U64) and,
    // for large enough totals, larger count accumulation.
    let echo_passes = (echo_ctl % 4) as usize;
    let base_len = ops.len();
    for pass in 0..echo_passes {
        for i in 0..base_len {
            let Op { value, incr } = ops[i];
            let scale = 1u64 << ((pass * 4 + 2).min(30));
            let boosted = incr.saturating_mul(scale);
            // Cap at 2^40 to keep the fuzzer responsive.
            if boosted > 0 && boosted <= (1u64 << 40) {
                ops.push(Op { value, incr: boosted });
            }
        }
    }

    let split = (partition_byte as usize) % (ops.len() + 1);
    let (left, right) = ops.split_at(split);

    check_merge_same::<8>(left, right);
    check_merge_same::<16>(left, right);
    check_merge_different::<8, 16>(left, right);
    check_merge_different::<16, 8>(left, right);
});

/// Map a selector byte into an increment that exercises different
/// bucket-width tiers. The `mode` byte picks a strategy so the fuzzer
/// can explore qualitatively different counter distributions.
fn decode_increment(sel: u8, mode: u8) -> u64 {
    match mode % 5 {
        // Powers of 2 — leap across width boundaries.
        0 => 1u64 << ((sel as u32) % 24),

        // Sit right at each width's maximum value.
        1 => match sel {
            0..=63 => 1,                 // B1 max
            64..=95 => 3,                // B2 max
            96..=127 => 15,              // B4 max
            128..=159 => 255,            // U8 max
            160..=191 => 65535,          // U16 max
            192..=223 => (1 << 24) - 1, // large U32
            224..=255 => 1 << 28,        // toward U32 overflow
        },

        // Ramp: selector scaled by mode, covering a smooth range.
        2 => (sel as u64 + 1) * ((mode as u64 / 5) + 1),

        // Just past each threshold — immediate widening trigger.
        3 => match sel % 6 {
            0 => 2,     // exceeds B1
            1 => 4,     // exceeds B2
            2 => 16,    // exceeds B4
            3 => 256,   // exceeds U8
            4 => 65536, // exceeds U16
            _ => 1,
        },

        // Mostly unit inserts with rare large spikes.
        _ => {
            if sel < 200 {
                1
            } else {
                1u64 << (sel - 200)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Merge checks
// ---------------------------------------------------------------------------

fn build<const N: usize>(ops: &[Op]) -> (Histogram<N>, Vec<Op>) {
    let mut h = Histogram::<N>::new().with_min_width(Width::B1);
    let mut ok = Vec::new();
    for &op in ops {
        if h.record_incr(op.value, op.incr).is_ok() {
            ok.push(op);
        }
    }
    (h, ok)
}

fn check_merge_same<const N: usize>(left: &[Op], right: &[Op]) {
    let (mut h1, mut ok) = build::<N>(left);
    let (h2, ok_right) = build::<N>(right);
    if h1.merge_from(&h2).is_err() {
        return;
    }
    ok.extend_from_slice(&ok_right);
    verify_histogram(&mut h1, &ok);
}

fn check_merge_different<const N: usize, const M: usize>(
    left: &[Op],
    right: &[Op],
) {
    let (mut h1, mut ok) = build::<N>(left);
    let (h2, ok_right) = build::<M>(right);
    if h1.merge_from(&h2).is_err() {
        return;
    }
    ok.extend_from_slice(&ok_right);
    verify_histogram(&mut h1, &ok);
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

fn verify_histogram<const N: usize>(hist: &mut Histogram<N>, inserted: &[Op]) {
    let ops = inserted
        .iter()
        .map(|op| (op.value, op.incr))
        .collect::<Vec<_>>();
    verify::verify_histogram(hist, &ops, "merge");
}
