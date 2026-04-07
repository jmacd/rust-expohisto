#![no_main]

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Width, Histogram, Scale, MAX_SCALE};

#[path = "verify.rs"]
mod verify;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let values: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();

    if values.is_empty() {
        return;
    }

    // Test with different histogram sizes.
    check_histogram::<8>(&values);
    check_histogram::<16>(&values);
});

/// Reference-oracle test: insert every value, then verify the histogram
/// state matches an independently-computed expectation.
fn check_histogram<const N: usize>(values: &[f64]) {
    let mut hist = Histogram::<N>::new().with_min_width(Width::B1);
    let mut inserted: Vec<f64> = Vec::new();

    for &v in values {
        if hist.update(v).is_ok() {
            inserted.push(v);
        }
    }

    let ops = inserted.iter().map(|&v| (v, 1u64)).collect::<Vec<_>>();
    verify::verify_histogram(&mut hist, &ops, "histogram_oracle");

    let non_zero: Vec<f64> = inserted.iter().copied().filter(|&v| v != 0.0).collect();
    if non_zero.is_empty() {
        return;
    }

    let scale = hist.view().scale();

    // ── 5. scale optimality ───────────────────────────────────────────
    // Verify that the scale is the highest one where the index span
    // fits in capacity. We use the *initial* B1 capacity (before any
    // widening) because the histogram only lowers scale when span
    // exceeds the capacity at whatever width it currently has.
    //
    // Counter overflow can force widen-steps (each costs 1 scale),
    // so we only assert scale <= span-optimal-scale.
    let b1_cap = (N * 64) as i32;

    // The histogram rounds subnormals to (biased_exp=1, significand=1).
    // Use the same rounded value so the oracle maps subnormals to the
    // same bucket index as the histogram.
    const SUBNORMAL_ROUNDED: f64 = f64::from_bits((1u64 << 52) | 1);

    // Find the highest scale where span fits at B1 capacity.
    let mut optimal = MAX_SCALE;
    for s in (otel_expohisto::MIN_SCALE..=MAX_SCALE).rev() {
        if let Ok(m) = Scale::new(s) {
            let indices: Vec<i32> = non_zero.iter().map(|&v| {
                if v.to_bits() >> 52 == 0 {
                    m.map_to_index(SUBNORMAL_ROUNDED)
                } else {
                    m.map_to_index(v)
                }
            }).collect();
            let lo = *indices.iter().min().unwrap();
            let hi = *indices.iter().max().unwrap();
            if hi - lo + 1 <= b1_cap {
                optimal = s;
                break;
            }
        }
    }

    assert!(
        scale <= optimal,
        "scale too high: hist={} but optimal span-fit is {} (b1_cap={})",
        scale,
        optimal,
        b1_cap,
    );
}
