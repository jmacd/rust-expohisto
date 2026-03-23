#![no_main]

use libfuzzer_sys::fuzz_target;
use otel_expohisto::{Width, Histogram, Scale, max_scale};

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

    // Test with literal mode (default).
    check_histogram::<8>(&values, true);
    check_histogram::<16>(&values, true);

    // Test with literal mode disabled (bucket mode from start).
    check_histogram::<8>(&values, false);
    check_histogram::<16>(&values, false);
});

/// Reference-oracle test: insert every value, then verify the histogram
/// state matches an independently-computed expectation.
fn check_histogram<const N: usize>(values: &[f64], literal_mode: bool) {
    let mut hist = Histogram::<N>::new().with_min_width(if literal_mode { Width::B0 } else { Width::B1 });
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

    // Find the highest scale where span fits at B1 capacity.
    let mut optimal = max_scale();
    for s in (otel_expohisto::MIN_SCALE..=max_scale()).rev() {
        if let Ok(m) = Scale::new(s) {
            let indices: Vec<i32> = non_zero.iter().map(|&v| m.map_to_index(v)).collect();
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
