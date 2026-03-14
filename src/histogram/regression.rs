use super::*;

/// Regression: merge_from_raw called trim_bucket_range() on literal-mode
/// histograms, corrupting literal_count by interpreting f64 bit patterns
/// as B1 counters.
#[test]
fn trim_bucket_range_preserves_literal_mode() {
    let v = f64::from_bits(0x4078014421f8ff58_u64.swap_bytes());

    let mut h: Histogram<8> = Histogram::new().with_literal_mode(true);

    // Insert non-zero value with incr=2
    assert!(h.update_by_incr(v, 2).is_ok());

    // Insert zero with incr=12
    assert!(h.update_by_incr(0.0, 12).is_ok());

    let view = h.view();
    let buckets = view.positive();
    let bucket_total: u64 = (0..buckets.len()).map(|i| buckets.at(i)).sum();
    assert_eq!(view.count(), 14);
    assert_eq!(bucket_total, 2, "non-zero observations should be in buckets");
}
