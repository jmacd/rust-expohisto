use otel_expohisto::Histogram;

fn main() {
    // Create a histogram with 16 u64 words (128 bytes) of data pool.
    // All 16 words are available for bucket data: 1024 one-bit buckets
    // at the default B1 width.
    let mut hist: Histogram<16> = Histogram::new();

    // Record observations
    hist.update(1.5).unwrap();
    hist.update(2.7).unwrap();
    hist.update(100.0).unwrap();

    // Access statistics through a view
    let v = hist.view();
    let stats = v.stats();
    println!("count: {}, sum: {}", stats.count, stats.sum);
    println!("scale: {}", v.scale());
}
