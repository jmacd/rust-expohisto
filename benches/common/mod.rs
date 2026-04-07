use otel_expohisto::Scale;

pub fn unique_values(scale: i32, count: usize) -> Vec<f64> {
    let m = Scale::new(scale).unwrap();
    let mut vals = Vec::with_capacity(count);
    for i in 0..count as i32 {
        let lo = m.lower_boundary(i).unwrap_or(1.0);
        let hi = m.lower_boundary(i + 1).unwrap_or(lo * 1.001);
        vals.push((lo + hi) / 2.0);
    }
    vals
}
