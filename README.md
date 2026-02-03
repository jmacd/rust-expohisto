# rust-expohisto

An allocation-free implementation of the [OpenTelemetry Exponential Histogram](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram) in Rust.

## Overview

Exponential histograms provide a compact, high-resolution representation of value distributions using logarithmically-spaced bucket boundaries. This implementation is designed for:

- **No heap allocation**: Fixed-size bucket storage using const generics
- **`no_std` compatible**: Works in embedded and kernel contexts (requires `libm`)
- **High performance**: Optional lookup tables for O(1) index mapping

## Usage

```rust
use rust_expohisto::{Histogram, Mapping};

// Create a histogram with 160 buckets at scale 4
let mut hist: Histogram<u64, 160> = Histogram::new(4);

// Record some values
hist.update(1.5);
hist.update(2.7);
hist.update(100.0);

// Access bucket counts
for (index, count) in hist.positive().iter() {
    println!("bucket {}: {} values", index, count);
}
```

## Exponential Scale

The histogram divides the positive real line into buckets with boundaries at powers of `base = 2^(2^(-scale))`:

| Scale | Base | Buckets per power of 2 | Relative error |
|-------|------|------------------------|----------------|
| 10 | 1.00068 | 1024 | ~0.034% |
| 8 | 1.00271 | 256 | ~0.14% |
| 6 | 1.01089 | 64 | ~0.54% |
| 4 | 1.04427 | 16 | ~2.2% |
| 0 | 2.0 | 1 | ~41% |

Higher scales provide finer resolution at the cost of more buckets.

## Bucket Inclusivity

Per the OpenTelemetry specification (for Prometheus compatibility), bucket boundaries are **upper-inclusive**:

> The bucket identified by `index` represents values **greater than** `base^index` and **less than or equal to** `base^(index+1)`.

This means exact powers of two require special handling—they fall into the bucket *below* what a naive logarithm would suggest:

```rust
// For a power of two, index = (exponent << scale) - 1
if significand == 0 {
    return (exponent << scale) - 1;
}
```

## Index Mapping Algorithms

### Scale ≤ 0: Exponent Extraction

For non-positive scales, the bucket index is derived directly from the IEEE 754 exponent bits—no floating-point math required.

### Scale > 0: Logarithm Method (default)

For positive scales, the standard approach uses:

```
index = floor(ln(value) × 2^scale / ln(2))
```

This requires a logarithm computation (~50-100 cycles with `libm`).

### Scale > 0: Lookup Table Method (optional)

For applications where mapping performance is critical, compile-time lookup tables provide O(1) integer-only mapping:

```toml
[dependencies]
rust-expohisto = { version = "0.1", features = ["lookup-256"] }
```

Available features:
- `lookup-64`: 1.5KB table, supports scales 1-6
- `lookup-256`: 6KB table, supports scales 1-8  
- `lookup-1024`: 24KB table, supports scales 1-10

## Lookup Table Algorithm

The lookup table approach eliminates floating-point operations by:

1. **Linear bucket approximation**: Divide the mantissa range `[0, 2^52)` into `2N` equal-width linear buckets
2. **Precomputed mapping**: Each linear bucket maps to a log-scale bucket (with at most 1 bucket of error)
3. **Boundary refinement**: A single integer comparison against the exact boundary corrects the approximation

This algorithm was developed independently by [Dynatrace](https://github.com/open-telemetry/opentelemetry-collector/pull/3841) and [NewRelic](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java), with similar designs.

### Exact Boundary Computation

Bucket boundaries are computed exactly at build time using the algorithm from [PR #3841](https://github.com/open-telemetry/opentelemetry-collector/pull/3841):

1. Compute `2^position` exactly (trivial for integer powers)
2. Apply `sqrt()` `scale` times: `√(√(...√(2^position)...)) = 2^(position/2^scale)`
3. Scale by `2^52` and truncate to get candidate significand
4. Verify using exact BigUint arithmetic: `candidate^N ≥ 2^(52N + position)`
5. Increment if needed to find the exact boundary

This guarantees boundaries are correct to 1 ULP (unit in last place).

## Crate Structure

- **`rust-expohisto`**: Main library with histogram and mapping implementations
- **`mapping-gen`**: Sub-crate for generating lookup tables (used at build time)

The `mapping-gen` crate can be tested independently:

```bash
cd mapping-gen && cargo test
```

## References

- [OpenTelemetry Exponential Histogram Specification](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram)
- [go-expohisto](https://github.com/open-telemetry/otel-go-contrib/tree/main/exp/expohisto) - Reference Go implementation by the same author
- [Dynatrace Lookup Table Prototype (PR #3841)](https://github.com/open-telemetry/opentelemetry-collector/pull/3841)
- [NewRelic SubBucketLookupIndexer](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java)
- [NewRelic Indexer Documentation](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/Indexer.md)

## License

Apache-2.0
