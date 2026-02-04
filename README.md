# rust-expohisto

An allocation-free implementation of the [OpenTelemetry Exponential Histogram](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram) in Rust.

## Overview

Exponential histograms provide a compact, high-resolution representation of value distributions using logarithmically-spaced bucket boundaries. This implementation is designed for:

- **No heap allocation**: Fixed-size bucket storage using const generics
- **High performance**: Lookup table provides 3.5× speedup over logarithm-based mapping
- **Configurable table size**: Trade static memory for lookup acceleration at higher scales

## Quick Start

```rust
use rust_expohisto::Histogram;

// Create a histogram with 64 buckets and u32 counters
let mut hist: Histogram<u32, 64> = Histogram::new();

// Record observations
hist.update(1.5);
hist.update(2.7);
hist.update(100.0);

// Access statistics
println!("count: {}, sum: {}", hist.count(), hist.sum());
println!("scale: {}", hist.scale());
```

## Performance

Benchmark results (15 test values, per-value timing):

| Method | Scale | Time | Notes |
|--------|-------|------|-------|
| Exponent | ≤0 | ~1.1 ns | Bit extraction only |
| Lookup | 1-14 | ~1.7 ns | Integer-only, 3.5× faster than log |
| Logarithm | 1-14 | ~6 ns | Fallback when lookup unavailable |
| Logarithm | 15-20 | ~6 ns | No lookup table for scales >14 |

The lookup table accelerates all scales from 1 up to the compiled maximum. Higher scales beyond the table fall back to logarithm computation.

## Lookup Table Features

Choose **one** feature based on your needs—each table supports all scales from 1 up to the maximum:

```toml
[dependencies]
rust-expohisto = { version = "0.1", features = ["lookup-10"] }  # default
```

| Feature | Table Size | Scales Accelerated | Use Case |
|---------|------------|-------------------|----------|
| `lookup-4` | 0.2 KB | 1–4 | Minimal memory |
| `lookup-6` | 0.8 KB | 1–6 | Embedded systems |
| `lookup-8` | 3 KB | 1–8 | Balanced |
| `lookup-10` | 12 KB | 1–10 | **Recommended** (default) |
| `lookup-12` | 48 KB | 1–12 | High resolution |
| `lookup-14` | 192 KB | 1–14 | Maximum coverage |

Only one feature may be enabled—a compile-time check enforces this.

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

### Scale > 0: Lookup Table (default)

When a lookup feature is enabled (default: `lookup-10`), mapping uses integer-only operations:

1. Extract mantissa and exponent from the IEEE 754 representation
2. Use the mantissa to index into a precomputed lookup table
3. Apply a single boundary check to correct the approximation
4. Combine with exponent to produce the final index

For scales beyond the table's maximum, the implementation falls back to logarithm computation.

### Scale > 0: Logarithm Fallback

When no lookup table is available (or for scales above the table's maximum), the standard formula is used:

```
index = floor(ln(value) × 2^scale / ln(2))
```

## Lookup Table Design

The lookup table eliminates floating-point operations by:

1. **Linear bucket approximation**: Divide the mantissa range `[0, 2^52)` into `2N` equal-width linear buckets
2. **Precomputed mapping**: Each linear bucket maps to a log-scale bucket (with at most 1 bucket of error)
3. **Boundary refinement**: A single integer comparison against the exact boundary corrects the approximation

A single table at scale N supports all scales 1 through N by computing the index at full resolution and right-shifting the result.

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
