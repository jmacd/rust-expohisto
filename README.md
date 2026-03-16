# Rust OpenTelemetry Exponential Histogram

[![CI](https://github.com/jmacd/rust-expohisto/actions/workflows/ci.yml/badge.svg)](https://github.com/jmacd/rust-expohisto/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/otel-expohisto.svg)](https://crates.io/crates/otel-expohisto)
[![docs.rs](https://docs.rs/otel-expohisto/badge.svg)](https://docs.rs/otel-expohisto)
[![License](https://img.shields.io/crates/l/otel-expohisto.svg)](https://github.com/jmacd/rust-expohisto/blob/main/LICENSE)

An allocation-free, compile-time table-lookup based implementation of the
[OpenTelemetry Exponential Histogram](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram)
in Rust.

## Overview

Exponential histograms provide a compact, high-resolution representation of value distributions using logarithmically-spaced bucket boundaries. This implementation is designed for:

- **No heap allocation**: Fixed-size bucket storage using const generics (`Histogram<N>`)
- **`Send + Sync`**: All fields are `Copy` primitives; safe to share across threads with external synchronization
- **High performance**: Lookup table provides 3.5× speedup over logarithm-based mapping
- **Sub-byte counters**: 1-bit bucket counters (B1) maximize resolution; auto-widen on overflow
- **Configurable table size**: Trade static memory for lookup acceleration at higher scales
- **Quantile estimation**: CDF-walk with linear interpolation over the bucket distribution
- **Atomic error recovery**: Snapshot/rollback ensures failed operations leave the histogram unchanged
- **Literal mode**: Cold-start optimization defers bucket allocation until the value range is known
- **`no_std` compatible**: Only the `std::error::Error` impls require the `std` feature
- **Zero `unsafe` code**: Entirely safe Rust; no `unsafe` blocks anywhere in the crate
- **Zero runtime dependencies**: Only a build dependency (`expohisto-mapping-gen`) for compile-time table generation
- **Comprehensive testing**: Unit tests and fuzz targets

**Minimum Supported Rust Version (MSRV):** 1.73

### Limitations

- **Positive buckets only** — This crate implements a single positive bucket set. The OTel spec defines both positive and negative bucket ranges; negative values are rejected. This is suitable for the common case of non-negative measurements (latencies, sizes, counts). Adding negative bucket support would double the per-histogram memory footprint.

## Quick Start

```rust
use otel_expohisto::Histogram;

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
println!("count: {}, sum: {}", v.count(), v.sum());
println!("scale: {}", v.scale());
```

## Performance

Benchmark results — `map_to_index` over 100 random f64 values (criterion, median):

| Method | Scale range | Per-value | Notes |
|--------|------------|-----------|-------|
| Exponent | ≤ 0 | ~3.4 ns | Bit extraction only |
| Dynatrace lookup | 1–14 | ~5.9 ns | Integer-only, N linear buckets, 2 corrections |
| NewRelic lookup | 1–14 | ~7.3 ns | Integer-only, 2N linear buckets, 1 correction |
| Logarithm | 1–20 | ~10.5 ns | `ln()`-based, works at any scale |

The lookup table accelerates all scales from 1 up to the compiled maximum. Scales beyond the table maximum are rejected by `Mapping::new()`.

## Lookup Table Features

Choose a **scale** feature to set the table size, and an **algorithm** feature to select the mapping method:

```toml
[dependencies]
otel-expohisto = { version = "0.1", features = ["newrelic", "scale-8"] }  # default: std + newrelic + scale-8
```

### Scale (table size)

Scale features `scale-1` through `scale-20` control the lookup table size.
Each table supports all scales from 1 up to its maximum; higher scales
are rejected by `Mapping::new()`. Selected examples:

| Feature | Table Size | Scales Accelerated | Use Case |
|---------|------------|-------------------|----------|
| `scale-4` | 152 B | 1–4 | Minimal memory |
| `scale-6` | 536 B | 1–6 | Embedded systems |
| `scale-8` | 2 KB | 1–8 | **Default** |
| `scale-10` | 8 KB | 1–10 | Recommended |
| `scale-12` | 32 KB | 1–12 | High resolution |
| `scale-14` | 128 KB | 1–14 | Maximum practical coverage |

### Algorithm

| Feature | Linear Buckets | Max Corrections | Notes |
|---------|---------------|-----------------|-------|
| `newrelic` | 2N | 1 | **Default**. Slightly larger index table |
| `dynatrace` | N | 2 | ~50% smaller index table |

The `newrelic` and `dynatrace` algorithms produce identical results, are equally tested, and perform the same at runtime — choose whichever you prefer. Memory differences are negligible (see [Lookup Table Design](docs/design.md#lookup-table-design)). At least one must be enabled.

### Other features

| Feature | Default | Effect |
|---------|---------|--------|
| `std` | ✓ | Enables `std::error::Error` impls for `Overflow` and `MappingError`. Disable for `#![no_std]` builds. |
| `logarithm` | | Pure `ln()`-based mapper for testing and benchmarking. Requires `std`. |
| `boundary` | | Enables `lower_boundary()` at positive scales and quantile estimation (`QuantileIter`). Requires `std`. |
| `bench-internals` | | Exposes internal methods (e.g., `downscale()`) for benchmarking |
| `bench-all` | | Enables `newrelic` + `dynatrace` + `logarithm` + `boundary` + `scale-8` + `bench-internals` for testing all algorithms together |

## Documentation

- **[Design & Architecture](docs/design.md)** — Exponential scale theory, index mapping algorithms, lookup table design, crate structure
- **[Implementation Details](docs/internals.md)** — Literal mode, sub-byte counters (SWAR), parameter selection, API overview
- **[Reference](docs/reference.md)** — Error handling, thread safety, `no_std` support, testing, OTel spec compatibility

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build, test, fuzz, and PR guidelines.

Quick validation:

```bash
cargo clippy --all-targets --features bench-all -- -D warnings
cargo test --features bench-all
cargo check --manifest-path fuzz/Cargo.toml --all-targets
```

## License

Apache-2.0
