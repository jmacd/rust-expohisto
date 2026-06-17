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
- **High performance**: Lookup table provides ~1.8× speedup over logarithm-based mapping
- **Sub-byte counters**: 1-bit bucket counters (B1) maximize resolution; auto-widen on overflow
- **Configurable table size**: Trade static memory for lookup acceleration at higher scales
- **Quantile estimation**: CDF-walk with linear interpolation over the bucket distribution
- **Guaranteed relative error**: `Sketch<N>` / `SketchPN<K, L>` offer a fixed-scale, DDSketch-style collapse-left mode with a worst-case relative-error bound (see [Relative-error mode](#relative-error-mode-sketch))
- **Compact configuration**: 2-byte `Settings` struct pairs `Scale` + `Width`; histograms track both initial and current settings
- **`no_std` compatible**: Only the `std::error::Error` impls require the `std` feature
- **Zero `unsafe` code**: Entirely safe Rust; no `unsafe` blocks anywhere in the crate
- **Zero runtime dependencies**: Only a build dependency (`expohisto-mapping-gen`) for compile-time table generation
- **Comprehensive testing**: Unit tests and fuzz targets

**Minimum Supported Rust Version (MSRV):** 1.83

### Limitations

- **`HistogramNN` is positive-only** — `HistogramNN<N>` (aliased as `Histogram<N>`) implements a single positive bucket set; negative values are rejected. This is suitable for the common case of non-negative measurements (latencies, sizes, counts). For values of any sign, use `HistogramPN<K, L>` which maintains independent positive and negative bucket ranges with synchronized scales.

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
let stats = v.stats();
println!("count: {}, sum: {}", stats.count, stats.sum);
println!("scale: {}", v.scale());
```

## Relative-error mode (`Sketch`)

`Sketch<N>` trades auto-scaling for a **guaranteed relative error**, in the
style of [DDSketch](https://arxiv.org/abs/1908.10693). The scale is fixed at
construction, so every retained bucket has a worst-case relative error
`α = (b − 1)/(b + 1)` with `b = 2^(2^−scale)`. When a value won't fit the
current window, the structure **collapses the left (small-value) side**
instead of downscaling — the largest values keep full resolution and the
dropped small buckets fold into an underflow counter. This is ideal for
latency tails, where the high quantiles matter and the low end does not.

```rust
use otel_expohisto::Sketch;

// Coarsest scale whose worst-case relative error is within 1%.
let mut s: Sketch<16> = Sketch::new().with_relative_error(0.01).unwrap();
for v in [1.0, 2.0, 5.0, 12.0, 30.0, 100.0, 1e6] {
    s.update(v).unwrap();
}

let v = s.view();
println!("scale: {}, p-tail accurate above underflow", v.scale());
println!("underflow (inaccurate low mass): {}", v.underflow_count());
// `v.positive()` yields an OTel-compatible contiguous `bucket_counts` run.
```

`SketchPN<K, L>` is the signed counterpart (a positive and a negative
`Sketch` plus a zero count). For the guarantee, the collapse mechanics, the
order-dependence tradeoff, and sizing, see
[Collapse-Left Design](docs/collapse-design.md).

## Performance

Benchmark results — `map_to_index` over 100 random f64 values (criterion, median):

| Method | Scale range | Per-value | Notes |
|--------|------------|-----------|-------|
| Exponent | ≤ 0 | ~3.4 ns | Bit extraction only |
| Lookup table | 1–14 | ~5.9 ns | Integer-only, compile-time generated |
| Logarithm | 1–20 | ~10.5 ns | `ln()`-based, works at any scale |

The lookup table accelerates all scales from 1 up to the compiled maximum. Scales beyond the table maximum are rejected by `Scale::new()`.

## Features

### Scale (table size)

Scale features `scale-1` through `scale-16` control the lookup table size.
Each table supports all scales from 1 up to its maximum; higher scales
are rejected by `Scale::new()`. Selected examples:

| Feature | Table Size | Scales Accelerated | Use Case |
|---------|------------|-------------------|----------|
| `scale-4` | 152 B | 1–4 | Minimal memory |
| `scale-6` | 536 B | 1–6 | Embedded systems |
| `scale-8` | 2 KB | 1–8 | **Default** |
| `scale-10` | 8 KB | 1–10 | Recommended |
| `scale-12` | 32 KB | 1–12 | High resolution |
| `scale-14` | 128 KB | 1–14 | Maximum practical coverage |

### Other features

| Feature | Default | Effect |
|---------|---------|--------|
| `std` | ✓ | Enables `std::error::Error` impls for `Overflow` and `ScaleError`. Disable for `#![no_std]` builds. |
| `logarithm` | | Pure `ln()`-based mapper for testing and benchmarking. Requires `std`. |
| `boundary` | | Enables `lower_boundary()` at positive scales. Requires `std`. |
| `quantile` | | Quantile estimation (`QuantileIter`). Requires `boundary`. |
| `bench-internals` | | Exposes internal methods (e.g., `downscale()`) for benchmarking |
| `bench-all` | | Enables `logarithm` + `quantile` + `scale-8` + `bench-internals` for comprehensive testing |

## Documentation

- **[Design & Architecture](docs/design.md)** — Exponential scale theory, index mapping algorithms, lookup table design, crate structure
- **[Implementation Details](docs/internals.md)** — Literal mode, sub-byte counters (SWAR), parameter selection, API overview
- **[Collapse-Left Design](docs/collapse-design.md)** — `Sketch` / `SketchPN`: the relative-error guarantee, the underflow side counter, collapse mechanics, the order-dependence tradeoff, and sizing
- **[Reference](docs/reference.md)** — Error handling, thread safety, `no_std` support, testing, OTel spec compatibility
- **[Historical Notes](docs/history.md)** — Origins of the lookup table algorithm

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
