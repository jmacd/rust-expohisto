# Reference

## Error Handling & Atomicity

All mutating operations (`update`, `record_incr`, `merge_from`) return `Result<(), Error>`. The `Error::Overflow` variant indicates that the total count would exceed `u64::MAX`.

**Snapshot/rollback guarantee:** Before any mutating operation, the histogram clones itself. If the operation fails (e.g., a U64 counter would overflow), the clone is restored and the histogram is left unchanged. This ensures that partial mutations from multi-step operations (downscale + widen + insert) never leak to the caller.

```rust,ignore
let snapshot_count = h.view().stats().count;
if h.update(value).is_err() {
    // h is unchanged — count, sum, buckets all identical to before
    assert_eq!(h.view().stats().count, snapshot_count);
}
```

## Thread Safety

`Histogram<N>` is `Send + Sync` — all fields are `Copy` primitives with no interior mutability. However, mutations require `&mut self`, so concurrent access needs external synchronization:

```rust,ignore
use std::sync::{Arc, Mutex};
use otel_expohisto::Histogram;

// Thread-safe shared histogram
let hist = Arc::new(Mutex::new(Histogram::<16>::new()));
```

For hot-path recording, consider a thread-local histogram per worker with periodic `merge_from` into a shared aggregate.

## Safety

The crate contains **zero `unsafe` code**. All bit-level manipulation (sub-byte get/set, SWAR pairwise merge, shift-and-mask widening) is implemented entirely in safe Rust. The `data: [u64; N]` pool is reinterpreted at different counter widths using arithmetic indexing, not pointer casts.

## `no_std` Support

The crate is `#![no_std]` compatible. The `std` feature (enabled by default) only gates `std::error::Error` implementations for `Overflow` and `ScaleError`. All core functionality — recording, merging, downscaling — works without `std`. Quantile estimation requires the `quantile` feature (which depends on `boundary`). Math operations (`ln`, `exp`, `floor`, `powi`) delegate to `libm` when `std` is disabled.

To use in a `no_std` environment, disable default features and re-enable the ones you need:

```toml
[dependencies]
otel-expohisto = { version = "0.1", default-features = false, features = ["scale-8"] }
```

The CI test matrix includes `--no-default-features` to ensure `no_std` compatibility is continuously validated.

## Testing & Quality

The crate includes comprehensive validation at multiple levels:

- **140 unit tests** covering basic operations, widening, downscaling, SWAR pairwise merge, all merge strategies, quantile estimation, and boundary conditions
- **4 fuzz targets** (`cargo +nightly fuzz run <target>`):
  - `histogram_oracle` — validates invariants (count, sum, min/max, bucket integrity) with random f64 sequences
  - `merge_oracle` — fuzzes weighted `record()` + cross-scale merge
  - `stateful_oracle` — state-machine fuzzer with interleaved update/merge/clear/read operations
  - `rng_stress` — large histogram (N=160) with millions of random values
- **Exhaustive boundary validation** — upper-inclusive semantics verified over all ~3 billion f64 values in the first sub-bucket at scale 16
- **CI matrix** — tests across 4 feature combinations (`bench-all`, `scale-8`, `--no-default-features --features scale-8`, `--no-default-features`), plus clippy, rustfmt, doc, MSRV (1.73), and example checks

## Examples

Four examples are included in the `examples/` directory:

| Example | Run command | Description |
|---------|------------|-------------|
| `basic` | `cargo run --example basic` | Record latencies, view stats and iterate buckets |
| `merge` | `cargo run --example merge` | Same-size and cross-size histogram merging |
| `sizing` | `cargo run --example sizing` | Interactive capacity explorer for choosing `N` and `min_width` |
| `quick_start_test` | `cargo run --example quick_start_test` | Minimal 3-value example |

## References

- [OpenTelemetry Exponential Histogram Specification](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram)
- [Upper-inclusive boundary discussion](https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261): The specification change (for Prometheus compatibility) that motivated the boundary condition re-engineering in this implementation
- [Golang OpenTelemetry Exponential Histogram](https://github.com/lightstep/go-expohisto): Golang reference implementation by the same author
- [Historical Notes](history.md): Origins of the lookup table algorithm, with links to the original Dynatrace and NewRelic implementations

## OTel SDK Specification Compatibility

This section documents compatibility with the [Base2 Exponential Bucket Histogram Aggregation](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/metrics/sdk.md#base2-exponential-bucket-histogram-aggregation) section of the OpenTelemetry Metrics SDK specification.

### Configuration Parameters

The spec defines three configuration parameters:

| Parameter | Spec Default | This Implementation | Notes |
|-----------|-------------|---------------------|-------|
| **MaxSize** | 160 | Any compile-time `N` via `Histogram<N>` | `N` is the data pool size in u64 words. Bucket capacity depends on the current counter width. |
| **MaxScale** | 20 | 16 (`MAX_SCALE`) | Scales 1–16 use the compile-time lookup table (selected via `scale-N` feature). The default feature is `scale-8`. `Histogram::new().with_scale()` lets the user set a lower starting scale. |
| **RecordMinMax** | true | Always on | `min` and `max` are tracked on every update. There is no option to disable them. |

### Collected Fields

The spec requires all histogram aggregations to collect count, sum, min, and max. This implementation provides:

| Field | Type | Notes |
|-------|------|-------|
| `count` | `u64` | Total measurement count |
| `sum` | `f64` | Arithmetic sum of all values (zero values excluded from sum) |
| `min` | `f64` | Minimum observed value |
| `max` | `f64` | Maximum observed value |
| `zero_count` | `u64` (derived) | Compute as `count - sum(positive buckets)` while encoding/exporting |
| `positive` | `BucketView` | Positive range bucket counts |
| `scale` | `i32` | Current mapping scale (adjusted automatically) |

### Handle All Normal Values

> Implementations are REQUIRED to accept the entire normal range of IEEE floating point values.

**Supported.** All normal positive f64 values (from $2^{-1022}$ through the largest finite f64) are mapped to the correct bucket index. Subnormal values are mapped to the lowest normal bucket rather than rejected.

> Implementations SHOULD NOT incorporate non-normal values (i.e., +Inf, -Inf, and NaNs) into the sum, min, and max fields.

**Supported.** `record_incr` rejects non-finite and negative values at runtime, returning `Error::Extreme`. Zero values are accepted (counted but not bucketed).

### Support a Minimum and Maximum Scale

> The implementation MUST maintain reasonable minimum and maximum scale parameters that the automatic scale parameter will not exceed.

**Supported.** Scale is bounded by `MIN_SCALE` (-10) and `MAX_SCALE` (16). The starting scale (configurable via `Histogram::new().with_scale()`) sets the upper bound for automatic scale selection.

### Use the Maximum Scale for Single Measurements

> When the histogram contains not more than one value in either of the positive or negative ranges, the implementation SHOULD use the maximum scale.

**Supported.** A new histogram starts at `max_scale`. The first observation is recorded at that scale. Scale only decreases when a second value doesn't fit within the current bucket capacity.

### Maintain the Ideal Scale

> Implementations SHOULD adjust the histogram scale as necessary to maintain the best resolution possible, within the constraint of maximum size.

**Supported.** When a new value's bucket index would exceed the current bucket capacity, the histogram computes the minimum downscale needed to accommodate both the existing range and the new value. It never downscales more than necessary. Creating a fresh histogram restores the configured starting scale.

### Negative Values

The spec defines both positive and negative bucket ranges. `HistogramNN<N>` (aliased as `Histogram<N>`) supports only non-negative values — there is a single positive bucket set. This is suitable for the common case of non-negative measurements (latencies, sizes, counts).

For values of any sign, use `HistogramPN<K, L>` which maintains independent positive (`K` words) and negative (`L` words) bucket ranges. Both sub-histograms are automatically synchronized to the same scale after each update or merge.

### Merging

The spec requires aggregations to be mergeable. This implementation supports:

- **Same- or cross-size merge:** `Histogram::merge_from()` merges histograms, computing the minimum common scale and downscaling as needed.  The source and destination may have different `N` parameters (e.g., `Histogram<16>` into `Histogram<8>`).
- **Cross-size merge:** The source and destination may have different `N` parameters (e.g., `Histogram<16>` into `Histogram<8>`).

### Counter Widening

Not part of the spec, but relevant to overflow handling: bucket counters start at 1-bit (B1) and automatically widen through the chain B1→B2→B4→U8→U16→U32→U64 when a counter would overflow during `update` or `merge`. Each widening step halves the bucket count and doubles counter capacity. This allows the common case to use extremely compact 1-bit counters (one bucket per bit) while still handling extreme counts.

### Summary

| Spec Requirement | Status |
|-----------------|--------|
| MaxSize = 160 default | Supported |
| MaxScale = 20 default | Supported (up to 16) |
| RecordMinMax | Always on |
| Handle all normal values | Supported |
| Reject +Inf, -Inf, NaN | Supported (runtime `Error::Extreme`) |
| Subnormal values | Mapped to lowest normal bucket |
| Minimum and maximum scale | Supported (MIN_SCALE = -10, MAX_SCALE = 16) |
| Max scale for single measurements | Supported |
| Maintain ideal scale | Supported |
| Positive bucket range | Supported |
| Negative bucket range | Not implemented |
| Zero count | Derived from bucket iteration |
| Count, sum, min, max | Supported |
| Merge | Supported (same-type, cross-size, raw) |
