# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added

- `Histogram<N, P>` — allocation-free exponential histogram with const-generic
  pool size `N` and precision tier `P` (`P32` or `P64`)
- **Literal mode cold start** — new histograms store raw f64 values until the
  pool fills, then promote to bucket mode at the optimal scale in one shot
- **Sub-byte bucket widths** — counters start at 1-bit and widen through
  B1→B2→B4→U8→U16→U32→U64 via SWAR (SIMD Within A Register) operations
- **Three mapping algorithms** — `newrelic` (default), `dynatrace`, and
  `logarithm`, selected at compile time via Cargo features
- **Configurable lookup table scale** — `scale-4` through `scale-14` features
  trade binary size for finer resolution support
- `Histogram::merge_from()` — same-size in-place merge with atomicity
- `Histogram::merge_from_other()` — cross-size merge (different `N` values)
- `Histogram::merge_from_raw()` — merge from raw bucket data
- `Histogram::with_max_scale()` — cap the histogram's maximum scale
- `Histogram::with_min_bucket_width()` — skip sub-byte widths for faster ops
- `Histogram::with_literal_mode()` — disable literal mode when value range
  is known upfront
- `BucketView` — borrow-based read access to bucket data with iteration
- OTel SDK specification compatibility (count, sum, min, max, positive buckets)
- +Inf and subnormal value handling
- 159 unit tests and 3 fuzz targets
- Comprehensive README with algorithm documentation, SWAR explanation,
  memory layout diagrams, and OTel spec compatibility matrix
