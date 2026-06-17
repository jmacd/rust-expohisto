# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added

- `Histogram<N>` — allocation-free exponential histogram with const-generic
  pool size `N` (in u64 words)
- **Literal mode cold start** — new histograms store raw f64 values until the
  pool fills, then promote to bucket mode at the optimal scale in one shot
- **Sub-byte bucket widths** — counters start at 1-bit and widen through
  B1→B2→B4→U8→U16→U32→U64 via SWAR (SIMD Within A Register) operations
- **Lookup table mapping** — compile-time generated index tables for O(1)
  integer-only bucket mapping at positive scales; a built-in `logarithm`
  mapper is always available for scales above the table maximum (or as the
  sole mapper when no lookup table is enabled)
- **Configurable lookup table scale** — `scale-1` through `scale-16` features
  trade binary size for finer resolution support
- `Histogram::merge_from()` — merge from any histogram (same or different `N`)
- `HistogramPN<K, L>` — positive+negative histogram with independent pool sizes
  and automatic scale synchronization

- `Sketch<N>` — fixed-scale, collapse-left histogram with a DDSketch-style
  worst-case relative-error guarantee; collapses the small-value side into an
  underflow side counter instead of downscaling, keeping full resolution on the
  top of the distribution
- `SketchPN<K, L>` — signed counterpart of `Sketch` (positive + negative ranges
  plus a zero count)
- `Sketch::with_relative_error()` — pick the coarsest scale meeting a target
  relative error; `with_scale()` / `with_min_width()` for explicit configuration
- `Sketch::merge_from()` — same-scale merge (any pool sizes); commutative
- `SketchView` / `SketchBucketView` — OTel-export-shaped read access, with the
  underflow folded into the lowest bucket and `underflow_count()` reported
  separately
- `Sketch::quantile()` (with `quantile` feature) — relative-error-optimal
  representative `2LU/(L+U)`, achieving `α` rather than `≈2α`
- `Error::ScaleMismatch` — returned when merging `Sketch`es of different scales
- SWAR `spread` primitive (inverse of `narrow`) — pure counter-width change at a
  fixed scale, shared by `Sketch` collapse and merge

- `Histogram::with_min_width()` — skip sub-byte widths for faster ops
- `BucketView` — borrow-based read access to bucket data with iteration
- OTel SDK specification compatibility (count, sum, min, max, positive buckets)
- +Inf and subnormal value handling
- 200+ unit tests and 6 fuzz targets
- Comprehensive documentation: design theory, implementation internals,
  collapse-left design, OTel spec compatibility matrix (see `docs/`), and
  README overview
