# Architectural Redesign: Byte-Array Buckets with In-Place Widening

## Overview

The `rust-expohisto` crate underwent a major architectural redesign. The previous
multi-type generic design (`Histogram<C: Counter, const SIZE: usize>` with a
runtime `ExpoHistogram` enum wrapper) was replaced with a single unified type
(`Histogram<const N: usize>`) that uses a byte array internally and widens
counters in place.

## What Changed

### 1. `max_scale` field

Added a `max_scale` field to `Histogram` so the scale can be capped at
construction time. On `clear()`, the histogram resets to `max_scale` rather than
the global maximum.

- `Histogram::with_max_scale(scale)` constructor
- `Histogram::max_scale()` getter
- `clear()` resets to `max_scale`

### 2. New const generic: `Histogram<const N: usize>`

**Before:** `Histogram<C: Counter, const SIZE: usize>` where `C` ∈ {u8, u16,
u32, u64} and `SIZE` = bucket count. Required choosing counter width at compile
time.

**After:** `Histogram<const N: usize>` where `N` = number of `u64` words of
bucket storage. Total bucket bytes = N×8. Counter width is determined at runtime,
starting at u8 and widening automatically.

Examples:
- `Histogram<1>` → 8 bytes → 8 u8 buckets initially
- `Histogram<2>` → 16 bytes → 16 u8 buckets initially
- `Histogram<27>` → 216 bytes → 216 u8 buckets initially

### 3. `BucketWidth` enum and in-place widening

Counters start at u8 (maximizing initial bucket count) and widen in place when a
counter saturates:

```
u8 (N×8 buckets) → u16 (N×4 buckets) → u32 (N×2 buckets) → u64 (N buckets)
```

The `widen_in_place()` method performs a combined downscale + counter-widen
operation:

1. Linearizes the circular buffer (rotate so index_start == index_base)
2. Determines the downscale amount (usually 1; auto-bumps to 2 when fully packed
   with odd index_start to avoid exceeding new-width capacity)
3. Group-sums adjacent counters and writes them at the wider type, left-to-right
   (output never overtakes input since each output element occupies the same
   bytes as its input group)
4. Updates index_start/index_end/index_base

The backing storage is `[u64; N]` reinterpreted via `bytemuck::cast_slice` as
`&[u8]`, `&[u16]`, `&[u32]`, or `&[u64]` depending on the current
`BucketWidth`.

### 4. Removed: `ExpoHistogram` enum, `MMSC`, Boxing, `Counter` trait

The entire `aggregator.rs` module was removed:

- **`ExpoHistogram`** enum (7 variants: Empty, Small16/32/64, Large16/32/64) —
  replaced by a single `Histogram<N>` with runtime widening
- **`MMSC`** struct (min/max/sum/count without buckets) — statistics are now
  always part of `Histogram`
- **`Counter` trait** — no longer needed; counter width is runtime via
  `BucketWidth`
- **`Box<Histogram<...>>`** — no boxing anywhere; everything is inline
- **Snapshot-based widening** (clone → widen_into → retry) — replaced by
  in-place widening with no allocation
- **`Resolution`**, **`CounterWidth`** enums from aggregator — removed
- **`SMALL_SIZE`**, **`LARGE_SIZE`** constants — removed

### 5. +Inf and subnormal handling

- Removed `debug_assert!(value.is_finite())` from all mapping functions
  (exponent.rs, newrelic.rs, dynatrace.rs, logarithm.rs, histogram.rs,
  aggregator.rs)
- +Inf maps to the same bucket as `f64::MAX` via the upper-inclusive correction
  (significand == 0 → correction = -1)
- Subnormals are handled naturally by `get_normal_base2()` returning -1023

### 6. OTel spec compatibility documentation

Added a comprehensive compatibility section to README.md covering zero handling,
scale semantics, merge behavior, and known deviations.

## Files Changed

| File | Status | Notes |
|------|--------|-------|
| `src/histogram.rs` | **Rewritten** | 1419 lines. New `Buckets<N>` and `Histogram<N>` with bytemuck-based byte array and in-place widening |
| `src/aggregator.rs` | **Removed** | Was ~530 lines. MMSC, ExpoHistogram enum, all boxing |
| `src/lib.rs` | **Updated** | Removed aggregator module and old exports. Updated doc example and size table for word-count semantics |
| `Cargo.toml` | **Updated** | Added `bytemuck = "1.25.0"` with `derive` feature |
| `src/exponent.rs` | **Minor** | Removed `debug_assert!(value.is_finite())` |
| `src/newrelic.rs` | **Minor** | Removed `debug_assert!(value.is_finite())` |
| `src/dynatrace.rs` | **Minor** | Removed `debug_assert!(value.is_finite())` |
| `src/logarithm.rs` | **Minor** | Removed `debug_assert!(value.is_finite())` |
| `README.md` | **Updated** | OTel spec compatibility section |

## Public API

```rust
// Types
pub struct Histogram<const N: usize>;  // N = u64 word count
pub struct Buckets<const N: usize>;
pub struct BucketsIter<'a, const N: usize>;
pub enum BucketWidth { U8, U16, U32, U64 }

// Histogram methods
Histogram::new() -> Self
Histogram::with_max_scale(scale: i32) -> Self
Histogram::with_scale(scale: i32) -> Self
h.update(value: f64) -> bool
h.update_by_incr(value: f64, incr: u64) -> bool
h.merge_from(other: &Self) -> bool
h.merge_from_raw(...) -> bool
h.clear()
h.swap(other: &mut Self)
h.sum() -> f64
h.count() -> u64
h.zero_count() -> u64
h.min() -> f64
h.max() -> f64
h.scale() -> i32
h.max_scale() -> i32
h.bucket_width() -> BucketWidth
h.positive() -> &Buckets<N>

// Buckets methods
b.at(pos: u32) -> u64
b.len() -> u32
b.is_empty() -> bool
b.offset() -> i32
b.width() -> BucketWidth
b.capacity() -> usize
b.iter() -> BucketsIter<N>
```

## Test Results

38 lib tests + 1 doc-test, all passing. Key test coverage:

- Basic operations (update, zero, multiple values)
- Automatic downscaling
- Merge (same-size, via `merge_from` and `merge_from_raw`)
- Comprehensive merge equivalence across 5 sizes × 44 value sets × all pairs
- Auto-widen: u8→u16, u16→u32, u32→u64
- Bucket count halves on widen
- Clear resets to u8 width and max_scale
- Widen preserves data (count + sum invariants)
- Edge values: +Inf, subnormals
- Exhaustive u8 overflow (all slots full, triggers widen)
