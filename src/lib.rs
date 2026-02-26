// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free OpenTelemetry Exponential Histogram implementation in Rust.
//!
//! This crate provides a fixed-size exponential histogram that uses const generics
//! to avoid any heap allocation. The histogram automatically adjusts its scale
//! to accommodate the range of input data.
//!
//! # Mapping Algorithms
//!
//! This crate supports multiple mapping algorithms, selected at compile time:
//!
//! - **`logarithm`**: Pure logarithm-based mapping using `floor(ln(value) * scaleFactor)`.
//!   Works for all scales, small binary size, but has floating-point precision errors near boundaries.
//!
//! - **`newrelic`**: NewRelic lookup table-based mapping. Exact (no FP errors),
//!   uses integer-only computation with 2N linear buckets and 1 correction.
//!
//! - **`dynatrace`**: Dynatrace lookup table-based mapping. Exact (no FP errors),
//!   ~50% smaller index table, uses N linear buckets and 2 corrections.
//!
//! Pair an algorithm feature with a scale feature (`scale-4` through `scale-14`)
//! to set the lookup table size.
//!
//! # Features
//!
//! - **Allocation-free**: Uses fixed-size arrays via const generics
//! - **Auto-widening counters**: Buckets start at u8 and widen in-place
//!   (u8 → u16 → u32 → u64) via combined downscale+widen when a counter saturates
//! - **Automatic scaling**: Scale adjusts automatically to fit data in available buckets
//! - **Merge support**: Histograms can be merged in-place without allocation
//! - **Algorithm choice**: Select mapping algorithm at compile time
//!
//! # Example
//!
//! ```
//! use rust_expohisto::Histogram;
//!
//! // Create a histogram with 2 u64 words (16 bytes) of bucket storage.
//! // Starts with 16 u8 buckets, widens to 8×u16 → 4×u32 → 2×u64.
//! let mut hist: Histogram<2> = Histogram::new();
//!
//! // Record observations
//! hist.update(0.5);
//! hist.update(1.0);
//! hist.update(2.0);
//! hist.update(100.0);
//!
//! // Access statistics
//! println!("count: {}", hist.count());
//! println!("sum: {}", hist.sum());
//! println!("min: {}", hist.min());
//! println!("max: {}", hist.max());
//! println!("scale: {}", hist.scale());
//!
//! // Access bucket data
//! let buckets = hist.positive();
//! println!("offset: {}", buckets.offset());
//! println!("bucket count: {}", buckets.len());
//! for i in 0..buckets.len() {
//!     println!("  bucket[{}]: {}", i, buckets.at(i));
//! }
//! ```
//!
//! # Size Considerations
//!
//! `N` is the number of `u64` words of bucket storage (total bytes = N×8).
//! Total struct size ≈ 80 bytes overhead + N×8.
//!
//! | Configuration | Bytes | Bucket Capacity (u8 → u16 → u32 → u64) |
//! |---------------|-------|------------------------------------------|
//! | `Histogram<1>` | 8 | 8 → 4 → 2 → 1 |
//! | `Histogram<2>` | 16 | 16 → 8 → 4 → 2 |
//! | `Histogram<4>` | 32 | 32 → 16 → 8 → 4 |
//! | `Histogram<8>` | 64 | 64 → 32 → 16 → 8 |
//! | `Histogram<16>` | 128 | 128 → 64 → 32 → 16 |
//! | `Histogram<27>` | 216 | 216 → 108 → 54 → 27 |
//!
//! # Scale and Resolution
//!
//! The histogram starts at the maximum supported scale (finest resolution) and
//! automatically downscales when the range of observed values exceeds the bucket
//! capacity. When a counter saturates, the histogram performs an in-place
//! widen+downscale: bucket count halves, counter width doubles, and scale
//! decreases by 1.

pub mod exponent;
pub mod float64;
pub mod histogram;
pub mod mapping;

// Algorithm modules - conditionally compiled
#[cfg(feature = "logarithm")]
pub mod logarithm;

#[cfg(any(
    feature = "scale-4",
    feature = "scale-6",
    feature = "scale-8",
    feature = "scale-10",
    feature = "scale-12",
    feature = "scale-14"
))]
pub mod lookup;

#[cfg(feature = "newrelic")]
pub mod newrelic;

#[cfg(feature = "dynatrace")]
pub mod dynatrace;

pub use histogram::{BucketWidth, Buckets, BucketsIter, Histogram};
pub use mapping::{Mapping, MappingError, MAX_SCALE, MIN_SCALE, max_scale};
