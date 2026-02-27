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
//! - **Auto-widening counters**: Buckets start at 1-bit and widen in-place
//!   (1→2→4 bits → u8 → u16 → u32 → u64) via combined downscale+widen when a
//!   counter saturates. Sub-byte transitions use parallel bit-sum (SWAR).
//! - **Automatic scaling**: Scale adjusts automatically to fit data in available buckets
//! - **Merge support**: Histograms can be merged in-place without allocation
//! - **Algorithm choice**: Select mapping algorithm at compile time
//!
//! # Example
//!
//! ```
//! use rust_expohisto::{Histogram, P64};
//!
//! // Create a histogram with full 64-bit precision and 2 u64 words (16 bytes) of bucket storage.
//! // Starts with 128 1-bit buckets, widening through 2-bit → 4-bit → u8 → u16 → u32 → u64.
//! let mut hist: Histogram<P64, 2> = Histogram::new();
//!
//! // Record observations
//! hist.update(0.5).unwrap();
//! hist.update(1.0).unwrap();
//! hist.update(2.0).unwrap();
//! hist.update(100.0).unwrap();
//!
//! // Access statistics
//! println!("count: {}", hist.count());
//! println!("sum: {:?}", hist.sum());
//! println!("min: {:?}", hist.min());
//! println!("max: {:?}", hist.max());
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
//! The first type parameter `P` selects the precision tier for statistics
//! (sum/min/max as float, count/zero_count as unsigned int):
//!
//! | Tier | Float | Count | Stats overhead |
//! |------|-------|-------|----------------|
//! | `P64` | `f64` | `u64` | 40 bytes |
//! | `P32` | `f32` | `u32` | 20 bytes |
//! | `P16`* | `f16` | `u16` | 10 bytes |
//!
//! \* `P16` requires the `half` feature.
//!
//! `N` is the number of `u64` words of bucket storage (total bytes = N×8).
//!
//! | Buckets (`N`) | Bytes | Bucket Capacity (1b → 2b → 4b → u8 → u16 → u32 → u64) |
//! |---------------|-------|---------------------------------------------------------|
//! | `Histogram<P, 1>` | 8 | 64 → 32 → 16 → 8 → 4 → 2 → 1 |
//! | `Histogram<P, 2>` | 16 | 128 → 64 → 32 → 16 → 8 → 4 → 2 |
//! | `Histogram<P, 4>` | 32 | 256 → 128 → 64 → 32 → 16 → 8 → 4 |
//! | `Histogram<P, 8>` | 64 | 512 → 256 → 128 → 64 → 32 → 16 → 8 |
//! | `Histogram<P, 16>` | 128 | 1024 → 512 → 256 → 128 → 64 → 32 → 16 |
//! | `Histogram<P, 27>` | 216 | 1728 → 864 → 432 → 216 → 108 → 54 → 27 |
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
pub mod precision;

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

pub use histogram::{BucketWidth, Buckets, BucketsIter, Histogram, Overflow};
pub use mapping::{Mapping, MappingError, MAX_SCALE, MIN_SCALE, max_scale};
pub use precision::{HistCount, HistFloat, P32, P64, Precision};
