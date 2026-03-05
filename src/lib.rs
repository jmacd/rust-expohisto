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
//! use rust_expohisto::{Histogram, P32};
//!
//! // Create a histogram with 16 u64 words (128 bytes) of data pool.
//! // P32 uses 2 words for MMSC fields (f32/u32), leaving 14 words
//! // for bucket data: 896 1-bit buckets, widening to 14 u64 counters.
//! let mut hist: Histogram<16, P32> = Histogram::new();
//!
//! // Record observations
//! hist.update(0.5).unwrap();
//! hist.update(1.0).unwrap();
//! hist.update(2.0).unwrap();
//! hist.update(100.0).unwrap();
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
//! `Histogram<N, P>` uses const generics for pool size (`N`) and the
//! precision tier `P` controls MMSC field width.
//! [`P32`] uses 2 words (f32/u32), [`P64`] uses 4 words (f64/u64).
//!
//! | `Histogram<N, P32>` | Pool bytes | Bucket words | B1 capacity |
//! |---------------------|-----------|--------------|-------------|
//! | `Histogram<8, P32>` | 64 | 6 | 384 |
//! | `Histogram<12, P32>` | 96 | 10 | 640 |
//! | `Histogram<16, P32>` | 128 | 14 | 896 |
//! | `Histogram<20, P32>` | 160 | 18 | 1152 |
//! | `Histogram<32, P32>` | 256 | 30 | 1920 |
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

pub use histogram::{BucketView, BucketWidth, BucketsIter, Histogram, Overflow};
pub use mapping::{Mapping, MappingError, MAX_SCALE, MIN_SCALE, max_scale};
pub use precision::{HistCount, HistFloat, P32, P64, Precision};
