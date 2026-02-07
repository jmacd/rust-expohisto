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
//! - **`logarithm`** (default): Pure logarithm-based mapping using `floor(ln(value) * scaleFactor)`.
//!   Works for all scales, small binary size, but has floating-point precision errors near boundaries.
//!
//! - **`newrelic-*`**: Lookup table-based mapping from NewRelic. Exact (no FP errors),
//!   uses integer-only computation. Choose table size based on your max scale needs:
//!   `newrelic-4`, `newrelic-6`, `newrelic-8`, `newrelic-10`, `newrelic-12`, `newrelic-14`.
//!
//! # Features
//!
//! - **Allocation-free**: Uses fixed-size arrays via const generics
//! - **Configurable counter type**: Choose between `u8`, `u16`, `u32`, or `u64`
//! - **Automatic scaling**: Scale adjusts automatically to fit data in available buckets
//! - **Merge support**: Histograms can be merged in-place without allocation
//! - **Algorithm choice**: Select mapping algorithm at compile time
//!
//! # Example
//!
//! ```
//! use rust_expohisto::Histogram;
//!
//! // Create a compact histogram: 16 buckets with u16 counters
//! // Total size: ~108 bytes
//! let mut hist: Histogram<u16, 16> = Histogram::new();
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
//! | Configuration | Approximate Size |
//! |---------------|------------------|
//! | `Histogram<u8, 8>` | ~72 bytes |
//! | `Histogram<u16, 16>` | ~108 bytes |
//! | `Histogram<u32, 32>` | ~200 bytes |
//! | `Histogram<u64, 64>` | ~580 bytes |
//!
//! # Choosing Parameters
//!
//! - **Counter type (`C`)**: Determines max count per bucket
//!   - `u8`: max 255 per bucket
//!   - `u16`: max 65,535 per bucket
//!   - `u32`: max ~4 billion per bucket
//!
//! - **Size (`SIZE`)**: Number of buckets, affects resolution
//!   - Smaller size = more downscaling = coarser resolution
//!   - Larger size = finer resolution = more memory
//!   - Should be a power of 2 for efficiency
//!
//! # Scale and Resolution
//!
//! The histogram starts at scale 20 (finest resolution) and automatically
//! downscales when the range of observed values exceeds the bucket capacity.
//! At scale 20, each bucket represents a ~0.0001% change in value.
//! At scale 0, each bucket represents a factor of 2 change.

pub mod exponent;
pub mod float64;
pub mod histogram;
pub mod mapping;

// Algorithm modules - conditionally compiled
#[cfg(feature = "logarithm")]
pub mod logarithm;

#[cfg(any(
    feature = "newrelic-4",
    feature = "newrelic-6",
    feature = "newrelic-8",
    feature = "newrelic-10",
    feature = "newrelic-12",
    feature = "newrelic-14"
))]
pub mod newrelic;

#[cfg(any(
    feature = "dynatrace-4",
    feature = "dynatrace-6",
    feature = "dynatrace-8",
    feature = "dynatrace-10",
    feature = "dynatrace-12",
    feature = "dynatrace-14"
))]
pub mod dynatrace;

pub use histogram::{Buckets, BucketsIter, Counter, Histogram, Histogram16, Histogram32, Histogram64};
pub use mapping::{Mapping, MappingError, MAX_SCALE, MIN_SCALE, max_scale};
