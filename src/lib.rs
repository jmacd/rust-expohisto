// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Allocation-free OpenTelemetry Exponential Histogram implementation in Rust.
//!
//! This crate provides a fixed-size exponential histogram that uses const generics
//! to avoid any heap allocation. The histogram automatically adjusts its scale
//! to accommodate the range of input data.
//!
//! # Features
//!
//! - **Allocation-free**: Uses fixed-size arrays via const generics
//! - **Configurable counter type**: Choose between `u8`, `u16`, `u32`, or `u64`
//! - **Automatic scaling**: Scale adjusts automatically to fit data in available buckets
//! - **Merge support**: Histograms can be merged in-place without allocation
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

pub mod float64;
pub mod histogram;
#[cfg(any(feature = "lookup-4", feature = "lookup-6", feature = "lookup-8", feature = "lookup-10", feature = "lookup-12", feature = "lookup-14"))]
pub mod lookup;
pub mod mapping;

pub use histogram::{Buckets, BucketsIter, Counter, Histogram, Histogram16, Histogram32, Histogram64};
pub use mapping::{Mapping, MappingError, MAX_SCALE, MIN_SCALE, map_to_index_lg};
