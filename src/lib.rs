// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub(crate) mod exponent;
pub(crate) mod float64;
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
