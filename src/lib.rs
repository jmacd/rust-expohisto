// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

pub(crate) mod exponent;
pub mod histogram;
pub mod mapping;

#[cfg(feature = "boundary")]
mod boundary;

// Algorithm modules - conditionally compiled.
// These are public for benchmark access but hidden from docs since users
// should go through `Scale` rather than calling algorithms directly.
#[cfg(feature = "logarithm")]
#[doc(hidden)]
pub mod logarithm;

#[doc(hidden)]
pub mod lookup;

pub use histogram::{
    BucketDescriptor, BucketView, BucketsIter, Error, Histogram, HistogramView, Settings, Stats,
    Width,
};
#[cfg(feature = "quantile")]
pub use histogram::{QuantileIter, QuantileValue};
pub use mapping::{MAX_SCALE, MIN_SCALE, Scale, ScaleError, table_scale};

// Note: public because mapping-gen uses it from another crate
pub mod float64;
