// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

pub(crate) mod exponent;
pub(crate) mod float64;
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
    BucketDescriptor, BucketView, Width, BucketsIter, Histogram, HistogramView, Overflow,
    Settings, Stats,
};
#[cfg(feature = "boundary")]
pub use histogram::{QuantileIter, QuantileValue};
pub use mapping::{max_scale, Scale, ScaleError, MAX_SCALE, MIN_SCALE};
