// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Lookup table generation for exponential histogram mapping.
//!
//! This crate provides utilities for generating exact lookup tables
//! used to map f64 values to histogram bucket indices efficiently.

mod float64;
mod newrelic_table;

pub use float64::*;
pub use newrelic_table::*;
