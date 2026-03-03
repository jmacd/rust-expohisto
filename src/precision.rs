// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Variable-width numeric types for histogram statistics.
//!
//! The [`Precision`] trait bundles a floating-point type ([`HistFloat`]) and an
//! unsigned integer type ([`HistCount`]) used for the histogram's min/max/sum
//! and count fields respectively.
//!
//! Two built-in precision tiers are provided:
//!
//! | Tier   | Float | Count | Overhead* |
//! |--------|-------|-------|----------|
//! | [`P32`] | `f32` | `u32` | 28 bytes |
//! | [`P64`] | `f64` | `u64` | 56 bytes |
//!
//! \* Overhead = 3×Float + 1×Count + 2×i32 (sum/min/max + count + scale fields).

use core::fmt::Debug;
use core::ops::{Add, AddAssign};

/// Floating-point type used for histogram sum, min, and max.
pub trait HistFloat:
    Copy + Clone + Debug + PartialEq + PartialOrd + Add<Output = Self> + AddAssign
{
    /// Converts from `f64`, possibly losing precision.
    fn from_f64(v: f64) -> Self;

    /// Converts to `f64`.
    fn to_f64(self) -> f64;

    /// Returns the zero value.
    fn zero() -> Self;

    /// Returns the minimum of two values.
    ///
    /// Matches IEEE 754 `minimum` semantics: propagates NaN, treats −0 < +0.
    fn min_of(self, other: Self) -> Self;

    /// Returns the maximum of two values.
    fn max_of(self, other: Self) -> Self;
}

/// Unsigned integer type used for histogram count.
pub trait HistCount: Copy + Clone + Debug + PartialEq + Eq + PartialOrd + Ord {
    /// Returns zero.
    fn zero() -> Self;

    /// Checked addition of two values of the same type.
    fn checked_add(self, rhs: Self) -> Option<Self>;

    /// Checked addition of a `u64` value. Returns `None` if the value
    /// does not fit or the addition overflows.
    fn checked_add_u64(self, rhs: u64) -> Option<Self>;

    /// Converts to `u64`.
    fn to_u64(self) -> u64;
}

/// Bundles a [`HistFloat`] and [`HistCount`] into a precision tier.
pub trait Precision {
    /// Floating-point type for sum/min/max.
    type Float: HistFloat;
    /// Unsigned integer type for count.
    type Count: HistCount;
}

// ---------------------------------------------------------------------------
// P64: f64 + u64 (default, full precision)
// ---------------------------------------------------------------------------

/// 64-bit precision tier: `f64` for floats, `u64` for counts.
#[derive(Debug, Clone, Copy)]
pub struct P64;

impl Precision for P64 {
    type Float = f64;
    type Count = u64;
}

impl HistFloat for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn min_of(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline]
    fn max_of(self, other: Self) -> Self {
        self.max(other)
    }
}

impl HistCount for u64 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        self.checked_add(rhs)
    }
    #[inline]
    fn checked_add_u64(self, rhs: u64) -> Option<Self> {
        self.checked_add(rhs)
    }
    #[inline]
    fn to_u64(self) -> u64 {
        self
    }
}

// ---------------------------------------------------------------------------
// P32: f32 + u32
// ---------------------------------------------------------------------------

/// 32-bit precision tier: `f32` for floats, `u32` for counts.
#[derive(Debug, Clone, Copy)]
pub struct P32;

impl Precision for P32 {
    type Float = f32;
    type Count = u32;
}

impl HistFloat for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn min_of(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline]
    fn max_of(self, other: Self) -> Self {
        self.max(other)
    }
}

impl HistCount for u32 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        self.checked_add(rhs)
    }
    #[inline]
    fn checked_add_u64(self, rhs: u64) -> Option<Self> {
        let rhs = u32::try_from(rhs).ok()?;
        self.checked_add(rhs)
    }
    #[inline]
    fn to_u64(self) -> u64 {
        self as u64
    }
}


