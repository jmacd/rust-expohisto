// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Thin math compatibility layer.
//!
//! Delegates to `std` f64 methods when available, `libm` otherwise.

#[cfg(feature = "std")]
mod imp {
    #[inline]
    pub fn exp(x: f64) -> f64 {
        x.exp()
    }
    #[inline]
    pub fn ln(x: f64) -> f64 {
        x.ln()
    }
    #[inline]
    pub fn floor(x: f64) -> f64 {
        x.floor()
    }
    #[inline]
    pub fn powi(x: f64, n: i32) -> f64 {
        x.powi(n)
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    #[inline]
    pub fn exp(x: f64) -> f64 {
        libm::exp(x)
    }
    #[inline]
    pub fn ln(x: f64) -> f64 {
        libm::log(x)
    }
    #[inline]
    pub fn floor(x: f64) -> f64 {
        libm::floor(x)
    }
    #[inline]
    pub fn powi(x: f64, n: i32) -> f64 {
        // All non-test call sites use base 2.0, but keep the general signature.
        // libm::pow handles the general case; ldexp is exact for base 2.
        if x == 2.0 {
            libm::ldexp(1.0, n)
        } else {
            libm::pow(x, n as f64)
        }
    }
}

pub use imp::*;
