// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! SWAR (SIMD Within A Register) — per-word parallel pairwise summation.
//!
//! [`swar_widen_max`] is the main Phase 1 primitive for downscale: it
//! widens lanes in place by `steps` doublings and returns the maximum
//! lane value across all words. Internally it applies [`swar_step`]
//! per step.
//!
//! Additional primitives ([`swar_has_overflow`], [`narrow_word`],
//! [`swar_narrow_compact`], [`swar_shift_up`]) are available for
//! testing but are not used by the current downscale implementation.

use super::width::Width;

// Widen a single word in palce.
#[inline]
pub(crate) fn widen_into(before: Width, after: Width, word: &mut u64) {
    match (before, after) {
        (Width::B1, Width::U64) => {
            *word = word.count_ones() as u64;
        }
        (Width::B1, Width::U32) => {
            let s0 = (*word & 0x0000_0000_FFFF_FFFF).count_ones() as u64;
            let s1 = (*word & 0xFFFF_FFFF_0000_0000).count_ones() as u64;
            *word = (s0 << 0) | (s1 << 32);
        }
        (Width::B1, Width::U16) => {
            let s0 = (*word & 0x0000_0000_0000_FFFF).count_ones() as u64;
            let s1 = (*word & 0x0000_0000_FFFF_0000).count_ones() as u64;
            let s2 = (*word & 0x0000_FFFF_0000_0000).count_ones() as u64;
            let s3 = (*word & 0xFFFF_0000_0000_0000).count_ones() as u64;
            *word = (s0 << 0) | (s1 << 16) | (s2 << 32) | (s3 << 48);
        } // @@@ TODO
    };
}
