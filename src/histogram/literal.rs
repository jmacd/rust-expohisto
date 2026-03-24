// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Literal mode: stores raw f64 bit patterns (including zeros) in
//! the data pool. The pool index is `count % N`, wrapping at batch
//! boundaries. All-zero batches stay in literal mode; the first
//! batch containing a non-zero value triggers promotion to buckets.

use crate::mapping::Scale;

use super::{Histogram, Overflow};

impl<const N: usize> Histogram<N> {
    /// Stores a value (including zero) in the literal pool.
    ///
    /// When the pool fills and contains non-zero values, promotes to
    /// bucket mode. All-zero batches wrap and stay in literal mode.
    pub(super) fn update_literal(&mut self, value: f64) {
        debug_assert!(self.current.width.is_literal());

        let used = self.stats.count as usize % N;
        self.data[used] = value.to_bits();

        // Pool just filled? Promote if any non-zero values exist.
        if used == N - 1 && self.stats.sum != 0.0 {
            self.promote_to_buckets(N);
        }
    }

    /// Promotes from literal mode to bucket mode.
    ///
    /// Reads `stats.min` / `stats.max` (already up-to-date) to insert
    /// the extremes first, establishing the full index range and
    /// minimizing intermediate downscale steps. Zeros are skipped
    /// during replay since they have no bucket representation.
    pub(super) fn promote(&mut self) {
        debug_assert!(self.current.width.is_literal());

        let r = self.stats.count as usize % N;
        let entries = if r == 0 && self.stats.count > 0 { N } else { r };
        self.promote_to_buckets(entries)
    }

    /// Promote a full data slice of literal values. These include zeros
    /// since the overall count field includes them and until promotion
    /// we have no other way to know how many entries are filled.
    fn promote_to_buckets(&mut self, entries: usize) {
        if entries == 0 || self.stats.sum == 0.0 {
            // Switch to B1.
            self.switch_to_b1();
            return;
        }

        // Collect stored values before we clobber the data pool.
        let mut literals = [0u64; N];
        literals[..entries].copy_from_slice(&self.data[..entries]);

        let lo = self.stats.min;
        let hi = self.stats.max;

        // Reset to empty bucket mode at the initial scale.
        self.switch_to_b1();

        // Replay remaining non-zero values, skipping one lo and one hi.
        let mut skip_lo = true;
        let mut skip_hi = lo != hi;
        // Insert extremes first.
        self.update_buckets(lo, 1).expect("literal safety");
        if skip_hi {
            self.update_buckets(hi, 1).expect("literal safety");
        }

        for &bits in &literals[..entries] {
            let v = f64::from_bits(bits);
            if v == 0.0 {
                continue;
            }
            if skip_lo && v == lo {
                skip_lo = false;
                continue;
            }
            if skip_hi && v == hi {
                skip_hi = false;
                continue;
            }
            self.update_buckets(v, 1).expect("literal safety");
        }
    }
}
