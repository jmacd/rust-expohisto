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
    pub(super) fn update_literal(&mut self, value: f64) -> Result<(), Overflow> {
        debug_assert!(self.current.width.is_literal());

        let used = self.stats.count as usize % N;
        self.data[used] = value.to_bits();

        // Pool just filled? Promote if any non-zero values exist.
        if used == N - 1 && self.stats.sum != 0.0 {
            return self.promote_pool(N);
        }

        Ok(())
    }

    /// Promotes from literal mode to bucket mode.
    ///
    /// Reads `stats.min` / `stats.max` (already up-to-date) to insert
    /// the extremes first, establishing the full index range and
    /// minimizing intermediate downscale steps. Zeros are skipped
    /// during replay since they have no bucket representation.
    pub(super) fn promote(&mut self) -> Result<(), Overflow> {
        debug_assert!(self.current.width.is_literal());

        let r = self.stats.count as usize % N;
        let entries = if r == 0 && self.stats.count > 0 { N } else { r };
        self.promote_pool(entries)
    }

    fn promote_pool(&mut self, entries: usize) -> Result<(), Overflow> {
        if entries == 0 || self.stats.sum == 0.0 {
            // Empty or all zeros — switch to bucket mode with no buckets.
            self.reset_bucket_state();
            return Ok(());
        }

        // Collect stored values before we clobber the data pool.
        let mut literals = [0u64; N];
        literals[..entries].copy_from_slice(&self.data[..entries]);

        let lo = self.stats.min;
        let hi = self.stats.max;

        // Reset to empty bucket mode at the initial scale.
        self.reset_bucket_state();
        self.current.scale = Scale::new(self.current.scale.scale()).map_err(|_| Overflow)?;

        // Insert extremes first.
        self.update_buckets(lo, 1)?;
        if hi != lo {
            self.update_buckets(hi, 1)?;
        }

        // Replay remaining non-zero values, skipping one lo and one hi.
        let mut skip_lo = true;
        let mut skip_hi = lo != hi;
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
            self.update_buckets(v, 1)?;
        }

        Ok(())
    }
}
