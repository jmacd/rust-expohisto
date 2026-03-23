// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Literal mode: stores raw f64 bit patterns instead of bucket counters
//! until the data pool overflows, then promotes to bucket mode.

use crate::mapping::Scale;

use super::{Histogram, Overflow};

impl<const N: usize> Histogram<N> {
    /// Stores a single value in literal mode, promoting to bucket mode
    /// when the data pool is full.
    pub(super) fn update_literal(&mut self, value: f64) -> Result<(), Overflow> {
        debug_assert!(self.current.width.is_literal());

        let count = self.literal_count();

        if count < self.literal_capacity() {
            self.data[count] = value.to_bits();
            self.index_end += 1;
            Ok(())
        } else {
            self.promote_with(value)
        }
    }

    /// Promotes from literal mode to bucket mode.
    ///
    /// Inserts min and max first to establish the full index range,
    /// minimizing intermediate downscale steps. Then replays remaining
    /// values (skipping the already-inserted min and max).
    fn promote_to_buckets(&mut self, trigger: Option<f64>) -> Result<(), Overflow> {
        debug_assert!(self.current.width.is_literal());

        let count = self.literal_count();

        if count == 0 && trigger.is_none() {
            self.reset_bucket_state();
            return Ok(());
        }

        // Collect stored literal values before we clobber the data pool.
        let mut literals = [0u64; N];
        literals[..count].copy_from_slice(self.literal_values());

        // Determine lo/hi across pool + trigger.
        let mut lo = self.stats.min;
        let mut hi = self.stats.max;
        if let Some(tv) = trigger {
            lo = lo.min(tv);
            hi = hi.max(tv);
        }

        // Reset to empty bucket mode at the initial scale.
        self.reset_bucket_state();
        self.current.scale = Scale::new(self.current.scale.scale()).map_err(|_| Overflow)?;

        // Insert lo and hi first.
        self.update_buckets(lo, 1)?;
        if hi != lo {
            self.update_buckets(hi, 1)?;
        }

        // Replay remaining values, skipping one lo and one hi.
        let mut skip_lo = true;
        let mut skip_hi = lo != hi;
        for &bits in &literals[..count] {
            let v = f64::from_bits(bits);
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
        if let Some(tv) = trigger {
            if !(skip_lo && tv == lo || skip_hi && tv == hi) {
                self.update_buckets(tv, 1)?;
            }
        }

        Ok(())
    }

    /// Promotes from literal mode to bucket mode, including a trigger
    /// value that caused overflow of literal capacity.
    #[inline]
    pub(super) fn promote_with(&mut self, trigger: f64) -> Result<(), Overflow> {
        self.promote_to_buckets(Some(trigger))
    }

    /// Promotes from literal mode to bucket mode without a trigger value.
    /// Used when self is a merge destination and needs to accept bucket data.
    #[inline]
    pub(super) fn promote(&mut self) -> Result<(), Overflow> {
        self.promote_to_buckets(None)
    }
}
