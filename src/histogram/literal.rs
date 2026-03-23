// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Literal mode: stores raw f64 bit patterns instead of bucket counters
//! until the data pool overflows, then promotes to bucket mode.

use crate::mapping::Mapping;

use super::{Histogram, Overflow};

impl<const N: usize> Histogram<N> {
    /// Stores a value in literal mode, promoting to bucket mode on overflow.
    pub(super) fn update_literal(&mut self, value: f64, incr: u64) -> Result<(), Overflow> {
        debug_assert!(self.current.width.is_literal());

        let count = self.literal_count();
        let remaining = self.literal_capacity() - count;

        if incr <= remaining as u64 {
            let needed = incr as usize;
            self.data[count..count + needed].fill(value.to_bits());
            self.index_end += needed as i32;
            Ok(())
        } else {
            self.promote_with(value, incr)
        }
    }

    /// Promotes from literal mode to bucket mode, optionally including
    /// a trigger value that caused overflow of literal capacity.
    fn promote_to_buckets(&mut self, trigger: Option<(f64, u64)>) -> Result<(), Overflow> {
        debug_assert!(self.current.width.is_literal());

        let count = self.literal_count();

        if count == 0 && trigger.is_none() {
            self.reset_bucket_state();
            return Ok(());
        }

        // Collect stored literal values before we clobber the data pool.
        let mut literals = [0u64; N];
        literals[..count].copy_from_slice(self.literal_values());

        // Reset to empty bucket mode at the initial scale and replay values
        // through the normal update path, which handles widening and
        // downscaling incrementally.
        self.reset_bucket_state();
        self.current.mapping = Mapping::new(self.current.mapping.scale()).map_err(|_| Overflow)?;

        for &bits in &literals[..count] {
            let v = f64::from_bits(bits);
            self.update_buckets(v, 1)?;
        }

        if let Some((trigger_val, trigger_incr)) = trigger {
            self.update_buckets(trigger_val, trigger_incr)?;
        }

        Ok(())
    }

    /// Promotes from literal mode to bucket mode, including a trigger value
    /// that caused overflow of literal capacity.
    #[inline]
    pub(super) fn promote_with(&mut self, trigger: f64, trigger_incr: u64) -> Result<(), Overflow> {
        self.promote_to_buckets(Some((trigger, trigger_incr)))
    }

    /// Promotes from literal mode to bucket mode without a trigger value.
    /// Used when self is a merge destination and needs to accept bucket data.
    #[inline]
    pub(super) fn promote(&mut self) -> Result<(), Overflow> {
        self.promote_to_buckets(None)
    }
}
