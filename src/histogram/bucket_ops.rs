// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket operations — widen and downscale.
//!
//! Both widen and downscale use SWAR pairwise-merge steps at sub-U64
//! widths, then scatter-write at U64.  Each step is self-contained:
//! any value displaced by an odd-base alignment shift is fixed up
//! immediately rather than being deferred to a later phase.

use super::bucket_width::BucketWidth;
use super::swar::{swar_has_overflow, swar_narrow_compact, swar_shift_up_one, swar_step};
use super::Histogram;

impl<const N: usize> Histogram<N> {
    /// Performs one SWAR pairwise-merge step (1-step downscale).
    ///
    /// When `index_base` is odd, the data must be shifted up by one
    /// physical slot before the SWAR step so that logical pairs align
    /// with physical pairs.  If the top slot is occupied, its value is
    /// saved, zeroed (freeing room for the shift), and re-inserted
    /// immediately after the step completes.
    ///
    /// When `force_widen` is true, always accepts the wider format
    /// (used on counter overflow).  Otherwise narrows back to the
    /// original width when possible (preserving bucket capacity).
    ///
    /// Returns `None` if already at U64.  Returns `Some(displaced)` on
    /// success, where `displaced` is an optional `(index, count)` that
    /// could not be placed after a widen (the caller must re-insert it).
    pub(super) fn pairwise_merge(
        &mut self,
        force_widen: bool,
    ) -> Option<Option<(i32, u64)>> {
        let width = self.bucket_width;
        if width == BucketWidth::U64 {
            return None;
        }

        debug_assert!(
            self.index_start >= self.index_base,
            "sub-U64 data must not wrap: start={} base={}",
            self.index_start,
            self.index_base,
        );

        let odd_base = self.index_base & 1 != 0;

        // Save the top slot if it would be clobbered by the shift.
        let saved = if odd_base {
            let cap = self.bucket_capacity() as i32;
            let top_physical = (cap - 1) as usize;

            // If the live range spans the full physical capacity, the
            // top slot holds a real value that must be preserved.
            let saved = if self.index_end == self.index_base + cap - 1 {
                let top_idx = self.index_end;
                let val = self.bucket_get(top_physical);
                self.index_end -= 1;
                Some((top_idx, val))
            } else {
                None
            };

            // The top slot must be empty for shift_up_one.  It may hold
            // the saved live value or stale data from a previous trim.
            self.bucket_set(top_physical, 0);
            swar_shift_up_one(self.bucket_data_mut(), width);
            saved
        } else {
            None
        };

        swar_step(self.bucket_data_mut(), width);

        let widened = force_widen || swar_has_overflow(self.bucket_data(), width);
        if widened {
            self.bucket_width = width.wider().unwrap();
        } else {
            swar_narrow_compact(self.bucket_data_mut(), width);
        }

        self.shift_indices(1);

        // Fix up the saved value.  After the SWAR step, the output
        // occupies the first ~half of the physical capacity; the saved
        // value's target is always in the empty second half (no-widen)
        // or wraps to slot 0 (widen — collision with the first pair sum).
        if let Some((idx, val)) = saved {
            let new_idx = idx >> 1;
            if !widened {
                // No widen → capacity unchanged → target slot is in the
                // zeroed second half of the buffer.
                if new_idx > self.index_end {
                    self.index_end = new_idx;
                }
                let slot = self.slot_for(new_idx);
                debug_assert_eq!(self.bucket_get(slot), 0);
                self.bucket_set(slot, val);
                Some(None)
            } else {
                // Widen → capacity halved → target wraps to slot 0, which
                // already holds a different bucket's data.  Return the
                // value so the caller can re-insert it (which may trigger
                // a further downscale to make room).
                Some(Some((new_idx, val)))
            }
        } else {
            Some(None)
        }
    }

    /// Downscales at U64 width by collapsing 2^by adjacent buckets.
    ///
    /// Rotates the ring buffer to make the live range contiguous, then
    /// folds groups in-place with saturating addition.  No temporary
    /// buffer is needed because the write position never overtakes the
    /// read position.
    pub(super) fn downscale_u64(&mut self, by: i32) {
        debug_assert_eq!(self.bucket_width, BucketWidth::U64);
        debug_assert!(by >= 1);

        if self.range_is_empty() {
            self.shift_indices(by);
            return;
        }

        let range_len = (self.index_end - self.index_start + 1) as usize;
        let new_start = self.index_start >> by;
        let new_end = self.index_end >> by;
        let new_len = (new_end - new_start + 1) as usize;

        // Rotate the ring so index_start maps to physical slot 0,
        // making the live range contiguous in data[0..range_len].
        let start_slot = self.slot_for(self.index_start);
        self.data.rotate_left(start_slot);

        // Fold in-place: sum adjacent entries that share the same
        // shifted index.  write ≤ k holds at every step because each
        // output group contains at least one input entry.
        let mut write = 0usize;
        let mut acc = 0u64;
        let mut cur_new = self.index_start >> by;

        for k in 0..range_len {
            let new_idx = (self.index_start + k as i32) >> by;
            if new_idx != cur_new {
                self.data[write] = acc;
                write += 1;
                acc = 0;
                cur_new = new_idx;
            }
            acc = acc.saturating_add(self.data[k]);
        }
        self.data[write] = acc;
        write += 1;
        debug_assert_eq!(write, new_len);

        // Zero all slots beyond the new live range.
        self.data[new_len..].fill(0);

        self.index_start = new_start;
        self.index_end = new_end;
        self.index_base = new_start;
    }
}
