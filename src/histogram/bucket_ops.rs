// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket operations — widen (SWAR) and downscale.

use super::bucket_width::BucketWidth;
use super::swar::{swar_has_overflow, swar_narrow_compact, swar_shift_up_one, swar_step};
use super::Histogram;

impl<const N: usize> Histogram<N> {
    /// Widens bucket counters by one scale-step using SWAR pairwise
    /// summation. Doubles the counter width and halves the index range
    /// (equivalent to a 1-step downscale).
    ///
    /// At sub-U64, data never wraps (indices are always in
    /// `[index_base, index_base + cap)`), so SWAR operates on a
    /// contiguous linear layout.
    ///
    /// Returns `None` if already at U64.
    /// Returns `Some((steps, deferred))` on success: `steps` is always 1,
    /// and `deferred` is an optional `(index, count)` displaced during
    /// an odd-base shift that the caller must re-insert.
    pub(super) fn bucket_widen(&mut self) -> Option<(i32, Option<(i32, u64)>)> {
        if self.is_effectively_empty() {
            let target = self.bucket_width.widen_by(1)?;
            self.bucket_width = target;
            self.shift_indices(1);
            return Some((1, None));
        }

        debug_assert!(
            self.bucket_width != BucketWidth::U64,
            "cannot widen past U64",
        );
        debug_assert!(
            self.index_start >= self.index_base,
            "sub-U64 data must not wrap: start={} base={}",
            self.index_start,
            self.index_base,
        );

        self.swar_merge_step(true)
    }

    /// Performs one SWAR pairwise-merge step.
    ///
    /// If `index_base` is odd, shifts data up by one slot to restore
    /// even alignment before the SWAR step.
    ///
    /// When the top slot is occupied (possible after a widen that filled
    /// all post-widen capacity), the top slot is saved, zeroed, and
    /// returned as a deferred value for the caller to re-insert.
    ///
    /// When `force_widen` is true, always accepts the wider format
    /// (used by `bucket_widen`). Otherwise checks for overflow and
    /// narrows back to the original width when possible (used by
    /// `do_downscale` to preserve bucket capacity).
    ///
    /// Returns `None` if already at U64.
    /// Returns `Some((steps, deferred))` on success: `steps` is always 1,
    /// and `deferred` is an optional `(index, count)` that was displaced
    /// by the shift and must be re-inserted at `index >> 1` by the caller.
    pub(super) fn swar_merge_step(
        &mut self,
        force_widen: bool,
    ) -> Option<(i32, Option<(i32, u64)>)> {
        let width = self.bucket_width;
        if width == BucketWidth::U64 {
            return None;
        }

        let shifted = self.index_base & 1 != 0;
        let mut deferred = None;

        if shifted {
            if self.top_slot_occupied() {
                // Save the top slot value (standalone at even index,
                // partner is out of range) and zero it so shift has room.
                let top_index = self.index_end;
                debug_assert_eq!(
                    top_index,
                    self.index_base + self.bucket_capacity() as i32 - 1,
                    "index_end must equal physical top when top_slot_occupied"
                );
                let top_slot = self.slot_for(top_index);
                let top_val = self.bucket_get(top_slot);
                self.bucket_set(top_slot, 0);
                deferred = Some((top_index, top_val));
            }
            swar_shift_up_one(self.bucket_data_mut(), width);
        }

        swar_step(self.bucket_data_mut(), width);

        if force_widen || swar_has_overflow(self.bucket_data(), width) {
            self.bucket_width = width.wider().unwrap();
        } else {
            swar_narrow_compact(self.bucket_data_mut(), width);
        }

        self.shift_indices(1);

        if shifted {
            self.clamp_index_end();
        }

        // Adjust deferred index for this merge step.
        let deferred = deferred.map(|(idx, val)| (idx >> 1, val));

        Some((1, deferred))
    }

    /// Clears bucket data and writes `sums` into the new index range.
    pub(super) fn rewrite_buckets(&mut self, start: i32, end: i32, base: i32, sums: &[u64]) {
        self.bucket_data_mut().fill(0);
        self.index_start = start;
        self.index_end = end;
        self.index_base = base;
        for (i, &v) in sums.iter().enumerate() {
            let idx = start + i as i32;
            self.bucket_set(self.slot_for(idx), v);
        }
    }

    /// Downscales at U64 width by collapsing 2^by adjacent buckets.
    ///
    /// At U64 width, sums use saturating arithmetic and cannot
    /// meaningfully overflow.
    pub(super) fn bucket_downscale_u64(&mut self, by: i32) {
        debug_assert_eq!(self.bucket_width, BucketWidth::U64);
        debug_assert!(by >= 1);

        if self.is_effectively_empty() {
            self.shift_indices(by);
            return;
        }

        let new_start = self.index_start >> by;
        let new_end = self.index_end >> by;
        let new_len = (new_end - new_start + 1) as usize;

        let mut sums = [0u64; 256];
        debug_assert!(new_len <= sums.len());

        for old_idx in self.index_start..=self.index_end {
            let val = self.bucket_get(self.slot_for(old_idx));
            let out = ((old_idx >> by) - new_start) as usize;
            sums[out] = sums[out].saturating_add(val);
        }

        self.rewrite_buckets(new_start, new_end, new_start, &sums[..new_len]);
    }
}
