// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket operations — widen and downscale.
//!
//! Downscale merges groups of 2^change adjacent buckets by summing
//! their counters.  The implementation clones the data array, then
//! scatter-adds from the clone into a fresh, aligned output buffer.
//! This linear read→fold→write pattern avoids in-place alignment
//! fixups and is SIMD-friendly.

use super::bucket_width::BucketWidth;
use super::Histogram;

impl<const N: usize> Histogram<N> {
    /// Downscales by `change` steps: merges groups of `2^change`
    /// adjacent bucket indices by summing their counters.
    ///
    /// `min_width` sets a floor on the output width (use the current
    /// width for a normal downscale, or the next wider width when
    /// a counter overflow forces widening).
    ///
    /// Clones the data array, determines the minimum output width
    /// that can hold all group sums (at least `min_width`), then
    /// scatter-adds each old counter into the corresponding output
    /// slot.  `index_base` is freshly computed and word-aligned, so
    /// no alignment fixups are needed.
    ///
    /// Returns `Err(Overflow)` if any group sum exceeds `u64::MAX`.
    pub(super) fn do_downscale(
        &mut self,
        change: i32,
        min_width: BucketWidth,
    ) -> Result<(), super::Overflow> {
        debug_assert!(change >= 1);

        if self.range_is_empty() {
            self.shift_indices(change);
            return Ok(());
        }

        let old_data = self.data;
        let old_width = self.bucket_width;
        let old_base = self.index_base;
        let old_start = self.index_start;
        let old_end = self.index_end;

        let new_start = old_start >> change;
        let new_end = old_end >> change;

        // Phase 1: Determine output width.
        //
        // Walk the old live range, summing each group of adjacent
        // counters that share the same shifted index.  Track the
        // maximum group sum to find the minimum width that fits.
        let mut max_sum: u64 = 0;
        let mut cur_group = old_start >> change;
        let mut group_acc: u64 = 0;

        for idx in old_start..=old_end {
            let new_idx = idx >> change;
            if new_idx != cur_group {
                if group_acc > max_sum {
                    max_sum = group_acc;
                }
                group_acc = 0;
                cur_group = new_idx;
            }
            let slot = Self::slot_in(idx, old_base, old_width);
            let val = Self::get_in(&old_data, slot, old_width);
            group_acc = group_acc.checked_add(val).ok_or(super::Overflow)?;
        }
        // Final group.
        if group_acc > max_sum {
            max_sum = group_acc;
        }

        // Find the minimum width that can hold max_sum, starting
        // from the requested floor (also respecting min_bucket_width).
        let mut new_width = if min_width > self.min_bucket_width {
            min_width
        } else {
            self.min_bucket_width
        };
        while max_sum > new_width.counter_max() {
            new_width = new_width.wider().ok_or(super::Overflow)?;
        }

        // Phase 2: Scatter-add into a fresh, aligned buffer.
        let new_spw = new_width.slots_per_word() as i32;
        let new_base = new_start & !(new_spw - 1);

        self.data = [0u64; N];
        self.bucket_width = new_width;
        self.index_base = new_base;
        self.index_start = new_start;
        self.index_end = new_end;

        cur_group = old_start >> change;
        group_acc = 0;

        for idx in old_start..=old_end {
            let new_idx = idx >> change;
            if new_idx != cur_group {
                // Write the completed group sum.
                let out_slot = Self::slot_in(cur_group, new_base, new_width);
                Self::set_in(&mut self.data, out_slot, new_width, group_acc);
                group_acc = 0;
                cur_group = new_idx;
            }
            let in_slot = Self::slot_in(idx, old_base, old_width);
            let val = Self::get_in(&old_data, in_slot, old_width);
            // Cannot overflow: we already checked in phase 1.
            group_acc += val;
        }
        // Write the final group.
        let out_slot = Self::slot_in(cur_group, new_base, new_width);
        Self::set_in(&mut self.data, out_slot, new_width, group_acc);

        self.trim_bucket_range();
        Ok(())
    }

    /// Downscales by 1 step with forced widening.
    ///
    /// Used when a counter overflows: the width must increase by at
    /// least one level.
    pub(super) fn widen_by_one(&mut self) -> Result<(), super::Overflow> {
        let min = self.bucket_width.wider().ok_or(super::Overflow)?;

        if self.range_is_empty() {
            self.bucket_width = min;
            self.shift_indices(1);
            return Ok(());
        }

        self.do_downscale(1, min)
    }

    // -- Static helpers for reading/writing packed counters in a
    //    data array, parameterized by base and width so they work
    //    on both the old (cloned) and new layouts.

    /// Physical slot index for a logical bucket index, given a base
    /// and width.
    #[inline]
    const fn slot_in(index: i32, base: i32, width: BucketWidth) -> usize {
        let cap = width.capacity(N) as i32;
        (index - base).rem_euclid(cap) as usize
    }

    /// Reads a counter from a data array at a physical slot.
    #[inline]
    const fn get_in(data: &[u64; N], slot: usize, width: BucketWidth) -> u64 {
        let bits = width.bits();
        let spw = 64 / bits;
        let wi = slot / spw;
        let shift = (slot % spw) * bits;
        let mask = width.counter_max();
        (data[wi] >> shift) & mask
    }

    /// Writes a counter into a data array at a physical slot.
    #[inline]
    fn set_in(data: &mut [u64; N], slot: usize, width: BucketWidth, value: u64) {
        let bits = width.bits();
        let spw = 64 / bits;
        let wi = slot / spw;
        let shift = (slot % spw) * bits;
        let mask = width.counter_max();
        let word = &mut data[wi];
        *word = (*word & !(mask << shift)) | ((value & mask) << shift);
    }
}
