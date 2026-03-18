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
    /// Uses a speculative single-pass approach: clones the data array,
    /// picks `min_width` as the output width, then scatter-adds in one
    /// pass.  If any group sum exceeds `counter_max` for the chosen
    /// width, the pass restarts at the next wider width.  In the common
    /// case (sums fit), this halves the work vs. a two-pass scan.
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

        let mut spec_width = if min_width > self.min_bucket_width {
            min_width
        } else {
            self.min_bucket_width
        };

        // Speculative single-pass: scatter-add at spec_width, retry
        // at the next wider width if any group sum overflows it.
        loop {
            let new_spw = spec_width.slots_per_word() as i32;
            let new_base = new_start & !(new_spw - 1);
            let counter_max = spec_width.counter_max();

            self.data = [0u64; N];
            self.bucket_width = spec_width;
            self.index_base = new_base;
            self.index_start = new_start;
            self.index_end = new_end;

            let mut cur_group = old_start >> change;
            let mut group_acc: u64 = 0;
            let mut retry = false;

            for idx in old_start..=old_end {
                let new_idx = idx >> change;
                if new_idx != cur_group {
                    if group_acc > counter_max {
                        retry = true;
                        break;
                    }
                    let out_slot = (cur_group - new_base) as usize;
                    Self::set_in(&mut self.data, out_slot, spec_width, group_acc);
                    group_acc = 0;
                    cur_group = new_idx;
                }
                let in_slot = Self::old_slot(idx, old_base, old_width);
                let val = Self::get_in(&old_data, in_slot, old_width);
                group_acc = group_acc.checked_add(val).ok_or(super::Overflow)?;
            }

            if !retry && group_acc > counter_max {
                retry = true;
            }

            if retry {
                spec_width = spec_width.wider().ok_or(super::Overflow)?;
                continue;
            }

            // Write the final group and finish.
            let out_slot = (cur_group - new_base) as usize;
            Self::set_in(&mut self.data, out_slot, spec_width, group_acc);

            self.trim_bucket_range();
            return Ok(());
        }
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

    /// Physical slot for reading from the *old* layout.
    ///
    /// At U64 width, `index_base` may not equal `index_start` (the
    /// ring buffer can wrap), so `rem_euclid` is needed.  At sub-U64
    /// widths the live range is always contiguous (`index >= base`),
    /// so a plain subtraction suffices.
    #[inline]
    const fn old_slot(index: i32, base: i32, width: BucketWidth) -> usize {
        if matches!(width, BucketWidth::U64) {
            (index - base).rem_euclid(N as i32) as usize
        } else {
            debug_assert!((index - base) >= 0);
            (index - base) as usize
        }
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
