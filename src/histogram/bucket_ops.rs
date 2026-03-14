// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket operations — widen (SWAR) and downscale.

use super::bucket_width::BucketWidth;
use super::swar::{swar_has_overflow, swar_narrow_compact, swar_shift_up_one, swar_step};
use super::Histogram;

impl<const N: usize> Histogram<N> {
    /// Widens bucket counters from the current width by `steps` scale-steps,
    /// using SWAR pairwise summation. Each step doubles the counter width
    /// and halves the index range (equivalent to a 1-step downscale).
    ///
    /// At sub-U64, data never wraps (indices are always in
    /// `[index_base, index_base + cap)`), so SWAR operates on a
    /// contiguous linear layout.
    ///
    /// Returns `None` if already at U64 or `steps` would exceed U64.
    /// Returns `Some(steps)` on success (= the scale decrease applied).
    pub(super) fn bucket_widen(&mut self, steps: i32) -> Option<i32> {
        debug_assert!(steps >= 1);

        if self.is_effectively_empty() {
            let target = self.bucket_width.widen_by(steps)?;
            self.bucket_width = target;
            self.shift_indices(steps);
            return Some(steps);
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

        let mut done = 0;
        while done < steps {
            done += self.swar_merge_step(true)?;
        }

        Some(done)
    }

    /// Performs one SWAR pairwise-merge step.
    ///
    /// If `index_base` is odd, shifts data up by one slot to restore
    /// even alignment. When the top slot is occupied, falls back to
    /// [`scalar_merge_step`](Self::scalar_merge_step).
    ///
    /// When `force_widen` is true, always accepts the wider format
    /// (used by `bucket_widen`). Otherwise checks for overflow and
    /// narrows back to the original width when possible (used by
    /// `do_downscale` to preserve bucket capacity).
    ///
    /// Returns the number of scale levels consumed, or `None` if
    /// already at U64.
    pub(super) fn swar_merge_step(&mut self, force_widen: bool) -> Option<i32> {
        let width = self.bucket_width;
        if width == BucketWidth::U64 {
            return None;
        }

        let shifted = self.index_base & 1 != 0;

        if shifted {
            if self.top_slot_occupied() {
                return self.scalar_merge_step(force_widen);
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

        Some(1)
    }

    /// Scalar gather-scatter merge. Used when SWAR cannot operate
    /// (odd base with top slot occupied).
    ///
    /// If `force_widen` is true, always widens by one level (for the
    /// `CounterOverflow` path). Otherwise, only widens if any pair sum
    /// exceeds the current width's maximum.
    ///
    /// When the merged range exceeds the new capacity (which happens
    /// when base is odd and the range is fully packed), additional
    /// merge levels are applied until the range fits.
    ///
    /// Returns the number of scale steps performed (>= 1).
    #[allow(clippy::needless_range_loop)]
    pub(super) fn scalar_merge_step(&mut self, force_widen: bool) -> Option<i32> {
        let width = self.bucket_width;
        let max_val = width.counter_max();
        let bucket_words = self.bucket_word_count();

        let mut cur_start = self.index_start >> 1;
        let mut cur_end = self.index_end >> 1;
        let mut cur_len = (cur_end - cur_start + 1) as usize;

        let mut sums = vec![0u64; cur_len];
        let mut needs_widen = force_widen;

        // Phase 1: gather from hardware layout, merging adjacent pairs.
        for old_idx in self.index_start..=self.index_end {
            let val = self.bucket_get(self.slot_for(old_idx));
            let out = ((old_idx >> 1) - cur_start) as usize;
            sums[out] = sums[out].saturating_add(val);
            if sums[out] > max_val {
                needs_widen = true;
            }
        }

        let mut target_width = if needs_widen { width.wider()? } else { width };

        let mut scale_steps = 1i32;

        // Phase 2: if the merged range doesn't fit in the new capacity,
        // keep merging pairs and widening until it does.
        loop {
            let target_cap = target_width.capacity(bucket_words);
            if cur_len <= target_cap {
                break;
            }
            // Re-pair the sums array for another merge level.
            let prev_start = cur_start;
            let prev_len = cur_len;
            cur_start >>= 1;
            cur_end >>= 1;
            cur_len = (cur_end - cur_start + 1) as usize;

            let mut next = vec![0u64; cur_len];
            for i in 0..prev_len {
                let old_idx = prev_start + i as i32;
                let out = ((old_idx >> 1) - cur_start) as usize;
                next[out] = next[out].saturating_add(sums[i]);
            }
            sums = next;

            if target_width != BucketWidth::U64 {
                target_width = target_width.wider()?;
            }
            scale_steps += 1;
        }

        // Phase 3: commit — clear data, write back.
        self.bucket_width = target_width;

        let spw = target_width.slots_per_word() as i32;
        self.rewrite_buckets(cur_start, cur_end, cur_start & !(spw - 1), &sums[..cur_len]);

        Some(scale_steps)
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
