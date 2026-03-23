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

/// Captured old-layout state for a downscale operation.
///
/// Bundles the immutable parameters shared by all downscale codepaths,
/// keeping their signatures compact.
struct DownscaleCtx<'a, const N: usize> {
    change: i32,
    old_data: &'a [u64; N],
    old_width: BucketWidth,
    old_base: i32,
    old_start: i32,
    old_end: i32,
    new_start: i32,
    new_end: i32,
}

impl<const N: usize> DownscaleCtx<'_, N> {
    /// Sum of old counters that map to output group `grp`, with
    /// checked addition for u64 overflow.
    fn group_sum(&self, grp: i32) -> Result<u64, super::Overflow> {
        // Each output group covers 2^change consecutive old indices.
        let group_size = 1i32 << self.change;
        let lo = (grp * group_size).max(self.old_start);
        let hi = (grp * group_size + group_size - 1).min(self.old_end);
        let mut acc: u64 = 0;
        for idx in lo..=hi {
            let slot = Histogram::<N>::old_slot(idx, self.old_base, self.old_width);
            let val = Histogram::<N>::get_in(self.old_data, slot, self.old_width);
            acc = acc.checked_add(val).ok_or(super::Overflow)?;
        }
        Ok(acc)
    }
}

impl<const N: usize> Histogram<N> {
    /// Downscales by `change` steps: merges groups of `2^change`
    /// adjacent bucket indices by summing their counters.
    ///
    /// `min_width` sets a floor on the output width (use the current
    /// width for a normal downscale, or the next wider width when
    /// a counter overflow forces widening).
    ///
    /// Two codepaths:
    ///
    /// - **Safe**: `count ≤ counter_max` — even the worst-case group
    ///   sum (all count in one bucket) fits, so no overflow checks.
    ///   This always covers U64 width (`counter_max` = `u64::MAX`).
    /// - **Speculative hybrid**: begins writing at `spec_width`; on
    ///   the first group overflow, scans the remainder to find the
    ///   true max group sum, computes the exact target width, then
    ///   repairs the already-written prefix.
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
        let ctx = DownscaleCtx {
            change,
            old_data: &old_data,
            old_width: self.bucket_width,
            old_base: self.index_base,
            old_start: self.index_start,
            old_end: self.index_end,
            new_start: self.index_start >> change,
            new_end: self.index_end >> change,
        };

        let spec_width = if min_width > self.min_bucket_width {
            min_width
        } else {
            self.min_bucket_width
        };

        if self.count() <= spec_width.counter_max() {
            self.downscale_safe(&ctx, spec_width);
            Ok(())
        } else {
            self.downscale_speculative(&ctx, spec_width)
        }
    }

    /// Safe path: `count ≤ counter_max`, so no group can overflow.
    /// This always covers the U64 case (count is u64, counter_max
    /// is u64::MAX).
    fn downscale_safe(
        &mut self,
        ctx: &DownscaleCtx<'_, N>,
        w: BucketWidth,
    ) {
        let base = w.word_start(ctx.new_start);
        self.init_output(w, base, ctx.new_start, ctx.new_end);

        for grp in ctx.new_start..=ctx.new_end {
            // Safety: count ≤ counter_max guarantees no group overflows.
            let acc = ctx.group_sum(grp).expect("safe path");
            Self::set_in(&mut self.data, (grp - base) as usize, w, acc);
        }
    }

    /// Speculative hybrid: begin writing at `spec_width`; on the
    /// first overflow, scan the remainder for the true max, compute
    /// the exact target width, then repair the prefix.
    fn downscale_speculative(
        &mut self,
        ctx: &DownscaleCtx<'_, N>,
        w: BucketWidth,
    ) -> Result<(), super::Overflow> {
        let base = w.word_start(ctx.new_start);
        let counter_max = w.counter_max();
        self.init_output(w, base, ctx.new_start, ctx.new_end);

        for grp in ctx.new_start..=ctx.new_end {
            let acc = ctx.group_sum(grp)?;
            if acc > counter_max {
                return self.downscale_repair(ctx, w, grp, acc);
            }
            Self::set_in(&mut self.data, (grp - base) as usize, w, acc);
        }

        Ok(())
    }

    /// Overflow recovery: scan remaining groups for the true max,
    /// compute exact target width, re-read the prefix from the
    /// speculative output, and write everything at the new width.
    fn downscale_repair(
        &mut self,
        ctx: &DownscaleCtx<'_, N>,
        spec_width: BucketWidth,
        overflow_group: i32,
        overflow_acc: u64,
    ) -> Result<(), super::Overflow> {
        let spec_base = spec_width.word_start(ctx.new_start);

        // Phase 1: find the true max group sum from overflow onward.
        let max_acc = (overflow_group..=ctx.new_end)
            .map(|grp| ctx.group_sum(grp))
            .try_fold(overflow_acc, |m, s| s.map(|v| m.max(v)))?;

        // Phase 2: compute exact target width.
        // max_acc > spec_width.counter_max() is guaranteed (that's
        // why we're here).  Since counter_max = (1 << bits) - 1 and
        // bits is a power of two, max_acc ≥ 1 << bits, which always
        // maps to a strictly wider BucketWidth via from_max_value.
        let tw = BucketWidth::from_max_value(max_acc).ok_or(super::Overflow)?;
        debug_assert!(tw > spec_width);

        // Phase 3: repair prefix — re-read groups already written
        // at spec_width and rewrite at the target width.
        let prefix_data = self.data;
        let tbase = tw.word_start(ctx.new_start);
        self.init_output(tw, tbase, ctx.new_start, ctx.new_end);

        for grp in ctx.new_start..overflow_group {
            let val = Self::get_in(&prefix_data, (grp - spec_base) as usize, spec_width);
            Self::set_in(&mut self.data, (grp - tbase) as usize, tw, val);
        }

        // Phase 4: re-sum from overflow_group onward at target width.
        for grp in overflow_group..=ctx.new_end {
            let acc = ctx.group_sum(grp).expect("already validated");
            Self::set_in(&mut self.data, (grp - tbase) as usize, tw, acc);
        }

        Ok(())
    }

    /// Initializes output state for a downscale pass.
    #[inline]
    fn init_output(
        &mut self,
        width: BucketWidth,
        base: i32,
        start: i32,
        end: i32,
    ) {
        self.data = [0u64; N];
        self.bucket_width = width;
        self.index_base = base;
        self.index_start = start;
        self.index_end = end;
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
        let spw = width.slots_per_word();
        let bits = width.bits();
        let wi = slot / spw;
        let shift = (slot % spw) * bits;
        let mask = width.counter_max();
        (data[wi] >> shift) & mask
    }

    /// Writes a counter into a data array at a physical slot.
    #[inline]
    fn set_in(data: &mut [u64; N], slot: usize, width: BucketWidth, value: u64) {
        let spw = width.slots_per_word();
        let bits = width.bits();
        let wi = slot / spw;
        let shift = (slot % spw) * bits;
        let mask = width.counter_max();
        let word = &mut data[wi];
        *word = (*word & !(mask << shift)) | ((value & mask) << shift);
    }
}
