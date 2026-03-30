// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale operations.

use super::Histogram;
use super::swar::{narrow, widen};
use super::width::{Width, ALL_WIDTHS};

impl<const N: usize> Histogram<N> {
    /// Downscales the histogram by at least `change` scale steps.
    ///
    /// Returns the actual number of scale steps applied, which may
    /// exceed `change` when bucket sums require a wider output width.
    pub(super) fn do_downscale(&mut self, change: u32) -> Result<u32, super::Error> {
        debug_assert!(change != 0);
        debug_assert!(!self.buckets_empty());

        if self.current.width == Width::U64 {
            self.downscale_u64(change);
            return Ok(change);
        }

        self.downscale_sub_u64(change)
    }

    /// Downscale when input is already at U64 width.
    ///
    /// Groups of 2^change consecutive words are summed in place,
    /// processing outward from word_base to avoid overwriting
    /// unread data.
    fn downscale_u64(&mut self, change: u32) {
        let group_size = 1i32 << change;
        let group_mask = group_size - 1;

        let from = self.word_base & !group_mask;
        let to = from + group_size;

        // Compute group containing word_base.
        self.data[0] = (from..to)
            .fold(0, |s, idx| s + self.data[idx as usize % N]);

        // Compute groups above word_base.
        let end_limit = self.word_end + 1;
        for slot in (to..end_limit).step_by(group_size as usize) {
            let widx = slot >> change;
            let end = (slot + group_size).min(end_limit);
            self.data[widx as usize % N] = (slot..end)
                .fold(0, |s, idx| s + self.data[idx as usize % N]);
        }

        // Compute groups below word_base.
        let start_limit = self.word_start & !group_mask;
        for slot in (start_limit..from).step_by(group_size as usize) {
            let widx = slot >> change;
            let begin = slot.max(self.word_start);
            self.data[widx as usize % N] = (begin..slot + group_size)
                .fold(0, |s, idx| s + self.data[idx as usize % N]);
        }

        // Clear the vacated range.
        let clear_from = (end_limit >> change) as usize + 1;
        let clear_to = start_limit as usize >> change;
        self.data[clear_from..clear_to].fill(0);

        self.shift_indices(change);
    }

    /// Downscale from a sub-U64 width by iterative in-place widening,
    /// then narrowing and repacking.
    ///
    /// Phase 1 widens by `change` steps so each lane holds the sum
    /// of 2^change input buckets. Phase 2 widens further one step at
    /// a time until the gap between lane width and required width
    /// reaches `change` — or we exhaust at U64. Phase 3 determines
    /// any cross-word grouping needed (when U64 was reached with
    /// insufficient gap). Phase 4 narrows and repacks.
    ///
    /// Word-level compression is always 2^change:
    ///   2^cross_steps words summed into one value, repeated
    ///   2^narrow_steps times and packed at output_width.
    ///
    /// Returns the actual scale change (≥ `change`).
    fn downscale_sub_u64(&mut self, change: u32) -> Result<u32, super::Error> {
        let input_width = self.current.width;
        let to_u64 = input_width.to_u64_widen_steps();

        // Phase 1: Widen by up to `change` steps (capped at U64).
        let first_widen = change.min(to_u64);
        let mut cur = input_width.wider_by(first_widen).expect("capped at U64");
        let mut total_widen = first_widen;
        let mut total_or = 0u64;

        for widx in self.word_start..=self.word_end {
            let di = widx as usize % N;
            self.data[di] = widen(input_width, cur, self.data[di]);
            total_or |= cur.or_fold_lanes(self.data[di]);
        }

        // Phase 2: Widen one step at a time until the gap between
        // current width and required width reaches `change`, or we
        // exhaust in-word widening at U64.
        loop {
            let required = Width::from_max_value(total_or);
            if cur.subtract(required) >= change as i32 {
                break;
            }
            if cur == Width::U64 {
                break;
            }

            let prev = cur;
            cur = cur.wider_by(1).expect("not yet U64");
            total_or = 0;
            for widx in self.word_start..=self.word_end {
                let di = widx as usize % N;
                self.data[di] = widen(prev, cur, self.data[di]);
                total_or |= cur.or_fold_lanes(self.data[di]);
            }
            total_widen += 1;
        }

        // Phase 3: Determine cross-word grouping steps.
        //
        // If the widen loop achieved gap >= change, no cross-word
        // grouping is needed (cross_steps = 0). Otherwise we reached
        // U64 and must sum consecutive words to make up the
        // difference. Each doubling adds at most 1 bit to the max,
        // so gap decreases by at most 1 per step while cross_steps
        // increases by 1 — the sum is non-decreasing and the loop
        // always terminates.
        let required = Width::from_max_value(total_or);
        let mut cross_steps = 0u32;

        if cur.subtract(required) < change as i32 {
            debug_assert_eq!(cur, Width::U64);

            loop {
                cross_steps += 1;
                let group_size = 1i32 << cross_steps;
                let aligned = self.word_start & !(group_size - 1);
                let mut max_sum = 0u64;
                let mut gstart = aligned;
                while gstart <= self.word_end {
                    let mut sum = 0u64;
                    for g in 0..group_size {
                        let widx = gstart + g;
                        if widx >= self.word_start && widx <= self.word_end {
                            sum += self.data[widx as usize % N];
                        }
                    }
                    max_sum = max_sum.max(sum);
                    gstart += group_size;
                }
                let required = Width::from_max_value(max_sum);
                let gap = Width::U64.subtract(required) as u32;
                if cross_steps + gap >= change {
                    break;
                }
            }
        }

        // Phase 4: Narrow and repack.
        let narrow_steps = change - cross_steps;
        let output_width = ALL_WIDTHS[cur as usize - narrow_steps as usize];

        let group = 1i32 << cross_steps;
        let total_merge = 1i32 << change;
        let aligned = self.word_start & !(total_merge - 1);
        let mut out_widx = aligned >> change;
        let mut ostart = aligned;

        if narrow_steps > 0 {
            let repack = 1i32 << narrow_steps;
            let chunk_bits = 64u32 >> narrow_steps;

            while ostart <= self.word_end {
                let mut acc = 0u64;
                for r in 0..repack {
                    let gstart = ostart + r * group;
                    let mut value = 0u64;
                    for g in 0..group {
                        let widx = gstart + g;
                        if widx >= self.word_start && widx <= self.word_end {
                            value += self.data[widx as usize % N];
                            self.data[widx as usize % N] = 0;
                        }
                    }
                    let narrowed = narrow(cur, output_width, value);
                    acc |= narrowed << (r as u32 * chunk_bits);
                }
                self.data[out_widx as usize % N] = acc;
                out_widx += 1;
                ostart += total_merge;
            }
        } else {
            // Pure cross-word grouping, output stays at U64.
            while ostart <= self.word_end {
                let mut sum = 0u64;
                for g in 0..total_merge {
                    let widx = ostart + g;
                    if widx >= self.word_start && widx <= self.word_end {
                        sum += self.data[widx as usize % N];
                        self.data[widx as usize % N] = 0;
                    }
                }
                self.data[out_widx as usize % N] = sum;
                out_widx += 1;
                ostart += total_merge;
            }
        }

        self.shift_indices(change);
        self.current.width = output_width;
        Ok(total_widen + cross_steps)
    }
}
