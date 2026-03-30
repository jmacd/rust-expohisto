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
    /// Phase 1 widens by `change` steps so each lane holds the sum of
    /// 2^change input buckets. Phase 2 widens further, one step at a
    /// time, until the gap between lane width and required width
    /// reaches `change` — meaning the values can be repacked into
    /// 2^change fewer words. Phase 3 narrows and merges the words.
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
        // current width and required width reaches `change`.
        loop {
            let required = Width::from_max_value(total_or);
            let gap = cur.subtract(required);
            if gap >= change as i32 {
                break;
            }

            if cur == Width::U64 {
                // Sub-U64 widening exhausted; cross-word grouping
                // is required for the remaining compression.
                return self.downscale_cross_word(change, total_widen);
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

        // Phase 3: Narrow and repack.
        //
        // The output width is `change` steps below `cur`. Since
        // gap >= change, narrowing preserves all values.
        let output_width = ALL_WIDTHS[cur as usize - change as usize];

        // Merge 2^change words into one by narrowing each word
        // (which packs values into the low 64>>change bits) and
        // OR-ing them at successive offsets within the output word.
        //
        // Forward scan is safe: the output position for group g
        // is always behind group g+1's first input word.
        let merge = 1i32 << change;
        let chunk_bits = 64u32 >> change;
        let aligned = self.word_start & !(merge - 1);
        let mut out_widx = aligned >> change;
        let mut gstart = aligned;

        while gstart <= self.word_end {
            let mut acc = 0u64;
            for i in 0..merge {
                let widx = gstart + i;
                if widx >= self.word_start && widx <= self.word_end {
                    let di = widx as usize % N;
                    let narrowed = narrow(cur, output_width, self.data[di]);
                    acc |= narrowed << (i as u32 * chunk_bits);
                    self.data[di] = 0;
                }
            }
            self.data[out_widx as usize % N] = acc;
            out_widx += 1;
            gstart += merge;
        }

        self.shift_indices(change);
        self.current.width = output_width;
        Ok(total_widen)
    }

    /// Downscale when in-word widening reached U64 without sufficient
    /// gap to narrow by `change` steps. Performs cross-word grouping
    /// of U64 values followed by narrowing and repacking.
    ///
    /// At entry, data is at U64 width (1 value per word) after
    /// `total_widen` in-word widen steps. Cross-word grouping sums
    /// consecutive U64 words to reduce word count, then narrowing
    /// repacks the sums at the tightest fitting width.
    ///
    /// The word-level compression is always 2^change:
    ///   cross_steps of grouping (2^cross_steps : 1) +
    ///   narrow_steps of narrow+repack (2^narrow_steps : 1).
    ///
    /// Returns the actual scale change (≥ `change`).
    fn downscale_cross_word(
        &mut self,
        change: u32,
        total_widen: u32,
    ) -> Result<u32, super::Error> {
        // Determine minimum cross_steps such that
        // cross_steps + gap(cross_steps) >= change.
        //
        // gap(k) = U64 - required_width for the max group sum at
        // group size 2^k. Each doubling adds at most 1 bit to the
        // max, so gap decreases by at most 1 per step while
        // cross_steps increases by 1 — the sum is non-decreasing.
        let mut cross_steps = 0u32;
        loop {
            let group_size = 1i32 << cross_steps;
            let group_mask = group_size - 1;
            let aligned = self.word_start & !group_mask;
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
            cross_steps += 1;
        }

        let narrow_steps = change - cross_steps;
        let output_width = ALL_WIDTHS[Width::U64 as usize - narrow_steps as usize];

        // Combined pass: group-sum + narrow + repack.
        //
        // Each output word holds 2^narrow_steps values at output_width,
        // where each value is the sum of 2^cross_steps consecutive U64
        // words. Total input words per output = 2^change.
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
                    let mut sum = 0u64;
                    for g in 0..group {
                        let widx = gstart + g;
                        if widx >= self.word_start && widx <= self.word_end {
                            sum += self.data[widx as usize % N];
                            self.data[widx as usize % N] = 0;
                        }
                    }
                    let narrowed = narrow(Width::U64, output_width, sum);
                    acc |= narrowed << (r as u32 * chunk_bits);
                }
                self.data[out_widx as usize % N] = acc;
                out_widx += 1;
                ostart += total_merge;
            }
        } else {
            // Pure cross-word grouping, output stays U64.
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
