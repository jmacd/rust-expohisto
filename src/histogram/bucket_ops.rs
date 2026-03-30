// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale operations.

use super::Histogram;
use super::swar::{narrow, widen};
use super::width::{ALL_WIDTHS, Width};

impl<const N: usize> Histogram<N> {
    /// Widen every active word from `before` to `after` and return
    /// the OR-fold of all lanes at the new width.
    fn widen_words(&mut self, before: Width, after: Width) -> u64 {
        let mut total_or = 0u64;
        for widx in self.word_start..=self.word_end {
            let di = widx as usize % N;
            self.data[di] = widen(before, after, self.data[di]);
            total_or |= after.or_fold_lanes(self.data[di]);
        }
        total_or
    }
    /// Downscales the histogram by at least `change` scale steps.
    ///
    /// Returns the actual number of scale steps applied, which may
    /// exceed `change` when bucket sums require a wider output width.
    pub(super) fn do_downscale(&mut self, change: u32) -> Result<u32, super::Error> {
        debug_assert!(change != 0);
        debug_assert!(!self.buckets_empty());

        let input_width = self.current.width;
        let to_u64 = input_width.to_u64_widen_steps();

        // Phase 1: Widen by up to `change` steps (capped at U64).
        let first_widen = change.min(to_u64);
        let mut cur = input_width;
        let mut total_widen = first_widen;
        let mut total_or = 0u64;

        if first_widen > 0 {
            cur = input_width.wider_by(first_widen).expect("capped at U64");
            total_or = self.widen_words(input_width, cur);
        }

        // Phase 2: Widen one step at a time until the gap between
        // current width and required width reaches `change`, or we
        // exhaust in-word widening at U64.
        loop {
            if cur == Width::U64 {
                break;
            }
            let required = Width::from_max_value(total_or);
            if cur.subtract(required) >= change as i32 {
                break;
            }

            let prev = cur;
            cur = cur.wider_by(1).expect("not yet U64");
            total_or = self.widen_words(prev, cur);
            total_widen += 1;
        }

        // Phase 3: Determine cross-word grouping steps.
        //
        // If the widen loop achieved gap >= change, no cross-word
        // grouping is needed (cross_steps = 0). Otherwise we are
        // at U64 and must sum consecutive words to make up the
        // difference. Each doubling adds at most 1 bit to the max,
        // so gap decreases by at most 1 per step while cross_steps
        // increases by 1 — the sum is non-decreasing and the loop
        // always terminates.
        let mut cross_steps = 0u32;

        if cur == Width::U64 {
            if total_or == 0 {
                // When width started at U64, phase 1 and 2 were skipped.
                for widx in self.word_start..=self.word_end {
                    total_or |= self.data[widx as usize % N];
                }
            }
            let required = Width::from_max_value(total_or);
            let gap = cur.subtract(required) as u32;

            if gap < change {
                loop {
                    cross_steps += 1;
                    let group_size = 1i32 << cross_steps;
                    let aligned = self.word_start & !(group_size - 1);
                    let mut or_sums = 0u64;
                    let mut gstart = aligned;
                    while gstart <= self.word_end {
                        let mut sum = 0u64;
                        for g in 0..group_size {
                            let widx = gstart + g;
                            if widx >= self.word_start && widx <= self.word_end {
                                sum += self.data[widx as usize % N];
                            }
                        }
                        or_sums |= sum;
                        gstart += group_size;
                    }
                    let required = Width::from_max_value(or_sums);
                    let gap = Width::U64.subtract(required) as u32;
                    if cross_steps + gap >= change {
                        break;
                    }
                }
            }
        }

        // Phase 4: Narrow and repack.
        //
        // Word-level compression is always 2^change:
        //   2^cross_steps words summed into one value, repeated
        //   2^narrow_steps times and packed at output_width.
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
