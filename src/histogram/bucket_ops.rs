// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale operations.

use super::Histogram;
use super::swar::widen;

impl<const N: usize> Histogram<N> {
    pub(super) fn do_downscale(&mut self, change: u32) -> Result<(), super::Error> {
        debug_assert!(change != 0);
        debug_assert!(self.buckets_empty());

        let input_width = self.current.width;
        let to_u64_widen = input_width.to_u64_widen_steps();
        let first_widen_by = change.min(to_u64_widen);
        let second_widen_by = change - first_widen_by;
        let group_mask = (1 << second_widen_by) - 1;

        let mut total_combined = 0;

        // Widen as much as possible before repacking.
        if to_u64_widen != 0 {
            let new_width = input_width.wider_by(first_widen_by).expect("checked");

            // Or-fold the values after widening. This computes the
            // maximum bit in each output bucket.
            let mut group_combined: u64 = 0;

            // Widen one word at a time
            for widx in self.word_start..=self.word_end {
                let di = widx as usize % N;

                let word = widen(input_width, new_width, self.data[di]);

                self.data[di] = word;

                if second_widen_by == 0 {
                    let folded = new_width.or_fold_lanes(word);
                    total_combined |= folded;
                } else if (widx & group_mask) != group_mask {
                    group_combined += word;
                } else {
                    total_combined |= group_combined;
                    group_combined = 0;
                }
            }
        }

        Ok(())
    }
}
