// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale operations.

use super::Histogram;

impl<const N: usize> Histogram<N> {
    pub(super) fn do_downscale(&mut self, change: u32) -> Result<(), super::Error> {
        debug_assert!(change != 0);
        debug_assert!(self.buckets_empty());

        let mut width = self.current.width;
        let to_u64_widen = width.to_u64_widen_steps();
        let first_widen_by = change.min(to_u64_widen);
        let second_widen_by = change - first_widen_by;
        let group_mask = (1 << second_widen_by) - 1;
        let mut max_group_count = 0;

        // Widen as much as possible before compacting.
        if to_u64_widen != 0 {
            let new_width = width.wider_by(first_widen_by).expect("checked");

            let mut group_count = 0;

            for widx in self.word_start..=self.word_end {
                let di = widx as usize % N;

                super::swar::widen_into(width, new_width, &mut self.data[di]);
                group_count += self.data[di];
            }

            width = new_width;
        }

        //         if second_widen_by {
        //             max_count = max_count.max(self.data[di]);
        //         }

        Ok(())
    }
}
