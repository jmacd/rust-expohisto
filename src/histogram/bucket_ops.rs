// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale operations.

use super::Histogram;
use super::swar::widen;
use super::width::Width;

impl<const N: usize> Histogram<N> {
    pub(super) fn do_downscale(&mut self, change: u32) -> Result<(), super::Error> {
        debug_assert!(change != 0);
        debug_assert!(self.buckets_empty());

        let input_width = self.current.width;
        let input_to_u64_widen = input_width.to_u64_widen_steps();
        let first_widen_by = change.min(input_to_u64_widen);
        let second_widen_by = change - first_widen_by;
        let group_mask = (1 << second_widen_by) - 1;

        if input_to_u64_widen == 0 {
            // Starting at U64 is simpler.
        } else {
            // Widen by up to the downscale factor.
            let wider_width = input_width.wider_by(first_widen_by).expect("checked");

            let mut total_combined = 0;
            let mut group_combined = 0;

            for widx in self.word_start..=self.word_end {
                let di = widx as usize % N;

                let word = widen(input_width, wider_width, self.data[di]);

                self.data[di] = word;

                if second_widen_by == 0 {
                    // When we are not combining u64 values, no sum required.
                    total_combined |= wider_width.or_fold_lanes(word);
                } else {
                    group_combined += word;

                    if (widx & group_mask) == group_mask {
                        total_combined |= group_combined;
                        group_combined = 0;
                    }
                }
            }

            total_combined |= group_combined;

            // We or-folded the group totals.
            let required_width = Width::from_max_value(wider_width.or_fold_lanes(total_combined));

            // The wider width has to be at least one greater than required.
            debug_assert!(wider_width.subtract(required_width) > 0);

            // Howeve, required can be more than, less than, or equal input_width.
            let difference = required_width.subtract(input_width);

            // When we change scale, sometimes require more in-word widening.
            if difference > 0 {
                // When required is greater than input width, it means
                // we haven't widened enough. The wider_width must not be
                // U64 or it implies an earlier overflow.
                let overflow_widen = difference as u32;
                let wider_to_u64_widen = wider_width.to_u64_widen_steps();
                debug_assert!(overflow_widen <= wider_to_u64_widen);

                let wider_again = wider_width.wider_by(overflow_widen).expect("checked");

                // In this case, we have to do it again.
                for widx in self.word_start..=self.word_end {
                    let di = widx as usize % N;

                    let word = widen(wider_width, wider_again, self.data[di]);

                    self.data[di] = word;
                }
            }

            // NOT YET
            // @@@ self.shift_indices(first_widen_by);
        }

        Ok(())
    }
}
