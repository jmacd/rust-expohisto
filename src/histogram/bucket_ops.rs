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
        let group_size = 1 << second_widen_by;
        let group_mask = group_size - 1;

        // Handle the U64 case, where no sub-u64 widening occurs.
        if input_to_u64_widen == 0 {

            // Range of output group 0 is (from..to)
            let from = self.word_base & !group_mask;
            let to = from + group_size;

            // Compute group 0.
            self.data[0] = (from..to).fold(0, |s, idx| s + self.data[idx as usize % N]);

            // Compute groups 1..end
            let end_limit = self.word_end+1;
            for slot in (to..end_limit).step_by(group_size as usize) {
                let widx = slot >> second_widen_by;
                let end = (slot + group_size).min(end_limit);
                self.data[widx as usize] = (slot..end)
                    .fold(0, |s, idx| s + self.data[idx as usize % N]);
            }

            // Compute groups start..0
            let start_limit = self.word_start & !group_mask;
            
            for slot in (start_limit..from).step_by(group_size as usize)
            {
                let widx = slot >> second_widen_by;
                let begin = slot.max(self.word_start);
                self.data[widx as usize] = (begin..slot + group_size)
                    .fold(0, |s, idx| s + self.data[idx as usize % N]);
            }

            // Clear the now-empty buckets.
            let clear_from = (end_limit >> second_widen_by) as usize + 1;
            let clear_to = start_limit as usize >> second_widen_by;
            self.data[clear_from..clear_to].fill(0);

            // Downscale the range.
            self.shift_indices(change);
            
            return Ok(())
        }

        // Widen by up to the downscale factor. In some cases, we will
        // be able to repack back down to the original width.
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
        
        // We OR-folded the group totals.
        let required_width = Width::from_max_value(wider_width.or_fold_lanes(total_combined));
        
        // The wider width has to be at least one greater than required.
        debug_assert!(wider_width.subtract(required_width) > 0);
        
        // However, required can be more than, less than, or equal input_width.
        let difference = required_width.subtract(input_width);
        debug_assert!(difference >= 0);
        
        // When we change scale, sometimes require more in-word widening.
        let (current_width, output_width) = if difference == 0 {
            (wider_width, input_width)
        } else {
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
            debug_assert!(wider_again as i32 - required_width as i32 == change as i32);
            (wider_again, required_width)
        };
        
        // TODO: CASE TWO
        if second_widen_by == 0 {
            // No further summing needed.

            // One word at a time, swar-narrow from current_width to required_width
        } else {

            // Compute group sums, bucket_set in correct location at output width.
        }

        // Zero-fill the slots we have invalidated.
        
        Ok(())
    }
}
