// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Merge logic for combining histograms.

use super::swar::widen;
use super::width::Width;
use super::{Error, HighLow, Histogram, Stats};

impl<const N: usize> Histogram<N> {
    /// Merges another histogram into this one.
    ///
    /// The source histogram may have a different pool size (`M`).
    /// Uses snapshot/rollback for atomicity: a failed merge leaves
    /// `self` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Overflow`] if the combined total count would
    /// exceed `u64::MAX`.
    pub fn merge_from<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Error> {
        if other.stats.count == 0 {
            return Ok(());
        }

        let new_count = self.checked_add_count(other.stats.count).ok_or(Error::Overflow)?;

        let snapshot = self.clone();
        match self.merge_buckets(other) {
            Ok(()) => {
                self.commit_stats(&Stats {
                    count: new_count,
                    sum: self.stats.sum + other.stats.sum,
                    min: other.stats.min,
                    max: other.stats.max,
                });
                Ok(())
            }
            Err(e) => {
                *self = snapshot;
                Err(e)
            }
        }
    }

    /// Core merge: downscale self, then word-by-word merge from source.
    fn merge_buckets<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Error> {
        if other.buckets_empty() {
            return Ok(());
        }

        let other_scale = other.current.scale.scale();
        let other_width = other.current.width;

        // When self is empty, adopt the source's width and scale
        // directly — there is no data to transform.
        if self.buckets_empty() {
            self.current.width = self.current.width.max(other_width);
            let target = self.current.scale.scale().min(other_scale);
            self.current.scale =
                crate::mapping::Scale::new(target).expect("valid scale");
        }

        // Phase 1: determine target scale from the combined range.
        let min_scale = self.current.scale.scale().min(other_scale);

        let self_hl = self.slot_range_at_scale(min_scale);
        let other_hl = other.slot_range_at_scale(min_scale);
        let combined = self_hl.merge(other_hl);

        let word_hl = HighLow {
            low: self.current.width.slot_to_word_index(combined.low),
            high: self.current.width.slot_to_word_index(combined.high),
        };
        let extra = word_hl.change_steps(N);
        let target_scale = min_scale - extra as i32;

        let self_change = self.current.scale.scale() - target_scale;
        if self_change > 0 && !self.buckets_empty() {
            self.downscale_by(self_change as u32)?;
        } else if self_change > 0 {
            // Empty histogram: just set the scale.
            self.current.scale =
                crate::mapping::Scale::new(target_scale).expect("valid scale");
        }

        // Phase 2: word-by-word merge from source.
        let shift = (other_scale - self.current.scale.scale()) as u32;
        let in_word_steps = shift.min(other_width.to_u64_widen_steps());
        let cross_steps = shift - in_word_steps;

        if cross_steps == 0 {
            self.merge_in_word(other, other_width, other_scale, in_word_steps)?;
        } else {
            self.merge_cross_word(other, other_width, other_scale, cross_steps)?;
        }

        Ok(())
    }

    /// Merge when the scale shift fits entirely within a single source word.
    /// Each source word is widened by `in_word_steps`, producing multiple
    /// destination-slot contributions per word.
    fn merge_in_word<const M: usize>(
        &mut self,
        other: &Histogram<M>,
        src_width: Width,
        src_scale: i32,
        in_word_steps: u32,
    ) -> Result<(), Error> {
        let widened_width = if in_word_steps > 0 {
            src_width.wider_by(in_word_steps).expect("capped by to_u64")
        } else {
            src_width
        };
        let lanes = widened_width.slots_per_u64();
        let lane_bits = widened_width.bits_per_slot();
        let lane_mask = widened_width.counter_max();
        let src_slots_per_lane = 1i32 << in_word_steps;

        for src_widx in other.word_start..=other.word_end {
            let word = other.data[other.data_idx(src_widx)];
            if word == 0 {
                continue;
            }

            let widened = if in_word_steps > 0 {
                widen(src_width, widened_width, word)
            } else {
                word
            };

            let first_src_slot = src_width.word_to_slot_index(src_widx);

            for lane in 0..lanes {
                let count = (widened >> (lane * lane_bits)) & lane_mask;
                if count == 0 {
                    continue;
                }
                let src_slot = first_src_slot + (lane as i32) * src_slots_per_lane;
                self.retry_increment(count, |h| {
                    let actual_shift = src_scale - h.current.scale.scale();
                    src_slot >> actual_shift
                })?;
            }
        }
        Ok(())
    }

    /// Merge when the scale shift requires cross-word grouping.
    /// Source words are widened to U64 and accumulated in aligned groups.
    fn merge_cross_word<const M: usize>(
        &mut self,
        other: &Histogram<M>,
        src_width: Width,
        src_scale: i32,
        cross_steps: u32,
    ) -> Result<(), Error> {
        let group_size = 1i32 << cross_steps;
        let aligned_start = other.word_start & !(group_size - 1);
        let need_widen = src_width != Width::U64;

        let mut gstart = aligned_start;
        while gstart <= other.word_end {
            let mut sum = 0u64;
            for w in 0..group_size {
                let widx = gstart + w;
                if widx >= other.word_start && widx <= other.word_end {
                    let word = other.data[other.data_idx(widx)];
                    sum += if need_widen {
                        widen(src_width, Width::U64, word)
                    } else {
                        word
                    };
                }
            }

            if sum > 0 {
                let first_src_slot = src_width.word_to_slot_index(gstart);
                self.retry_increment(sum, |h| {
                    let actual_shift = src_scale - h.current.scale.scale();
                    first_src_slot >> actual_shift
                })?;
            }

            gstart += group_size;
        }
        Ok(())
    }
}
