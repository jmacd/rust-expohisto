// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale operations.

use super::Histogram;
//use super::width::Width;

impl<const N: usize> Histogram<N> {
    /// Downscales by `change` steps: merges groups of `2^change`
    /// adjacent bucket indices by summing their counters in place.
    pub(super) fn do_downscale(&mut self, change: u32) -> Result<(), super::Error> {
        debug_assert!(change != 0);
        debug_assert!(self.buckets_empty());

        let width = self.current.width;
        let to_u64_widen = width.to_u64_widen_steps();

        if to_u64_widen != 0 {
            let actual_widen_by = change.min(to_u64_widen);
            let new_width = width.wider_by(actual_widen_by).expect("checked");

            let word_start_idx = self.word_start >> to_u64_widen;
            let word_end_idx = self.word_end >> to_u64_widen;

            for idx in word_start_idx..=word_end_idx {
                super::swar::widen_into(width, new_width, &mut self.data[idx as usize % N]);
            }
        }
        // @@@

        Ok(())
    }

    // if swar_steps == change {
    //     self.downscale_case_a(
    //         change, swar_steps, new_start, new_end, num_groups, spec_width,
    //     )
    // } else {
    //     self.downscale_case_b(
    //         change, swar_steps, new_start, new_end, num_groups, spec_width,
    //     )
    // }
    // // ---------------------------------------------------------------
    // // Case A — intra-word: all group sums are SWAR lanes
    // // ---------------------------------------------------------------

    // /// After SWAR-widen by `change` steps, each group sum lives in a
    // /// widened SWAR lane. Repack front-to-back at `actual_width`.
    // ///
    // /// In-place safety: `actual_width ≤ swar_width`, so the output
    // /// word for group g is always ≤ the input word for lane g.
    // /// Front-to-back writing never overwrites unread source data.
    // fn downscale_case_a(
    //     &mut self,
    //     change: u32,
    //     swar_steps: u32,
    //     new_start: i32,
    //     new_end: i32,
    //     num_groups: usize,
    //     spec_width: Width,
    // ) -> Result<(), super::Error> {
    //     let width = self.current.width;
    //     let swar_width = ALL_WIDTHS[width.level() + swar_steps as usize];
    //     let swar_spw = swar_width.slots_per_word();
    //     let swar_bits = swar_width.bits();
    //     let swar_mask = swar_width.counter_max();
    //     let group_size = 1i32 << change;

    //     // Phase 1: SWAR-widen in place and track max.
    //     let data_words = self.data_word_count();
    //     let max_lane_value = swar_widen_max(&mut self.data[..data_words], width, swar_steps);

    //     // Phase 2: In-place front-to-back repack.
    //     let actual_width = Width::from_max_value(max_lane_value)
    //         .unwrap_or(Width::B1)
    //         .max(spec_width);
    //     let base = actual_width.word_start(new_start);

    //     // Lane 0 corresponds to index_base. After swar_steps, each
    //     // lane covers 2^swar_steps original slots.
    //     let lane_offset =
    //         ((new_start * group_size - self.index_base) / (1i32 << swar_steps)) as usize;

    //     // Accumulator for building output words.
    //     let out_bits = actual_width.bits();
    //     let out_spw = actual_width.slots_per_word();
    //     let out_mask = actual_width.counter_max();

    //     // Track current output word being built.
    //     let mut out_word: u64 = 0;
    //     let mut out_word_idx: usize = usize::MAX; // sentinel

    //     for g in 0..num_groups {
    //         // Read widened lane.
    //         let lane = lane_offset + g;
    //         let src_wi = lane / swar_spw;
    //         let src_li = lane % swar_spw;
    //         let val = (self.data[src_wi] >> (src_li * swar_bits)) & swar_mask;

    //         // Compute output position.
    //         let out_slot = (new_start + g as i32 - base) as usize;
    //         let dst_wi = out_slot / out_spw;
    //         let dst_li = out_slot % out_spw;

    //         if dst_wi != out_word_idx {
    //             // Flush previous output word.
    //             if out_word_idx != usize::MAX {
    //                 self.data[out_word_idx] = out_word;
    //             }
    //             // Start new output word. If it's the same as a future
    //             // source word, the source will be read before this
    //             // word is modified (front-to-back guarantee).
    //             out_word = 0;
    //             out_word_idx = dst_wi;
    //         }

    //         out_word |= (val & out_mask) << (dst_li * out_bits);
    //     }
    //     // Flush last output word.
    //     if out_word_idx != usize::MAX {
    //         self.data[out_word_idx] = out_word;
    //     }

    //     // Zero words beyond the output.
    //     let last_out_word = (new_end - base) as usize / out_spw;
    //     for w in &mut self.data[(last_out_word + 1)..data_words] {
    //         *w = 0;
    //     }

    //     self.current.width = actual_width;
    //     self.index_base = base;
    //     self.index_start = new_start;
    //     self.index_end = new_end;
    //     Ok(())
    // }

    // // ---------------------------------------------------------------
    // // Case B — inter-word: group-sum consecutive U64 words
    // // ---------------------------------------------------------------

    // /// Groups span multiple words. SWAR-widen to U64, then sum groups
    // /// of consecutive words in place, then repack.
    // fn downscale_case_b(
    //     &mut self,
    //     change: u32,
    //     swar_steps: u32,
    //     new_start: i32,
    //     new_end: i32,
    //     num_groups: usize,
    //     spec_width: Width,
    // ) -> Result<(), super::Error> {
    //     let width = self.current.width;
    //     let spw = width.slots_per_word() as i32;

    //     // Phase 1: SWAR-widen each word to a single U64 value.
    //     if width == Width::U64 {
    //         // Ring buffer may wrap at U64. Linearize so word w
    //         // covers index_base + w contiguously.
    //         let range = (self.index_end - self.index_start + 1) as usize;
    //         // Rotate the ring buffer so that index_start maps to data[0].
    //         let start_slot = (self.index_start - self.index_base).rem_euclid(N as i32) as usize;
    //         if start_slot != 0 {
    //             Self::rotate_left(&mut self.data, start_slot, range);
    //         }
    //         self.index_base = self.index_start;
    //     } else {
    //         let data_words = self.data_word_count();
    //         swar_widen_max(&mut self.data[..data_words], width, swar_steps);
    //         // After SWAR: each word is one U64 sum. Data is
    //         // contiguous (sub-U64 never wraps).
    //     }

    //     // After phase 1: data word w holds the sum of original
    //     // indices [index_base + w*spw, index_base + (w+1)*spw).
    //     // words_per_group consecutive words form one output group.
    //     let words_per_group = (1i32 << change) / spw;
    //     debug_assert!(words_per_group >= 2);

    //     // First word (linear, may be negative) for output group 0.
    //     let first_word = (new_start * (1i32 << change) - self.index_base) / spw;

    //     // Safety of the forward pass: first_word ≥ -(wpg - 1)
    //     // because index_base ≤ index_start and new_start * K ≤
    //     // index_start (floor division). For group g ≥ 1 its
    //     // lowest source word is first_word + g*wpg ≥ (g-1)*wpg + 1
    //     // > g - 1, so no read overlaps a previous write.
    //     debug_assert!(first_word >= -(words_per_group - 1));

    //     // Phase 2: In-place forward-pass group-sum.
    //     let mut max_sum: u64 = 0;
    //     for g in 0..num_groups {
    //         let base_w = first_word + (g as i32) * words_per_group;
    //         let mut acc: u64 = 0;
    //         for k in 0..words_per_group {
    //             let w = base_w + k;
    //             if w >= 0 && (w as usize) < N {
    //                 acc = acc
    //                     .checked_add(self.data[w as usize])
    //                     .ok_or(super::Error::Overflow)?;
    //             }
    //         }
    //         self.data[g] = acc;
    //         max_sum = max_sum.max(acc);
    //     }

    //     // Zero tail.
    //     for w in &mut self.data[num_groups..N] {
    //         *w = 0;
    //     }

    //     // Phase 3: In-place repack data[0..num_groups] from U64 to
    //     // actual_width. Since actual_width ≤ U64, the output fits
    //     // in ≤ num_groups words. Repack back-to-front so we don't
    //     // overwrite unread U64 values.
    //     let actual_width = Width::from_max_value(max_sum)
    //         .unwrap_or(Width::B1)
    //         .max(spec_width);

    //     if actual_width != Width::U64 {
    //         let base = actual_width.word_start(new_start);
    //         let out_bits = actual_width.bits();
    //         let out_spw = actual_width.slots_per_word();
    //         let out_mask = actual_width.counter_max();
    //         let first_slot = (new_start - base) as usize;
    //         let last_word = (first_slot + num_groups - 1) / out_spw;

    //         // Back-to-front: for each output word (highest first),
    //         // determine which groups map to it, read their U64 sums,
    //         // and pack into the word. Because actual_width < U64,
    //         // the output word index ≤ the group index, so reads
    //         // always precede writes.
    //         let mut g = num_groups;
    //         for ow in (0..=last_word).rev() {
    //             let w_slot_lo = ow * out_spw;
    //             let w_slot_hi = w_slot_lo + out_spw;
    //             let our_lo = first_slot.max(w_slot_lo);
    //             let our_hi = (first_slot + num_groups).min(w_slot_hi);
    //             let mut word: u64 = 0;
    //             if our_hi > our_lo {
    //                 for slot in (our_lo..our_hi).rev() {
    //                     g -= 1;
    //                     let li = slot - w_slot_lo;
    //                     word |= (self.data[g] & out_mask) << (li * out_bits);
    //                 }
    //             }
    //             self.data[ow] = word;
    //         }
    //         // Zero beyond the output range (may already be zero).
    //         for w in &mut self.data[(last_word + 1)..num_groups] {
    //             *w = 0;
    //         }
    //         self.current.width = actual_width;
    //         self.index_base = base;
    //     } else {
    //         self.current.width = Width::U64;
    //         self.index_base = Width::U64.word_start(new_start);
    //     }

    //     self.index_start = new_start;
    //     self.index_end = new_end;
    //     Ok(())
    // }

    // // ---------------------------------------------------------------
    // // Helpers
    // // ---------------------------------------------------------------

    // /// Computes a speculative output width for the downscale pass.
    // ///
    // /// - **Floor** (pigeonhole): `count / num_groups` is the average
    // ///   group sum; at least one group must be ≥ this value.
    // /// - **Ceiling** (worst case): each group sums at most `2^change`
    // ///   counters at `counter_max(current_width)`.
    // ///
    // /// The result is clamped to at least `min_width`.
    // fn speculative_width(
    //     count: u64,
    //     num_groups: u64,
    //     current_width: Width,
    //     change: u32,
    //     min_width: Width,
    // ) -> Width {
    //     let avg = count / num_groups.max(1);
    //     let floor = Width::from_max_value(avg).unwrap_or(Width::B1);

    //     let target_level = current_width.level() + change as usize;
    //     let ceiling = if target_level <= 6 {
    //         ALL_WIDTHS[target_level]
    //     } else {
    //         Width::U64
    //     };

    //     floor.max(min_width).min(ceiling)
    // }

    // /// Downscales by 1 step with forced widening.
    // ///
    // /// Used when a counter overflows: the width must increase by at
    // /// least one level.
    // pub(super) fn widen_by_one(&mut self) -> Result<(), super::Error> {
    //     let min = self.current.width.wider().ok_or(super::Error::Overflow)?;

    //     if self.buckets_empty() {
    //         self.current.width = min;
    //         self.shift_indices(1);
    //         return Ok(());
    //     }

    //     self.do_downscale(1, min)
    // }

    // /// Number of data words that may contain live counters.
    // fn data_word_count(&self) -> usize {
    //     if self.buckets_empty() {
    //         return 0;
    //     }
    //     let last_slot = (self.index_end - self.index_base) as usize;
    //     let spw = self.current.width.slots_per_word();
    //     last_slot / spw + 1
    // }

    // /// Rotates `data[0..len]` left by `mid` positions in place.
    // fn rotate_left(data: &mut [u64; N], mid: usize, len: usize) {
    //     if mid == 0 || len == 0 || mid >= len {
    //         return;
    //     }
    //     data[..len].rotate_left(mid);
    //     // Zero positions beyond the live range that may now contain
    //     // stale data from the rotation.
    //     for w in &mut data[len..N] {
    //         *w = 0;
    //     }
    // }
}
