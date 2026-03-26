// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket operations — widen and downscale.
//!
//! Downscale merges groups of `2^change` adjacent buckets by summing
//! their counters.  When groups fit in a single word the
//! implementation uses SWAR (SIMD Within A Register) to sum packed
//! counters in parallel.  When groups span multiple words (including
//! U64 width), SWAR widens each word to a single U64 value, then
//! each output group's words are summed directly by index.

use super::Histogram;
use super::swar::swar_step;
use super::width::{ALL_WIDTHS, Width};

impl<const N: usize> Histogram<N> {
    /// Downscales by `change` steps: merges groups of `2^change`
    /// adjacent bucket indices by summing their counters.
    ///
    /// `min_width` sets a floor on the output width (use the current
    /// width for a normal downscale, or the next wider width when
    /// a counter overflow forces widening).
    pub(super) fn do_downscale(
        &mut self,
        change: u32,
        min_width: Width,
    ) -> Result<(), super::Error> {
        debug_assert!(change >= 1);

        if self.buckets_empty() {
            return Ok(());
        }

        let new_start = self.index_start >> change;
        let new_end = self.index_end >> change;
        let num_groups = (new_end - new_start + 1) as usize;

        let width = self.current.width;
        let spw = width.slots_per_word();

        // How many SWAR doublings fit within a word.
        let swar_steps = if width == Width::U64 {
            0u32
        } else {
            change.min(spw.trailing_zeros())
        };

        let spec_width = Self::speculative_width(
            self.stats.count,
            num_groups as u64,
            width,
            change,
            min_width.max(self.initial.width),
        );

        if swar_steps == change {
            // Groups fit in one word — full SWAR, extract lanes.
            self.downscale_intra_word(
                change, swar_steps, new_start, new_end, num_groups, spec_width,
            )
        } else {
            // Groups span words — SWAR to U64, then direct
            // group-sum by word index.
            self.downscale_inter_word(
                change, swar_steps, new_start, new_end, num_groups, spec_width,
            )
        }
    }

    /// Intra-word path: SWAR reduces each group entirely within a
    /// word, then extract the widened lanes.
    ///
    /// `index_base` is always a multiple of `slots_per_word` (set by
    /// `word_start`).  Since `group_size ≤ spw`, `index_base` is
    /// also a multiple of `group_size`, so SWAR pair boundaries
    /// naturally align with group boundaries.
    fn downscale_intra_word(
        &mut self,
        change: u32,
        swar_steps: u32,
        new_start: i32,
        _new_end: i32,
        num_groups: usize,
        spec_width: Width,
    ) -> Result<(), super::Error> {
        let group_size = 1i32 << change;
        let width = self.current.width;
        let swar_width = ALL_WIDTHS[width.level() + swar_steps as usize];
        let swar_spw = swar_width.slots_per_word();
        let swar_bits = swar_width.bits();
        let swar_mask = swar_width.counter_max();

        // SWAR reduce.
        let data_words = self.data_word_count();
        {
            let mut cw = width;
            for _ in 0..swar_steps {
                swar_step(&mut self.data[..data_words], cw);
                cw = ALL_WIDTHS[cw.level() + 1];
            }
        }

        // Lane 0 corresponds to index_base (which is group-aligned).
        // Output group g is at lane (lane_offset + g).
        let lane_offset = ((new_start * group_size - self.index_base)
            / (1i32 << swar_steps)) as usize;

        // Find max group sum for width selection.
        let mut max_sum: u64 = 0;
        for g in 0..num_groups {
            let lane = lane_offset + g;
            let wi = lane / swar_spw;
            let li = lane % swar_spw;
            let val = (self.data[wi] >> (li * swar_bits)) & swar_mask;
            max_sum = max_sum.max(val);
        }

        // Repack at target width.
        let actual_width = Width::from_max_value(max_sum)
            .unwrap_or(Width::B1)
            .max(spec_width);
        let base = actual_width.word_start(new_start);
        let widened_data = self.data;
        self.data = [0u64; N];

        for g in 0..num_groups {
            let lane = lane_offset + g;
            let wi = lane / swar_spw;
            let li = lane % swar_spw;
            let val = (widened_data[wi] >> (li * swar_bits)) & swar_mask;
            let slot = (new_start + g as i32 - base) as usize;
            Self::set_in(&mut self.data, slot, actual_width, val);
        }

        self.current.width = actual_width;
        self.index_base = base;
        self.index_start = new_start;
        self.index_end = _new_end;
        Ok(())
    }

    /// Inter-word path: groups span multiple words.
    ///
    /// After SWAR-widening to U64 (if sub-U64), each data word holds
    /// one U64 counter.  For each output group we compute which data
    /// words fall in that group's index range and sum them directly.
    /// The group sums are written to `data[0..num_groups]` in-place
    /// (safe because output index g < first source index for g ≥ 1),
    /// then repacked at the target width.
    fn downscale_inter_word(
        &mut self,
        change: u32,
        swar_steps: u32,
        new_start: i32,
        new_end: i32,
        num_groups: usize,
        spec_width: Width,
    ) -> Result<(), super::Error> {
        let group_size = 1i32 << change;
        let width = self.current.width;
        let spw = width.slots_per_word() as i32;
        let data_words = self.data_word_count();

        // Step 1: SWAR-widen each word to a single U64 value.
        if width == Width::U64 {
            // Ring buffer → linearize so word w = index_base + w.
            let mut linear = [0u64; N];
            for idx in self.index_start..=self.index_end {
                let ring_slot = (idx - self.index_base).rem_euclid(N as i32) as usize;
                let dest = (idx - self.index_start) as usize;
                linear[dest] = self.data[ring_slot];
            }
            self.data = linear;
            self.index_base = self.index_start;
        } else {
            let mut cw = width;
            for _ in 0..swar_steps {
                swar_step(&mut self.data[..data_words], cw);
                cw = ALL_WIDTHS[cw.level() + 1];
            }
        }

        // Step 2: Sum each output group's words directly.
        //
        // After step 1, data word w covers original indices
        // [index_base + w*spw, index_base + (w+1)*spw).  Output
        // group g covers original indices
        // [(new_start+g)*group_size, (new_start+g+1)*group_size).
        //
        // Map group boundaries to word indices and sum the
        // overlapping range, clamped to [0, data_words).
        let mut max_sum: u64 = 0;
        for g in 0..num_groups {
            let group_lo = (new_start + g as i32) * group_size;
            let group_hi = group_lo + group_size; // exclusive
            // Word range (may extend beyond actual data).
            let w_lo = ((group_lo - self.index_base) / spw).max(0) as usize;
            let w_hi = (((group_hi - self.index_base) + spw - 1) / spw)
                .max(0) as usize;
            let w_lo = w_lo.min(data_words);
            let w_hi = w_hi.min(data_words);

            let mut acc: u64 = 0;
            for w in w_lo..w_hi {
                acc = acc
                    .checked_add(self.data[w])
                    .ok_or(super::Error::Overflow)?;
            }
            self.data[g] = acc;
            max_sum = max_sum.max(acc);
        }
        // Zero tail.
        for w in num_groups..N {
            self.data[w] = 0;
        }

        // Step 3: Repack the U64 group sums at the target width.
        let actual_width = Width::from_max_value(max_sum)
            .unwrap_or(Width::B1)
            .max(spec_width);
        let base = actual_width.word_start(new_start);
        let sums = self.data;
        self.data = [0u64; N];
        for (g, &val) in sums[..num_groups].iter().enumerate() {
            let slot = (new_start + g as i32 - base) as usize;
            Self::set_in(&mut self.data, slot, actual_width, val);
        }

        self.current.width = actual_width;
        self.index_base = base;
        self.index_start = new_start;
        self.index_end = new_end;
        Ok(())
    }

    /// Computes a speculative output width for the downscale pass.
    ///
    /// Uses two bounds to bracket the needed width:
    ///
    /// - **Floor** (pigeonhole): `count / num_groups` is the average
    ///   group sum; at least one group must be ≥ this value, so any
    ///   width below `from_max_value(average)` is provably too small.
    ///
    /// - **Ceiling** (worst case): each group sums at most `2^change`
    ///   counters each at `counter_max(current_width)`, so the output
    ///   width never exceeds `level(current_width) + change`.  When
    ///   floor == ceiling the width is exact and no repair is needed.
    ///
    /// The result is clamped to at least `min_width`.
    fn speculative_width(
        count: u64,
        num_groups: u64,
        current_width: Width,
        change: u32,
        min_width: Width,
    ) -> Width {
        // Floor: average group sum.  At least one group has ≥ this.
        let avg = count / num_groups.max(1);
        let floor = Width::from_max_value(avg).unwrap_or(Width::B1);

        // Ceiling: worst-case group sum.
        let target_level = current_width.level() + change as usize;
        let ceiling = if target_level <= 6 {
            ALL_WIDTHS[target_level]
        } else {
            Width::U64
        };

        // Pick the tightest bound, clamped to [min_width, ceiling].
        floor.max(min_width).min(ceiling)
    }

    /// Downscales by 1 step with forced widening.
    ///
    /// Used when a counter overflows: the width must increase by at
    /// least one level.
    pub(super) fn widen_by_one(&mut self) -> Result<(), super::Error> {
        let min = self.current.width.wider().ok_or(super::Error::Overflow)?;

        if self.buckets_empty() {
            self.current.width = min;
            self.shift_indices(1);
            return Ok(());
        }

        self.do_downscale(1, min)
    }

    /// Number of data words that may contain live counters.
    fn data_word_count(&self) -> usize {
        if self.buckets_empty() {
            return 0;
        }
        let last_slot = (self.index_end - self.index_base) as usize;
        let spw = self.current.width.slots_per_word();
        last_slot / spw + 1
    }

    /// Writes a counter into a data array at a physical slot.
    #[inline]
    fn set_in(data: &mut [u64; N], slot: usize, width: Width, value: u64) {
        let spw = width.slots_per_word();
        let bits = width.bits();
        let wi = slot / spw;
        let shift = (slot % spw) * bits;
        let mask = width.counter_max();
        let word = &mut data[wi];
        *word = (*word & !(mask << shift)) | ((value & mask) << shift);
    }
}
