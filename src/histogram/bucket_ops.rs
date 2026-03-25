// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket operations — widen and downscale.
//!
//! Downscale merges groups of `2^change` adjacent buckets by summing
//! their counters.  When alignment permits, the implementation uses
//! SWAR (SIMD Within A Register) to sum packed counters in parallel.
//! Otherwise it falls back to a scalar read–sum–write path.

use super::Histogram;
use super::swar::{swar_shift_up, swar_step};
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

        let group_size = 1i32 << change;
        let new_start = self.index_start >> change;
        let new_end = self.index_end >> change;
        let num_groups = (new_end - new_start + 1) as usize;

        let width = self.current.width;
        let spw = width.slots_per_word();

        // How many SWAR steps fit within a word.
        let swar_steps = if width == Width::U64 {
            0u32
        } else {
            change.min(spw.trailing_zeros())
        };

        // Check if SWAR alignment fits in the data array.
        let can_swar = swar_steps > 0 && {
            let misalign = self.index_base.rem_euclid(group_size) as usize;
            let shift = if misalign != 0 {
                group_size as usize - misalign
            } else {
                0
            };
            let new_last_slot = (self.index_end - self.index_base) as usize + shift;
            new_last_slot / spw < N
        };

        let spec_width = Self::speculative_width(
            self.stats.count,
            num_groups as u64,
            width,
            change,
            min_width.max(self.initial.width),
        );

        if can_swar {
            self.do_downscale_swar(
                change, new_start, new_end, num_groups, spec_width,
            )
        } else {
            self.do_downscale_scalar(
                change, new_start, new_end, num_groups, spec_width,
            )
        }
    }

    /// SWAR path: align, reduce in parallel, extract, repack.
    fn do_downscale_swar(
        &mut self,
        change: u32,
        new_start: i32,
        new_end: i32,
        num_groups: usize,
        spec_width: Width,
    ) -> Result<(), super::Error> {
        let group_size = 1i32 << change;
        let width = self.current.width;
        let spw = width.slots_per_word();
        let swar_steps = change.min(spw.trailing_zeros());
        let swar_width = ALL_WIDTHS[width.level() + swar_steps as usize];
        let swar_spw = swar_width.slots_per_word();
        let swar_bits = swar_width.bits();
        let swar_mask = swar_width.counter_max();
        let words_per_group: usize = if swar_steps < change {
            1 << (change - swar_steps)
        } else {
            1
        };

        // --- Align ---
        let misalign = self.index_base.rem_euclid(group_size) as usize;
        if misalign != 0 {
            let shift = group_size as usize - misalign;
            let new_last_slot = (self.index_end - self.index_base) as usize + shift;
            let needed_words = new_last_slot / spw + 1;
            swar_shift_up(&mut self.data[..needed_words], width, shift);
            self.index_base -= shift as i32;
        }

        // --- SWAR reduce ---
        let data_words = self.data_word_count();
        {
            let mut cw = width;
            for _ in 0..swar_steps {
                swar_step(&mut self.data[..data_words], cw);
                cw = ALL_WIDTHS[cw.level() + 1];
            }
        }

        // --- Find max group sum ---
        let swar_base = self.index_base >> change;
        let lane_offset = (new_start - swar_base) as usize;
        let mut max_sum: u64 = 0;

        if words_per_group > 1 {
            for g in 0..num_groups {
                let first_word = (lane_offset + g) * words_per_group;
                let mut acc: u64 = 0;
                for w in 0..words_per_group {
                    let wi = first_word + w;
                    if wi < data_words {
                        acc = acc
                            .checked_add(self.data[wi])
                            .ok_or(super::Error::Overflow)?;
                    }
                }
                max_sum = max_sum.max(acc);
            }
        } else {
            for g in 0..num_groups {
                let lane = lane_offset + g;
                let wi = lane / swar_spw;
                let li = lane % swar_spw;
                let val = (self.data[wi] >> (li * swar_bits)) & swar_mask;
                max_sum = max_sum.max(val);
            }
        }

        // --- Repack at target width ---
        let actual_width = Width::from_max_value(max_sum)
            .unwrap_or(Width::B1)
            .max(spec_width);
        let base = actual_width.word_start(new_start);
        let widened_data = self.data;
        self.data = [0u64; N];

        if words_per_group > 1 {
            for g in 0..num_groups {
                let first_word = (lane_offset + g) * words_per_group;
                let mut acc: u64 = 0;
                for w in 0..words_per_group {
                    let wi = first_word + w;
                    if wi < data_words {
                        acc = acc.wrapping_add(widened_data[wi]);
                    }
                }
                let slot = (new_start + g as i32 - base) as usize;
                Self::set_in(&mut self.data, slot, actual_width, acc);
            }
        } else {
            for g in 0..num_groups {
                let lane = lane_offset + g;
                let wi = lane / swar_spw;
                let li = lane % swar_spw;
                let val = (widened_data[wi] >> (li * swar_bits)) & swar_mask;
                let slot = (new_start + g as i32 - base) as usize;
                Self::set_in(&mut self.data, slot, actual_width, val);
            }
        }

        self.current.width = actual_width;
        self.index_base = base;
        self.index_start = new_start;
        self.index_end = new_end;
        Ok(())
    }

    /// Scalar fallback: read each counter individually, sum groups,
    /// write output.  Used when SWAR alignment doesn't fit or at U64
    /// width.
    fn do_downscale_scalar(
        &mut self,
        change: u32,
        new_start: i32,
        new_end: i32,
        num_groups: usize,
        spec_width: Width,
    ) -> Result<(), super::Error> {
        let group_size = 1i32 << change;
        let old_data = self.data;
        let old_width = self.current.width;
        let old_base = self.index_base;
        let old_start = self.index_start;
        let old_end = self.index_end;
        let old_spw = old_width.slots_per_word();
        let old_bits = old_width.bits();
        let old_mask = old_width.counter_max();

        // First pass: find the max group sum.
        let mut max_sum: u64 = 0;
        for g in 0..num_groups {
            let ni = new_start + g as i32;
            let lo = (ni * group_size).max(old_start);
            let hi = (ni * group_size + group_size - 1).min(old_end);
            let mut acc: u64 = 0;
            for idx in lo..=hi {
                let slot = if matches!(old_width, Width::U64) {
                    (idx - old_base).rem_euclid(N as i32) as usize
                } else {
                    (idx - old_base) as usize
                };
                let wi = slot / old_spw;
                let li = slot % old_spw;
                let val = (old_data[wi] >> (li * old_bits)) & old_mask;
                acc = acc.checked_add(val).ok_or(super::Error::Overflow)?;
            }
            max_sum = max_sum.max(acc);
        }

        // Determine output width.
        let actual_width = Width::from_max_value(max_sum)
            .unwrap_or(Width::B1)
            .max(spec_width);
        let base = actual_width.word_start(new_start);

        // Second pass: repack at target width.
        self.data = [0u64; N];
        for g in 0..num_groups {
            let ni = new_start + g as i32;
            let lo = (ni * group_size).max(old_start);
            let hi = (ni * group_size + group_size - 1).min(old_end);
            let mut acc: u64 = 0;
            for idx in lo..=hi {
                let slot = if matches!(old_width, Width::U64) {
                    (idx - old_base).rem_euclid(N as i32) as usize
                } else {
                    (idx - old_base) as usize
                };
                let wi = slot / old_spw;
                let li = slot % old_spw;
                let val = (old_data[wi] >> (li * old_bits)) & old_mask;
                acc = acc.wrapping_add(val);
            }
            let out_slot = (ni - base) as usize;
            Self::set_in(&mut self.data, out_slot, actual_width, acc);
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
