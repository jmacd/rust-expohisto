// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Merge logic for combining histograms.
//!
//! Uses a unified word-by-word merge with on-the-fly repacking:
//!
//! 1. **Phase 1** — Downscale self so the combined slot range fits in N words.
//! 2. **Phase 1.5** — Further downscale if needed so that `tm_log ≥ 0`
//!    (at least one source word maps to each dest word).
//! 3. **Phase 2** — For each dest word, repack its contributing source words
//!    (widen → cross-word sum → narrow → pack) then `swar_add_checked`.
//!    Overflow triggers iterative widening of self; `tm_log` is invariant
//!    under widening so group boundaries never change.

use super::swar::{narrow, swar_add_checked, widen};
use super::width::Width;
use super::{Error, HighLow, Histogram, Stats};

impl<const N: usize> Histogram<N> {
    /// Merges another histogram into this one.
    ///
    /// The source histogram may have a different pool size (`M`).
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

        self.merge_buckets(other);

        self.commit_stats(&Stats {
            count: new_count,
            sum: self.stats.sum + other.stats.sum,
            min: other.stats.min,
            max: other.stats.max,
        });
        Ok(())
    }

    /// Core merge: downscale self for range, ensure `tm_log ≥ 0`,
    /// then word-by-word merge with on-the-fly repacking.
    ///
    /// Infallible: count overflow is checked by the caller, and all
    /// internal operations (downscale, widen) always succeed.
    fn merge_buckets<const M: usize>(&mut self, other: &Histogram<M>) {
        if other.buckets_empty() {
            return;
        }

        let src_scale = other.current.scale.scale();
        let src_width = other.current.width;

        // When self is empty, adopt the source's width and scale
        // directly — there is no data to transform.
        if self.buckets_empty() {
            self.current.width = self.current.width.max(src_width);
            let target = self.current.scale.scale().min(src_scale);
            self.current.scale =
                crate::mapping::Scale::new(target).expect("valid scale");
        }

        // Phase 1: determine target scale from the combined range.
        //
        // Use the wider of the two widths for word-range calculation
        // so that Phase 1 accounts for the slot capacity the merge
        // will actually need. downscale_by_min prevents do_downscale
        // from narrowing back below src_width.
        let merge_width = self.current.width.max(src_width);
        let min_scale = self.current.scale.scale().min(src_scale);

        let self_hl = self.slot_range_at_scale(min_scale);
        let other_hl = other.slot_range_at_scale(min_scale);
        let combined = self_hl.merge(other_hl);

        let word_hl = HighLow {
            low: merge_width.slot_to_word_index(combined.low),
            high: merge_width.slot_to_word_index(combined.high),
        };
        let extra = word_hl.change_steps(N);
        let target_scale = min_scale - extra as i32;

        let self_change = self.current.scale.scale() - target_scale;
        if self_change > 0 && !self.buckets_empty() {
            self.downscale_by_min(self_change as u32, src_width)
                .expect("downscale is infallible");
        } else if self_change > 0 {
            // Empty histogram: just set the scale.
            self.current.scale =
                crate::mapping::Scale::new(target_scale).expect("valid scale");
        }

        // Phase 1.5: ensure tm_log ≥ 0 (every dest word has ≥ 1
        // source word mapping to it).
        //
        // tm_log = shift + src_w - dest_w (all in log₂-of-bits).
        // downscale_by(d) increases tm_log by exactly d.
        let shift = (src_scale - self.current.scale.scale()) as u32;
        let src_w = src_width as u32;
        let dest_w = self.current.width as u32;
        if shift + src_w < dest_w {
            let deficit = dest_w - shift - src_w;
            if !self.buckets_empty() {
                self.downscale_by_min(deficit, src_width)
                    .expect("downscale is infallible");
            } else {
                let new_scale = self.current.scale.scale() - deficit as i32;
                self.current.scale =
                    crate::mapping::Scale::new(new_scale).expect("valid scale");
            }
        }

        // Phase 2: word-by-word merge with on-the-fly repacking.
        self.merge_words(other, src_width, src_scale);
    }

    /// Pre-extend the word range to cover `[lo_widx, hi_widx]` and
    /// zero-fill any newly exposed words.
    fn extend_word_range(&mut self, lo_widx: i32, hi_widx: i32) {
        if self.buckets_empty() {
            self.word_start = lo_widx;
            self.word_end = hi_widx;
            self.word_base = lo_widx;
            return;
        }
        if lo_widx < self.word_start {
            for w in lo_widx..self.word_start {
                self.data[self.data_idx(w)] = 0;
            }
            self.word_start = lo_widx;
        }
        if hi_widx > self.word_end {
            for w in (self.word_end + 1)..=hi_widx {
                self.data[self.data_idx(w)] = 0;
            }
            self.word_end = hi_widx;
        }
    }

    /// Word-by-word merge with on-the-fly repacking.
    ///
    /// For each dest word, gathers its contributing source words,
    /// widens/sums/narrows them into a dest-width SWAR word, and adds
    /// via `swar_add_checked`. On overflow, widens self and retries.
    ///
    /// `tm_log = shift + src_w - dest_w` (the log₂ of source words
    /// per dest word) is invariant under self-widening, so group
    /// boundaries never change across retries.
    fn merge_words<const M: usize>(
        &mut self,
        other: &Histogram<M>,
        src_width: Width,
        src_scale: i32,
    ) {
        // tm_log is invariant: widening self increases both shift and
        // dest_w by the same amount, so they cancel.
        let shift0 = (src_scale - self.current.scale.scale()) as u32;
        let tm_log = shift0 + src_width as u32 - self.current.width as u32;

        // When tm_log ≥ 31, total_merge overflows i32. Since source
        // has at most M ≤ 250 words, ALL source words collapse into a
        // single dest slot. Fall back to retry_increment which handles
        // widening/downscaling naturally.
        if tm_log >= 31 {
            self.merge_words_collapsed(other, src_width, src_scale);
            return;
        }

        let total_merge = 1i32 << tm_log;

        // dest_word = src_word >> tm_log.
        let aligned_start = other.word_start & !(total_merge - 1);
        let dest_lo = aligned_start >> tm_log;
        let dest_hi = other.word_end >> tm_log;

        self.extend_word_range(dest_lo, dest_hi);

        for dest_widx in dest_lo..=dest_hi {
            let src_start = dest_widx << tm_log;

            'retry: loop {
                // Recompute decomposition (changes after widen).
                let shift = (src_scale - self.current.scale.scale()) as u32;
                let in_word_steps = shift.min(src_width.to_u64_widen_steps());
                let cross_steps = shift - in_word_steps;
                let cur = if in_word_steps > 0 {
                    src_width.wider_by(in_word_steps).expect("capped at U64")
                } else {
                    src_width
                };
                let dest_width = self.current.width;
                let narrow_steps = cur as u32 - dest_width as u32;
                let group = 1i32 << cross_steps;
                let repack_count = 1i32 << narrow_steps;
                let need_widen_src = in_word_steps > 0;

                // Pass 1: compute sub-group sums at `cur` width.
                // Track the or-fold to detect pre-narrow overflow.
                let mut sums = [0u64; 64]; // max repack_count = 2^6
                let mut or_sums = 0u64;

                for r in 0..repack_count {
                    let gstart = src_start + r * group;
                    let mut value = 0u64;
                    for g in 0..group {
                        let widx = gstart + g;
                        if widx >= other.word_start && widx <= other.word_end {
                            let word = other.data[other.data_idx(widx)];
                            value += if need_widen_src {
                                widen(src_width, cur, word)
                            } else {
                                word
                            };
                        }
                    }
                    sums[r as usize] = value;
                    or_sums |= cur.or_fold_lanes(value);
                }

                // Pre-narrow overflow: source sums exceed dest counter
                // capacity. Widen self so they fit, then retry.
                if or_sums > dest_width.counter_max() {
                    let new_width = Width::from_max_value(or_sums);
                    let change = new_width.subtract(dest_width) as u32;
                    self.widen_words(dest_width, new_width);
                    self.change_scale(change);
                    self.current.width = new_width;
                    continue 'retry;
                }

                // Pass 2: narrow and pack into one dest-width SWAR word.
                let acc = if narrow_steps > 0 {
                    let chunk_bits = 64u32 >> narrow_steps;
                    let mut a = 0u64;
                    for r in 0..repack_count {
                        a |= narrow(cur, dest_width, sums[r as usize])
                            << (r as u32 * chunk_bits);
                    }
                    a
                } else {
                    sums[0]
                };

                if acc == 0 {
                    break;
                }

                // Add to dest word.
                let didx = self.data_idx(dest_widx);
                match swar_add_checked(self.data[didx], acc, dest_width) {
                    Some(result) => {
                        self.data[didx] = result;
                        break;
                    }
                    None => {
                        // Widen self to fit the sum of dest + source.
                        let max_a = dest_width.or_fold_lanes(self.data[didx]);
                        let max_b = dest_width.or_fold_lanes(acc);
                        let new_width = Width::from_max_value(max_a + max_b);
                        let change = new_width.subtract(dest_width) as u32;
                        self.widen_words(dest_width, new_width);
                        self.change_scale(change);
                        self.current.width = new_width;
                        // Retry: repack at wider dest width.
                    }
                }
            }
        }
    }

    /// Fallback for extreme tm_log (≥ 31): all source words map to a
    /// single dest slot. Widen source to U64, sum, and retry_increment.
    fn merge_words_collapsed<const M: usize>(
        &mut self,
        other: &Histogram<M>,
        src_width: Width,
        src_scale: i32,
    ) {
        let need_widen = src_width != Width::U64;
        let mut sum = 0u64;
        for widx in other.word_start..=other.word_end {
            let word = other.data[other.data_idx(widx)];
            sum += if need_widen {
                widen(src_width, Width::U64, word)
            } else {
                word
            };
        }
        if sum == 0 {
            return;
        }
        let first_src_slot = src_width.word_to_slot_index(other.word_start);
        self.retry_increment(sum, |h| {
            let shift = src_scale - h.current.scale.scale();
            if shift <= 0 {
                first_src_slot << (-shift)
            } else if shift >= 31 {
                if first_src_slot >= 0 { 0 } else { -1 }
            } else {
                first_src_slot >> shift
            }
        })
        .expect("retry_increment is infallible after count check");
    }
}
