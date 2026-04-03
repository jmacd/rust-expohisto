// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Merge logic for combining histograms.
//!
//! 1. **Prepare** — Set width to `max(W_self, W_src)` and downscale
//!    self so the combined slot range fits in N words.
//! 2. **Merge** — For each dest word, repack its contributing source
//!    words (widen, cross-word sum, narrow, pack) then
//!    `swar_add_checked`. Overflow widens self and retries;
//!    `tm_log` is invariant under widening so group boundaries are
//!    stable.

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

        let new_count = self
            .checked_add_count(other.stats.count)
            .ok_or(Error::Overflow)?;

        self.merge_buckets(other);

        self.commit_stats(&Stats {
            count: new_count,
            sum: self.stats.sum + other.stats.sum,
            min: other.stats.min,
            max: other.stats.max,
        });
        Ok(())
    }

    /// Core merge: prepare self (width + scale), then word-by-word
    /// merge with on-the-fly repacking.
    ///
    /// Infallible: count overflow is checked by the caller, and all
    /// internal operations (downscale, widen) always succeed.
    fn merge_buckets<const M: usize>(&mut self, other: &Histogram<M>) {
        if other.buckets_empty() {
            return;
        }

        let src_scale = other.current.scale.scale();
        let src_width = other.current.width;
        let merge_width = self.current.width.max(src_width);
        let min_scale = self.current.scale.scale().min(src_scale);

        // Combined slot range at min_scale, using merge_width for
        // word capacity.
        let self_hl = self.slot_range_at_scale(min_scale);
        let other_hl = other.slot_range_at_scale(min_scale);
        let combined = self_hl.merge(other_hl);

        let word_hl = HighLow {
            low: merge_width.slot_to_word_index(combined.low),
            high: merge_width.slot_to_word_index(combined.high),
        };
        let extra = word_hl.change_steps(N);
        let target_scale = min_scale - extra as i32;

        // Three requirements on self_change:
        //  - range: enough to fit combined range in N words
        //  - width: enough to widen self to merge_width
        //  - tm_log >= 0: enough shift so each dest word has >= 1
        //    source word (only matters when self is wider than source)
        let range_change = (self.current.scale.scale() - target_scale).max(0) as u32;
        let width_change = (merge_width as u32).saturating_sub(self.current.width as u32);
        let self_change = range_change.max(width_change);

        if self.buckets_empty() {
            // No data to transform — just set scale and width.
            let new_scale = self.current.scale.scale() - self_change as i32;
            self.current.scale =
                crate::mapping::Scale::new(new_scale).expect("valid scale");
            self.current.width = merge_width;
        } else if self_change > 0 {
            self.downscale_by_min(self_change, merge_width);
        }

        // Ensure tm_log >= 0: after downscale, self's width and scale
        // are final. If self is wider than source + shift, we need
        // more shift so each dest word has at least one source word.
        let shift = (src_scale - self.current.scale.scale()) as u32;
        let dest_w = self.current.width as u32;
        if shift + (src_width as u32) < dest_w {
            let deficit = dest_w - shift - src_width as u32;
            if self.buckets_empty() {
                let new_scale = self.current.scale.scale() - deficit as i32;
                self.current.scale =
                    crate::mapping::Scale::new(new_scale).expect("valid scale");
            } else {
                self.downscale_by_min(deficit, merge_width);
            }
        }

        // Word-by-word merge with on-the-fly repacking.
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
    /// `tm_log = shift + src_w - dest_w` is the log2 of source words
    /// per dest word. It is invariant under self-widening, so group
    /// boundaries are stable across overflow retries.
    fn merge_words<const M: usize>(
        &mut self,
        other: &Histogram<M>,
        src_width: Width,
        src_scale: i32,
    ) {
        let shift0 = (src_scale - self.current.scale.scale()) as u32;
        let tm_log = shift0 + src_width as u32 - self.current.width as u32;

        debug_assert!(
            tm_log < 31,
            "tm_log={tm_log} (shift={shift0}, src_w={}, dest_w={}): \
             scale bounds violated",
            src_width as u32,
            self.current.width as u32,
        );

        let total_merge = 1i32 << tm_log;
        let aligned_start = other.word_start & !(total_merge - 1);
        let dest_lo = aligned_start >> tm_log;
        let dest_hi = other.word_end >> tm_log;

        self.extend_word_range(dest_lo, dest_hi);

        for dest_widx in dest_lo..=dest_hi {
            let src_start = dest_widx << tm_log;

            loop {
                let acc = match Self::repack_source(
                    other, src_width, src_scale,
                    self.current.scale.scale(), self.current.width,
                    src_start,
                ) {
                    Err(or_sums) => {
                        self.widen_to(Width::from_max_value(or_sums));
                        continue;
                    }
                    Ok(0) => break,
                    Ok(acc) => acc,
                };

                let dest_width = self.current.width;
                let didx = self.data_idx(dest_widx);
                if let Some(result) = swar_add_checked(self.data[didx], acc, dest_width) {
                    self.data[didx] = result;
                    break;
                }

                // Overflow: widen self and retry.
                let max_a = dest_width.or_fold_lanes(self.data[didx]);
                let max_b = dest_width.or_fold_lanes(acc);
                self.widen_to(Width::from_max_value(max_a + max_b));
            }
        }
    }

    /// Repack source words `[src_start .. src_start + total_merge)`
    /// into a single dest-width SWAR word.
    ///
    /// Decomposes the scale shift into in-word widening (src lanes
    /// toward U64) and cross-word grouping (sum adjacent words), then
    /// narrows the result to dest_width.
    ///
    /// Returns `Err(or_sums)` if the source sums overflow dest_width
    /// lanes (caller must widen self and retry).
    fn repack_source<const M: usize>(
        other: &Histogram<M>,
        src_width: Width,
        src_scale: i32,
        dest_scale: i32,
        dest_width: Width,
        src_start: i32,
    ) -> Result<u64, u64> {
        let shift = (src_scale - dest_scale) as u32;
        let in_word = shift.min(src_width.to_u64_widen_steps());
        let cross = shift - in_word;
        let cur = if in_word > 0 {
            src_width.wider_by(in_word).expect("capped at U64")
        } else {
            src_width
        };
        let narrow_steps = cur as u32 - dest_width as u32;
        let group = 1i32 << cross;
        let repack_count = 1i32 << narrow_steps;

        // Gather sub-group sums at `cur` width.
        let mut sums = [0u64; 64];
        let mut or_sums = 0u64;
        for r in 0..repack_count {
            let gstart = src_start + r * group;
            let mut value = 0u64;
            for g in 0..group {
                let widx = gstart + g;
                if widx >= other.word_start && widx <= other.word_end {
                    let word = other.data[other.data_idx(widx)];
                    value += if in_word > 0 {
                        widen(src_width, cur, word)
                    } else {
                        word
                    };
                }
            }
            sums[r as usize] = value;
            or_sums |= cur.or_fold_lanes(value);
        }

        // Source sums exceed dest counter capacity.
        if or_sums > dest_width.counter_max() {
            return Err(or_sums);
        }

        // Narrow and pack into one dest-width SWAR word.
        Ok(if narrow_steps > 0 {
            let chunk_bits = 64u32 >> narrow_steps;
            let mut acc = 0u64;
            for r in 0..repack_count {
                acc |= narrow(cur, dest_width, sums[r as usize])
                    << (r as u32 * chunk_bits);
            }
            acc
        } else {
            sums[0]
        })
    }

    /// Widen self to `new_width`, updating scale and all words.
    fn widen_to(&mut self, new_width: Width) {
        let old_width = self.current.width;
        let change = new_width.subtract(old_width) as u32;
        self.widen_words(old_width, new_width);
        self.change_scale(change);
        self.current.width = new_width;
    }
}
