// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Merge logic for combining histograms.

use super::{BucketDescriptor, HighLow, Histogram, Overflow, Stats, scale_reduction};

impl<const N: usize> Histogram<N> {
    /// Merges another histogram into this one.
    ///
    /// The source histogram may have a different pool size (`M`).
    /// When the destination is empty, the source's bucket width is
    /// adopted to avoid unnecessary widening steps.
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if the combined total count would exceed
    /// `u64::MAX`. See [`record()`](Self::record) for details.
    pub fn merge_from<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Overflow> {
        if other.literal {
            return self.merge_literal_from(other);
        }
        if !other.buckets_empty() && self.buckets_empty() {
            self.bucket_width = self.bucket_width.max(other.bucket_width);
        }
        self.merge_as_raw(other)
    }

    /// Extracts stats and bucket data from `other` and delegates to
    /// [`merge_from_raw`](Self::merge_from_raw).
    fn merge_as_raw<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Overflow> {
        self.merge_from_raw(
            &Stats {
                count: other.count(),
                sum: other.sum(),
                min: other.min(),
                max: other.max(),
            },
            &BucketDescriptor {
                scale: other.mapping.scale(),
                offset: other.index_start,
                len: other.range_len(),
            },
            |i| {
                let index = other.index_start + i as i32;
                other.bucket_get(other.slot_for(index))
            },
        )
    }

    /// Merges raw bucket data from an external source.
    ///
    /// # Arguments
    ///
    /// * `stats` — aggregate statistics (count, sum, min, max) of the source
    /// * `buckets` — bucket layout (scale, offset, len) of the source
    /// * `at` — returns the count at bucket position `i` (0-indexed from offset)
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if the combined total count would exceed
    /// `u64::MAX`. See [`record()`](Self::record) for details.
    pub fn merge_from_raw(
        &mut self,
        stats: &Stats,
        buckets: &BucketDescriptor,
        at: impl Fn(u32) -> u64,
    ) -> Result<(), Overflow> {
        if stats.count == 0 {
            return Ok(());
        }
        self.merge_raw_buckets(stats, buckets, &at)
    }

    fn merge_raw_buckets(
        &mut self,
        stats: &Stats,
        buckets: &BucketDescriptor,
        at: &impl Fn(u32) -> u64,
    ) -> Result<(), Overflow> {
        let new_count = self.checked_add_count(stats.count).ok_or(Overflow)?;
        let new_sum = self.sum() + stats.sum;

        if buckets.len > 0 {
            if self.literal {
                self.promote()?;
            }

            let other_end = buckets.offset + buckets.len as i32 - 1;
            let cap = self.bucket_capacity() as i32;
            let min_scale = self.mapping.scale().min(buckets.scale);

            let self_hl = self.index_range_at_scale(min_scale);
            let other_hl = {
                let shift = buckets.scale - min_scale;
                HighLow {
                    low: buckets.offset >> shift,
                    high: other_end >> shift,
                }
            };
            let hlp = self_hl.merge(other_hl);
            let target_scale = min_scale - scale_reduction(hlp, cap);

            self.downscale_to(target_scale)?;

            for i in 0..buckets.len {
                let count = at(i);
                if count == 0 {
                    continue;
                }
                self.retry_increment(count, |h| {
                    let shift = buckets.scale - h.mapping.scale();
                    (buckets.offset + i as i32) >> shift
                })?;
            }
        }

        self.commit_stats(new_sum, new_count, stats.min, stats.max);
        Ok(())
    }

    /// Merges literal values from another histogram into this one.
    fn merge_literal_from<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Overflow> {
        debug_assert!(other.literal);
        if other.count() == 0 {
            return Ok(());
        }
        self.merge_literal_values(other)
    }

    fn merge_literal_values<const M: usize>(
        &mut self,
        other: &Histogram<M>,
    ) -> Result<(), Overflow> {
        let new_count = self.checked_add_count(other.count()).ok_or(Overflow)?;
        let new_sum = self.sum() + other.sum();
        if self.literal {
            self.promote()?;
        }
        for &bits in other.literal_values() {
            self.update_buckets(f64::from_bits(bits), 1)?;
        }
        self.commit_stats(new_sum, new_count, other.min(), other.max());
        Ok(())
    }

    pub(super) fn index_range_at_scale(&self, target_scale: i32) -> HighLow {
        if self.buckets_empty() {
            return HighLow::empty();
        }
        let shift = self.mapping.scale() - target_scale;
        HighLow {
            low: self.index_start >> shift,
            high: self.index_end >> shift,
        }
    }
}
