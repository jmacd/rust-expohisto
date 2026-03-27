// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Merge logic for combining histograms.

// use super::{BucketDescriptor, Error, HighLow, Histogram, Stats};

// impl<const N: usize> Histogram<N> {
//     /// Merges another histogram into this one.
//     ///
//     /// The source histogram may have a different pool size (`M`).
//     /// When the destination is empty, the source's bucket width is
//     /// adopted to avoid unnecessary widening steps.
//     ///
//     /// # Errors
//     ///
//     /// Returns [`Error`] if the combined total count would exceed
//     /// `u64::MAX`. See [`record()`](Self::record) for details.
//     pub fn merge_from<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Error> {
//         if !other.buckets_empty() && self.buckets_empty() {
//             self.current.width = self.current.width.max(other.current.width);
//         }
//         self.merge_as_raw(other)
//     }

//     /// Extracts stats and bucket data from `other` and delegates to
//     /// [`merge_from_raw`](Self::merge_from_raw).
//     fn merge_as_raw<const M: usize>(&mut self, other: &Histogram<M>) -> Result<(), Error> {
//         self.merge_from_raw(
//             &other.stats(),
//             &BucketDescriptor {
//                 scale: other.current.scale.scale(),
//                 offset: other.index_start,
//                 len: other.range_len(),
//             },
//             |i| {
//                 let index = other.index_start + i as i32;
//                 other.bucket_get(other.slot_for(index))
//             },
//         )
//     }

//     /// Merges raw bucket data from an external source.
//     ///
//     /// # Arguments
//     ///
//     /// * `stats` — aggregate statistics (count, sum, min, max) of the source
//     /// * `buckets` — bucket layout (scale, offset, len) of the source
//     /// * `at` — returns the count at bucket position `i` (0-indexed from offset)
//     ///
//     /// # Errors
//     ///
//     /// Returns [`Error`] if the combined total count would exceed
//     /// `u64::MAX`. See [`record()`](Self::record) for details.
//     pub fn merge_from_raw(
//         &mut self,
//         stats: &Stats,
//         buckets: &BucketDescriptor,
//         at: impl Fn(u32) -> u64,
//     ) -> Result<(), Error> {
//         if stats.count == 0 {
//             return Ok(());
//         }
//         self.merge_raw_buckets(stats, buckets, &at)
//     }

//     fn merge_raw_buckets(
//         &mut self,
//         stats: &Stats,
//         buckets: &BucketDescriptor,
//         at: &impl Fn(u32) -> u64,
//     ) -> Result<(), Error> {
//         let new_count = self.checked_add_count(stats.count).ok_or(Error::Overflow)?;
//         let new_sum = self.stats.sum + stats.sum;

//         if buckets.len > 0 {
//             let other_end = buckets.offset + buckets.len as i32 - 1;
//             let cap = self.bucket_count();
//             let min_scale = self.current.scale.scale().min(buckets.scale);

//             let self_hl = self.index_range_at_scale(min_scale);
//             let other_hl = {
//                 let shift = buckets.scale - min_scale;
//                 HighLow {
//                     low: buckets.offset >> shift,
//                     high: other_end >> shift,
//                 }
//             };
//             let hlp = self_hl.merge(other_hl);
//             let target_scale = min_scale - hlp.change_steps(cap) as i32;

//             self.downscale_to(target_scale)?;

//             // Set count early so that any downscale triggered inside
//             // retry_increment sees the true total.  This maintains the
//             // invariant count ≥ any_bucket_value that do_downscale
//             // relies on for its safe-path decision.
//             self.stats.count = new_count;

//             for i in 0..buckets.len {
//                 let count = at(i);
//                 if count == 0 {
//                     continue;
//                 }
//                 self.retry_increment(count, |h| {
//                     let shift = buckets.scale - h.current.scale.scale();
//                     (buckets.offset + i as i32) >> shift
//                 })?;
//             }
//         }

//         self.commit_stats(&Stats {
//             count: new_count,
//             sum: new_sum,
//             min: stats.min,
//             max: stats.max,
//         });
//         Ok(())
//     }

//     pub(super) fn index_range_at_scale(&self, target_scale: i32) -> HighLow {
//         if self.buckets_empty() {
//             return HighLow::empty();
//         }
//         let shift = self.current.scale.scale() - target_scale;
//         HighLow {
//             low: self.index_start >> shift,
//             high: self.index_end >> shift,
//         }
//     }
// }
