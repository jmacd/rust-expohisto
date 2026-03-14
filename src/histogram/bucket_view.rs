// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Read-only view of bucket data in a histogram.

use super::bucket_width::BucketWidth;
use super::Histogram;

/// Read-only view of bucket data in a histogram.
///
/// Obtaining a `BucketView` via [`positive()`](Histogram::positive)
/// requires `&mut self` because literal-mode histograms are lazily
/// promoted to bucket mode on first read. After promotion, subsequent
/// reads are normal bucket lookups with no extra cost.
#[derive(Debug)]
pub struct BucketView<'a, const N: usize> {
    pub(super) hist: &'a Histogram<N>,
}

impl<const N: usize> BucketView<'_, N> {
    /// Returns the offset (smallest index).
    #[inline]
    pub fn offset(&self) -> i32 {
        // BucketView is only created after promotion, so we're in bucket mode.
        self.hist.index_start
    }

    /// Number of logical buckets in use.
    #[inline]
    pub fn len(&self) -> u32 {
        self.hist.bucket_range_len()
    }

    /// Returns the current counter width.
    #[inline]
    pub fn width(&self) -> BucketWidth {
        self.hist.bucket_width
    }

    /// Returns true if no buckets are in use.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of logical buckets available at the current width.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.hist.bucket_capacity()
    }

    /// Returns the count at position `pos` (0-indexed from offset).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= len()`.
    #[inline]
    pub fn at(&self, pos: u32) -> u64 {
        let len = self.len();
        assert!(
            pos < len,
            "BucketView::at: pos {} out of range (len {})",
            pos,
            len
        );
        let index = self.hist.index_start + pos as i32;
        self.hist.bucket_get(self.hist.slot_for(index))
    }

    /// Returns an iterator over bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, N> {
        BucketsIter {
            hist: self.hist,
            pos: 0,
            len: self.len(),
        }
    }
}

impl<'a, const N: usize> IntoIterator for &'a BucketView<'a, N> {
    type Item = u64;
    type IntoIter = BucketsIter<'a, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over bucket counts.
#[derive(Debug)]
pub struct BucketsIter<'a, const N: usize> {
    hist: &'a Histogram<N>,
    pos: u32,
    len: u32,
}

impl<const N: usize> Iterator for BucketsIter<'_, N> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let index = self.hist.index_start + self.pos as i32;
        let count = self.hist.bucket_get(self.hist.slot_for(index));
        self.pos += 1;
        Some(count)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for BucketsIter<'_, N> {}
