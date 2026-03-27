// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Read-only view of bucket data in a histogram.

use super::Histogram;
use super::width::{SlotAddr, Width};

/// Read-only view of bucket data in a histogram.
#[derive(Debug)]
pub struct BucketView<'a, const N: usize> {
    pub(super) hist: &'a Histogram<N>,
}

impl<const N: usize> BucketView<'_, N> {
    /// Returns the base bucket offset.
    #[inline]
    pub fn offset(&self) -> i32 {
        self.hist.word_start
    }

    /// Number of logical buckets in use.
    #[inline]
    pub fn bucket_count(&self) -> u32 {
        self.hist.current_slot_count() as u32
    }

    /// Returns the current counter width.
    #[inline]
    pub fn width(&self) -> Width {
        self.hist.current.width
    }

    /// Returns true if no buckets are in use.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hist.buckets_empty()
    }

    /// Returns an iterator over bucket counts.
    #[inline]
    pub fn iter(&self) -> BucketsIter<'_, N> {
        BucketsIter {
            hist: self.hist,
            addr: self.is_empty().then(|| self.hist.start_addr()),
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
    addr: Option<SlotAddr<'a>>,
}

impl<const N: usize> Iterator for BucketsIter<'_, N> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let count = self.addr.as_ref().map(|addr| self.hist.bucket_get(&addr));
        if let Some(addr) = self.addr.take() {
            self.addr = addr.next_addr(self.hist.word_end);
        }
        count
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.addr {
            None => (0, None),
            Some(addr) => {
                let remaining = self.hist.size_hint(addr);
                (remaining, Some(remaining))
            }
        }
    }
}

impl<const N: usize> ExactSizeIterator for BucketsIter<'_, N> {}
