// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Downscale.

use super::Histogram;
use super::width::Width;

impl<const N: usize> Histogram<N> {
    /// Downscales by `change` steps: merges groups of `2^change`
    /// adjacent bucket indices by summing their counters in place.
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

        // ...
    }
}
