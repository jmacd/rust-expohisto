// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Exact boundary computation for exponential histogram mapping.
//!
//! The core function [`compute_boundaries_exact`] produces the 52-bit
//! significand boundaries used by the lookup-table mapping algorithm.
//! Each boundary is verified with exact integer arithmetic (bignum `pow`)
//! to be within 1 ULP of the true mathematical value.

use crate::float64::SIGNIFICAND_MASK;
use rug::ops::Pow;
use rug::{Float, Integer};

/// Computes log bucket end boundaries as 52-bit significands using exact arithmetic.
///
/// `index_bits` is the scale parameter; the function computes `N = 2^index_bits`
/// boundaries.  `boundaries[k]` is the 52-bit significand of `2^(k/N)`,
/// with `boundaries[0] == 0`.  Each non-zero boundary is the smallest
/// integer significand `S` such that `S^N >= 2^(52*N + k)`, verified by
/// exact integer arithmetic.
pub fn compute_boundaries_exact(index_bits: u32) -> Vec<u64> {
    let n = 1usize << index_bits;

    // 128-bit precision is sufficient for index_bits up to 20:
    // repeated sqrt accumulates ~index_bits * 2^-128 error, well
    // below the 1-ULP threshold at 52-bit significand width.
    const PRECISION: u32 = 128;

    let mut boundaries = Vec::with_capacity(n);

    for position in 0..n {
        if position == 0 {
            // 2^(0/N) = 1.0, significand bits are all zero
            boundaries.push(0);
            continue;
        }

        // Compute 2^(position/N) via repeated square root:
        // start with 2^position (exact integer), then take sqrt
        // `index_bits` times to obtain 2^(position / 2^index_bits).
        let mut x = Float::with_val(PRECISION, 1u32) << position as u32;
        for _ in 0..index_bits {
            x = x.sqrt();
        }

        // Scale by 2^52 to get the IEEE significand + 2^52
        x <<= 52;
        let integer = x
            .to_integer()
            .unwrap_or_else(|| panic!("boundary float at position {position} is NaN/Inf"));
        let mut ieee_normalized = integer.to_u64().unwrap_or_else(|| {
            panic!("boundary at position {position} does not fit in u64: {integer}")
        });

        // Verify using exact integer arithmetic:
        // We need the smallest significand S such that S^N >= 2^(52*N + position).
        let compare_to = Integer::from(1u32) << (52 * n + position) as u32;

        let sig = Integer::from(ieee_normalized).pow(n as u32);
        if sig < compare_to {
            ieee_normalized += 1;
        }

        // Validate: (ieee_normalized - 1)^N must be < compare_to
        let sig_less_one = Integer::from(ieee_normalized - 1).pow(n as u32);
        assert!(
            sig_less_one < compare_to,
            "incorrect boundary at position {}: off by more than 1 ULP",
            position
        );

        boundaries.push(ieee_normalized & SIGNIFICAND_MASK);
    }

    boundaries
}

/// Computes the exact bucket index for a value at a given scale,
/// using precomputed boundaries.
///
/// `boundaries` must have `2^scale` entries as returned by
/// [`compute_boundaries_exact`].
pub fn map_to_index(value: f64, scale: i32, boundaries: &[u64]) -> i32 {
    let significand = crate::float64::get_significand(value);
    let exponent = crate::float64::get_unbiased_exponent(value);

    // Power of two: significand is 0, index is (exp << scale) - 1.
    // This handles the upper-inclusive case: value 2^exp belongs to
    // bucket (exp << scale) - 1.
    if significand == 0 {
        return (exponent << scale) - 1;
    }

    // Find the first k in 1..=n where significand < boundaries[k].
    // boundaries[0] = 0 and significand > 0, so we search from index 1.
    let lo = boundaries[1..].partition_point(|&b| b <= significand) + 1;

    // lo is the first k where significand < boundaries[k],
    // so significand is in [boundaries[lo-1], boundaries[lo]).
    let subbucket = (lo - 1) as i32;
    (exponent << scale) + subbucket
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundaries_position_0() {
        // Position 0 should always be 0 (2^0 = 1.0, significand = 0)
        for index_bits in 1..=10 {
            let boundaries = compute_boundaries_exact(index_bits);
            assert_eq!(
                boundaries[0], 0,
                "boundary[0] should be 0 at index_bits {}",
                index_bits
            );
        }
    }

    #[test]
    fn test_boundaries_monotonic() {
        for index_bits in 1..=10 {
            let boundaries = compute_boundaries_exact(index_bits);
            for i in 1..boundaries.len() {
                assert!(
                    boundaries[i] > boundaries[i - 1],
                    "boundaries not monotonic at index_bits {}, position {}",
                    index_bits,
                    i
                );
            }
        }
    }

    #[test]
    fn test_map_to_index_sanity() {
        // At scale 1, base = sqrt(2) ≈ 1.414
        // Bucket -1 contains (1/sqrt(2), 1]
        // Bucket 0 contains (1, sqrt(2)]
        // Bucket 1 contains (sqrt(2), 2]

        let boundaries = compute_boundaries_exact(1);

        // Power of two: 2^0 = 1.0 at scale 1 -> index = (0 << 1) - 1 = -1
        assert_eq!(map_to_index(1.0, 1, &boundaries), -1);

        // Power of two: 2^1 = 2.0 at scale 1 -> index = (1 << 1) - 1 = 1
        assert_eq!(map_to_index(2.0, 1, &boundaries), 1);

        // 1.1 is in (1, sqrt(2)] so index should be 0
        let idx = map_to_index(1.1, 1, &boundaries);
        assert_eq!(idx, 0, "1.1 at scale 1 should be in bucket 0, got {}", idx);

        // 1.5 is in (sqrt(2), 2] so index should be 1  (since 1.5 > 1.414)
        let idx = map_to_index(1.5, 1, &boundaries);
        assert_eq!(idx, 1, "1.5 at scale 1 should be in bucket 1, got {}", idx);

        // 0.9 is in (1/sqrt(2), 1] so index should be -1
        let idx = map_to_index(0.9, 1, &boundaries);
        assert_eq!(
            idx, -1,
            "0.9 at scale 1 should be in bucket -1, got {}",
            idx
        );
    }
}
