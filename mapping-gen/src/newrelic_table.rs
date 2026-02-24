// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Lookup table generation for exponential histogram mapping.

use crate::float64::SIGNIFICAND_MASK;
use rug::ops::Pow;
use rug::{Float, Integer};

/// Generated lookup tables for a specific size.
#[derive(Debug, Clone)]
pub struct LookupTables {
    /// log_2 of table size (N = 2^index_bits buckets).
    /// Also equals the maximum histogram scale supported.
    pub index_bits: u32,
    /// Number of log buckets (N = 2^index_bits).
    pub n: usize,
    /// Maps linear bucket index to approximate log bucket index.
    /// Has 2*N entries.
    pub log_bucket_index: Vec<u16>,
    /// End significand (52-bit) for each log bucket.
    /// Has N+1 entries (last is sentinel).
    pub log_bucket_end: Vec<u64>,
    /// Shift to convert 52-bit significand to linear bucket index.
    pub significand_shift: u32,
}

impl LookupTables {
    /// Generates lookup tables for a given number of index bits.
    pub fn generate(index_bits: u32) -> Self {
        let n = 1usize << index_bits;
        let mut boundaries = compute_boundaries_exact(n, index_bits);

        // Upper-inclusive adjustment: change boundary[0] from 0 to 1.
        // This makes the ">=" comparison naturally exclude significand=0
        // (exact powers of two) from sub-bucket 0, placing them in the
        // bucket below — matching OTel's upper-inclusive bucket semantics.
        // All other boundaries are ceilings of irrational values, so this
        // change only affects exact powers of two.
        debug_assert_eq!(boundaries[0], 0);
        boundaries[0] = 1;

        let log_bucket_index = compute_linear_to_log_mapping(n, &boundaries);

        // LOG_BUCKET_END stores the adjusted boundaries plus a sentinel.
        // boundaries[0] = 1 (upper-inclusive), boundaries[k] = significand of 2^(k/N) for k>0.
        let mut log_bucket_end = boundaries.clone();
        log_bucket_end.push(1u64 << 52); // sentinel = 2^52

        let significand_shift = 52 - (index_bits + 1); // +1 for 2N linear buckets

        Self {
            index_bits,
            n,
            log_bucket_index,
            log_bucket_end,
            significand_shift,
        }
    }

    /// Write the lookup tables as Rust source code.
    pub fn write_rust_source<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(
            w,
            "// Auto-generated lookup tables with {} index bits ({} buckets)",
            self.index_bits,
            self.n
        )?;
        writeln!(w)?;

        writeln!(w, "use crate::float64::SIGNIFICAND_WIDTH;")?;
        writeln!(w)?;

        writeln!(w, "/// Maximum histogram scale supported by this lookup table.")?;
        writeln!(w, "pub const LOOKUP_SCALE: i32 = {};", self.index_bits)?;
        writeln!(w)?;

        writeln!(w, "/// Number of bits to index into 2*N linear buckets.")?;
        writeln!(
            w,
            "const LINEAR_BUCKET_BITS: u32 = LOOKUP_SCALE as u32 + 1;"
        )?;
        writeln!(w)?;

        writeln!(
            w,
            "/// Shift to convert 52-bit significand to linear bucket index."
        )?;
        writeln!(
            w,
            "/// significand >> SIGNIFICAND_SHIFT yields an index in 0..2*N."
        )?;
        writeln!(
            w,
            "pub const SIGNIFICAND_SHIFT: u32 = SIGNIFICAND_WIDTH - LINEAR_BUCKET_BITS;"
        )?;
        writeln!(w)?;

        writeln!(
            w,
            "/// Maps linear bucket index to approximate log bucket index."
        )?;
        writeln!(
            w,
            "/// Linear bucket i starts at significand (i * 2^52) / (2 * N)."
        )?;
        writeln!(
            w,
            "pub const LOG_BUCKET_INDEX: [u16; 1 << LINEAR_BUCKET_BITS] = ["
        )?;
        for (i, &idx) in self.log_bucket_index.iter().enumerate() {
            if i % 16 == 0 {
                write!(w, "    ")?;
            }
            write!(w, "{:4},", idx)?;
            if i % 16 == 15 || i == self.log_bucket_index.len() - 1 {
                writeln!(w)?;
            }
        }
        writeln!(w, "];")?;
        writeln!(w)?;

        writeln!(w, "/// End significand (52-bit) for each log bucket (upper-inclusive).")?;
        writeln!(
            w,
            "/// boundary[0] = 1 handles upper-inclusive semantics: significand 0"
        )?;
        writeln!(
            w,
            "/// (exact powers of two) falls below boundary[0], mapping to sub-bucket -1."
        )?;
        writeln!(
            w,
            "/// Last entry is a sentinel (2^52) for boundary checks."
        )?;
        writeln!(
            w,
            "pub const LOG_BUCKET_END: [u64; (1 << LOOKUP_SCALE) + 1] = ["
        )?;
        for (i, &boundary) in self.log_bucket_end.iter().enumerate() {
            if i % 4 == 0 {
                write!(w, "    ")?;
            }
            if i == self.n {
                writeln!(w, "0x{:013X}, // sentinel = 2^52", boundary)?;
            } else {
                write!(w, "0x{:013X},", boundary)?;
                if i % 4 == 3 {
                    writeln!(w)?;
                }
            }
        }
        writeln!(w, "];")?;

        Ok(())
    }
}

/// Computes log bucket end boundaries as 52-bit significands using exact arithmetic.
pub fn compute_boundaries_exact(n: usize, index_bits: u32) -> Vec<u64> {
    // Use sufficient precision for exact computation
    // 128 bits is plenty for index_bits up to 20
    const PRECISION: u32 = 128;

    let mut boundaries = Vec::with_capacity(n);

    for position in 0..n {
        if position == 0 {
            // 2^(0/N) = 1.0, significand bits are all zero
            boundaries.push(0);
            continue;
        }

        // Compute 2^(position/N) using repeated square root
        // Start with 2^position, then take sqrt `index_bits` times
        let mut x = Float::with_val(PRECISION, 1u32) << position as u32;
        for _ in 0..index_bits {
            x = x.sqrt();
        }

        // Scale by 2^52 to get the IEEE significand + 2^52
        x <<= 52;
        let mut ieee_normalized = x.to_integer().unwrap().to_u64().unwrap();

        // Verify using exact integer arithmetic:
        // We need the smallest significand S such that S^N >= 2^(52*N + position)
        let compare_to = Integer::from(1u32) << (52 * n + position) as u32;

        // Check if ieee_normalized^N >= compare_to
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

/// Computes the exact bucket index for a value at a given scale.
pub fn map_to_index_exact(value: f64, scale: i32) -> i32 {
    let significand = crate::float64::get_significand(value);
    let exponent = crate::float64::get_normal_base2(value);

    // Power of two: significand is 0, index is (exp << scale) - 1
    // This handles the upper-inclusive case: value 2^exp is in bucket (exp << scale) - 1
    if significand == 0 {
        return (exponent << scale) - 1;
    }

    // For non-powers-of-two, we use the formula: index = floor(log2(value) * N)
    // where N = 2^scale.
    //
    // For value = (1 + s/2^52) * 2^exp, this is:
    // index = floor((exp + log2(1 + s/2^52)) * N)
    //       = exp * N + floor(log2(1 + s/2^52) * N)
    //
    // The subbucket floor(log2(1 + s/2^52) * N) is found by comparing against
    // exact boundaries: boundaries[k] = significand of 2^(k/N).
    //
    // If s is in [boundaries[k], boundaries[k+1]), then subbucket = k.

    let n = 1usize << scale;
    let boundaries = compute_boundaries_exact(n, scale as u32);

    // Binary search for the smallest k such that s < boundaries[k]
    // We search in 1..=n because boundaries[0] = 0 and s > 0.
    let mut lo = 1usize;
    let mut hi = n;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if significand >= boundaries[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // lo is the first k where s < boundaries[k]
    // So s is in [boundaries[lo-1], boundaries[lo]), and subbucket = lo - 1
    let subbucket = (lo - 1) as i32;

    (exponent << scale) + subbucket
}

/// Compute which log bucket each linear bucket's start falls into.
pub fn compute_linear_to_log_mapping(n: usize, boundaries: &[u64]) -> Vec<u16> {
    let linear_count = 2 * n;
    let mut mapping = Vec::with_capacity(linear_count);

    // Linear bucket i starts at significand = (i * 2^52) / (2N)
    for i in 0..linear_count {
        let linear_start = ((i as u128) << 52) / (linear_count as u128);
        let linear_start = linear_start as u64;

        // Find the log bucket containing this start
        let mut log_bucket = 0u16;
        for (k, &boundary) in boundaries.iter().enumerate() {
            if linear_start < boundary {
                log_bucket = k as u16;
                break;
            }
            if k == n - 1 {
                log_bucket = n as u16;
            }
        }
        mapping.push(log_bucket);
    }

    mapping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float64::{get_normal_base2, get_significand};

    #[test]
    fn test_generate_6_bits() {
        let tables = LookupTables::generate(6);
        assert_eq!(tables.index_bits, 6);
        assert_eq!(tables.n, 64);
        assert_eq!(tables.log_bucket_index.len(), 128);
        assert_eq!(tables.log_bucket_end.len(), 65); // 64 + sentinel
        assert_eq!(tables.significand_shift, 45); // 52 - 7
    }

    #[test]
    fn test_generate_8_bits() {
        let tables = LookupTables::generate(8);
        assert_eq!(tables.index_bits, 8);
        assert_eq!(tables.n, 256);
        assert_eq!(tables.log_bucket_index.len(), 512);
        assert_eq!(tables.log_bucket_end.len(), 257);
        assert_eq!(tables.significand_shift, 43); // 52 - 9
    }

    #[test]
    fn test_generate_10_bits() {
        let tables = LookupTables::generate(10);
        assert_eq!(tables.index_bits, 10);
        assert_eq!(tables.n, 1024);
        assert_eq!(tables.log_bucket_index.len(), 2048);
        assert_eq!(tables.log_bucket_end.len(), 1025);
        assert_eq!(tables.significand_shift, 41); // 52 - 11
    }

    #[test]
    fn test_generate_12_bits() {
        let tables = LookupTables::generate(12);
        assert_eq!(tables.index_bits, 12);
        assert_eq!(tables.n, 4096);
        assert_eq!(tables.log_bucket_index.len(), 8192);
        assert_eq!(tables.log_bucket_end.len(), 4097);
        assert_eq!(tables.significand_shift, 39); // 52 - 13
    }

    #[test]
    fn test_generate_14_bits() {
        let tables = LookupTables::generate(14);
        assert_eq!(tables.index_bits, 14);
        assert_eq!(tables.n, 16384);
        assert_eq!(tables.log_bucket_index.len(), 32768);
        assert_eq!(tables.log_bucket_end.len(), 16385);
        assert_eq!(tables.significand_shift, 37); // 52 - 15
    }

    #[test]
    fn test_boundaries_position_0() {
        // Position 0 should always be 0 (2^0 = 1.0, significand = 0)
        for index_bits in 1..=10 {
            let n = 1usize << index_bits;
            let boundaries = compute_boundaries_exact(n, index_bits);
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
            let n = 1usize << index_bits;
            let boundaries = compute_boundaries_exact(n, index_bits);
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
    fn test_lookup_correctness() {
        // Test that using the lookup tables gives correct results
        // Compare against floor(ln(value) * 2^scale / ln(2))
        for scale in 6..=10 {
            let tables = LookupTables::generate(scale);
            let scale_factor = (1u64 << scale) as f64 / core::f64::consts::LN_2;

            // Test values throughout the [1, 2) range
            let test_values: Vec<f64> = (1..tables.n)
                .map(|i| 2.0_f64.powf((i as f64 + 0.5) / tables.n as f64))
                .collect();

            for &value in &test_values {
                let significand = get_significand(value);
                let exponent = get_normal_base2(value);

                // Compute expected index using logarithm
                let expected_idx = (value.ln() * scale_factor).floor() as i32;

                // Use lookup
                let linear_idx = (significand >> tables.significand_shift) as usize;
                let approx_bucket = tables.log_bucket_index[linear_idx] as usize;
                let bucket = if significand >= tables.log_bucket_end[approx_bucket] {
                    approx_bucket + 1
                } else {
                    approx_bucket
                };
                let actual_idx = (exponent << scale) + bucket as i32 - 1;

                assert_eq!(
                    actual_idx, expected_idx,
                    "mismatch at scale={}, value={}: got {}, expected {}",
                    scale, value, actual_idx, expected_idx
                );
            }
        }
    }

    #[test]
    fn test_boundary_values_upper_inclusive() {
        // Test that values exactly on bucket boundaries fall into the correct bucket
        // For upper-inclusive: value = 2^(k/N) should give index = k - 1
        for scale in 6..=10 {
            let tables = LookupTables::generate(scale);

            for k in 1..tables.n {
                // Value exactly at 2^(k/N)
                let value = 2.0_f64.powf(k as f64 / tables.n as f64);
                let significand = get_significand(value);
                let exponent = get_normal_base2(value);

                // Use lookup — upper-inclusive semantics are baked into the
                // table (boundary[0] = 1), so no special case needed even
                // for significand == 0 (exact powers of two).
                let linear_idx = (significand >> tables.significand_shift) as usize;
                let approx_bucket = tables.log_bucket_index[linear_idx] as usize;
                let bucket = if significand >= tables.log_bucket_end[approx_bucket] {
                    approx_bucket + 1
                } else {
                    approx_bucket
                };
                let actual_idx = (exponent << scale) + bucket as i32 - 1;

                // For upper-inclusive boundaries:
                // Value 2^(k/N) is exactly on the boundary, should give index k-1
                // (since bucket k-1 includes values <= 2^(k/N))
                let expected_idx = k as i32 - 1;

                // Note: Due to floating point, value may be slightly above exact boundary
                // Allow being in bucket k-1 or k
                assert!(
                    actual_idx == expected_idx || actual_idx == expected_idx + 1,
                    "boundary value 2^({}/{}) should give index {} or {}, got {}",
                    k,
                    tables.n,
                    expected_idx,
                    expected_idx + 1,
                    actual_idx
                );
            }
        }
    }

    #[test]
    fn test_map_to_index_exact_sanity() {
        // Sanity check the exact function with known values

        // Power of two: 2^0 = 1.0 at scale 1 -> index = (0 << 1) - 1 = -1
        assert_eq!(map_to_index_exact(1.0, 1), -1);

        // Power of two: 2^1 = 2.0 at scale 1 -> index = (1 << 1) - 1 = 1
        assert_eq!(map_to_index_exact(2.0, 1), 1);

        // At scale 1, base = sqrt(2) ≈ 1.414
        // Bucket -1 contains (1/sqrt(2), 1]
        // Bucket 0 contains (1, sqrt(2)]
        // Bucket 1 contains (sqrt(2), 2]

        // 1.1 is in (1, sqrt(2)] so index should be 0
        let idx = map_to_index_exact(1.1, 1);
        assert_eq!(idx, 0, "1.1 at scale 1 should be in bucket 0, got {}", idx);

        // 1.5 is in (sqrt(2), 2] so index should be 1  (since 1.5 > 1.414)
        let idx = map_to_index_exact(1.5, 1);
        assert_eq!(idx, 1, "1.5 at scale 1 should be in bucket 1, got {}", idx);

        // 0.9 is in (1/sqrt(2), 1] so index should be -1
        let idx = map_to_index_exact(0.9, 1);
        assert_eq!(
            idx, -1,
            "0.9 at scale 1 should be in bucket -1, got {}",
            idx
        );
    }

    #[test]
    fn test_write_rust_source() {
        let tables = LookupTables::generate(6);
        let mut output = Vec::new();
        tables.write_rust_source(&mut output).unwrap();
        let source = String::from_utf8(output).unwrap();

        assert!(source.contains("LOOKUP_SCALE: i32 = 6"));
        assert!(source.contains("LOG_BUCKET_INDEX: [u16; 1 << LINEAR_BUCKET_BITS]"));
        assert!(source.contains("LOG_BUCKET_END: [u64; (1 << LOOKUP_SCALE) + 1]"));
        assert!(source.contains("SIGNIFICAND_SHIFT: u32 = SIGNIFICAND_WIDTH - LINEAR_BUCKET_BITS"));
    }

    /// Get the next representable f64 value greater than v.
    fn next_up(v: f64) -> f64 {
        f64::from_bits(v.to_bits() + 1)
    }

    /// Get the next representable f64 value less than v.
    fn next_down(v: f64) -> f64 {
        f64::from_bits(v.to_bits() - 1)
    }

    /// Reference lg implementation (pure logarithm)
    fn map_to_index_lg(value: f64, scale: i32, scale_factor: f64) -> i32 {
        if get_significand(value) == 0 {
            let exp = get_normal_base2(value);
            return (exp << scale) - 1;
        }
        (value.ln() * scale_factor).floor() as i32
    }

    /// Lookup table implementation at native table resolution (no downscaling)
    fn map_to_index_lookup_at_native_scale(value: f64, tables: &LookupTables) -> i32 {
        let significand = get_significand(value);
        let exponent = get_normal_base2(value);
        let scale = tables.index_bits as i32; // At native resolution, histogram scale = index_bits

        // Upper-inclusive semantics are baked into the table (boundary[0] = 1),
        // so no special case for significand == 0 is needed.
        let linear_idx = (significand >> tables.significand_shift) as usize;
        let approx_bucket = tables.log_bucket_index[linear_idx] as usize;
        let bucket = if significand >= tables.log_bucket_end[approx_bucket] {
            approx_bucket + 1
        } else {
            approx_bucket
        } as i32;

        (exponent << scale) + bucket - 1
    }

    #[test]
    fn test_lg_and_lookup_vs_exact() {
        // Compare lg (logarithm) and lookup against exact (boundary table) computation.
        //
        // Key insight from NewRelic: lookup table has NO computational error because
        // it uses integer operations only. The boundaries are verified with BigUint
        // during table generation.
        //
        // We precompute boundaries once per scale (not per value!) to avoid slowness.

        println!("=== Accuracy Analysis vs Exact ===");
        println!();

        const STEPS_PER_BOUNDARY: i32 = 20;

        // Test several table sizes
        for lookup_scale in [6u32, 8, 10] {
            let tables = LookupTables::generate(lookup_scale);
            let scale = lookup_scale as i32;
            let scale_factor = std::f64::consts::LOG2_E * (1u64 << scale) as f64;
            let buckets_per_octave = 1i32 << scale;

            // Precompute boundaries ONCE
            let n = 1usize << scale;
            let boundaries = compute_boundaries_exact(n, scale as u32);

            println!(
                "LOOKUP_SCALE={} ({} buckets per octave)",
                lookup_scale, tables.n
            );

            let mut total_tests = 0u64;
            let mut lg_errors = 0u64;
            let mut lookup_errors = 0u64;

            // Test boundaries within each octave from 2^-5 to 2^5
            for exp in -5..=5 {
                for bucket_offset in 0..buckets_per_octave {
                    let boundary_exp =
                        (exp as f64) + (bucket_offset as f64) / (buckets_per_octave as f64);
                    let boundary_value = 2.0_f64.powf(boundary_exp);

                    // Step through values near this boundary
                    let mut v = boundary_value;
                    for _ in 0..STEPS_PER_BOUNDARY {
                        v = next_up(v);
                        total_tests += 1;

                        // Exact using precomputed boundaries
                        let exact_idx = map_to_index_with_boundaries(v, scale, &boundaries);
                        let lg_idx = map_to_index_lg(v, scale, scale_factor);
                        let lookup_idx = map_to_index_lookup_at_native_scale(v, &tables);

                        if lg_idx != exact_idx {
                            lg_errors += 1;
                        }
                        if lookup_idx != exact_idx {
                            lookup_errors += 1;
                        }
                    }

                    let mut v = boundary_value;
                    for _ in 0..STEPS_PER_BOUNDARY {
                        v = next_down(v);
                        total_tests += 1;

                        let exact_idx = map_to_index_with_boundaries(v, scale, &boundaries);
                        let lg_idx = map_to_index_lg(v, scale, scale_factor);
                        let lookup_idx = map_to_index_lookup_at_native_scale(v, &tables);

                        if lg_idx != exact_idx {
                            lg_errors += 1;
                        }
                        if lookup_idx != exact_idx {
                            lookup_errors += 1;
                        }
                    }
                }
            }

            let lg_rate = (lg_errors as f64 / total_tests as f64) * 100.0;
            let lookup_rate = (lookup_errors as f64 / total_tests as f64) * 100.0;

            println!("  Tests: {}", total_tests);
            println!("  lg errors:     {:>6} ({:.4}%)", lg_errors, lg_rate);
            println!(
                "  lookup errors: {:>6} ({:.4}%)",
                lookup_errors, lookup_rate
            );
            println!();

            // Lookup should be exact (0 errors) - same boundaries used
            assert_eq!(
                lookup_errors, 0,
                "Lookup table should be exact at LOOKUP_SCALE={}, but had {} errors",
                lookup_scale, lookup_errors
            );
        }

        println!("Summary:");
        println!("  - Lookup is EXACT (0% error) - uses same precomputed boundaries");
        println!("  - lg has precision errors due to floating-point log()");
    }

    /// Fast exact mapping using precomputed boundaries
    fn map_to_index_with_boundaries(value: f64, scale: i32, boundaries: &[u64]) -> i32 {
        let significand = get_significand(value);
        let exponent = get_normal_base2(value);

        if significand == 0 {
            return (exponent << scale) - 1;
        }

        let n = boundaries.len();

        // Binary search for the smallest k such that significand < boundaries[k]
        let mut lo = 1usize;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if significand >= boundaries[mid] {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let subbucket = (lo - 1) as i32;
        (exponent << scale) + subbucket
    }
}
