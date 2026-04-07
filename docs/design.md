# Design & Architecture

## Exponential Scale

The histogram divides the positive real line into buckets with boundaries at powers of the base:

$$
\text{base} = 2^{2^{-S}}
$$

where `S` is the scale. There are `N = 2^S` buckets per power of two, so the k-th sub-bucket boundary within an octave is `base^k = 2^(k/N)`. The relative width of each bucket is `base - 1`:

| Scale | Base | Buckets per power of 2 | Relative error |
|-------|------|------------------------|----------------|
| 10 | 1.00068 | 1024 | ~0.034% |
| 8 | 1.00271 | 256 | ~0.14% |
| 6 | 1.01089 | 64 | ~0.54% |
| 4 | 1.04427 | 16 | ~2.2% |
| 0 | 2.0 | 1 | ~41% |

Higher scales provide finer resolution at the cost of more buckets.

Bucket boundaries between 1 and 4, at different scales:

```text
Scale 0 (1 bucket per octave):
  1               2               4
  ├───────────────┤───────────────┤
  │   bucket -1   │   bucket 0    │   bucket 1

Scale 1 (2 buckets per octave, base = √2 ≈ 1.414):
  1       √2      2      2√2      4
  ├───────┤───────┤───────┤───────┤
  │  b -1 │  b 0  │  b 1  │  b 2  │   b 3

Scale 4 (16 buckets per octave, base = 2^(1/16) ≈ 1.044):
  1    1.04 1.09 1.14 1.19 ... 1.91  2    2.09 ...  4
  ├────┤────┤────┤────┤────···──┤────┤────┤────···──┤
  │b-1 │ b0 │ b1 │ b2 │ b3     │b15 │b16 │b17      │ b31
  ◄──────── 16 buckets ────────►◄──── 16 buckets ──►
```

Each scale doubles the number of buckets per octave and halves the relative error.

## Bucket Inclusivity

Per the OpenTelemetry specification, bucket boundaries are **upper-inclusive**:

> The bucket identified by `index` represents values **greater than** `base^index` and **less than or equal to** `base^(index+1)`.

This convention was adopted for Prometheus compatibility in a [specification change](https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261) that post-dates the original lookup table algorithms (see [Historical Notes](history.md)). This implementation re-engineers the boundary condition for upper-inclusive semantics, validated by an exhaustive test over all ~3 billion f64 values in the first sub-bucket at scale 16.

The practical effect: exact powers of two must fall into the bucket *below* what a naive `floor(log(value))` would suggest.

### Why the lookup table can handle this without a branch

The lookup table boundary arrays use `>=` comparisons against precomputed significand values. Within one octave (i.e., from `2^e` to `2^(e+1)`), the sub-bucket boundaries are at `2^(k/N)` for `k = 0, 1, ..., N-1` where `N = 2^scale`.

The crucial observation: **all sub-bucket boundaries except `k = 0` are irrational numbers.** The value `2^(k/N)` is irrational whenever `k/N` is not an integer (by the [Gelfond–Schneider theorem](https://en.wikipedia.org/wiki/Gelfond%E2%80%93Schneider_theorem)), so no IEEE 754 f64 can ever land exactly on these boundaries. The `>=` comparison against the integer ceiling of an irrational boundary always gives the correct bucket regardless of the inclusivity convention—the f64 is always strictly above or strictly below the ideal boundary.

The **only** boundary that's rational (and representable) is `2^(0/N) = 1.0`, which has significand `0`. This is the only case where upper- vs. lower-inclusive matters. By changing `boundary[0]` from `0` to `1` in the lookup table, the `>=` check naturally excludes `significand == 0` from sub-bucket 0, placing exact powers of two in the bucket below—exactly matching the upper-inclusive convention, without any branch.

In the lookup table, the upper-inclusive fix works as follows: `BOUNDARIES[1] = 1` (instead of `0`) ensures that `significand == 0` never passes the correction check, so `bucket` stays at `approx = 0` and the `- 1` in the final formula places exact powers of two one bucket lower.

For the **logarithm fallback** and **exponent mapping** (scale ≤ 0), an explicit correction is still needed since these don't use the boundary table:

```rust,ignore
// Exact powers of two: use exponent directly, subtract 1
if significand == 0 { return (exponent << scale) - 1; }
```

## Index Scale Algorithms

### Scale ≤ 0: Exponent Extraction

For non-positive scales, the bucket index is derived directly from the IEEE 754 exponent bits—no floating-point math required.

### Scale > 0: Lookup Table (default)

When a scale feature is enabled (default: `scale-8`), mapping uses integer-only operations:

1. Extract significand and exponent from the IEEE 754 representation
2. Use the significand to index into a precomputed lookup table
3. Apply boundary check(s) to correct the approximation
4. Combine with exponent to produce the final index

For scales beyond the table's maximum, the mapping automatically falls back to the built-in logarithm mapper.

### Scale > 0: Logarithm (fallback)

The logarithm mapper is always available and handles scales above the compiled table maximum (or all scales when no lookup table is enabled). It uses the standard formula:

```text
index = ceil(ln(value) × 2^scale / ln(2)) - 1
```

This is the upper-inclusive form. For non-powers of two (the vast majority of f64 values), `ceil(x) - 1 = floor(x)`, so the implementation uses `floor()` for the general case and an explicit correction for exact powers of two (see [Bucket Inclusivity](#bucket-inclusivity)).

## Lookup Table Design

The lookup table algorithm maps f64 values to bucket indices using integer-only operations at runtime. It was inspired by independently-developed algorithms from [Dynatrace](https://github.com/dynatrace-oss/dynahist) and [NewRelic](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java) (see [Historical Notes](history.md) for background). This implementation uses `2N` linear buckets with a single boundary correction.

### Data structures

Two tables are generated at compile time at the highest compiled scale `H` (where `N = 2^H`):

**`BOUNDARIES[N+3]`** — Exact sub-bucket boundary significands:

```text
[sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]
```

- `b[k]` is the 52-bit significand of `ceil(2^(k/N))` for `k > 0`
- `b[0] = 1` (not 0) implements upper-inclusive semantics (see [Bucket Inclusivity](#bucket-inclusivity))
- Trailing sentinels simplify bounds checking

**`INDEX_TABLE[2N]`** — Linear-to-log bucket mapping, derived from `BOUNDARIES`:

```text
SHIFT = 52 - log2(2N) = 51 - H

for i in 0..2N:
    INDEX_TABLE[i] = largest j such that BOUNDARIES[j+1] <= (i << SHIFT)
```

Each entry gives the approximate log bucket for the significand range starting at `i << SHIFT`.

### Scale algorithm

```text
fn map_to_index(value, scale) -> index:
    significand = bits 0..51 of value         // IEEE 754 significand
    exponent    = biased_exponent - 1023      // IEEE 754 exponent

    // Step 1: Linear approximation
    approx = INDEX_TABLE[significand >> SHIFT] // O(1) lookup

    // Step 2: Boundary correction (at most 1 needed)
    bucket = approx
    if significand >= BOUNDARIES[approx + 1]:
        bucket += 1

    // Step 3: Combine with exponent, downscale to requested scale
    fine_index = (exponent << H) + bucket - 1
    return fine_index >> (H - scale)
```

All operations are integer: bit extraction, array indexing, comparison, shift, and addition. No floating-point arithmetic is performed at runtime.

The `- 1` in step 3, combined with `b[0] = 1` in the boundary table, is how upper-inclusive semantics emerge: when `significand == 0` (exact power of two), `approx` is 0 and the correction doesn't fire, so the result is `(exponent << H) - 1` — one bucket lower than the naive formula.

### Why one correction suffices

Each linear bucket spans a width `W = 2^52 / 2N` of significand space. The number of log-scale boundaries that can fall within one linear bucket is at most:

$$
\Delta = N \cdot \log_2\!\left(1 + \frac{W}{2^{52}}\right) = N \cdot \log_2\!\left(1 + \frac{1}{2N}\right) \approx \frac{1}{2\ln 2} \approx 0.72 < 1
$$

Because `log` is concave, the worst case is always the **first** linear bucket (starting at significand 0), where the log function is steepest. Since the bound is < 1, at most one log-scale boundary can fall within any linear bucket, so **one correction suffices**.

### Exact boundary computation

The k-th sub-bucket boundary at scale `S` (where `N = 2^S`) is:

$$
\text{boundary}(k) = 2^{k/N} = 2^{k \cdot 2^{-S}}
$$

Note that `boundary(1) = 2^(1/N) = 2^(2^(-S))` is the exponential base (see [Exponential Scale](#exponential-scale)), and `boundary(k) = base^k`.

These are computed exactly at build time using repeated square roots and bignum verification (following [PR #3841](https://github.com/open-telemetry/opentelemetry-collector/pull/3841)):

```text
fn compute_boundary(k, S) -> u64:
    // Start with 2^k as a high-precision float
    x = 2^k                          // exact

    // Take sqrt S times: 2^k → 2^(k/2) → 2^(k/4) → ... → 2^(k/2^S)
    repeat S times:
        x = sqrt(x)

    // Convert to 52-bit significand
    candidate = floor(x × 2^52)

    // Verify and correct using exact bignum arithmetic:
    // We need the smallest integer c such that c^N ≥ 2^(52N + k)
    if candidate^N < 2^(52N + k):
        candidate += 1

    return candidate & SIGNIFICAND_MASK
```

The key identity is that applying `sqrt` `S` times divides the exponent by `2^S = N`:

$$
\underbrace{\sqrt{\sqrt{\cdots\sqrt{2^k}}}}_{S \text{ times}} = 2^{k/2^S} = 2^{k/N}
$$

The float computation uses 128-bit precision, which is far more than enough for the 52-bit significand. The bignum verification step guarantees the result is the exact ceiling — correct to 1 ULP (unit in last place) — regardless of any floating-point rounding in the sqrt chain.

### Multi-scale support

A table generated at scale `H` supports all scales `1..H` via arithmetic right shift. The index at scale `S` is simply `fine_index >> (H - S)`. This works because the exponential histogram has a nested structure: bucket `k` at scale `S` contains buckets `2k` and `2k+1` at scale `S+1`.

## Crate Structure

```text
otel-expohisto/
├── src/
│   ├── lib.rs            # Crate root and public re-exports
│   ├── histogram/        # Core histogram implementation
│   │   ├── mod.rs        #   Histogram<N>: update, merge, downscale, widen
│   │   ├── view.rs       #   HistogramView, BucketView, BucketsIter
│   │   ├── width.rs      #   Width enum and SlotAddr
│   │   ├── downscale.rs  #   Downscale and widen operations
│   │   ├── merge.rs      #   merge_from
│   │   ├── swar.rs       #   SWAR pairwise merge, shift, narrow-compact
│   │   ├── quantile.rs   #   QuantileIter: CDF-walk quantile estimation (feature = "quantile")
│   │   └── tests.rs      #   140 unit tests
│   ├── mapping.rs        # Scale-to-index dispatch (Scale struct)
│   ├── exponent.rs       # Scale ≤ 0: IEEE 754 exponent extraction
│   ├── logarithm.rs      # Scale > 0 fallback: ln()-based mapping
│   ├── lookup.rs         # Lookup table mapping (2N linear buckets, 1 correction)
│   └── float64.rs        # IEEE 754 bit-manipulation helpers
├── mapping-gen/          # Build-time sub-crate: lookup table generation
├── build.rs              # Generates lookup_tables.rs from mapping-gen
├── benches/              # Criterion benchmarks (mapping, downscale, literal, sub_byte)
├── examples/             # Usage examples (basic, merge, sizing, quick_start_test)
└── fuzz/                 # Fuzz targets (histogram_oracle, merge_oracle, stateful_oracle, rng_stress)
```

- **`otel-expohisto`**: Main library with histogram and mapping implementations
- **`mapping-gen`**: Sub-crate for generating lookup tables (used at build time via `build.rs`)

The `mapping-gen` crate can be tested independently:

```bash
cd mapping-gen && cargo test
```
