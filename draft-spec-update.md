# Draft Specification Update: ExponentialHistogram Enhancements

> **Status**: Draft for discussion
> **Scope**: Updates to the [Metrics Data Model § ExponentialHistogram][data-model]
> and the [Metrics SDK § Base2 Exponential Bucket Histogram Aggregation][sdk-spec]
>
> This document proposes two enhancements to the ExponentialHistogram
> specification:
>
> 1. **Lookup Table Mapping**: An exact, integer-only alternative to the
>    logarithm-based mapping function for positive scales.
> 2. **Variable-Width Counters**: A compact representation using sub-byte
>    bucket counters that trade counter depth for slot count, enabling
>    allocation-free fixed-size histograms.

[data-model]: https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram
[sdk-spec]: https://opentelemetry.io/docs/specs/otel/metrics/sdk/#base2-exponential-bucket-histogram-aggregation

---

## 1. Lookup Table Mapping Function

### Motivation

The existing specification describes two mapping approaches for
ExponentialHistogram:

- **Scale ≤ 0**: Extract and shift the IEEE 754 exponent (exact).
- **Scale > 0**: Use the built-in logarithm function (inexact).

The logarithm approach has known limitations:

> The use of `math.Log()` to calculate the bucket index is not
> guaranteed to be exactly correct near powers of two. Values near a
> boundary could be mapped into the incorrect bucket due to inaccuracy.
> Defining an exact mapping function is out of scope for this document.

This section defines an exact mapping function for positive scales
using a compile-time generated lookup table.

### Overview

The lookup table algorithm maps IEEE 754 double-precision floating-point
values to bucket indices using only integer operations at runtime: bit
extraction, array indexing, integer comparison, shift, and addition.
No floating-point arithmetic is performed.

The key insight is that the significand space `[0, 2^52)` can be
partitioned into `2N` equal-width linear buckets (where `N = 2^S` for
table scale `S`), and because the logarithm function is concave, at most
one exponential bucket boundary can fall within any linear bucket. This
means a linear lookup gives the correct bucket index or is off by exactly
one, requiring a single correction step.

### Data Structures

Two tables are generated at compile time for a chosen maximum table
scale `H` (where `N = 2^H`):

**`BOUNDARIES[N+3]`** — Exact sub-bucket boundary significands:

```
BOUNDARIES = [sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]
```

where `b[k]` for `k > 0` is the 52-bit significand of the smallest
IEEE 754 double whose value is ≥ `2^(k/N)`. That is:

```
b[k] = significand_bits(ceil_to_double(2^(k/N)))
```

The entry `b[0] = 1` (rather than 0) implements upper-inclusive bucket
semantics: it ensures that exact powers of two are mapped into the
bucket below, consistent with the specification's bucket inclusivity rule.

Trailing sentinel values `2^52` simplify bounds checking by eliminating
special cases at the end of the array.

**`INDEX_TABLE[2N]`** — Linear-to-exponential bucket mapping:

```
SHIFT = 52 - log2(2N) = 51 - H

for i in 0..2N:
    INDEX_TABLE[i] = max { j : BOUNDARIES[j+1] <= (i << SHIFT) }
```

Each entry maps a linear significand region to the largest exponential
bucket index whose boundary does not exceed the region's starting
significand. Since at most one boundary can fall within any linear
region, this gives the correct index or one less than the correct index.

### Algorithm

Given a positive IEEE 754 double-precision value with significand `s`
(52-bit unsigned integer, with implicit leading 1 removed) and unbiased
exponent `e` (signed integer, with the IEEE 754 bias of 1023 removed):

```
function MapToIndex(s, e, scale):
    // Step 1: Linear approximation at the table scale H
    linear_idx = s >> SHIFT           // index into INDEX_TABLE
    approx = INDEX_TABLE[linear_idx]  // approximate exponential bucket

    // Step 2: Boundary correction (at most one step needed)
    bucket = approx
    if s >= BOUNDARIES[approx + 1]:
        bucket = bucket + 1

    // Step 3: Combine with exponent, adjust to requested scale
    fine_index = (e << H) + bucket - 1
    return fine_index >> (H - scale)
```

**Step 1** partitions the 52-bit significand space into `2N` equal
regions and looks up the pre-computed exponential bucket for each region.
This is a constant-time array access.

**Step 2** corrects for the case where an exponential bucket boundary
falls within the linear region. Since the logarithm is concave, the
linear approximation can only underestimate by at most 1 (see
[Correctness Proof](#correctness-proof) below). A single comparison
against the pre-computed exact boundary suffices.

**Step 3** assembles the full index by combining the within-octave
bucket with the octave (exponent). The `- 1` implements upper-inclusive
semantics: when `s = 0` (exact power of two), `approx = 0` and the
correction does not fire, so the result is `(e << H) - 1` — one bucket
below the boundary, as required by the specification.

The final right-shift `>> (H - scale)` downscales from the table's
native resolution to the requested scale. This preserves the perfect
subsetting property: the result is identical to computing directly at
the requested scale.

### Correctness Proof

The maximum number of exponential bucket boundaries that can fall within
a single linear bucket of width `W = 2^52 / (2N)` is:

```
Δ = N × log₂(1 + W/2^52) = N × log₂(1 + 1/(2N)) ≈ 1/(2·ln2) ≈ 0.721
```

Because `log₂` is concave, the worst case occurs at the start of the
significand range (significand = 0), where the function is steepest.
Since `Δ < 1`, at most one exponential boundary can fall within any
linear bucket. Therefore, the linear approximation in Step 1 is correct
or off by exactly one, and a single comparison in Step 2 suffices.

This bound holds for all scales `H ≥ 1`. The value `≈ 0.721` is
independent of the table scale — doubling the table scale doubles both
`N` and the number of linear buckets, keeping `Δ` constant.

### Table Size and Scale Selection

The table dimensions are determined by the maximum table scale `H`:

| Table scale H | N = 2^H | INDEX_TABLE entries | BOUNDARIES entries | Total size |
|---------------|---------|--------------------|--------------------|------------|
| 4             | 16      | 32                 | 19                 | ~100 bytes |
| 6             | 64      | 128                | 67                 | ~400 bytes |
| 8 (default)   | 256     | 512                | 259                | ~1.5 KiB |
| 10            | 1024    | 2048               | 1027               | ~6 KiB |

The table supports all scales from 1 to H inclusive. Scales above H
fall back to the logarithm method. Scale 0 and below use the exponent
extraction method (unchanged from the current specification).

A table scale of 8 (the recommended default) provides 256 sub-buckets
per octave with ~0.14% relative error. This covers the practical range
of scales used in metrics collection. Higher table scales (10 or above)
may be useful for scientific or financial applications requiring
sub-0.1% bucket resolution.

Implementations SHOULD select a table scale that covers their maximum
configured `MaxScale`. Implementations MAY generate the table at compile
time or at initialization time.

### Properties

1. **Exactness**: The mapping is exact for all representable IEEE 754
   double-precision values. There is no floating-point rounding error.

2. **Constant time**: The algorithm performs a fixed number of
   operations (one shift, one array access, one comparison, one
   conditional increment, two shifts, one addition) regardless of input.

3. **No floating-point arithmetic**: All runtime operations are integer.
   This makes the algorithm suitable for `no_std` environments,
   embedded systems, and environments where floating-point performance
   is unpredictable (e.g., soft-float targets).

4. **Perfect subsetting**: Computing at table scale H and right-shifting
   produces identical results to computing directly at any lower scale,
   preserving the exponential histogram's perfect subsetting property.

### Comparison with the Logarithm Method

| Property | Logarithm | Lookup Table |
|----------|-----------|--------------|
| Exactness | Approximate (off-by-one near boundaries) | Exact |
| Runtime ops | 1 `ln()` + 1 `floor()`/`ceil()` + multiply | 2 shifts + 1 lookup + 1 compare + add |
| Floating-point | Required | Not used at runtime |
| Power-of-two special case | Required | Handled by table design |
| Code complexity | ~10 lines | ~10 lines + table data |
| Memory | None | 1.5 KiB (scale-8 default) |
| `no_std` / embedded | Requires libm or equivalent | Integer-only |

Implementations SHOULD prefer the lookup table method when the maximum
scale is known at compile time or initialization time. The logarithm
method remains suitable as a fallback for dynamically-chosen scales that
exceed the table range.

---

## 2. Variable-Width Bucket Counters

### Motivation

The current specification defines ExponentialHistogram with a fixed
`MaxSize` (default 160) of full-width counters, typically 64-bit
unsigned integers. For 160 buckets, this requires at minimum 160 × 8 =
1,280 bytes for the counter array alone (plus per-range offset and
metadata), and typically more due to dynamic array allocation overhead.

Variable-width counters provide an alternative representation where the
total storage is a fixed-size pool of `N` 64-bit words. Bucket counters
start at 1 bit per counter (giving `N × 64` counters) and widen
automatically as counts grow, using SWAR (SIMD Within A Register)
techniques for sub-byte counter manipulation.

This enables **allocation-free, fixed-size histogram structures** that
fit in a cache line or a small number of cache lines, suitable for
high-frequency metrics collection in performance-critical paths.

### Counter Width Levels

The counter pool consists of `N` words of 64 bits each, giving a total
of `N × 64` bits. These bits are partitioned into counters of uniform
width:

| Width | Bits per counter | Max count | Available slots (N=10) | Available slots (N=16) |
|-------|-----------------|-----------|------------------------|------------------------|
| B1    | 1               | 1         | 640                    | 1024                   |
| B2    | 2               | 3         | 320                    | 512                    |
| B4    | 4               | 15        | 160                    | 256                    |
| U8    | 8               | 255       | 80                     | 128                    |
| U16   | 16              | 65,535    | 40                     | 64                     |
| U32   | 32              | ~4.3×10⁹  | 20                     | 32                     |
| U64   | 64              | ~1.8×10¹⁹ | 10                     | 16                     |

Note that at B4 width, `Histogram<10>` (10 words, 80 bytes of data)
provides exactly 160 slots — the same as the specification's default
`MaxSize` — in 640 bits rather than 10,240 bits, a **16× reduction**
in counter storage.

### Relationship to MaxSize

The `MaxSize` concept from the current specification maps to the
variable-width model as follows:

```
effective_max_size = N × 64 / bits_per_counter
```

As counter width increases, the effective `MaxSize` decreases. The
histogram automatically downscales (reduces `scale`) to fit the bucket
range within the reduced slot count, maintaining the ideal scale
invariant from the current specification.

This is equivalent to the current specification's behavior when
`MaxSize` is reduced, but happens automatically and reversibly within a
single histogram instance.

### Scale and Width Dynamics

When a measurement causes either a counter overflow or a range overflow,
the histogram adjusts:

1. **Counter overflow**: A bucket count exceeds the current counter
   maximum. The histogram widens all counters to the next width level
   (e.g., B4 → U8), which halves the available slots. If the current
   bucket range exceeds the new slot count, a downscale is also
   performed.

2. **Range overflow**: The measurement's bucket index falls outside the
   current range. The histogram downscales (reduces `scale` by 1),
   which halves the index span by merging adjacent bucket pairs. If the
   range still doesn't fit, the process repeats.

Each width increase costs approximately one scale level (because halving
slots is equivalent to halving the index range, which is what one
downscale step achieves).

### Sizing Guidance

The relationship between histogram size, counter width, and effective
resolution is summarized by:

```
range_in_octaves = (N × 64 / bits_per_counter) / 2^scale
relative_error ≈ ln(2) / 2^(scale+1) ≈ 35% / 2^scale
```

The following table shows the effective range and error for common
configurations at U16 width (the typical steady-state for measurement
counts in the range 1,000–100,000):

| Size | Bytes | U16 slots | Scale | Range | Contrast | Rel. error |
|------|-------|-----------|-------|-------|----------|------------|
| N=10 | 128*  | 40        | 2     | 10 octaves | 1,024× | 8.6% |
| N=10 | 128*  | 40        | 3     | 5 octaves  | 32×    | 4.3% |
| N=16 | 176*  | 64        | 3     | 8 octaves  | 256×   | 4.3% |
| N=16 | 176*  | 64        | 4     | 4 octaves  | 16×    | 2.2% |
| N=32 | 304*  | 128       | 4     | 8 octaves  | 256×   | 2.2% |

(*) Total struct size including metadata fields.

For comparison, the current 160-bucket default at scale 3 provides
20 octaves of range with ~4.3% relative error, using at least 1,280
bytes for counters alone. `Histogram<16>` at U16/scale 3 achieves the
same error rate with 8 octaves of range in 176 bytes total — sufficient
for most metrics workloads and **7× smaller**.

### Sizing Intuition: Steps Down From the Ideal Scale

The behavior of a fixed-size variable-width histogram can be reasoned
about with a single observation:

> **The bucket with the highest density grows linearly with the
> measurement count, and its growth rate determines how often counters
> widen.**

Call that growth rate `p_max` — the fraction of measurements that land
in the densest bucket. Different distributions give different values:

- **Uniform** over the histogram's range: `p_max = 1 / slots`. Every
  bucket grows at the same rate; widening is the slowest possible.
- **Normal** with coefficient of variation CV: at scale 0 (one bucket
  per octave), roughly `0.4 / CV` of measurements land in the modal
  bucket. Each finer scale step halves the bucket width and so halves
  `p_max`. Smaller CV concentrates more mass in the mode and forces
  widening sooner.
- **Heavy-tailed** (log-normal, exponential, Pareto): `p_max` is
  smaller than the normal case at the same scale, because mass is
  spread over more octaves. These distributions widen counters more
  slowly but consume more octaves of range.

Because counter widths progress B1 → B2 → B4 → U8 → U16 → U32 → U64
(roughly doubling capacity each step), a histogram receiving `n`
measurements widens until the modal counter holds `n · p_max`.
Each widening **halves the available slot count**, which is equivalent
to **one downscale step from the ideal scale**.

The terminal configuration is therefore determined by a simple budget:

```
steps_down_from_ideal  ≈  widenings_from_p_max  +  downscales_for_contrast
```

where:

- `widenings_from_p_max` is the number of width levels needed to hold
  `n · p_max` (roughly one level per squaring of `n`).
- `downscales_for_contrast` is the number of scale reductions needed
  to fit the distribution's observed span into the remaining slots.

For typical observability workloads — normal-ish core distributions
with CV in 0.05–0.20, contrast in the 100×–10,000× range, and
n ≈ 10³–10⁵ per collection interval — the histogram settles at
**U16 width with scale 2–4**, two to three steps below the ideal
scale supported by the slot count alone. Heavier tails or larger `n`
push the terminal width to U32; wider CV or smaller `n` permits a
finer terminal scale.

### A Calculator Recipe for Sizing

If you know your **contrast** `C` (ratio of largest to smallest value
you care about, e.g. `p99 / p1`), your **measurement count** `N` per
collection interval, and your **target relative error** `E` (e.g. `0.05`
for 5%), you can size a histogram in five steps using a basic scientific
calculator. The recipe is built on the factorization

```
slots  =  (octaves of range)  ×  (buckets per octave)
       =  B  ×  2^K
```

where `K` is the histogram's scale. The specification's default
`160 = 10 × 2^4` corresponds to `B = 10` octaves at scale `K = 4`.
Choosing `B` and `K` is the whole sizing problem.

**Step 1 — Octaves of range you need.**
```
B  =  log(C) / log(2)
```
*Example:* contrast 1000× → `log(1000) / log(2)  ≈  3 / 0.301  ≈  10` octaves.

**Step 2 — Scale needed for your error target.**
Relative error halves each time scale doubles, with the rule
`error ≈ 0.35 / 2^K`. Solve for `K`:
```
K  =  log(0.35 / E) / log(2)        (round up)
```
*Example:* `E = 0.05` → `log(7) / log(2)  ≈  2.81` → `K = 3`. The error
will then be `0.35 / 8 ≈ 4.3%`.

**Step 3 — Slots needed.**
```
slots  =  B × 2^K
```
*Example:* `10 × 2^3 = 80` slots. (At the spec's default `K = 4`, the
same 10 octaves would require 160 slots — twice as many for half the
error.)

**Step 4 — Counter width.**
Estimate the densest bucket count. For a rough upper bound that doesn't
require knowing CV, assume the mass is spread evenly across the slots:
```
mode_count  ≈  N / slots          (uniform assumption, lower bound on width)
```
For sharply peaked normal-ish data, multiply by `2` to `4` as a safety
factor (this corresponds to a CV-based correction; see "Sizing
Intuition" above). Then pick the smallest counter width whose maximum
holds `mode_count`:

| If mode_count fits in… | Use width | Bits per counter |
|------------------------|-----------|------------------|
| 1                      | B1        | 1                |
| 3                      | B2        | 2                |
| 15                     | B4        | 4                |
| 255                    | U8        | 8                |
| 65,535                 | U16       | 16               |
| ~4 × 10⁹               | U32       | 32               |

*Example:* `N = 100,000`, `slots = 80` → `mode_count ≈ 1,250`, with a
×4 safety factor → `~5,000` → fits in **U16** (16 bits).

**Step 5 — Total storage.**
```
bytes  =  slots × bits_per_counter / 8
words  =  bytes / 8
```
Choose `Histogram<words>` (rounded up).

*Example:* `80 × 16 / 8 = 160` bytes of counter data → `Histogram<20>`
(20 × 64-bit words). Compare with the spec's default 160-bucket
histogram of full-width 64-bit counters: 1,280 bytes for the same range
and error — an **8× reduction**.

**Quick sanity table** (from the recipe above):

| Contrast | N        | Error | Octaves B | Scale K | Slots | Width | `Histogram<W>` |
|----------|----------|-------|-----------|---------|-------|-------|----------------|
| 100×     | 10,000   | 5%    | 7         | 3       | 56    | U16   | W = 14         |
| 1,000×   | 100,000  | 5%    | 10        | 3       | 80    | U16   | W = 20         |
| 1,000×   | 1,000,000| 2%    | 10        | 4       | 160   | U16   | W = 40         |
| 10,000×  | 100,000  | 10%   | 14        | 2       | 56    | U16   | W = 14         |

The recipe gives a steady-state provisioning size. The histogram begins
at B1 width (8× more slots than the U16 steady state) and widens
automatically as data accumulates, so it tolerates conservative sizing
without runtime cost.

### Implementation Notes

Variable-width counters are manipulated using SWAR (SIMD Within A
Register) techniques. For example, at B4 width each 64-bit word holds
16 four-bit counters. Incrementing a specific counter within a word
requires:

```
word += 1 << (counter_index × 4)
```

Overflow detection checks whether any 4-bit field exceeds 15 (i.e.,
whether carry propagated into the next field). These operations use
only bitwise AND, OR, shift, and addition — no branching or
floating-point arithmetic.

The downscale operation merges adjacent bucket pairs by adding their
counts, which for sub-byte widths is performed using SWAR addition
with masking. This is equivalent to the specification's existing
downscale semantics (merging buckets whose indices differ only in the
lowest bit).

### Wire Format Compatibility

Variable-width counters are an internal implementation detail of the
SDK. On the wire (OTLP), the histogram is represented using the
existing `ExponentialHistogramDataPoint` message with standard bucket
count arrays. The SDK converts from the compact internal representation
to the wire format during export.

This means variable-width histograms are **fully compatible** with all
existing ExponentialHistogram consumers. No changes to the OTLP
protocol or data model are required.

---

## 3. Recommended Specification Changes

### Data Model Changes

In the [ExponentialHistogram § Producer Expectations][producer] section,
after the paragraph beginning "Producers MAY use an inexact mapping
function", ADD:

> ##### Positive Scales: Use a Lookup Table
>
> For positive scales, producers MAY use a lookup table to compute exact
> bucket indices using integer-only operations. The lookup table method
> uses `2N` linear buckets (where `N = 2^S` for the table scale `S`)
> and a single boundary correction to map IEEE 754 significand bits to
> exponential bucket indices without floating-point arithmetic.
>
> The algorithm is:
>
> 1. Extract the 52-bit significand and unbiased exponent from the
>    IEEE 754 representation.
> 2. Use the upper bits of the significand to index into a pre-computed
>    `INDEX_TABLE` of size `2N`, obtaining an approximate bucket index.
> 3. Compare the significand against the pre-computed exact boundary
>    for the next bucket. If the significand is greater than or equal
>    to the boundary, increment the bucket index by one.
> 4. Combine the bucket index with the exponent and shift to the
>    requested scale.
>
> This method is exact for all representable IEEE 754 double-precision
> values and requires no special case for powers of two. The table can
> be generated at compile time or at initialization time.
>
> Implementations SHOULD prefer this method over the logarithm method
> when the maximum scale is known in advance, as it eliminates the
> possibility of off-by-one errors near bucket boundaries.

[producer]: https://opentelemetry.io/docs/specs/otel/metrics/data-model/#producer-expectations

### SDK Specification Changes

In the [Base2 Exponential Bucket Histogram Aggregation][sdk-agg]
section, after the configuration parameters table, ADD:

> #### Variable-Width Counter Representation
>
> Implementations MAY use a fixed-size pool of storage with
> variable-width bucket counters as an alternative to a dynamically-sized
> array of fixed-width counters. In this representation:
>
> - The total storage is a fixed number of machine words.
> - Bucket counters start at 1 bit per counter and widen automatically
>   (to 2, 4, 8, 16, 32, or 64 bits per counter) as counts grow.
> - Widening halves the number of available slots, which may trigger
>   a reduction in scale to maintain the ideal scale invariant.
>
> This representation enables allocation-free histogram instances with
> predictable memory usage, suitable for high-frequency instrumentation.
>
> The effective `MaxSize` at any point in time is:
>
> ```
> effective_max_size = total_bits / bits_per_counter
> ```
>
> For example, a 10-word (640-bit) pool provides 160 slots at 4-bit
> counter width — equivalent to the default `MaxSize` of 160 — in
> 80 bytes of counter storage rather than 1,280 bytes.
>
> When using variable-width counters, the `MaxSize` configuration
> parameter represents the **maximum possible** number of slots (at
> the narrowest counter width). The effective slot count decreases as
> counters widen to accommodate larger counts.

[sdk-agg]: https://opentelemetry.io/docs/specs/otel/metrics/sdk/#base2-exponential-bucket-histogram-aggregation

---

## Appendix A: Sizing Reference

### The 10-per-octave equivalence

The specification's default of 160 buckets uses the factoring
`160 = 10 × 2^K`. At scale K, this gives 10 buckets per octave and
`2^10 = 1024×` contrast (approximately 3 decades of range). The same
structure appears in the variable-width model:

For a pool of `N` words at counter width `W`:

```
slots = N × 64 / bits_per_counter(W)
```

At width B4 (4 bits per counter), `N = 10` gives exactly 160 slots.
The `10 × 2^K` factoring applies identically, with the same contrast
table from the specification:

| Scale | Maximum contrast at 10 × 2^K slots |
|-------|------------------------------------|
| K+2   | 5.657 (2^(10/4))                   |
| K+1   | 32 (2^(10/2))                      |
| K     | 1024 (2^10)                        |
| K-1   | 1,048,576 (2^20)                   |

As counters widen to U16 (16 bits), the effective pool becomes
`N × 4` slots. For `N = 16`, that is 64 slots = `8 × 2^K`, giving:

| Scale | Maximum contrast at 8 × 2^K slots |
|-------|------------------------------------|
| K+2   | 4 (2^(8/4))                        |
| K+1   | 16 (2^(8/2))                       |
| K     | 256 (2^8)                          |
| K-1   | 65,536 (2^16)                      |

### Counter pressure timeline

For a continuous distribution (e.g., response-time measurements), the
width progression over time follows a characteristic pattern:

| Width | Typical transition point | Cost |
|-------|--------------------------|------|
| B1→B2 | n ≈ 20 (birthday-paradox collision) | 1 scale level |
| B2→B4 | n ≈ 50–100 | 1 scale level |
| B4→U8 | n ≈ 500–2,000 | 1 scale level |
| U8→U16 | n ≈ 5,000–20,000 | 1 scale level |
| U16→U32 | n ≈ 10⁷–10⁸ | 1 scale level |
| U32→U64 | n ≈ 10¹²–10¹³ | 1 scale level |

The histogram spends most of its operational life at U16 width. The
early narrow-width phases (B1 through B4) are transient, lasting only
the first few hundred measurements. Pre-setting a minimum width of U8
or B4 eliminates this startup churn at the cost of 2–3 initial scale
levels.

### Comparison with the fixed-width default

| Property | 160 × 64-bit (spec default) | Histogram<10> (variable) | Histogram<16> (variable) |
|----------|----------------------------|--------------------------|--------------------------|
| Counter storage | 1,280 bytes | 80 bytes | 128 bytes |
| Total struct size | ~1,300+ bytes | 128 bytes | 176 bytes |
| Allocation | Dynamic (array) | None (stack) | None (stack) |
| Initial slots | 160 | 640 (B1) → 160 (B4) | 1024 (B1) → 256 (B4) |
| Steady-state slots (n≈10K) | 160 | 40 (U16) | 64 (U16) |
| Scale at 10 oct range | 4 (2.2% error) | 2 (8.6% error) | 3 (4.3% error) |
| `no_std` / embedded | Requires allocator | Yes | Yes |
