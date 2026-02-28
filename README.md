# Rust OpenTelemetry Exponential Histogram using Table Lookup

An allocation-free, table-lookup based implementation of the
[OpenTelemetry Exponential Histogram](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram)
in Rust.

## Overview

Exponential histograms provide a compact, high-resolution representation of value distributions using logarithmically-spaced bucket boundaries. This implementation is designed for:

- **No heap allocation**: Fixed-size bucket storage using const generics
- **High performance**: Lookup table provides 3.5× speedup over logarithm-based mapping
- **Configurable table size**: Trade static memory for lookup acceleration at higher scales

## Quick Start

```rust
use rust_expohisto::Histogram;

// Create a histogram with 64 buckets and u32 counters
let mut hist: Histogram<u32, 64> = Histogram::new();

// Record observations
hist.update(1.5);
hist.update(2.7);
hist.update(100.0);

// Access statistics
println!("count: {}, sum: {}", hist.count(), hist.sum());
println!("scale: {}", hist.scale());
```

## Performance

Benchmark results (100 test values, per-iteration timing):

| Method | Scale | Time | Notes |
|--------|-------|------|-------|
| Exponent | ≤0 | ~1.1 ns | Bit extraction only |
| NewRelic lookup | 1-8 | ~1.7 ns | Integer-only, 2N linear buckets, 1 correction |
| Dynatrace lookup | 1-8 | ~1.7 ns | Integer-only, N linear buckets, 2 corrections |
| Logarithm | 1-20 | ~6 ns | Fallback when lookup unavailable |

The lookup table accelerates all scales from 1 up to the compiled maximum. Higher scales beyond the table fall back to logarithm computation.

## Lookup Table Features

Choose a **scale** feature to set the table size, and an **algorithm** feature to select the mapping method:

```toml
[dependencies]
rust-expohisto = { version = "0.1", features = ["newrelic", "scale-8"] }  # default
```

### Scale (table size)

| Feature | Table Size | Scales Accelerated | Use Case |
|---------|------------|-------------------|----------|
| `scale-4` | 0.2 KB | 1–4 | Minimal memory |
| `scale-6` | 0.6–0.8 KB | 1–6 | Embedded systems |
| `scale-8` | 2.5–3.0 KB | 1–8 | **Default** |
| `scale-10` | 10–12 KB | 1–10 | Recommended |
| `scale-12` | 40–48 KB | 1–12 | High resolution |
| `scale-14` | 160–192 KB | 1–14 | Maximum coverage |

### Algorithm

| Feature | Linear Buckets | Max Corrections | Notes |
|---------|---------------|-----------------|-------|
| `newrelic` | 2N | 1 | Slightly larger index table |
| `dynatrace` | N | 2 | ~50% smaller index table |
| `logarithm` | — | — | No table needed, FP precision errors |

The `newrelic` and `dynatrace` algorithms produce identical results, are equally tested, and perform the same at runtime — choose whichever you prefer. Memory differences are negligible (see [Lookup Table Design](#lookup-table-design)).

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

## Bucket Inclusivity

Per the OpenTelemetry specification, bucket boundaries are **upper-inclusive**:

> The bucket identified by `index` represents values **greater than** `base^index` and **less than or equal to** `base^(index+1)`.

This convention was adopted for Prometheus compatibility in a [specification change](https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261) that post-dates both the Dynatrace and NewRelic lookup table algorithms. Both algorithms were originally designed with **lower-inclusive** boundaries. This implementation re-engineers the boundary condition for upper-inclusive semantics, validated by an exhaustive test over all ~3 billion f64 values in the first sub-bucket at scale 20.

The practical effect: exact powers of two must fall into the bucket *below* what a naive `floor(log(value))` would suggest.

### Why the lookup table can handle this without a branch

The lookup table boundary arrays use `>=` comparisons against precomputed significand values. Within one octave (i.e., from `2^e` to `2^(e+1)`), the sub-bucket boundaries are at `2^(k/N)` for `k = 0, 1, ..., N-1` where `N = 2^scale`.

The crucial observation: **all sub-bucket boundaries except `k = 0` are irrational numbers.** The value `2^(k/N)` is irrational whenever `k/N` is not an integer (by the [Gelfond–Schneider theorem](https://en.wikipedia.org/wiki/Gelfond%E2%80%93Schneider_theorem)), so no IEEE 754 f64 can ever land exactly on these boundaries. The `>=` comparison against the integer ceiling of an irrational boundary always gives the correct bucket regardless of the inclusivity convention—the f64 is always strictly above or strictly below the ideal boundary.

The **only** boundary that's rational (and representable) is `2^(0/N) = 1.0`, which has significand `0`. This is the only case where upper- vs. lower-inclusive matters. By changing `boundary[0]` from `0` to `1` in the lookup table, the `>=` check naturally excludes `significand == 0` from sub-bucket 0, placing exact powers of two in the bucket below—exactly matching the upper-inclusive convention, without any branch.

In this implementation both algorithms are re-engineered into a single skeleton parameterized by linear bucket count and correction count (see [Lookup Table Design](#lookup-table-design)). The upper-inclusive fix is the same for both: `BOUNDARIES[1] = 1` (instead of `0`) ensures that `significand == 0` never passes any correction check, so `bucket` stays at `approx = 0` and the `- 1` in the final formula places exact powers of two one bucket lower.

For the **logarithm fallback** and **exponent mapping** (scale ≤ 0), an explicit correction is still needed since these don't use the boundary table:

```rust
// Exact powers of two: use exponent directly, subtract 1
if significand == 0 { return (exponent << scale) - 1; }
```

## Index Mapping Algorithms

### Scale ≤ 0: Exponent Extraction

For non-positive scales, the bucket index is derived directly from the IEEE 754 exponent bits—no floating-point math required.

### Scale > 0: Lookup Table (default)

When a lookup feature is enabled (default: `newrelic` + `scale-8`), mapping uses integer-only operations:

1. Extract significand and exponent from the IEEE 754 representation
2. Use the significand to index into a precomputed lookup table
3. Apply boundary check(s) to correct the approximation
4. Combine with exponent to produce the final index

For scales beyond the table's maximum, the implementation falls back to logarithm computation.

### Scale > 0: Logarithm Fallback

When no lookup table is available (or for scales above the table's maximum), the standard formula is used:

```
index = ceil(ln(value) × 2^scale / ln(2)) - 1
```

This is the upper-inclusive form. For non-powers of two (the vast majority of f64 values), `ceil(x) - 1 = floor(x)`, so the implementation uses `floor()` for the general case and an explicit correction for exact powers of two (see [Bucket Inclusivity](#bucket-inclusivity)).

## Lookup Table Design

The [Dynatrace](https://github.com/dynatrace-oss/dynahist) and [NewRelic](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java) algorithms were developed independently with similar designs. In this implementation they are re-engineered so that both share a single boundary table, a single index-table derivation function, and an identical mapping skeleton. The only differences are two compile-time constants:

| Parameter | NewRelic | Dynatrace |
|-----------|----------|-----------|
| Linear buckets (`L`) | `2N` | `N` |
| Max corrections | `1` | `2` |
| Index table | `2N × u16` | `N × u16` |
| Memory per sub-bucket | 12 bytes | 10 bytes |
| Total at scale 8 (N=256) | 3.0 KB | 2.5 KB |

Both algorithms share the same `BOUNDARIES` array (`N+3` × `u64` = 8 bytes per entry), which dominates the memory footprint. The index tables use `u16` entries, so the per-sub-bucket cost is 8 + 4 = 12 bytes (NewRelic) vs. 8 + 2 = 10 bytes (Dynatrace)—a ~20% difference, not the 2× that the `2N` vs. `N` entry counts might suggest. Fewer linear buckets means each one can span more log-scale buckets, requiring more boundary corrections at runtime. In practice both are equally fast (see [Performance](#performance)).

### Shared data structures

Both algorithms use the same two tables, generated once at the highest compiled scale `H` (where `N = 2^H`):

**`BOUNDARIES[N+3]`** — Exact sub-bucket boundary significands, shared by both algorithms:

```
[sentinel=0, b[0]=1, b[1], ..., b[N-1], sentinel=2^52, sentinel=2^52]
```

- `b[k]` is the 52-bit significand of `ceil(2^(k/N))` for `k > 0`
- `b[0] = 1` (not 0) implements upper-inclusive semantics (see [Bucket Inclusivity](#bucket-inclusivity))
- Trailing sentinels allow unchecked `BOUNDARIES[approx + 2]` access in the Dynatrace variant

**`INDEX_TABLE[L]`** — Linear-to-log bucket mapping, derived from `BOUNDARIES` by the same function for both algorithms (with `L = 2N` for NewRelic, `L = N` for Dynatrace):

```
SHIFT = 52 - log2(L)

for i in 0..L:
    INDEX_TABLE[i] = largest j such that BOUNDARIES[j+1] <= (i << SHIFT)
```

Each entry gives the approximate log bucket for the significand range starting at `i << SHIFT`.

### Unified mapping algorithm

The runtime mapping is identical for both variants, parameterized only by `L` (linear bucket count) and `MAX_CORRECTIONS` (1 or 2):

```
fn map_to_index(value, scale) -> index:
    significand = bits 0..51 of value         // IEEE 754 significand
    exponent    = biased_exponent - 1023      // IEEE 754 exponent

    // Step 1: Linear approximation
    approx = INDEX_TABLE[significand >> SHIFT] // O(1) lookup

    // Step 2: Boundary corrections (1 for NewRelic, 2 for Dynatrace)
    bucket = approx
    for i in 1..=MAX_CORRECTIONS:
        if significand >= BOUNDARIES[approx + i]:
            bucket += 1

    // Step 3: Combine with exponent, downscale to requested scale
    fine_index = (exponent << H) + bucket - 1
    return fine_index >> (H - scale)
```

All operations are integer: bit extraction, array indexing, comparison, shift, and addition. No floating-point arithmetic is performed at runtime.

The `- 1` in step 3, combined with `b[0] = 1` in the boundary table, is how upper-inclusive semantics emerge: when `significand == 0` (exact power of two), `approx` is 0 and no correction fires, so the result is `(exponent << H) - 1` — one bucket lower than the naive formula.

### Why the correction count is bounded

Each linear bucket spans a width `W = 2^52 / L` of significand space. The number of log-scale boundaries that can fall within one linear bucket is at most:

$$
\Delta = N \cdot \log_2\!\left(1 + \frac{W}{2^{52}}\right) = N \cdot \log_2\!\left(1 + \frac{1}{L}\right)
$$

Because `log` is concave, the worst case is always the **first** linear bucket (starting at significand 0), where the log function is steepest. Substituting `L`:

- **NewRelic** (`L = 2N`): $\Delta = N \cdot \log_2(1 + 1/2N) \approx 1/(2\ln 2) \approx 0.72 < 1$ → **1 correction suffices**
- **Dynatrace** (`L = N`): $\Delta = N \cdot \log_2(1 + 1/N) \approx 1/\ln 2 \approx 1.44 < 2$ → **2 corrections suffice**

Since the worst case is the first linear bucket, and `significand == 0` lives at the start of that bucket, it is already covered by the bound — no extra correction is needed for exact powers of two.

### Exact boundary computation

The k-th sub-bucket boundary at scale `S` (where `N = 2^S`) is:

$$
\text{boundary}(k) = 2^{k/N} = 2^{k \cdot 2^{-S}}
$$

Note that `boundary(1) = 2^(1/N) = 2^(2^(-S))` is the exponential base (see [Exponential Scale](#exponential-scale)), and `boundary(k) = base^k`.

These are computed exactly at build time using repeated square roots and bignum verification (following [PR #3841](https://github.com/open-telemetry/opentelemetry-collector/pull/3841)):

```
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

- **`rust-expohisto`**: Main library with histogram and mapping implementations
- **`mapping-gen`**: Sub-crate for generating lookup tables (used at build time)

The `mapping-gen` crate can be tested independently:

```bash
cd mapping-gen && cargo test
```

## Sub-Byte Bucket Widths and Bit-Level Arithmetic

Bucket counters start at 1 bit per counter, maximizing the initial bucket count for a given memory budget. As counters saturate, they widen in place through the chain **B1→B2→B4→U8→U16→U32→U64**, each transition halving the bucket count and doubling counter capacity. The sub-byte widths (B1, B2, B4) are the novel part — once you reach U8, it's just `bytemuck::cast_slice` for free reinterpretation. This section describes the bit-level machinery that makes sub-byte widths work.

### Memory layout

All bucket data lives in a flat `[u64; N]` array. Each counter occupies a fixed number of bits, densely packed with **no padding**: the k-th counter is at bits `k*W..(k+1)*W` across the array, where `W` is the bit width. For N=4 (32 bytes), the capacity per width is:

| Width | Bits per counter | Buckets |
|-------|-----------------|---------|
| B1 | 1 | 256 |
| B2 | 2 | 128 |
| B4 | 4 | 64 |
| U8 | 8 | 32 |
| U16 | 16 | 16 |
| U32 | 32 | 8 |
| U64 | 64 | 4 |

All widths use the same physical `[u64; N]` backing array. This means a `Histogram<P64, 4>` always occupies the same number of bytes regardless of the current counter width — it just interprets the same bits differently.

### Sub-byte get/set

Slot access extracts or replaces a bitfield within a u64 word:

```
fn get(slot) -> u64:
    match width:
        B1:  data[slot / 64] >> (slot % 64)         & 1
        B2:  data[slot / 32] >> ((slot % 32) * 2)   & 3
        B4:  data[slot / 16] >> ((slot % 16) * 4)   & 0xF

fn set(slot, value):
    match width:
        B1:  word = &data[slot / 64]; bit = slot % 64
             *word = (*word & !(1 << bit)) | ((value & 1) << bit)
        B2:  word = &data[slot / 32]; shift = (slot % 32) * 2
             *word = (*word & !(3 << shift)) | ((value & 3) << shift)
        B4:  word = &data[slot / 16]; shift = (slot % 16) * 4
             *word = (*word & !(0xF << shift)) | ((value & 0xF) << shift)
```

The pattern is: divide by slots-per-word, multiply the intra-word index by the bit width, mask with `(1 << W) - 1`. The `set` path clears the target field with an AND-NOT and writes the new value with an OR.

### Pairwise sum via SWAR

When a B1 counter saturates (value goes from 1 to 2), the histogram needs to widen all counters from 1-bit to 2-bit. Naively this requires reading each pair of adjacent 1-bit counters, summing them, and writing a 2-bit result — a serial loop over potentially hundreds of slots.

Instead, the widening uses **SWAR** (SIMD Within A Register): each stage is one step of the textbook popcount algorithm. The key insight is that pairwise-summing N-bit fields into 2N-bit fields is exactly what popcount does at each stage, and the bitmask constants are the same.

```
B1 → B2:   w = ((x >> 1) & 0x5555...) + (x & 0x5555...)
B2 → B4:   w = ((x >> 2) & 0x3333...) + (x & 0x3333...)
B4 → U8:   w = ((x >> 4) & 0x0F0F...) + (x & 0x0F0F...)
```

Each formula processes all counters in one u64 word simultaneously:

- **B1→B2**: The mask `0x5555...5555` selects the odd-indexed bits. Shifting right by 1 aligns even-indexed bits with them. Adding gives a 2-bit sum of each adjacent pair. All 32 pairs in a u64 are processed in 3 operations.

- **B2→B4**: The mask `0x3333...3333` selects alternating 2-bit fields. Shifting right by 2 aligns adjacent 2-bit fields. Adding gives a 4-bit sum. All 16 pairs in 3 operations.

- **B4→U8**: The mask `0x0F0F...0F0F` selects alternating nibbles. Shifting right by 4 aligns them. Adding gives an 8-bit (byte) sum. Beyond this point, the value fits in a byte and the transition to U8 needs no further reinterpretation — `bytemuck::cast_slice` views the same `[u64]` as `[u8]`.

The inner loop is:
```
for w in data.iter_mut() {
    let x = *w;
    *w = ((x >> FIELD_WIDTH) & MASK) + (x & MASK);
}
```

No branches, no cross-word dependencies. With `-C target-cpu=native`, LLVM auto-vectorizes this into AVX2 or NEON instructions.

#### Overflow safety

In each stage, the maximum possible sum equals twice the maximum value of the source field: B1 max 1+1=2 (fits in 2 bits), B2 max 3+3=6 (fits in 4 bits), B4 max 15+15=30 (fits in 8 bits). The destination field is always wide enough.

#### Stale data zeroing

Because SWAR processes every word (not just the used range), stale bits outside the active bucket range are transformed rather than cleared. After the SWAR pass, slots beyond the new used count are explicitly zeroed to prevent stale data from becoming visible if the range is later extended.

### Bit-level circular buffer rotation

Buckets use a circular buffer: `index_base` marks which histogram index corresponds to physical slot 0. When the histogram needs to linearize the buffer (for widening or downscaling), it rotates the entire bit array so that `index_base == index_start`.

For byte-aligned widths, this delegates to Rust's `[T]::rotate_right(n)`. For sub-byte widths, the rotation operates at bit granularity on the raw `[u64; N]` array:

```
bit_rotate_right(data, shift):
    total_bits = N * 64
    shift = shift % total_bits

    // Step 1: whole-word rotate
    word_shift = shift / 64
    bit_shift  = shift % 64
    data.rotate_right(word_shift)

    // Step 2: sub-word carry shift (LEFT across the array)
    saved = data[N-1] >> (64 - bit_shift)
    for i in (1..N).rev():
        data[i] = (data[i] << bit_shift) | (data[i-1] >> (64 - bit_shift))
    data[0] = (data[0] << bit_shift) | saved
```

This matches `[T]::rotate_right` semantics: the bit at flat position `p` moves to `(p + shift) % total_bits`.

**Decomposition**: The shift is split into a whole-word component (handled by `[u64]::rotate_right`) and a sub-word residual. The sub-word step shifts each word LEFT by `bit_shift` bits, carrying the overflow into the next higher word. The top bits of `data[N-1]` wrap around to the bottom of `data[0]`.

**Carry direction**: The loop iterates in reverse (`N-1` down to `1`) so that each word reads from `data[i-1]` before that word is overwritten. The carry propagates from lower words to higher words, matching the left-shift direction. `saved` captures the wrap-around bits from the last word before the loop begins.

### Downscale: transactional group-sum

Downscaling by `k` merges groups of `2^k` adjacent buckets by summing their counters. At sub-byte widths, a group sum can exceed the counter maximum (e.g., four 1-bit counters summing to 4 doesn't fit in a 1-bit counter). When this happens, the downscale is aborted and the caller widens the counters first.

To avoid partial corruption, the implementation uses a **two-phase approach**:

1. **Pre-check** (read-only): Iterate over the circular buffer using `at()` (which handles wrap-around without mutation) and compute each group sum. If any sum exceeds `counter_max()`, return `false` immediately — no state has been modified.

2. **Mutate**: Only after the pre-check passes, linearize the buffer with `rotate()` and perform the actual in-place relocations. Each slot's count is added to its group's destination slot and the source is zeroed. After all relocations, unused trailing slots are zeroed.

The `relocate(dest, src)` helper reads the source, adds to the destination via `try_increment`, and only zeros the source on success. This avoids the destructive-read-before-write pitfall where zeroing the source before confirming the destination increment would lose data on overflow.

### Widen-in-place: combined counter-widen + downscale

When a bucket counter saturates (e.g., a B1 counter already holds 1 and needs to record another observation), the histogram performs `widen_in_place()`:

1. **Linearize**: `rotate()` aligns the circular buffer so `index_base == index_start`.

2. **Compute downscale amount**: Usually `by = 1` (pairwise grouping, halving bucket count). In rare cases where the span at `by = 1` exceeds the new capacity (due to odd `index_start`), bumps to `by = 2`.

3. **Group-sum and widen**: For sub-byte `by = 1`, uses the SWAR pairwise sum (fast path). Otherwise, uses the sequential group-sum (general path) which reads at the old width and writes at the new width by temporarily switching the width field during the loop.

4. **Update metadata**: Shift `index_start` and `index_end` right by `by`, set `index_base = index_start`, update `width`.

The transition preserves the total count across all buckets: the sum of all counters before and after widening is identical. Resolution is lost (adjacent buckets are merged), but no data is destroyed.

## References

- [OpenTelemetry Exponential Histogram Specification](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram)
- [Upper-inclusive boundary discussion](https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261): The specification change (for Prometheus compatibility) that motivated the boundary condition re-engineering in this implementation
- [Golang OpenTelemetry Exponential Histogram](https://github.com/lightstep/go-expohisto): Golang reference implementation by the same author
- [Dynatrace DynaHist library by Otmar Ertl](https://github.com/dynatrace-oss/dynahist) (see [ExponentialHistogramLargeInclusiveLayout](https://github.com/dynatrace-oss/dynahist/blob/main/src/main/java/com/dynatrace/dynahist/layout/ExponentialHistogramLargeInclusiveLayout.java))
- [NewRelic lookup table algorithm by Yuke Zhuge](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/Indexer.md)
- [NewRelic algorithm implementation](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java)

## OTel SDK Specification Compatibility

This section documents compatibility with the [Base2 Exponential Bucket Histogram Aggregation](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/metrics/sdk.md#base2-exponential-bucket-histogram-aggregation) section of the OpenTelemetry Metrics SDK specification.

### Configuration Parameters

The spec defines three configuration parameters:

| Parameter | Spec Default | This Implementation | Notes |
|-----------|-------------|---------------------|-------|
| **MaxSize** | 160 | 160 (`LARGE_SIZE`), 16 (`SMALL_SIZE`), or any compile-time `SIZE` | `ExpoHistogram` offers 160 and 16 at runtime; the generic `Histogram<C, SIZE>` accepts any const `SIZE` |
| **MaxScale** | 20 | 20 (`MAX_SCALE`) | Effective max depends on the mapping feature: table-based features cap at `TABLE_SCALE` (e.g. 8 for `newrelic-8`); the `logarithm` feature reaches 20. `Histogram::with_max_scale()` lets the user set a lower cap. |
| **RecordMinMax** | true | Always on | `min` and `max` are tracked on every update. There is no option to disable them. |

### Collected Fields

The spec requires all histogram aggregations to collect count, sum, min, and max. This implementation provides:

| Field | Type | Notes |
|-------|------|-------|
| `count` | `u64` | Total measurement count |
| `sum` | `f64` | Arithmetic sum of all values (zero values excluded from sum) |
| `min` | `f64` | Minimum observed value |
| `max` | `f64` | Maximum observed value |
| `zero_count` | `u64` | Count of zero-valued measurements |
| `positive` | `Buckets<C, SIZE>` | Positive range bucket counts in a circular buffer |
| `scale` | `i32` | Current mapping scale (adjusted automatically) |

### Handle All Normal Values

> Implementations are REQUIRED to accept the entire normal range of IEEE floating point values.

**Supported.** All normal positive f64 values (from $2^{-1022}$ through the largest finite f64) are mapped to the correct bucket index. Subnormal values are mapped to the lowest normal bucket rather than rejected.

> Implementations SHOULD NOT incorporate non-normal values (i.e., +Inf, -Inf, and NaNs) into the sum, min, and max fields.

**Caller responsibility.** `debug_assert!` guards reject non-finite and negative values during development, but there is no runtime check in release builds. The crate expects the SDK caller to filter these before recording.

### Support a Minimum and Maximum Scale

> The implementation MUST maintain reasonable minimum and maximum scale parameters that the automatic scale parameter will not exceed.

**Supported.** Scale is bounded by `MIN_SCALE` (-10) and `MAX_SCALE` (20). The `max_scale` field (configurable via `Histogram::with_max_scale()` or `ExpoHistogram::with_max_scale()`) sets the upper bound for automatic scale selection.

### Use the Maximum Scale for Single Measurements

> When the histogram contains not more than one value in either of the positive or negative ranges, the implementation SHOULD use the maximum scale.

**Supported.** A new histogram starts at `max_scale`. The first observation is recorded at that scale. Scale only decreases when a second value doesn't fit within the `SIZE` bucket span.

### Maintain the Ideal Scale

> Implementations SHOULD adjust the histogram scale as necessary to maintain the best resolution possible, within the constraint of maximum size.

**Supported.** When a new value's bucket index would exceed the `SIZE`-bucket span, the histogram computes the minimum downscale needed to accommodate both the existing range and the new value. It never downscales more than necessary. On `clear()`, scale resets to `max_scale`.

### Negative Values

The spec defines both positive and negative bucket ranges. **This implementation only supports non-negative values** — there is a single `positive` bucket set and no `negative` counterpart. The use case is recording non-negative measurements (latencies, sizes, counts) which is the overwhelmingly common case. Adding negative bucket support would double the per-histogram memory footprint.

### Merging

The spec requires aggregations to be mergeable. This implementation supports:

- **Same-type merge:** `Histogram::merge_from()` merges identically-typed histograms, computing the minimum common scale and downscaling as needed.
- **Cross-counter merge:** `Histogram::merge_from_histogram()` merges histograms with different counter types.
- **Cross-size merge:** `Histogram::merge_from_raw()` merges histograms with different `SIZE` parameters via a closure-based bucket accessor.
- **Runtime merge:** `ExpoHistogram::merge_from()` merges across resolutions (Small/Large) and counter widths (u16/u32/u64) with automatic counter widening on overflow.

### Counter Widening

Not part of the spec, but relevant to overflow handling: `ExpoHistogram` starts with `u16` bucket counters and automatically widens to `u32`, then `u64`, if a bucket counter would overflow during `update` or `merge`. This allows the common case to use compact 16-bit counters while still handling extreme counts.

### Summary

| Spec Requirement | Status |
|-----------------|--------|
| MaxSize = 160 default | Supported |
| MaxScale = 20 default | Supported |
| RecordMinMax | Always on |
| Handle all normal values | Supported |
| Reject +Inf, -Inf, NaN | Debug-only (caller responsibility) |
| Subnormal values | Mapped to lowest normal bucket |
| Minimum and maximum scale | Supported (MIN_SCALE = -10, MAX_SCALE = 20) |
| Max scale for single measurements | Supported |
| Maintain ideal scale | Supported |
| Positive bucket range | Supported |
| Negative bucket range | Not implemented |
| Zero count | Supported |
| Count, sum, min, max | Supported |
| Merge | Supported (same-type, cross-counter, cross-size) |

## License

Apache-2.0
