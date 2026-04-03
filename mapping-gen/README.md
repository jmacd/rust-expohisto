# Exponential Histogram Lookup Table Generator

This crate generates precomputed lookup tables that enable **integer-only** mapping
from IEEE 754 double-precision floating-point values to exponential histogram bucket
indices. The algorithm uses 2N linear buckets with a single boundary correction,
inspired by independently-developed approaches from
[NewRelic](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java)
(Yuke Zhuge) and
[Dynatrace](https://github.com/open-telemetry/opentelemetry-collector/pull/3841)
(Otmar Ertl).

## The Problem

An exponential histogram with **scale** parameter `S` divides the positive real line
into buckets with boundaries at consecutive powers of `base = 2^(2^(-S))`:

```
..., base^(k-1), base^k, base^(k+1), ...
```

For scale `S`, there are `N = 2^S` buckets per power of two. The bucket containing
a value `v` has index:

```
index = floor(log_base(v)) = floor(ln(v) × 2^S / ln(2))
```

The naive implementation using floating-point `ln()` is both slow (~6 ns) and
susceptible to precision errors near bucket boundaries. The lookup table approach
eliminates both problems using only integer operations (~1.7 ns, **3.5× faster**).

## IEEE 754 Double-Precision Representation

Every positive, normal f64 value can be written as:

```
value = (1 + m/2^52) × 2^exp
```

where:
- `m` is the 52-bit **significand**, stored explicitly in bits 0-51
- `exp` is the **exponent**, derived from bits 52-62 as `raw_exponent - 1023`
- The leading `1` is implicit (not stored)

Given this representation:
```
log_base(value) = log_base(2^exp) + log_base(1 + m/2^52)
                = exp × N + log_base(1 + m/2^52)
                           ╰────────────────────╯
                           ↳ subbucket ∈ [0, N)
```

The **exponent part** `exp × N` is trivially computed via bit shift: `exp << S`.
The challenge is computing the **subbucket** from the significand without floating-point math.

## Algorithm Overview

The lookup table approach has three key insights:

1. **Linear approximation**: Divide the significand range `[0, 2^52)` into `2N`
   equal-width "linear buckets" and map each to an approximate log-scale bucket
2. **Bounded error**: Because logarithm is concave, each linear bucket spans at
   most 2 log-scale buckets, so the approximation is off by at most 1
3. **Exact correction**: A single integer comparison against the precomputed
   bucket boundary corrects any approximation error

### Data Structures

The generator produces two tables:

**`LOG_BUCKET_INDEX[2N]`** — Maps each linear bucket to its approximate log bucket:
```
linear_bucket_i starts at significand = (i × 2^52) / (2N)
LOG_BUCKET_INDEX[i] = smallest log bucket k such that k's start ≤ linear_bucket_i's start
```

**`LOG_BUCKET_END[N+1]`** — Exact boundary (as 52-bit significand) for each log bucket:
```
LOG_BUCKET_END[k] = significand of 2^(k/N) = the upper boundary of bucket k
```
The first entry is `1` (not `0`) to implement upper-inclusive bucket semantics — this ensures exact powers of two (significand 0) map one bucket lower without a branch. The last entry is a sentinel value `2^52` for bounds checking.

### Runtime Lookup (Integer-Only)

```rust
fn map_to_index_lookup(value: f64, scale: i32) -> i32 {
    let significand = value.to_bits() & SIGNIFICAND_MASK;  // bits 0-51
    let exponent = ((value.to_bits() >> 52) & 0x7FF) as i32 - 1023;

    // Step 1: Linear approximation
    let linear_idx = significand >> SIGNIFICAND_SHIFT;  // top bits select linear bucket
    let approx_bucket = LOG_BUCKET_INDEX[linear_idx];

    // Step 2: Boundary correction (at most +1)
    // Upper-inclusive is baked in: boundary[0] = 1, so significand == 0
    // (exact powers of two) fails this check, staying in the bucket below.
    let bucket = if significand >= LOG_BUCKET_END[approx_bucket] {
        approx_bucket + 1
    } else {
        approx_bucket
    };

    // Step 3: Combine with exponent
    (exponent << scale) + bucket - 1
}
```

All operations are integer: bit extraction, array indexing, comparison, shifts, and addition.
No floating-point arithmetic is performed.

## Computing Exact Bucket Boundaries

The bucket boundaries must be computed **exactly** to ensure the lookup table has zero error.
A boundary at position `k` (for `k ∈ [0, N)`) corresponds to the value `2^(k/N)`.

We need to express this as a 52-bit significand. The mathematical relationship is:

```
2^(k/N) = (1 + s_k/2^52) × 2^0    for k ∈ [0, N)

Therefore: s_k = (2^(k/N) - 1) × 2^52
                = 2^(k/N + 52) - 2^52
```

Since `k/N` is not an integer for most `k`, we cannot compute `2^(k/N)` exactly using
standard floating-point. Instead, we use **repeated square roots**—a technique related
to Euler's method for computing nth roots.

### Repeated Square Root Method

The key observation is that when `N = 2^S` (a power of two):

```
2^(k/N) = 2^(k/2^S) = √(√(...√(2^k)...))
                      ╰─────────────────╯
                          S times
```

This is because:
- √(2^k) = 2^(k/2)
- √(√(2^k)) = 2^(k/4)
- √(√(√(2^k))) = 2^(k/8)
- etc.

After `S` applications of square root, we get `2^(k/2^S) = 2^(k/N)`.

This approach has a key advantage: **2^k is an exact integer**, and arbitrary-precision
square root algorithms can maintain exact precision throughout the computation.

### Why "Euler's Method"?

Euler's method for computing nth roots reduces the problem to repeated square roots:

```
x^(1/n) = √(√(...√(x × 2^(b×n))...)) / 2^b
```

where `b` is chosen to make the intermediate results manageable. In our case:
- We start with `x = 2^k` (exact integer)
- `n = N = 2^S` (power of two)
- We apply `S` square roots directly without the scaling factor

This is Euler's observation: computing `N`th roots is tractable when `N` is a power of two,
because we only need a square root algorithm (which is well-understood for arbitrary
precision).

### Implementation with Arbitrary-Precision Arithmetic

```rust
use rug::{Float, Integer};

fn compute_boundary(k: usize, n: usize, scale: u32) -> u64 {
    if k == 0 {
        return 0;  // 2^0 = 1.0, significand is 0
    }

    // Start with 2^k (exact)
    let mut x = Float::with_val(128, 1u32) << k as u32;

    // Apply √ exactly `scale` times to get 2^(k/N)
    for _ in 0..scale {
        x = x.sqrt();
    }

    // Scale by 2^52 to get IEEE significand representation
    x <<= 52;
    let mut candidate = x.to_integer().unwrap().to_u64().unwrap();

    // Verify and adjust (see next section)
    // ...

    candidate & SIGNIFICAND_MASK
}
```

The `rug` crate uses GMP/MPFR for arbitrary-precision computation, ensuring
sufficient precision (128 bits is more than enough for scale ≤ 20).

## Verification with Exact Integer Arithmetic

Floating-point arbitrary precision can still have subtle rounding. We verify
the computed boundary using **exact integer arithmetic**:

For boundary `s_k` at position `k`, we need the **smallest** significand such that:

```
(1 + s_k/2^52) ≥ 2^(k/N)
```

Multiplying both sides by `2^52`:

```
2^52 + s_k ≥ 2^(k/N + 52)
```

Since `2^52 + s_k` represents our boundary, and we have `N = 2^S`, we need:

```
(2^52 + s_k)^N ≥ 2^(52N + k)
```

This is an exact integer comparison! Both sides are integers:
- Left: The Nth power of our candidate boundary
- Right: 2 raised to an integer power

### Verification Algorithm

```rust
use rug::Integer;

fn verify_boundary(candidate: u64, k: usize, n: usize) -> u64 {
    // Both sides are exact integers
    let compare_to = Integer::from(1u32) << (52 * n + k) as u32;
    let sig = Integer::from(candidate).pow(n as u32);

    // If candidate^N < 2^(52N + k), we need to round up
    let mut result = candidate;
    if sig < compare_to {
        result += 1;
    }

    // Sanity check: (result - 1)^N must be < compare_to
    let sig_less = Integer::from(result - 1).pow(n as u32);
    assert!(sig_less < compare_to, "boundary error > 1 ULP");

    result
}
```

This guarantees the boundary is correct to within **1 ULP** (unit in last place)
of the true mathematical value.

## Multi-Scale Support via Right-Shifting

A table generated at scale `S_max` supports **all scales from 1 to S_max** through
a clever index transformation.

At scale `S_max`, we compute:
```
full_index = (exponent << S_max) + subbucket - 1
```

For a lower scale `S < S_max`, the index is:
```
index_at_S = full_index >> (S_max - S)
```

This works because the exponential histogram has a nested structure: bucket `k` at
scale `S` contains buckets `2k` and `2k+1` at scale `S+1`. Arithmetic right shift
by `d = S_max - S` correctly merges `2^d` consecutive buckets.

The `-1` adjustment for upper-inclusive boundaries must be applied **before** the
shift to ensure correct rounding for negative indices.

## Why the Lookup Table Has Zero Error

Unlike the floating-point logarithm formula which can produce off-by-one errors
due to precision limits, the lookup table approach is **exactly correct** because:

1. **Boundaries are exact**: Computed using arbitrary-precision arithmetic and
   verified with exact integer comparison
2. **Runtime uses only integers**: No floating-point operations that could round
3. **Linear approximation error is bounded and corrected**: The single comparison
   against `LOG_BUCKET_END` fixes any approximation error

Empirical testing shows:
- Logarithm formula: ~0.05% error rate at bucket boundaries
- Lookup table: **0% error rate** (provably exact)

## Memory vs. Coverage Trade-off

| Scale | N (buckets per 2×) | Table Size | Relative Error per Bucket |
|-------|-------------------|------------|---------------------------|
| 4     | 16                | 0.2 KB     | ~2.2%                    |
| 6     | 64                | 0.8 KB     | ~0.54%                   |
| 8     | 256               | 3 KB       | ~0.14%                   |
| 10    | 1024              | 12 KB      | ~0.034%                  |
| 12    | 4096              | 48 KB      | ~0.0085%                 |
| 14    | 16384             | 192 KB     | ~0.0021%                 |

Table size is approximately:
- `LOG_BUCKET_INDEX`: `2N × 2 bytes = 4N bytes`
- `LOG_BUCKET_END`: `(N+1) × 8 bytes ≈ 8N bytes`
- **Total**: `~12N bytes`

## Usage

This crate is used at build time by `otel-expohisto` to generate lookup tables:

```rust
use expohisto_mapping_gen::{generate_boundaries, write_boundaries, write_index_table};

let scale = 10; // 1024 buckets per power of 2

// Compute exact boundaries once (expensive bignum arithmetic).
let boundaries = generate_boundaries(scale);

// Write as Rust source code.
let mut output = std::fs::File::create("lookup_tables.rs").unwrap();
write_boundaries(&mut output, scale, &boundaries).unwrap();
write_index_table(&mut output, scale, &boundaries).unwrap();
```

The generated file contains:
```rust
pub const TABLE_SCALE: i32 = 10;
pub const INDEX_SHIFT: u32 = ...;
pub static BOUNDARIES: [u64; 1027] = [...];
pub static INDEX_TABLE: [u16; 2048] = [...];
```

## Mathematical Details

### Why Linear Approximation Works

For significand `s ∈ [0, 2^52)`, the subbucket is:

```
subbucket(s) = floor(N × log_2(1 + s/2^52))
```

The function `f(s) = log_2(1 + s/2^52)` is:
- Monotonically increasing
- Concave (second derivative negative)
- Maps [0, 2^52) to [0, 1)

Linear buckets divide the domain uniformly. Due to concavity, each linear bucket
of width `2^52 / (2N)` maps to at most 2 consecutive values of `floor(N × f(s))`.

More precisely: since `f''(s) < 0`, the function grows more slowly than linear.
A linear bucket's image under `f` has width at most `1/N` (when the bucket
straddles where `f` is steepest, near 0), ensuring the floor differs by at most 1.

### Upper-Inclusive Boundary Convention

Per OpenTelemetry spec (for Prometheus compatibility), bucket `k` contains values
`(base^k, base^(k+1)]`—the upper boundary is **inclusive**.

This means exact powers of two fall into the bucket **below** the naive logarithm
result. For the lookup table algorithm, this is handled without a branch by setting
`boundary[0] = 1` instead of `0`. Since the only exact bucket boundary that
coincides with an IEEE 754 representable value is `2^(0/N) = 1.0` (significand 0),
changing boundary[0] from 0 to 1 makes the `>=` comparison naturally exclude
exact powers of two from sub-bucket 0, placing them in the bucket below.

For the logarithm fallback (not table-based), an explicit check is still needed:

```rust
if significand == 0 {
    return (exponent << scale) - 1;  // subtract 1 for inclusive upper
}
```

## References

- [NewRelic Indexer Algorithm (Yuke Zhuge)](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/Indexer.md) — Original description
- [NewRelic SubBucketLookupIndexer](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java) — Java implementation
- [Dynatrace PR #3841](https://github.com/open-telemetry/opentelemetry-collector/pull/3841) — Independent discovery by Otmar Ertl
- [OpenTelemetry Exponential Histogram Spec](https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram)
- [IEEE 754 Double-Precision Format](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)

## License

Apache-2.0
