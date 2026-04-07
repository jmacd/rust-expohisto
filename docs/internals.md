# Implementation Details

## Sub-Byte Bucket Widths and Bit-Level Arithmetic

Bucket counters start at 1 bit per counter, maximizing the initial bucket count for a given memory budget. As counters saturate, they widen in place through the chain **B1→B2→B4→U8→U16→U32→U64**, each transition halving the bucket count and doubling counter capacity. All widths use a single shift-and-mask formula over the `[u64]` pool — sub-byte widths extract packed bitfields, while byte-aligned widths reduce to ordinary word-sized reads. This section describes the bit-level machinery that makes sub-byte widths work.

### Memory layout

`Histogram<N>` stores aggregate statistics in separate struct fields plus a flat
`[u64; N]` data pool. Because stats live outside the pool, all `N` words are
available for bucket data:

```text
Histogram<16>                     128 bytes data pool + 32 bytes stats

┌─────────────────────────────────────────────────────────────────┐
│ stats.count (u64) │ stats.sum (f64) │ stats.min/max (f64, f64) │
│ separate struct fields; not stored in `data`                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ data[0] │ data[1] │ data[2] │ ... │ data[15]                   │
│ 16 words: bucket data                                           │
└─────────────────────────────────────────────────────────────────┘
 ◄───────────── stats overhead: 32 bytes ─────────────►
 ◄──────────── data pool: N = 16 words = 128 bytes ───►
```

The data pool is interpreted at the current counter width.
All widths use the same physical `[u64; N]` backing array — the histogram just
interprets the same bits differently:

```text
16 bucket words at each width (Histogram<16>):

B1  ┌──────────────────────────────────────────────┐  1024 slots
    │ 64 bits per word × 16 words = 1024 1-bit slots│  max count: 1
    └──────────────────────────────────────────────┘

B2  ┌──────────────────────────────────────────────┐   512 slots
    │ 32 × 2-bit slots per word × 16 words         │  max count: 3
    └──────────────────────────────────────────────┘

B4  ┌──────────────────────────────────────────────┐   256 slots
    │ 16 nibbles per word × 16 words               │  max count: 15
    └──────────────────────────────────────────────┘

U8  ┌──────────────────────────────────────────────┐   128 slots
    │ 8 bytes per word × 16 words                  │  max count: 255
    └──────────────────────────────────────────────┘

U16 ┌──────────────────────────────────────────────┐    64 slots
    │ 4 u16s per word × 16 words                   │  max count: 65,535
    └──────────────────────────────────────────────┘

U32 ┌──────────────────────────────────────────────┐    32 slots
    │ 2 u32s per word × 16 words                   │  max count: ~4.3 billion
    └──────────────────────────────────────────────┘

U64 ┌──────────────────────────────────────────────┐    16 slots
    │ 1 u64 per word × 16 words                    │  max count: ~1.8 × 10¹⁹
    └──────────────────────────────────────────────┘
```

Each counter occupies a fixed number of bits, densely packed with **no padding**: the k-th counter is at bits `k*W..(k+1)*W` across the array, where `W` is the bit width.

### Sub-byte get/set

Slot access extracts or replaces a bitfield within a u64 word:

```text
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

```text
B1 → B2:   w = ((x >> 1) & 0x5555...) + (x & 0x5555...)
B2 → B4:   w = ((x >> 2) & 0x3333...) + (x & 0x3333...)
B4 → U8:   w = ((x >> 4) & 0x0F0F...) + (x & 0x0F0F...)
```

Each formula processes all counters in one u64 word simultaneously:

- **B1→B2**: The mask `0x5555...5555` selects the odd-indexed bits. Shifting right by 1 aligns even-indexed bits with them. Adding gives a 2-bit sum of each adjacent pair. All 32 pairs in a u64 are processed in 3 operations.

- **B2→B4**: The mask `0x3333...3333` selects alternating 2-bit fields. Shifting right by 2 aligns adjacent 2-bit fields. Adding gives a 4-bit sum. All 16 pairs in 3 operations.

- **B4→U8**: The mask `0x0F0F...0F0F` selects alternating nibbles. Shifting right by 4 aligns them. Adding gives an 8-bit (byte) sum. Beyond this point, each counter occupies at least a full byte, so the same shift-and-mask formula reduces to ordinary word-sized reads at the compiler level.

The inner loop is:
```text
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

### Downscale: SWAR merge (even and odd base)

Downscaling by 1 step merges pairs of adjacent buckets by summing their counters. At sub-byte widths, a pair sum can exceed the counter maximum (e.g., two 1-bit counters both set to 1 sum to 2, which doesn't fit in 1 bit). When this happens, the counters are widened to the next width.

The histogram tracks `index_base`, the logical index that corresponds to physical slot 0. Because SWAR operates on fixed positions within each word, it can only correctly pair adjacent counters when `index_base` is even. The downscale loop dispatches to one of two algorithms based on `index_base & 1`:

#### Even base: direct SWAR merge

When `index_base` is even, the k-th counter and its neighbor at k+1 are already adjacent within the same word. The SWAR pairwise sum processes all words in one pass:

1. **SWAR step**: Sum adjacent counter pairs into wider counters (same masks as the popcount stages described above).
2. **Overflow check**: Scan the widened data for any value that exceeds the original width's maximum. This uses a single bit-mask AND per word.
3. **If no overflow**: Narrow the widened sums back to the original width and compact pairs of words into one (`swar_narrow_compact`). Width is preserved, capacity stays the same.
4. **If overflow**: Keep the widened result, widen the counter width by one step.

Either way, one downscale step is consumed: indices are halved by right-shifting `index_start`, `index_end`, and `index_base`.

#### Clone + scatter-add (unified path)

Both downscale and counter-widening use a single `do_downscale(change, min_width)` function that works identically at all counter widths (B1 through U64):

1. **Clone** the `[u64; N]` data array.
2. **Determine output width**: walk the old live range, summing each group of `2^change` adjacent counters that share the same shifted index.  Track the maximum group sum to find the minimum `Width` whose `counter_max` accommodates all sums (at least `min_width`).
3. **Zero the data array** and compute a fresh, word-aligned `index_base`.
4. **Scatter-add**: for each old bucket, read its counter from the clone, compute the shifted output index, and add the value into the corresponding slot of the zeroed output buffer.

This linear read→fold→write pattern avoids in-place alignment fixups and is SIMD-friendly.  The alignment invariant (`index_base` aligned to `slots_per_word`) is trivially maintained because the base is computed fresh each time.

### Counter widening

When a bucket counter saturates (e.g., a B1 counter already holds 1 and needs to record another observation), the histogram must widen all counters.  This calls `do_downscale(1, current_width.wider())`, which merges adjacent pairs and ensures the output width is at least one level wider than the input:

```text
Widening cascade for Histogram<16> (16 bucket words):

B1 ──saturate──► B2 ──saturate──► B4 ──saturate──► U8 ──► U16 ──► U32 ──► U64
│                │                │                │       │       │       │
1024 slots       512 slots        256 slots        128     64      32      16
max=1            max=3            max=15           max=255 max=64K max=4G  max=2⁶⁴

Each transition: counters merge pairwise, scale decreases by 1.
```

The transition preserves the total count across all buckets: the sum of all counters before and after widening is identical. Resolution is lost (adjacent buckets are merged), but no data is destroyed.

## Choosing Your Parameters

`Histogram<N>` has one compile-time parameter:

### Pool size `N` (u64 words)

`N` controls data-pool size: each histogram uses exactly `N × 8` bytes of
bucket storage. Aggregate stats are stored separately as `count: u64`
and `sum/min/max: f64` (32 bytes total).
Larger `N` means more buckets, which means finer resolution before downscaling.

| N | Data pool bytes | Bucket words | B1 slots | U64 slots |
|---:|---:|---:|---:|---:|
| 8 | 64 | 8 | 512 | 8 |
| 16 | 128 | 16 | 1024 | 16 |
| 32 | 256 | 32 | 2048 | 32 |

**Rule of thumb**: For typical latency distributions (0.1ms–10s), `N=16`
provides 1024 B1 buckets — enough for scale 4 (16 buckets/octave, ~2.2%
relative error) across the full range.

### Other configuration

| Method | Effect |
|--------|--------|
| `with_scale(s)` | Set exact starting scale (returns `Err` if invalid; does not clamp) |
| `with_min_width(w)` | Skip sub-byte widths — e.g., `U8` for faster ops at the cost of fewer initial buckets |

Run `cargo run --example sizing` for an interactive capacity explorer.

## API Overview

### Recording values

```rust,ignore
use otel_expohisto::Histogram;

let mut h: Histogram<16> = Histogram::new();

// Record a single observation
h.update(42.0).unwrap();

// Record a value with a count (weighted recording)
h.record(3.14, 5).unwrap();   // records 3.14 five times
```

Both `update` and `record` return `Result<(), Overflow>`. On error the histogram is unchanged (see [Error Handling](reference.md#error-handling--atomicity)).

### Reading via `HistogramView`

All read access goes through `view()`, which returns an immutable `HistogramView`:

```rust,ignore
let v = h.view();
let s = v.stats();
s.count                        // u64  — total observations
s.sum                          // f64  — arithmetic sum
s.min                          // f64  — minimum value
s.max                          // f64  — maximum value
v.scale()                      // i32  — current mapping scale

// Iterate over non-empty positive buckets
let buckets = v.positive();
buckets.offset()               // i32  — index of the first bucket
buckets.len()                  // u32  — number of contiguous buckets
buckets.width()                // Width — current counter width
for count in &buckets {
    // each count is u64
}
```

### Quantile estimation

`HistogramView::quantiles` walks the histogram CDF and yields estimated values via linear interpolation within each straddling bucket. Quantile 0.0 returns `min`, quantile 1.0 returns `max`.

```rust,ignore
let v = h.view();
for qv in v.quantiles(&[0.5, 0.9, 0.99]) {
    println!("p{:.0} ≈ {:.3}", qv.quantile * 100.0, qv.value);
}
```

The returned `QuantileIter` implements `Iterator<Item = QuantileValue>` and `ExactSizeIterator`.

### Merging

Three merge strategies enable flexible aggregation:

```rust,ignore
use otel_expohisto::Histogram;

let mut a: Histogram<16> = Histogram::new();
let b: Histogram<16> = Histogram::new();

// Same-size merge
a.merge_from(&b).unwrap();

// Cross-size merge (different N parameters)
let c: Histogram<32> = Histogram::new();
a.merge_from(&c).unwrap();
```

Merge computes the minimum common scale, downscales as needed, and uses snapshot/rollback for atomicity.

### Lifecycle

| Method | Description |
|--------|-------------|
| `new()` | Create at maximum table scale with default settings |
| `with_scale(s)` | Create at exact scale (returns `Err` if invalid; does not clamp) |
| `swap(&mut other)` | Exchange contents with another histogram (O(N) memswap) |
| `width()` | Current counter width (`B1`..`U64`) |

### Constructor chain

```rust,ignore
use otel_expohisto::{Histogram, Width};

let h: Histogram<16> = Histogram::new()
    .with_scale(8)?                                   // set starting scale
    .with_min_width(Width::U8)           // skip sub-byte widths
    .with_min_width(Width::B1);    // disable cold-start optimization
```

### Standalone `Scale` API

The `Scale` struct is re-exported at the crate root for direct value-to-index conversion, independent of any histogram instance:

```rust,ignore
use otel_expohisto::{Scale, ScaleError, MAX_SCALE, MIN_SCALE, table_scale};

// Create a mapping at scale 8
let m = Scale::new(8).unwrap();
assert_eq!(m.scale(), 8);

// Map a value to its bucket index
let idx = m.map_to_index(3.14);

// Get the lower boundary of a bucket
let boundary = m.lower_boundary(idx).unwrap();
assert!(boundary <= 3.14);

// Scale constants
assert_eq!(MIN_SCALE, -10);
assert_eq!(MAX_SCALE, 16);
assert!(table_scale() <= MAX_SCALE);
```

`Scale::new(scale)` returns `Err(ScaleError::InvalidScale)` for scales outside \[-10, `table_scale()`\]. `lower_boundary()` returns `Err(ScaleError::Underflow)` or `Err(ScaleError::Overflow)` when the index corresponds to a subnormal or infinite value.

### Trait implementations

| Type | Traits |
|------|--------|
| `Histogram<N>` | `Clone`, `Default` (calls `new()`), `Debug` |
| `HistogramView` | `Debug` |
| `BucketView` | `Debug`, `IntoIterator` |
| `BucketsIter` | `Iterator<Item = u64>`, `ExactSizeIterator`, `Debug` |
| `QuantileIter` | `Iterator<Item = QuantileValue>`, `ExactSizeIterator`, `Debug` |
| `QuantileValue` | `Clone`, `Copy`, `Debug`, `PartialEq` |
| `Width` | `Clone`, `Copy`, `Debug`, `PartialEq`, `Eq`, `PartialOrd`, `Ord` |
| `Stats` | `Clone`, `Copy`, `Debug` (also has `Stats::EMPTY` constant) |
| `Error` | `Clone`, `Copy`, `Debug`, `Display`, `PartialEq` |
| `ScaleError` | `Clone`, `Copy`, `Debug`, `Display`, `Error`, `PartialEq`, `Eq` |
| `Scale` | `Clone`, `Copy`, `Debug` |
