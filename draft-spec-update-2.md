# Draft Specification Update v2: Allocation-Free ExponentialHistogram

> **Status**: Draft for discussion
> **Scope**: Updates to the [Metrics Data Model § ExponentialHistogram][data-model]
> and the [Metrics SDK § Base2 Exponential Bucket Histogram Aggregation][sdk-spec]
>
> Two enhancements:
>
> 1. **Exact lookup-table mapping** for positive scales (integer-only).
> 2. **Allocation-free fixed-size layout** using U64-aligned variable-width
>    counters in a ring buffer, with SWAR widening and narrowing.

[data-model]: https://opentelemetry.io/docs/specs/otel/metrics/data-model/#exponentialhistogram
[sdk-spec]: https://opentelemetry.io/docs/specs/otel/metrics/sdk/#base2-exponential-bucket-histogram-aggregation

---

## 1. Exact Lookup-Table Mapping

### Tables

Two tables are generated at compile time for a chosen table scale `H`,
where `N = 2^H` and `SHIFT = 51 − H`:

- **`BOUNDARIES[N + 3]`** — exact 52-bit significands at each of the
  `N` sub-bucket boundaries within one octave:

  ```
  BOUNDARIES[0]     = 0           // sentinel for upper-inclusive semantics
  BOUNDARIES[1]     = 1           // forces exact powers of two below the boundary
  BOUNDARIES[k+1]   = significand_bits(ceil_to_double(2^(k/N)))   for k = 1..N−1
  BOUNDARIES[N+1]   = 2^52        // trailing sentinel
  BOUNDARIES[N+2]   = 2^52        // trailing sentinel
  ```

- **`INDEX_TABLE[2N]`** — for each of `2N` equal-width significand
  regions, the largest exponential bucket index whose boundary does
  not exceed the region's start:

  ```
  INDEX_TABLE[i] = max { j : BOUNDARIES[j+1] ≤ (i << SHIFT) }       for i = 0..2N−1
  ```

### Mapping algorithm

Given a positive IEEE-754 double with 52-bit significand `s` (implicit
leading 1 stripped) and unbiased exponent `e`:

```
function map_to_index(s, e, scale):                # 1 ≤ scale ≤ H
    approx  = INDEX_TABLE[s >> SHIFT]              # linear approximation
    bucket  = approx + (1 if s ≥ BOUNDARIES[approx + 1] else 0)
    fine    = (e << H) + bucket − 1                # H-scale index, upper-inclusive
    return fine >> (H − scale)                     # downscale to requested scale
```

### Why one correction suffices

The maximum number of exponential boundaries in any single linear
region of width `2^52 / 2N` is

```
Δ = N · log2(1 + 1/(2N))  ≤  1/(2·ln 2)  ≈  0.721
```

Since `Δ < 1`, the linear lookup is exact or off by exactly one;
a single comparison resolves the case.

### Properties

- Exact for every IEEE-754 double; no `ln` required at runtime.
- Constant time: shift, table lookup, compare, two adds, shift.
- Perfect subsetting: computing at `H` then shifting is identical to
  computing directly at any lower scale.
- Table size for the recommended default `H = 8`: ~1.5 KiB.

---

## 2. Allocation-Free Fixed-Size Layout

### Storage

The histogram is a fixed struct holding:

```
scale            : i32                 # current resolution
width            : enum {B1,B2,B4,U8,U16,U32,U64}    # current counter width
word_base        : i32                 # word index that maps to data[0]
word_start       : i32                 # lowest active word index
word_end         : i32                 # highest active word index (inclusive)
data[N]          : [u64; N]            # ring buffer of packed counters
stats            : { count, sum, min, max }
```

There is no allocation; `N` is a compile-time constant.

### Counter widths

Each `u64` of `data` holds `64 / bits_per_slot(width)` counters of the
chosen width:

```
B1: 64 slots/u64,  max=1          U16: 4 slots/u64,  max=65535
B2: 32 slots/u64,  max=3          U32: 2 slots/u64,  max≈4·10⁹
B4: 16 slots/u64,  max=15         U64: 1 slot /u64
U8:  8 slots/u64,  max=255
```

A bucket index `i` decomposes into:

```
word_index  = i >> log2(slots_per_u64(width))
sub_offset  = i  &  (slots_per_u64(width) − 1)
shift       = sub_offset · bits_per_slot(width)
mask        = counter_max(width) << shift
```

The counter value at `i` lives in `data[ data_idx(word_index) ]` at
`shift`, masked by `mask`.

### Ring buffer addressing

`data` is a circular buffer over word indices; physical position is

```
data_idx(widx) = (widx − word_base) mod N
```

An insertion that extends `[word_start, word_end]` by one word costs a
single zero-fill of the new physical slot. Downscale shifts
`word_base, word_start, word_end` right by the scale change so that the
ring rotation tracks the new index space.

### SWAR widen and narrow

Widening one width step doubles each lane by pair-summing within a
`u64`. The seven primitives are:

```
b1→b2(w): (w & 0x5555…) + ((w >> 1) & 0x5555…)
b2→b4(w): (w & 0x3333…) + ((w >> 2) & 0x3333…)
b4→u8(w): (w & 0x0F0F…) + ((w >> 4) & 0x0F0F…)
u8→u16(w):(w & 0x00FF…) + ((w >> 8) & 0x00FF…)
u16→u32(w):(w & 0x0000FFFF…) + ((w >> 16) & 0x0000FFFF…)
u32→u64(w):(w & 0xFFFFFFFF) + (w >> 32)
```

Multi-step widening chains them; `B1 → U32` and `B1 → U64` shortcut to
`count_ones`. Each widen halves slots per word and doubles bytes per
counter; the lane sums never overflow because doubling lane width
exactly fits the worst-case sum of two narrower lanes.

Narrowing reverses the operation by repacking pairs of consecutive
*output-width* words into one *narrower-width* output word, masking
each lane to the new width. Narrowing is allowed only when no lane
exceeds the target width's `counter_max` (verified by an OR-fold
across all active words).

---

## 3. Pseudocode

Notation: `slot_addr(width, i)` returns `(word_index, shift, mask)`.
`counter_max(w)` is the lane saturation limit at width `w`.

### 3.1 Insert

```
function update(value, incr):
    if value is NaN, ±Inf, or negative:        return Error::Extreme
    if value == 0:    stats.count += incr;     return Ok
    (s, e) = decompose_ieee754(|value|)        # subnormals fold to (1, MIN_EXP)
    update_stats(value, incr)
    loop:
        index  = map_to_index(s, e, scale)     # § 1
        result = try_increment(index, incr)
        if resolve_increment(result):          return Ok

function try_increment(index, incr):
    (widx, shift, mask) = slot_addr(width, index)

    if buckets_empty():
        word_start = word_end = word_base = widx
    else if widx < word_start:
        if (word_end − widx) ≥ N:              return NeedsDownscale(widx, word_end)
        zero data[data_idx(w)] for w in widx .. word_start−1
        word_start = widx
    else if widx > word_end:
        if (widx − word_start) ≥ N:            return NeedsDownscale(word_start, widx)
        zero data[data_idx(w)] for w in word_end+1 .. widx
        word_end = widx

    word    = data[data_idx(widx)]
    current = (word >> shift) & mask
    new     = current + incr
    if new > counter_max(width):               return CounterOverflow(new)
    data[data_idx(widx)] = (word & ~mask) | (new << shift)
    return Ok

function resolve_increment(result):
    case Ok:                       return true
    case NeedsDownscale(lo, hi):
        steps = ⌈log2((hi − lo + 1) / N)⌉
        downscale(steps, min_output_width = width)
        return false
    case CounterOverflow(total):
        new_width = smallest width with counter_max ≥ total
        steps     = level(new_width) − level(width)
        if buckets_empty(): width = new_width
        else:               downscale(steps, min_output_width = new_width)
        return false
```

### 3.2 Downscale

`downscale(change, min_output_width)` reduces scale by at least
`change` steps while keeping the active range in `N` words. Three
phases (widen, optional cross-word grouping, narrow-and-repack) can
each contribute scale steps; the function returns the actual number
of steps consumed.

```
function downscale(change, min_output_width):
    abs_budget = scale − MIN_SCALE             # never push below MIN_SCALE
    cur        = width
    total_or   = 0
    total_widen = 0

    # Phase 1: widen up to `change` steps in place (capped at U64 / budget).
    first_widen = min(change, U64 − cur, abs_budget)
    if first_widen > 0:
        new_cur = cur + first_widen
        for widx in word_start..=word_end:
            data[data_idx(widx)] = swar_widen(cur, new_cur, data[data_idx(widx)])
            total_or |= or_fold_lanes(new_cur, data[data_idx(widx)])
        cur, total_widen = new_cur, first_widen

    # Phase 2: keep widening one step at a time until lanes have headroom
    # OR we've reached U64 / budget.
    while cur < U64 and total_widen < abs_budget:
        required = smallest width holding total_or
        if (cur − required) ≥ change and cur ≥ min_output_width: break
        prev = cur; cur = cur + 1
        total_or = 0
        for widx in word_start..=word_end:
            data[data_idx(widx)] = swar_widen(prev, cur, data[data_idx(widx)])
            total_or |= or_fold_lanes(cur, data[data_idx(widx)])
        total_widen += 1

    # Phase 3: at U64, sum aligned groups of words to consume the rest.
    cross_steps = 0
    if cur == U64:
        while not (total_widen + cross_steps ≥ change and headroom_ok()):
            if total_widen + cross_steps ≥ abs_budget: break
            cross_steps += 1
            recompute total_or as OR of group sums of size 2^cross_steps

    # Phase 4: narrow and repack into N-aligned output words.
    max_narrow   = cur − max(min_output_width, width)
    headroom_cap = scale − (total_widen + cross_steps) − MIN_SCALE
    narrow_steps = min(change − cross_steps, max_narrow, headroom_cap)
    word_shift   = cross_steps + narrow_steps
    output_width = cur − narrow_steps
    new_word_base = word_base >> word_shift

    for each aligned input group of 2^word_shift words, in two passes
        (forward from word_base group, then backward) to avoid clobber:
            value     = sum of `data[data_idx(w)]` for the `2^cross_steps` words
                        × `2^narrow_steps` lane-narrowing repacks
            packed    = swar_narrow(cur, output_width, value)
            data[(out_widx − new_word_base) mod N] = packed

    word_start >>= word_shift
    word_end   >>= word_shift
    word_base  >>= word_shift
    width       = output_width
    scale      −= total_widen + cross_steps
    return total_widen + cross_steps
```

The forward/reverse pass split prevents read/write aliasing because
input words at indices `< word_base` map to high physical addresses
(ring wrap-around) and must be read before being overwritten. The
phase-2 loop can consume **more** scale than `change` (returning a
larger value) when the OR-fold demands a wider lane.

### 3.3 Merge

```
function merge_from(other):
    if other.count == 0: return Ok
    new_count = self.count + other.count           # check overflow

    src_scale  = other.scale
    src_width  = other.width
    merge_w    = max(self.width, src_width)
    min_scale  = min(self.scale, src_scale)

    # 1. Prepare self: align scale + width to fit the union range in N words.
    combined   = union(slot_range_at_scale(self,  min_scale),
                       slot_range_at_scale(other, min_scale))
    extra      = ⌈log2((combined.high − combined.low + 1) / N)⌉   at width merge_w
    range_chg  = max(self.scale − (min_scale − extra), 0)
    width_chg  = level(merge_w) − level(self.width)
    self_chg   = max(range_chg, width_chg)
    downscale(min(self_chg, self.scale − MIN_SCALE), min_output_width = merge_w)

    # 2. Per-output-word: gather contributing source words, repack, add.
    scale_diff = src_scale − self.scale            # ≥ 0 after step 1
    for each output word_index `wo` overlapping the combined range:
        source_value = 0
        for each source word `ws` whose buckets fall into `wo` after
            shifting indices by `scale_diff` and lane-narrowing from
            src_width to self.width:
                source_value = swar_repack_into(source_value, ws, scale_diff,
                                                src_width, self.width)
        loop:
            result = swar_add_checked(self.data[data_idx(wo)], source_value, self.width)
            if result is Ok(sum):
                self.data[data_idx(wo)] = sum
                break
            else:                                  # any lane overflowed
                widen self by one step (in place) and retry; lane-group
                boundaries are invariant under widening so source_value
                can be re-derived consistently.

    update_stats(self, other)
    return Ok
```

`swar_add_checked(a, b, width)` performs lane-parallel addition and
detects overflow by checking whether any lane sum exceeds
`counter_max(width)`. Because each widening step doubles lane capacity
and the worst-case lane sum after one step at most doubles, retry
terminates in at most `U64 − width` iterations.

---

## 4. Recommended Specification Changes

### Data model

In *ExponentialHistogram § Producer Expectations*, after the paragraph
beginning "Producers MAY use an inexact mapping function", add:

> Producers MAY use a compile-time lookup table to compute exact
> bucket indices for positive scales using only integer operations.
> The table consists of `2N` linear-region indices (`N = 2^H` for
> table scale `H`) and `N + 3` exact 52-bit boundary significands.
> A single comparison after the linear lookup yields the exact
> bucket index. This method has no rounding error and requires no
> special case for powers of two.

### SDK

In *Base2 Exponential Bucket Histogram Aggregation*, after the
configuration parameters, add:

> Implementations MAY represent buckets as a fixed-size pool of
> `N` 64-bit words holding variable-width counters
> (1, 2, 4, 8, 16, 32, or 64 bits). Counters start at the narrowest
> width and widen automatically when any counter would overflow.
> Each widening halves the number of available slots and may
> trigger a corresponding scale reduction to maintain the ideal
> scale invariant. The pool is addressed as a ring buffer rotated
> by a `word_base` field, so insertions in either direction cost
> only a single zero-fill.
>
> The effective `MaxSize` at any moment is
> `total_bits / bits_per_counter`. Implementations using this
> representation report `MaxSize` as the maximum slot count
> (at the narrowest counter width).
