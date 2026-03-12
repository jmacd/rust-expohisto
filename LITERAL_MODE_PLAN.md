# Literal Mode Optimization Plan

## Problem

When the first few values are inserted into a histogram, they may span a wide
range, causing repeated downscale + widen operations as each value pushes the
index range further. This work is wasted — the histogram settles to a final
scale only after enough values reveal the true range.

## Solution

Store raw `f64` bit patterns ("literals") in the bucket data words for the first
few observations. On overflow (the `N+1`th non-zero value), compute the optimal
scale from the full observed range and populate buckets directly at that scale.

**Key advantage**: optimal scale and bucket width are determined from the full
set of initial values in one shot, eliminating all incremental downscale/widen
work during the cold-start phase.

## Design

### 1. Mode Representation

Add a `literal: bool` field to the `Histogram` struct. It fits in the existing
1-byte alignment gap between `bucket_width` (offset 18) and `index_base`
(offset 20), so struct size is unchanged.

In literal mode:

- `data[STAT_WORDS .. STAT_WORDS + literal_count]` holds `f64` bit patterns
  (one u64 per observation, via `f64::to_bits()`)
- `index_end` is repurposed to store the literal count (how many non-zero
  values are stored)
- `index_start` and `index_base` are unused (zeroed)
- MMSC fields (count, sum, min, max) are updated normally on every `update()`

Literal capacity = `bucket_word_count()` = `N - P::STAT_WORDS`.

### 2. Update Path (Literal Mode)

On `update_by_incr(value, incr)` when `literal == true`:

1. **Zero values** (`value == 0.0`): No literal storage needed. Only MMSC
   fields are updated (same as current behavior).

2. **Fits**: If `literal_count + incr <= literal_capacity`, store `incr` copies
   of `value.to_bits()` in consecutive data words and increment literal count.

3. **Overflow → Promotion**: If `literal_count + incr > literal_capacity`, call
   `promote_with(value, incr)` to convert to bucket mode, including the
   trigger value in the scale computation.

### 3. Promotion Algorithm

`promote_with(trigger_value, trigger_incr)`:

1. **Snapshot** `self.data` (contains MMSC + literals).
2. **Compute index range** at `max_scale` for all stored literals + the trigger
   value.
3. **Compute optimal scale**: use `change_scale(HighLow { min_idx, max_idx },
   capacity)` to find the minimum downscale needed.
4. **Compute optimal bucket width**: count per-index occurrences across all
   literals + trigger. Pick the narrowest `BucketWidth` whose `counter_max()`
   is ≥ the maximum per-index count. This avoids iterative widening during
   reinsertion.
5. **Switch to bucket mode**: clear bucket data words, set `literal = false`,
   set `bucket_width` to the computed width, set mapping to the computed scale.
6. **Reinsert all values**: for each literal in the snapshot + the trigger
   (with its incr), call `update_buckets()`.
7. **Rollback on error**: if any reinsertion fails (Overflow), restore from the
   snapshot.

A bare `promote()` (no trigger value) is also needed for merge-as-destination,
where self is in literal mode and needs to accept bucket data from the source.

### 4. Transparent Read API (BucketView)

**No breaking API changes.** `positive()` stays `&self`.

In literal mode, `BucketView` presents a *virtual view* that maps literals to
bucket indices on the fly:

- **`offset()`**: compute indices at `effective_scale` for all literals; return
  the minimum index.
- **`len()`**: `max_index - min_index + 1`.
- **`at(pos)`**: count how many literals map to `offset + pos` at the effective
  scale. O(n) per call, where n ≤ literal_capacity (small).

The **effective scale** is what promotion would choose: start from `max_scale`,
apply `change_scale()` to find the scale where the index span fits in
`bucket_capacity()`.

Cache `effective_scale`, `offset`, and `len` in the `BucketView` struct to
avoid recomputing per `at()` call. (Computed once in `positive()`.)

- **`scale()`** in literal mode: return `effective_scale` (not raw `max_scale`).
- **`bucket_width()`** in literal mode: return `min_bucket_width` (the target).
- **`non_zero_count()`** in literal mode: return `literal_count as u64`.

### 5. Merge Handling

All three merge entry points need literal-mode awareness:

#### `merge_from(&mut self, other: &Self)`

- **Self is literal**: call `promote()` first, then proceed normally.
- **Other is literal**: iterate other's literal values. For each literal,
  insert into self's buckets via `update_buckets()`. Commit MMSC
  (count/sum/min/max) from other. Atomicity via snapshot/rollback.
- **Both literal**: promote self first, then insert other's literals.

#### `merge_from_other<M>(&mut self, other: &Histogram<M, P>)`

- If other is literal: iterate its literals and insert via `update_buckets()`.
- If self is literal: promote first.
- Otherwise: delegate to `merge_from_raw` as today.

#### `merge_from_raw(...)`

- If self is literal: promote first, then proceed normally.
- (The source is always raw bucket data, never literal.)

### 6. Clear and Constructors

- **`new()`**: starts in literal mode (`literal = true`).
- **`with_max_scale()`**, **`with_scale()`**: start in literal mode.
- **`clear()`**: reset to literal mode (`literal = true`, bucket data zeroed,
  `index_end = 0`).

### 7. Builder Opt-Out (Optional)

Add `with_literal_mode(enabled: bool)` builder. Default is `true`. Setting
`false` disables literal mode (histogram starts in bucket mode as it does
today). Useful for benchmarks or cases where the caller knows the value range
upfront.

### 8. Debug and Display

`Debug` impl shows `mode: "literal"` or `mode: "bucket"` and, in literal mode,
the stored literal count and values.

## Edge Cases

- **All values identical**: all literals map to the same index. Promotion
  produces one bucket with count = N. Smart width selection (step 3.4) picks
  the right width without iterative widening.
- **update_by_incr with large incr**: if `incr` alone exceeds literal capacity,
  promote immediately with the trigger value and incr.
- **Subnormal values**: stored as literals, handled correctly by
  `map_to_index()` during promotion.
- **Zero-only histograms**: never consume literal slots. Stay in literal mode
  indefinitely (correct: no bucket data needed for zeros).

## Todos

1. `add-literal-field` — Add `literal: bool` field and literal_count accessors
2. `literal-update` — Implement literal-mode path in `update_by_incr`
3. `promote-algorithm` — Implement `promote()` and `promote_with()`
4. `transparent-read` — Update BucketView, scale(), non_zero_count() for literal mode
5. `merge-literal` — Update merge_from, merge_from_other, merge_from_raw
6. `clear-constructors` — Update new(), with_scale(), clear() for literal mode
7. `builder-opt-out` — Add with_literal_mode(bool) builder
8. `debug-impl` — Update Debug to show literal mode
9. `unit-tests` — Tests for literal storage, promotion, merge, edge cases
10. `fuzz-targets` — Update histogram_oracle, merge_oracle, stateful_oracle
11. `docs` — Update module/struct docs and lib.rs example
