# Merge Implementation Summary — March 31, 2026

## Overview

Implemented `merge_from` for `Histogram<N>`, the allocation-free
exponential histogram crate on branch `jmacd/v2`.  Two commits:

- **`d1fbaee`** — Core merge implementation (word-by-word with SWAR widen + group_sum)
- **`be87264`** — Word-level fast path optimization (swar_add_checked)

## Merge Algorithm

### Phase 1: Target scale computation

Project both histograms' slot ranges to `min(self_scale, other_scale)`,
compute the union, determine how many extra downscale steps are needed
to fit the combined word range in N words, then downscale self to the
target scale.  When self is empty, just adopt the target scale/width
directly.

### Phase 2: Word-by-word merge from source

Decompose `shift = other_scale - self.scale` into in-word and
cross-word parts:

```
in_word_steps  = min(shift, src_width.to_u64_widen_steps())
cross_steps    = shift - in_word_steps
```

Three merge paths depending on the situation:

**Fast path** (`cross_steps == 0` AND `widened_width == dest_width`):
Each source word maps 1:1 to a destination word (always aligned, no
straddling).  Uses `swar_add_checked` to add whole words directly.
On overflow: widen ALL dest words in place, widen the source
contribution to match, retry.  No fallback to lane-by-lane.

**In-word slow path** (`cross_steps == 0`, widths don't match):
Widen each source word, extract lanes, `retry_increment` each lane.
The closure `|h| src_slot >> (src_scale - h.scale)` recomputes the
destination slot index on each retry, adapting to any scale/width
changes caused by overflow handling.

**Cross-word path** (`cross_steps > 0`):
Widen source words to U64, accumulate aligned groups of `2^cross_steps`
words, one `retry_increment` per group sum.

### Atomicity

Snapshot/rollback: `self.clone()` before merge, restore on `Err`.
Stats committed only on success.

## New SWAR Primitives

### `swar_add_checked(a, b, width) -> Option<u64>`

SWAR addition with per-lane overflow detection.  Algorithm:

1. Zero the MSB of each lane in both operands
2. Add the lower bits — no inter-lane carry since MSB was zeroed
3. Detect overflow via the majority function:
   `overflow = (a_msb & b_msb) | (a_msb & carry) | (b_msb & carry)`
4. Compute correct result: `sum_lo ^ a_msb ^ b_msb`

Returns `None` without modifying anything if any lane would overflow.
For U64 width, delegates to `u64::checked_add`.

### `Width::msb_mask()`

Returns the MSB bit of each SWAR lane:
B1=`0xFFFF...`, B2=`0xAAAA...`, B4=`0x8888...`, U8=`0x8080...`, etc.

## Key Property: Word Index Invariance

The 1:1 word mapping is **preserved across widening**.  Proof:

```
dest_slot = (src_widx × src_slots_per_word) >> shift
          = src_widx × dest_slots_per_word
dest_word = src_widx
```

When both sides widen by `c` steps, the slot count halves but the
word index stays the same (half the slots × double the bits = same
u64).  This means the fast path can widen in place and continue
word-by-word without recomputing any index mapping.

## Supporting Changes

- **`HighLow::merge()` / `HighLow::empty()`** — range union helpers
- **`slot_range()` / `slot_range_at_scale()`** — project histogram
  slot ranges to a given scale for overlap computation
- **`extend_word_range()`** — pre-extend and zero-fill dest word
  range before the merge loop
- **`BucketView::offset()`** — fixed to return slot-level index
  (was word-level)
- **`BucketView::len()`** — new, returns total slot count
- **`BucketView::iter()`** — fixed inverted `is_empty` condition

## Bug Fixes Found Along the Way

1. **BucketView iterator inversion**: `is_empty().then(|| start)`
   created iterators for EMPTY histograms.  Fixed to
   `(!self.is_empty()).then(...)`.

2. **Counter overflow on empty histogram**: When the first-ever
   increment exceeds B1 max (e.g., `record_incr(1.0, 1000)`),
   `resolve_increment` tried to `downscale_by` on empty data,
   hitting a debug assert.  Fixed: when `buckets_empty()`, just
   set `self.current.width` directly.

3. **Fast path source widen after overflow**: After the overflow
   handler widens dest and updates `widened_width`, subsequent source
   words must be widened from `src_width` to the new `widened_width`.
   The original code checked `in_word_steps > 0` (a stale pre-overflow
   value).  Fixed: check `widened_width != src_width` instead.

## Files Changed

| File | Changes |
|------|---------|
| `src/histogram/merge.rs` | New: full merge implementation + fast path |
| `src/histogram/swar.rs` | New: `swar_add_checked`, `msb_mask`, 14 tests |
| `src/histogram/mod.rs` | New: `HighLow` helpers, `slot_range`, `extend_word_range`, empty-histogram overflow fix |
| `src/histogram/bucket_ops.rs` | `widen_words` visibility → `pub(crate)` |
| `src/histogram/bucket_view.rs` | `offset()` fix, `len()`, iterator fix |
| `src/histogram/width.rs` | `slot_to_word_index` → `pub(crate)` |
| `src/histogram/tests.rs` | 9 merge tests |

## Test Results

51 tests pass (28 pre-existing + 9 merge + 14 swar_add_checked).
All three fuzz targets compile clean.
