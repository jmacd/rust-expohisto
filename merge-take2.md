# Merge Take 2 — Discussion Summary

## Starting Point

The current `merge.rs` (from `merge-summary-mar31.md`) has three merge
paths: a **fast path** (swar_add_checked word-by-word), a **slow path**
(per-lane retry_increment), and a **cross-word path** (group sums via
retry_increment).  It uses snapshot/rollback for atomicity.

## Problems Identified

1. **Snapshot/rollback is premature** — the recording path
   (`retry_increment` → `resolve_increment`) never snapshots; it
   mutates in place and iterates.  Merge should do the same.

2. **The slow path is unnecessary** — it exists only for the
   width-mismatch case (`widened_width != self.current.width`).
   An iterative widening approach (like `downscale.rs` Phase 2)
   should eliminate the need for per-lane processing entirely.

## Key Insight: Repack the Source On-the-Fly

Instead of adjusting self's width to match the widened source (which
creates circular dependencies — widening self increases shift, which
increases widened_width, which stays ahead), **downscale the source
to self's scale and width** during the merge.

This uses the same `repack_group` logic from `downscale.rs`, applied
to the source histogram's data.  Each group of source words is
widened, grouped, and narrowed to produce a single dest-width SWAR
word, which is then added via `swar_add_checked`.

## Unified Algorithm

### Phase 1: Target Scale (unchanged)

Project both histograms' slot ranges to `min(self_scale, other_scale)`,
compute the union, determine how many extra downscale steps are needed
to fit the combined word range in N words, then downscale self.

### Phase 1.5: Ensure Total Merge ≥ 1

`total_merge = 2^(shift + src_width - dest_width)` is the number of
source words that map to each dest word.  When `total_merge < 1`
(self has much wider counters than source at a similar scale),
iteratively downscale self until `shift + src_width >= self.width`.

### Phase 2: Word-by-Word Merge with On-the-Fly Repacking

For each dest word, repack its contributing source words into a
dest-width SWAR word, then `swar_add_checked`.

The repack decomposition (for a group of `total_merge` source words):

```
if shift >= src_width.to_u64_widen_steps():
    cur = U64                              // full widen
    cross_steps = shift - (U64 - src_width)
else:
    cur = src_width.wider_by(shift)        // partial widen
    cross_steps = 0

narrow_steps = cur - dest_width
group = 2^cross_steps       // source words summed per sub-group
repack = 2^narrow_steps     // sub-groups packed per output word
```

Cross-word summing only happens at U64 width, avoiding inter-lane
carry corruption — the same constraint `do_downscale` enforces.

**Overflow handling (iterative widening):**

Before narrowing, check `cur.or_fold_lanes(or_of_sums)` against
`dest_width.counter_max()`.  If too large, widen self
(`widen_words` + `change_scale`), recompute decomposition, retry.

After narrowing, `swar_add_checked` may also overflow.  Same
response: widen self, retry.

### Key Invariant: `total_merge` Is Stable Under Self-Widening

When self widens by δ steps: `dest_width += δ`, `shift += δ`.
Since `tm_log = shift + src_width - dest_width`, the deltas cancel.
Group boundaries don't change — only the internal decomposition
(widen/cross/narrow split) is recomputed.

## What Gets Removed

| Removed | Reason |
|---------|--------|
| Snapshot/rollback in `merge_from` | Iterative widening handles overflow in place |
| `merge_in_word` dispatcher | Subsumed by unified repack |
| `merge_in_word_fast` | Subsumed — the tm_log=0 same-width case naturally produces an identity repack + swar_add_checked |
| `merge_in_word_slow` | Subsumed — width mismatches are resolved by the repack's widen/narrow decomposition |
| `merge_cross_word` | Subsumed — cross-word grouping is part of the repack decomposition |

## What Stays

| Kept | Role |
|------|------|
| `extend_word_range` | Pre-extend and zero-fill dest word range |
| `swar_add_checked` | Per-word addition with overflow detection |
| `widen_words` + `change_scale` | Iterative widening on overflow |

## Cases Unified

| Scenario | Old Path | New Behavior |
|----------|----------|-------------|
| Same scale, same width | Fast path | tm_log=0, identity repack, swar_add_checked |
| Same scale, dest wider | Slow path (per-lane) | tm_log=0, source widened to dest_width, swar_add_checked |
| Same scale, source wider | Slow path (per-lane) | Phase 1.5 downscales self; then repack narrows source |
| Different scale, small shift | Fast or slow path | Partial widen + narrow repack |
| Different scale, large shift | Cross-word path | Full widen to U64, cross-word grouping, narrow repack |

## Resolution Trade-off

The old slow path widened self only as much as actual lane sums
required (data-dependent, optimal).  The new approach widens self
based on the source's *maximum possible* lane values after repacking,
which may over-widen.  This is acceptable for immature code —
correctness and simplicity matter more than optimality.  The
data-dependent approach can be added back later as a fast-path
optimization.

## Open Questions

- Should `trim_bucket_range` be called after the merge loop?
  Currently absent from both old and new merge code.
- The Phase 1 target-scale computation uses `self.current.width`
  for `slot_to_word_index`.  Phase 1.5 may change the width.
  Verify (via tests/fuzzing) that the combined word range still
  fits after Phase 1.5.
