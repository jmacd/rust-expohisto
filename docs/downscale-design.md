# Downscale Algorithm Redesign

## Key Principle

Data `[u64; N]` is always a ring buffer at all widths.  Wrapping is
aligned to u64 word boundaries so SWAR operations never cross a word
boundary.  `slot_for()` uses `rem_euclid(cap)` uniformly.

## Algorithm

Given: downscale by `change` steps (merge groups of K = 2^change
adjacent buckets).

### Phase 1: SWAR-Widen-and-Max (single pass)

`swar_steps = min(change, log2(spw))`.  Instead of iterating
`swar_step` N times, apply a **single fused operation** per word
that widens from `current_width` to `current_width << swar_steps`
and simultaneously computes the max lane value.

There are 6 distinct step-count algorithms (1-step through 6-step),
each parameterized on input width for mask constants:

- **1-step**: one shift-mask-add (= existing `swar_step`)
- **2-step**: two rounds of shift-mask-add (e.g. B1→B4 = classic
  "count pairs of bits")
- **3-step through 5-step**: chained shift-mask-add rounds
- **6-step**: B1→U64 = `popcount`

Each function processes one word: widens lanes, returns the max lane
value in that word.  The caller loops over all N words, tracking the
global max.

After this phase, lane width = `current_width << swar_steps`.  Each
lane holds the sum of `2^swar_steps` original counters.
`max_lane_value` is known.

### Phase 2: Dispatch

- **swar_steps == change → Case A** (intra-word): all group sums are
  SWAR lanes.  Repack at target width.
- **swar_steps < change → Case B** (inter-word): each word is now a
  single U64 value.  Groups of `2^(change - swar_steps)` consecutive
  words must be summed, then repacked.

### Case A — Intra-Word Repack

All group sums live in SWAR lanes at width `W << change`.

1. Compute `actual_width = from_max_value(max_lane_value).max(min_width)`.
2. Scan words front-to-back: extract widened lanes, write at
   `actual_width` packing.  Since output lanes are narrower (or
   equal), output fits in ≤ source words — safe in-place.
3. `index_base >>= change`.

No alignment issues: `index_base` is word-aligned, group boundaries
align with SWAR lane boundaries, ring buffer just shifts.

### Case B — Inter-Word Group-Sum

After SWAR to U64, each data word holds one U64 value in a ring
buffer.  Output groups span `words_per_group = 2^(change - swar_steps)`
consecutive words.

Use pigeonhole (`count / num_groups`) to estimate the output width
before computing exact sums.

**In-place group-sum:**

1. **First group (wrapping):** Group 0's words may span the end and
   start of the data array (ring buffer wrap).  Sum all its words
   (reading with modular indexing), write result to `data[0]`.
2. **Forward pass:** For groups 1..num_groups, each group's source
   words come after the previous group's.  Sum each group's words,
   write to `data[g]`.  Safe because `g < g * words_per_group`
   (first source word of group g) for `words_per_group ≥ 2`.
3. Zero `data[num_groups..N]`.
4. Repack `data[0..num_groups]` from U64 to `actual_width`.

## Todos

1. **swar-max-tracking** — Fused SWAR-widen-and-max
   Implement 6 step-count functions (1-step through 6-step), each
   parameterized on input width.  Each processes one word: widens
   lanes from input to output width and returns max lane value.
   1-step = existing swar_step + max.  6-step (B1→U64) = popcount.
   Replaces iterative swar_step loop.

2. **case-a-repack** — Case A intra-word repack
   *Depends on: swar-max-tracking.*
   After SWAR-widen by change steps, scan words front-to-back:
   extract widened lanes, write at actual_width.  index_base >>=
   change.  Ring buffer aware, in-place safe because output lanes
   are narrower.

3. **case-b-first-group** — Case B wrap-aware first group
   *Depends on: swar-max-tracking.*
   Inter-word path: first output group may wrap around the ring
   buffer.  Sum its words with modular indexing, write to data[0].

4. **case-b-forward** — Case B forward-pass group-sum
   *Depends on: case-b-first-group.*
   Groups 1..num_groups: sum consecutive words, write data[g].
   Safe in-place because g < g * words_per_group.

5. **case-b-repack** — Case B repack U64 to target
   *Depends on: case-b-forward.*
   Repack data[0..num_groups] from U64 group sums to actual_width.

6. **remove-linearize** — Remove U64 linearization
   *Depends on: case-b-repack.*
   Remove the ring buffer linearization step from current inter-word
   path.  Ring buffer handled natively now.

7. **cleanup** — Update docs and stale code
   *Depends on: case-a-repack, remove-linearize.*
   Update module docs, remove stale comments about
   rotation/alignment.

8. **test-fuzz** — Test and fuzz validation
   *Depends on: cleanup.*
   Run full test suite and fuzz targets to validate.
