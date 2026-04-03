# Second Review: `src/histogram/merge.rs` — April 2, 2026

## 🔴 Potential Bugs

### 1. `debug_assert` guarding `1i32 << tm_log` (line 151–159)

The `debug_assert!(tm_log < 31)` only fires in debug builds.  In release,
if `tm_log ≥ 31`, `1i32 << tm_log` wraps (Rust shift semantics mask the
shift count), producing garbage for `total_merge`, `aligned_start`, and all
downstream index arithmetic.  While scale bounds may prevent this in
practice, there is no hard proof — this should be a proper `assert!` or the
`debug_assert!` should be promoted.

### 2. `widen_to` has no guard against narrowing (line 268–274)

```rust
let change = new_width.subtract(old_width) as u32;
```

If `new_width < old_width` (due to a bug elsewhere), `subtract` returns a
negative `i32`, the `as u32` cast wraps to a huge value, and `change_scale`
panics or corrupts the scale.  A `debug_assert!(new_width >= old_width)`
would catch this cheaply.

### 3. `max_a + max_b` can theoretically overflow `u64` (line 193)

At `Width::U64`, `or_fold_lanes` is the identity — it returns the full word
value.  If two words sum to > `u64::MAX`, the addition wraps silently, and
`Width::from_max_value` returns a bogus width.  In practice this is
unreachable (the combined count was checked at the top), but the local code
doesn't prove it.  A `saturating_add` would make the safety self-evident
with no performance cost.

---

## 🟡 Design / Robustness Concerns

### 4. Two-phase tm_log fixup (lines 75–108)

The code first computes `self_change = max(range_change, width_change)`
covering range and width requirements, then has a *second* fixup block
(lines 96–108) for the `tm_log ≥ 0` constraint.  The comment on line 75
lists three requirements, but only two are computed in `self_change`.  This
makes the algorithm harder to reason about and harder to prove correct.
Folding the `tm_log` constraint into the initial computation would simplify
the logic and eliminate the second `buckets_empty` check + potential
double-downscale.

### 5. `Scale::new(...).expect(...)` can panic (lines 88, 103–104)

Two paths set the scale directly via `Scale::new(new_scale).expect("valid
scale")` when self is empty.  If extreme inputs produce a scale below
`MIN_SCALE`, this panics rather than returning an error.  Since `merge_from`
already returns `Result`, propagating these as errors would be more robust.

### 6. `buckets_empty()` checks `data[0]` unconditionally (mod.rs:249)

This is outside `merge.rs` but affects it critically.  The function checks
`self.data[0] == 0` rather than `self.data[self.data_idx(self.word_start)]`.
The invariant that "the active word is always at physical index 0 when
`word_start == word_end`" holds today due to careful rebasing in
`extend_word_range` and `do_downscale`, but it is fragile and undocumented.
One wrong rebase would silently break this.

### 7. No snapshot/rollback for partially-modified self

The old design used clone-and-restore.  The current approach documents
`merge_buckets` as "infallible" and commits stats only after buckets are
merged.  This is cleaner, but if any `expect()` panics mid-merge or a future
code change introduces a fallible path, self is left in an inconsistent
state.  A brief comment explaining why atomicity is not needed
("`merge_buckets` is infallible after count-overflow check") would help
future maintainers.

---

## 🟢 Observations (Correct but Subtle)

### 8. Plain `+=` in `repack_source` inner loop (line 240) is safe

At first glance, `value += widen(...)` looks like it could cause cross-lane
contamination via carry.  However, when `group > 1` (cross-word summing),
`cur` is always `U64` (1 lane per word), so plain addition IS correct.  When
`group == 1`, there is no addition.  This invariant is non-obvious and
deserves a brief comment.

### 9. `or_sums` overflow check is conservative but safe (line 249)

The OR-accumulation of max-lane values can overestimate the required width
(e.g., `0b0101 | 0b1010 = 0b1111`).  This triggers unnecessary widening but
never misses actual overflow.  The wasted work is bounded (at most 6 widen
steps to U64).

### 10. The retry loop terminates (lines 169–195)

Each iteration either succeeds (`break`) or widens by at least one step.  At
`U64`, every lane is one `u64`, and `swar_add_checked` becomes
`checked_add` which succeeds because the combined count fits in `u64`.  So
the loop terminates in at most 6 iterations.

---

## 📝 Code Quality

### 11. ~80% of merge tests are commented out (tests.rs)

Lines 301–1885 contain dozens of commented-out merge tests including
comprehensive equivalence tests, regression tests, cross-size tests, and
fuzz reproducers.  These represent significant coverage that is not running.
Either restore them or remove the dead code.

### 12. `sums` fixed-size array (line 228)

`let mut sums = [0u64; 64]` is stack-allocated at 512 bytes per call.
Since `repack_source` is called in a hot loop per destination word, this is
fine for correctness but worth noting for performance awareness.

### 13. The `Stats` struct passed to `commit_stats` is confusing (line 38–43)

The `Stats` passed to `commit_stats` has `min: other.stats.min` (not the
combined min).  The combining happens inside `commit_stats` via `f64::min`.
While correct, constructing a `Stats` with half-baked values and relying on
the callee to fix them is a readability trap.
