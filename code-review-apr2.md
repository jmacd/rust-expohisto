# Code Review: `src/histogram/merge.rs`

## Overview

This file implements histogram merging — combining the bucket data and
stats of one `Histogram<M>` into another `Histogram<N>`, where the two
may differ in pool size, scale, and counter width. The approach is:

1. Check count overflow up front.
2. Reconcile scale and width so `self` can hold the combined range.
3. Word-by-word merge with on-the-fly SWAR repacking.
4. Commit aggregate stats.

The design is clever and the SWAR repacking strategy is genuinely
interesting. That said, I have several concerns ranging from
correctness risk to maintainability.

---

## Concern 1: Two-phase scale adjustment is fragile

`merge_buckets` computes `self_change` (lines 80–82) as
`max(range_change, width_change)`, then applies it. But immediately
after (lines 98–109), there's a *second* adjustment to ensure
`tm_log >= 0` (i.e., that each dest word maps to at least one source
word). This two-phase approach suggests the first calculation doesn't
fully capture the constraint.

The comment on line 75–79 even lists three requirements, but
`self_change` only takes the max of two of them. The third (tm_log ≥ 0)
is patched up after the fact. This works, but it's fragile — if
someone modifies the first phase without understanding the second
exists, they could break the invariant. Consider computing all three
constraints together:

```rust
let tm_log_change = (dest_w as i32 - shift0 as i32 - src_width as i32).max(0) as u32;
let self_change = range_change.max(width_change).max(tm_log_change);
```

The wrinkle is that `shift0` depends on `self_change` (circular), which
is likely why you split it. If so, the comment should explain why the
two-phase approach is necessary, not just what the three requirements
are.

## Concern 2: `buckets_empty()` checks only `data[0]`

```rust
pub(crate) const fn buckets_empty(&self) -> bool {
    if self.word_end != self.word_start {
        return false;
    }
    self.data[0] == 0
}
```

This checks the physical `data[0]`, not `data[self.data_idx(self.word_start)]`.
For a freshly constructed or `clear()`ed histogram where `word_base == 0`,
this is fine. But if `word_base` has drifted (e.g., after downscaling),
`word_start == word_end` doesn't mean the active word is at physical
index 0. If this is an invariant (single-word always lives at
physical 0), it should be documented or asserted.

This matters in merge because `buckets_empty()` guards multiple
branches (lines 53, 84, 102, 119).

## Concern 3: `from_max_value` can panic on large inputs

```rust
pub(crate) const fn from_max_value(value: u64) -> Self {
    let leading = 64 - value.leading_zeros();  // 0..=64
    let width = leading.next_power_of_two();    // panics if leading > 32
    ALL_WIDTHS[width.trailing_zeros() as usize] // index 0..=6
}
```

If `value` has more than 32 significant bits, `leading` > 32, and
`next_power_of_two()` yields 64, giving `trailing_zeros() == 6` →
`Width::U64`. That's fine. But `leading` itself can be 33–64, and
`next_power_of_two()` on a `u32` overflows at 33+ (returns 0 in
release, panics in debug). In practice the merge path calls this with
`or_sums` (line 178) and `max_a + max_b` (line 195), both of which
could be large. If `or_sums` or the sum exceeds 2^32, this function
has UB in release mode.

**Edit:** Actually `leading` is a `u32` from `leading_zeros()`. Values
0..=32 are safe (next_power_of_two maps to 1..=32, trailing_zeros
0..=5). Value 33..=64 would call `next_power_of_two()` on 33..=64,
which for u32 overflows at anything > 32. Values with >32 significant
bits should map to U64, but the path through `next_power_of_two` is
broken for them. This needs a `if leading > 32 { return Self::U64 }`
guard, or use `leading_zeros` differently.

## Concern 4: `repack_source` uses a fixed `[0u64; 64]` array

Line 230: `let mut sums = [0u64; 64];`

`repack_count = 1 << narrow_steps`, and `narrow_steps = cur as u32 -
dest_width as u32`. Since `cur` is at most `U64` (6) and `dest_width`
is at least `B1` (0), `narrow_steps` can be at most 6, giving
`repack_count` up to 64. So the array is exactly sized for the worst
case.

This is correct but the 64-element stack array is allocated on every
call regardless of `repack_count`. For the common case (small
`narrow_steps`), most of it is wasted. Not a bug, but worth noting
that this is a 512-byte stack allocation in a tight loop. If this is
performance-sensitive, consider sizing dynamically or using a smaller
fixed array with a match on `narrow_steps`.

## Concern 5: No atomicity / snapshot-rollback on `merge_from`

The repository memories note that `update_by_incr`, `merge_from_raw`,
and `merge_literal_from` all use snapshot/rollback (clone self before,
restore on Err). This `merge_from` does not — it checks count overflow
first, then calls `merge_buckets` (which is documented as infallible),
then `commit_stats`. If `merge_buckets` were to ever become fallible
(e.g., scale bounds), self would be left in a partially merged state.
The "infallible" doc comment is load-bearing and should perhaps be
backed by the type system (no `Result` return, no `expect` calls) —
but there are two `.expect("...")` calls inside `merge_buckets` and
`widen_to` paths.

The `.expect("valid scale")` on lines 88 and 105 and
`.expect("downscale is infallible")` on lines 92 and 108 mean this
function *can* panic, contradicting the "infallible" claim. Either
these truly cannot fail (in which case `unwrap_unchecked` or compile-time
proof is more appropriate), or they can fail and you need rollback.

## Concern 6: `extend_word_range` doesn't check capacity

Lines 118–137: `extend_word_range` expands `[word_start, word_end]`
and zero-fills, but never checks that `hi_widx - lo_widx + 1 <= N`.
The `data_idx` uses modular arithmetic (`rem_euclid`), so if the range
exceeds N, it silently wraps around and overwrites live data. The
caller (`merge_words`) relies on the scale computation in
`merge_buckets` to guarantee the range fits. A `debug_assert!` here
would be cheap insurance.

## Concern 7: Signed arithmetic and casting throughout

The code freely mixes `i32` and `u32` via `as` casts:

- Line 73: `min_scale - extra as i32` (extra is u32)
- Line 80: `(...).max(0) as u32`
- Line 81: `merge_width as u32` (Width is repr(u8))
- Line 98: `(src_scale - self.current.scale.scale()) as u32`
- Line 151: `(src_scale - self.current.scale.scale()) as u32`

Most of these are safe given the domain constraints, but on line 98
and 151, if `src_scale < self.current.scale.scale()` (which shouldn't
happen after the adjustment, but could if there's a logic error), the
cast wraps to a huge u32, causing `tm_log` to be enormous and the
`1i32 << tm_log` on line 161 to overflow. The `debug_assert!(tm_log < 31)`
on line 153 catches this in debug, but not in release.

## Concern 8: `aligned_start` negative-number rounding

Line 162: `let aligned_start = other.word_start & !(total_merge - 1);`

For negative `word_start`, bitwise AND with a negative mask rounds
*toward negative infinity* (e.g., `word_start = -3, total_merge = 4`
→ `aligned_start = -4`). This is the correct behavior for aligning
source words to dest boundaries. But it's subtle with signed integers
and deserves a brief comment or test case demonstrating negative
alignment.

## Minor Items

- **Line 14 imports**: `widen` is imported from `super::swar` but
  also exists conceptually as `self.widen_to`. The name collision is
  manageable but could confuse readers. `swar::widen` operates on a
  single word; `widen_to` widens the whole histogram.

- **Doc comment on `merge_from` (line 19–26)**: says "The source
  histogram may have a different pool size (`M`)" but doesn't mention
  that scale and width can also differ, which is the more interesting
  case.

- The `loop` + `continue` pattern in `merge_words` (lines 171–197) for
  retry-on-overflow is clean but unusual. A bounded retry with
  `debug_assert` on iteration count would guard against infinite loops
  if there's ever a bug in `widen_to` that doesn't actually widen.

## Summary

The algorithmic approach is solid — decomposing scale shift into
in-word widening + cross-word grouping, with overflow-triggered
widening, is elegant. The main risks are:

1. **`from_max_value` overflow** for large counter values (Concern 3) — likely a real bug.
2. **Two-phase scale adjustment** (Concern 1) — correct but fragile.
3. **Panic paths in "infallible" code** (Concern 5) — contradicts the safety claim.
4. **No capacity check in `extend_word_range`** (Concern 6) — relies on caller correctness.

I'd prioritize fixing Concern 3, adding debug_asserts for Concern 6,
and documenting the two-phase rationale for Concern 1.
