# Collapse-Left Design (`Sketch` and `SketchPN`)

`Sketch<N>` is a sibling of `HistogramNN<N>` that trades auto-scaling for a
**guaranteed relative error**, in the style of
[DDSketch](https://arxiv.org/abs/1908.10693). It reuses the crate's bucket
machinery — the same fixed-scale index mapping, sub-byte SWAR counters, and
ring-buffer window — but changes the overflow policy: instead of
downscaling (which loses precision across the whole range), it **collapses
the left (small-value) side**, keeping full resolution on the top of the
distribution. `SketchPN<K, L>` pairs two `Sketch`es for values of any sign.

This is the right structure for latency tails and similar one-sided
distributions, where the high quantiles (p99, p99.9) matter and the low end
does not.

## The guarantee

Fix a scale `S`. Let `b = 2^(2^-S)` and `α = (b - 1) / (b + 1)`. Every
*retained* (accurate) bucket has worst-case relative error `≤ α`. Because
collapsing only ever removes buckets on the **low** side, the accurate
window always covers the top of the observed distribution.

Promise (mirroring DDSketch's collapsing-lowest theorem): let
`f = (zero_count + underflow_count) / count`. For any quantile `q > f`, the
estimate `x̃_q` satisfies `|x̃_q − x_q| ≤ α · x_q`. For `q ≤ f` the estimate
is the underflow placeholder — bounded, but not relative-error accurate —
exactly DDSketch's behavior.

`α` by scale (the crate's resolution table):

| Scale | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| `α` | 33% | 17% | 8.6% | 4.3% | 2.2% | 1.1% | 0.54% | 0.27% | 0.14% |

`Sketch::with_relative_error(target)` picks the coarsest scale whose `α`
meets `target` (maximizing range per word before collapse).

## DDSketch comparison

| Aspect | DDSketch | `HistogramNN` | `Sketch` |
|---|---|---|---|
| Base | arbitrary `γ = (1+α)/(1−α)` | `2^(2^−scale)` (quantized) | `2^(2^−scale)` (quantized) |
| Error target | any `α` | nearest scale's `α` | nearest scale's `α`, fixed |
| Memory bound | # buckets `m` | total bytes `N` (width-traded) | total bytes `N` (width-traded) |
| Overflow policy | collapse lowest buckets | **downscale** (precision lost over full range) | **collapse lowest** (precision kept on top) |
| Accurate region | top; low collapsed | full range; precision can degrade | top; low collapsed |
| Wire format | DDSketch proto | OTel ExponentialHistogram | OTel ExponentialHistogram |

`Sketch` imports DDSketch's mechanism #1 (collapse-low) and rejects #2
(arbitrary base), so the result remains a valid OTel exponential histogram:
mergeable, exportable, no protocol change.

## Structure

```
Sketch<N> {
    scale:      Scale,     // fixed at construction
    width:      Width,     // B1..U64, sticky (only grows)
    word_*:     i32,       // ring-buffer window over the ACCURATE buckets
    underflow:  u64,       // side counter: mass below the floor
    live_total: u64,       // accurate + underflow (excludes zeros)
    stats:      Stats,     // exact count/sum/min/max
    data:       [u64; N],  // ACCURATE buckets only (sub-byte SWAR counters)
}
```

The `[u64; N]` pool holds only accurate buckets. The window is a
bidirectional ring buffer (as in `HistogramNN`), word-aligned so SWAR
operations never cross a `u64`. The **floor** is the lowest window slot:
`low_slot = word_to_slot_index(word_start)`. Invariant: `underflow` equals
the total mass whose bucket index is strictly below the floor, and the
floor only ever rises.

### Why the underflow is a side counter (not bucket 0)

The natural choice — fold the collapsed mass into the lowest *bucket* — was
rejected after measurement. The underflow absorbs the dominant mass (90%+
in the wide-distribution tests), so storing it as a normal counter forces
the whole array to its widest counter width, which **steals slots (`M`)
from the accurate window** and makes every below-window insert risk an
O(N) repack to widen that one slot.

Holding the underflow in a dedicated `u64` instead means:

- the accurate buckets' counter `Width` is driven **only by their own
  counts** — far more usable tail resolution;
- a below-window insert is an **O(1)** `u64` add (no widen, no relocate);
- the collapse never cascades (folding mass off the low end cannot
  overflow a side counter).

At export the underflow is folded back into `bucket_counts[0]`, so the wire
format is unchanged (see [Export](#export)).

## Collapse mechanics

Three placement outcomes for a value mapping to bucket `slot`:

1. **In/adjacent to the window** → increment the bucket (may overflow →
   *counter overflow* below). The window grows up or down while it fits in
   `N` words.
2. **Above a full window** (*range slide*) → slide the window up so its top
   tracks the new max; the words that fall off the bottom fold into
   `underflow`. Same counter width, so no re-spread — just word moves and a
   SWAR lane-sum of the evicted words.
3. **Below a full window** → `underflow += incr` directly (O(1)). The first
   such insert first expands the window to the canonical `N` words so the
   floor is a pure function of `(top word, width)`.

**Counter overflow** (a bucket exceeds its width) widens the array. At the
wider width the same `N` words hold fewer slots, so the window shrinks from
the bottom and the displaced low buckets fold into `underflow`. The
retained buckets are re-expressed at the wider width with the SWAR `spread`
op (see [SWAR](#swar-spread-vs-widen)).

Both rebuilds go through one helper, `scatter_source`, which adds a
source's accurate buckets into a fresh buffer at the new `(width, floor)`
and folds everything below the floor into `underflow` via a horizontal
lane-sum. Collapse calls it once; merge calls it twice.

## Order dependence (a deliberate tradeoff)

The collapse floor is `top − M` slots, where `M = N·64 / 2^width`. So the
resolution — and therefore the exact underflow/accurate boundary — depends
on the counter `width`, which is driven by the accurate-bucket counts and
is **sticky** (only grows). A low bucket that is briefly hot (widening the
counters) and later evicted leaves the width, hence the floor, dependent on
**insert/merge order**.

What this means:

- **Exactly order-independent:** `count`, `sum` (modulo f64 summation
  order), `min`, `max`, and the **α-guarantee itself** (every retained
  bucket is α-accurate in every order).
- **Order/merge dependent:** the exact slot at which the underflow/accurate
  boundary falls, hence the precise contents of buckets near the floor.

Crucially the divergence is confined to the **low end, at/below the
underflow floor** — the region the guarantee already disclaims. The **top
of the distribution never collapses**, so high quantiles (the entire point
of collapse-left) are exactly order-independent and gain resolution under
this design.

The alternative — coupling the width to the (order-independent) underflow
magnitude — restores an exactly-reproducible boundary but spends the tail
resolution to do it. This crate chooses resolution; a general-purpose,
bit-identical-merge aggregator would choose the coupling.

## Merge

`Sketch::merge_from` requires the **same fixed scale** (a `Sketch` cannot
rescale; mismatched scales return `Error::ScaleMismatch`). It unions the
two sources' accurate buckets, sizes the merge width to fit the combined
counts, and scatters both into a fresh buffer, folding the rest — plus both
inputs' `underflow` counters — into the merged `underflow`. Pool sizes
(`N`, `M`) may differ.

The merged floor is raised to cover each **collapsed input's floor**
(`ws ≥ max(natural, self_floor, other_floor)` in word terms). This is
required: a collapsed input's underflow is opaque, so if the merged floor
dropped below it that mass — really *below* the input floor — would be
wrongly re-exposed as accurate buckets. With the flooring, a single merge
is **exactly commutative**: `a.merge(b)` and `b.merge(a)` produce identical
state (the rebuild is symmetric in both operands). Sequential merges share
the same order caveat as inserts.

## SWAR: `spread` vs `widen`

There are three width-changing SWAR ops (`src/histogram/swar.rs`):

| op | effect on lanes | used for |
|---|---|---|
| `widen` | **sums** adjacent pairs | downscaling (`HistogramNN`): merge buckets + widen counter |
| `narrow` | truncate + pack | downscale repack |
| `spread` | **zero-extend** each lane | **`Sketch`**: pure counter widening at fixed scale |

`HistogramNN` only ever widens its counters *by downscaling* (merging
adjacent buckets), so its widen is `widen` (pairwise sum). A fixed-scale
`Sketch` must widen counters while **keeping each bucket's identity**, which
is the zero-extend `spread` (the inverse of `narrow`). `spread` is the one
primitive this work added; with it, `Sketch`'s collapse and merge run
through the shared `swar.rs` layer. Eviction folds a word into the
underflow with `widen(width, U64, word)`, a horizontal lane-sum.

## Export

The view (`Sketch::view()` → `SketchView`) projects onto the OTel
`ExponentialHistogram` wire fields: `scale`, `stats`, `zero_count`, and a
contiguous positive `bucket_counts` run starting at `offset`. When
collapsed, `offset` is the floor and the underflow is folded into the first
bucket, so the bytes are wire-compatible with no protocol change.

`underflow_count()` additionally reports the collapsed mass separately, so
a consumer that understands collapse-left knows the rank below which the
guarantee does not hold (and a future spec could carry it as a distinct
field). `SketchPN` exposes `positive()` / `negative()` (negative bucket
indices describe `|value|`) plus the shared `zero_count`.

## Quantiles

With the `quantile` feature, `Sketch::quantile(q)` walks the CDF and
returns the rank-selected bucket's **relative-error-optimal
representative** `2·L·U / (L + U)` (for boundaries `L ≤ U`), whose
worst-case relative error over `[L, U]` is exactly `α` — not the `≈ 2α` a
lower-boundary estimator gives. For ranks within the underflow it returns
the floor bucket's representative (bounded, not α-accurate).

## Sizing

Pick the three knobs from the workload:

- **Scale `S`** — from the target relative error via the `α` table (or
  `with_relative_error`).
- **Pool `N`** — the accurate window spans `M = N·64 / 2^width` slots,
  covering `M / 2^S` octaves before collapse. Choose `N` so the expected
  spread of the *upper* part of the distribution fits with a small
  underflow fraction `f`.
- **Min width** — `with_min_width` raises the starting counter width if
  most buckets are known to be hot, trading initial resolution for fewer
  early widenings.

The accurate octave-span shrinks as counters widen (more bits per slot),
but `α` never degrades — only how much of the low tail collapses.
