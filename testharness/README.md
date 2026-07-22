# Cross-Implementation Exponential Histogram Test Harness

Verifies that this Rust exponential histogram produces identical bucket
assignments as the Go reference implementation
([`open-telemetry/opentelemetry-collector-contrib/pkg/expohisto`](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/pkg/expohisto)).

## Quick Start

```bash
./run.sh                     # 30-second test at default scale
./run.sh -duration 5m        # Longer soak test
./run.sh -seed 42            # Reproducible run
./run.sh -scale 4            # Test at scale 4
```

## Architecture

### Three programs

| Component | Language | Purpose |
|-----------|----------|---------|
| `rust-cli/` | Rust | Reads hex-encoded f64s → histogram → canonical text output |
| `go-cli/` | Go | Reads hex-encoded f64s → `structure.Float64` → same canonical output |
| `orchestrator/` | Go | Generates inputs, filters, spawns both CLIs, compares semantically |

### How it works

1. **Generate**: The orchestrator produces batches of f64 test values from
   multiple generators (random, powers of two, boundary values, zeros, etc.)

2. **Filter**: Each value is checked with `math/big` (256-bit precision) to
   compute the exact bucket index. Values where Go's `floor(log(v) * scaleFactor)`
   gives the wrong answer are removed—they would cause a false failure.

3. **Run**: Both CLIs receive the same hex-encoded f64 values on stdin and produce
   a canonical text output:
   ```
   scale=7
   count=3
   sum=4018000000000000
   min=3ff0000000000000
   max=4008000000000000
   zero_count=0
   positive_offset=-1
   positive_counts=[1,0,...,1]
   ```
   In PN mode (`--pn` / `-pn`), two additional fields are output:
   ```
   negative_offset=-1
   negative_counts=[1,0,...,1]
   ```

4. **Compare**: Since Rust and Go may start at different maximum scales (Rust at
   `table_scale`, Go at 20), the orchestrator normalizes both outputs to the lower
   scale by downscaling (shifting indices right, merging adjacent buckets) before
   comparing. Stats (count, sum, min, max, zero_count) are compared directly.
   In PN mode, both positive and negative bucket arrays are normalized and
   compared independently.

### Modes

| Mode | CLI flags | Histogram type | Values accepted |
|------|-----------|---------------|-----------------|
| NN (default) | (none) | `Histogram<N>` / `structure.Float64` | Positive + zero only |
| PN | `--pn` / `-pn` | `HistogramPN<K,L>` / `structure.Float64` | Positive + negative + zero |

### PN test generators

The orchestrator includes 8 PN-specific generators that exercise
positive+negative histogram behavior:

| Generator | Purpose |
|-----------|---------|
| `pn_symmetric` | Equal positive/negative magnitudes; scale sync is a no-op |
| `pn_asymmetric` | Narrow positive + wide negative; forces scale coupling |
| `pn_scale_coupling` | Alternating tight/wide clusters on both sides |
| `pn_negative_only` | Exclusively negative values; positive side stays empty |
| `pn_zeros_mixed` | Mix of positive, negative, and zero values |
| `pn_extreme_spread` | Full f64 range on both sides; maximum downscale pressure |
| `pn_near_boundary` | Positive and negative values near bucket boundaries |
| `pn_single_neg_repeated` | Single negative value repeated; tests negative counting |

### Scale alignment

Rust starts at `table_scale()` (default: 8 via the `scale-8` feature). Go always
starts at scale 20. Both auto-downscale as needed based on their bucket capacity.
The orchestrator handles this by normalizing both outputs to the lower of their
final scales before comparison.

To test at higher Rust scales, recompile with the appropriate feature:
```bash
cd rust-cli
cargo build --release --features scale-20  # slow: generates 1M-entry lookup table
```

### Scale coupling (PN mode)

In PN mode, both implementations synchronize the positive and negative
sub-histograms to the same (lower) scale. The scale-coupling test generators
specifically create scenarios where:

- One side forces downscaling while the other doesn't need it
- Both sides independently downscale to different levels
- Alternating updates on each side ratchet the shared scale down

The orchestrator verifies that both implementations arrive at the same
final scale and bucket assignments on both sides.

### Known differences

- **Subnormals**: Rust promotes subnormal inputs to the smallest normal bucket
  (`significand=1, biased_exp=1`), while Go maps them to the bucket at-or-below
  `MIN_VALUE` (`(-1022 << scale) - 1`). Both CLIs normalize subnormals to
  `MIN_POSITIVE` before recording to avoid this discrepancy.

## Input/Output Format

Both CLIs use **hex-encoded IEEE-754 f64 bits** for all float values (input and
output), eliminating float→string→float parsing discrepancies.

- Input: one 16-digit hex string per line (e.g., `3ff0000000000000` = 1.0)
- Output: `sum`, `min`, `max` fields use hex bits; `±0` is canonicalized to `+0`

## Building individually

```bash
# Rust CLI
cd rust-cli && cargo build --release

# Go CLI
cd go-cli && go build -o go-cli .

# Orchestrator
cd orchestrator && go build -o orchestrator .

# Run manually
./orchestrator/orchestrator \
  -rust-bin ./rust-cli/target/release/expohisto-cli \
  -go-bin ./go-cli/go-cli \
  -duration 30s -scale 8
```
