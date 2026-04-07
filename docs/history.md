# Historical Notes

## Lookup Table Algorithm Origins

The lookup table approach used in this crate for mapping f64 values to exponential histogram bucket indices was independently developed by two teams:

- **Dynatrace** — Otmar Ertl's [DynaHist library](https://github.com/dynatrace-oss/dynahist) ([ExponentialHistogramLargeInclusiveLayout](https://github.com/dynatrace-oss/dynahist/blob/main/src/main/java/com/dynatrace/dynahist/layout/ExponentialHistogramLargeInclusiveLayout.java)) uses `N` linear buckets with 2 boundary corrections.

- **NewRelic** — Yuke Zhuge's [NrSketch library](https://github.com/newrelic-experimental/newrelic-sketch-java) ([SubBucketLookupIndexer](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/src/main/java/com/newrelic/nrsketch/indexer/SubBucketLookupIndexer.java), [algorithm description](https://github.com/newrelic-experimental/newrelic-sketch-java/blob/main/Indexer.md)) uses `2N` linear buckets with 1 boundary correction.

Both algorithms share the same core idea: divide the 52-bit IEEE 754 significand space into equidistant linear buckets, use a precomputed table to approximate the logarithmic bucket index, then apply a small number of boundary corrections. They produce identical bucket indices for all inputs and differ only in the space/correction tradeoff:

| Variant | Linear buckets | Corrections | Index table size |
|---------|---------------|-------------|------------------|
| Dynatrace | N | 2 | N × u16 |
| NewRelic | 2N | 1 | 2N × u16 |

Both share the same `BOUNDARIES` array (which dominates memory), so the total memory difference is ~20% — negligible in practice.

This implementation uses the `2N`/1-correction variant. The choice is essentially arbitrary; both are equally correct and perform similarly at runtime.

## Upper-inclusive boundary decision

See the lengthy discussion on this topic in OpenTelemetry
[here](https://github.com/open-telemetry/opentelemetry-specification/issues/2611#issuecomment-1178119261).
