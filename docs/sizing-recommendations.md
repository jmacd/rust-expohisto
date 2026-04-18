# Histogram Sizing Recommendations

For each (contrast, count) cell, shows the smallest `Histogram<N>`
whose **empirical median** relative error meets the target.
Based on 51 seeds per configuration with lognormal data.

Three quality tiers:
- **Fine** (≤1%): scale ≥ 6
- **Medium** (≤5%): scale ≥ 4
- **Coarse** (≤10%): scale ≥ 3

## Fine — target ≤1% error (scale ≥ 6)

Each cell: `N` (bytes) scale·width — or `—` if no N ≤ 64 suffices.

|    C \ n |        10        |       100        |        1K        |       10K        |       100K       |        1M        |
|----------|------------------|------------------|------------------|------------------|------------------|------------------|
|      10× |  N=6 (96B) 6·B1  | N=14 (160B) 6·B2 | N=52 (464B) 7·B4 |        —         |        —         |        —         |
|      10² | N=8 (112B) 6·B1  | N=20 (208B) 6·B2 | N=72 (624B) 6·B4 |        —         |        —         |        —         |
|      10³ | N=12 (144B) 6·B1 | N=22 (224B) 6·B2 | N=80 (688B) 6·B4 |        —         |        —         |        —         |
|      10⁴ | N=14 (160B) 6·B1 | N=32 (304B) 6·B2 | N=110 (928B) 6·B4 |        —         |        —         |        —         |
|      10⁵ | N=18 (192B) 6·B1 | N=40 (368B) 6·B2 | N=122 (1024B) 6·B4 |        —         |        —         |        —         |
|      10⁶ | N=20 (208B) 6·B1 | N=48 (432B) 6·B2 | N=160 (1328B) 6·B4 |        —         |        —         |        —         |

## Medium — target ≤5% error (scale ≥ 3)

Each cell: `N` (bytes) scale·width — or `—` if no N ≤ 64 suffices.

|    C \ n |        10        |       100        |        1K        |       10K        |       100K       |        1M        |
|----------|------------------|------------------|------------------|------------------|------------------|------------------|
|      10× |  N=4 (80B) 5·B1  |  N=4 (80B) 3·B4  | N=12 (144B) 3·U8 | N=44 (400B) 3·U16 | N=36 (336B) 3·U16 |        —         |
|      10² |  N=4 (80B) 5·B1  | N=8 (112B) 3·B4  | N=24 (240B) 3·U8 |        —         | N=90 (768B) 3·U16 |        —         |
|      10³ |  N=4 (80B) 4·B1  | N=12 (144B) 3·B4 | N=36 (336B) 3·U8 | N=72 (624B) 4·U8 | N=122 (1024B) 3·U16 |        —         |
|      10⁴ |  N=4 (80B) 3·B1  | N=14 (160B) 4·B4 | N=52 (464B) 5·B4 | N=90 (768B) 4·U8 | N=160 (1328B) 3·U16 | N=140 (1168B) 3·U16 |
|      10⁵ |  N=4 (80B) 3·B1  | N=14 (160B) 3·B4 | N=60 (528B) 4·B4 | N=80 (688B) 3·U8 | N=200 (1648B) 3·U16 | N=200 (1648B) 3·U16 |
|      10⁶ |  N=4 (80B) 3·B1  | N=16 (176B) 3·B4 | N=56 (496B) 4·B4 | N=72 (624B) 3·U8 | N=250 (2048B) 3·U16 | N=250 (2048B) 3·U16 |

## Coarse — target ≤10% error (scale ≥ 2)

Each cell: `N` (bytes) scale·width — or `—` if no N ≤ 64 suffices.

|    C \ n |        10        |       100        |        1K        |       10K        |       100K       |        1M        |
|----------|------------------|------------------|------------------|------------------|------------------|------------------|
|      10× |  N=4 (80B) 5·B1  |  N=4 (80B) 3·B4  |  N=6 (96B) 2·U8  | N=20 (208B) 2·U16 | N=20 (208B) 2·U16 |        —         |
|      10² |  N=4 (80B) 5·B1  |  N=4 (80B) 2·B4  | N=12 (144B) 2·U8 | N=40 (368B) 2·U16 | N=36 (336B) 2·U16 |        —         |
|      10³ |  N=4 (80B) 4·B1  |  N=6 (96B) 2·B4  | N=20 (208B) 2·U8 | N=72 (624B) 4·U8 | N=72 (624B) 2·U16 |        —         |
|      10⁴ |  N=4 (80B) 3·B1  | N=8 (112B) 2·B4  | N=24 (240B) 2·U8 | N=90 (768B) 4·U8 | N=80 (688B) 2·U16 | N=140 (1168B) 3·U16 |
|      10⁵ |  N=4 (80B) 3·B1  | N=10 (128B) 2·B4 | N=30 (288B) 2·U8 | N=80 (688B) 3·U8 | N=100 (848B) 2·U16 | N=90 (768B) 2·U16 |
|      10⁶ |  N=4 (80B) 3·B1  | N=12 (144B) 2·B4 | N=40 (368B) 2·U8 | N=72 (624B) 3·U8 | N=160 (1328B) 2·U16 | N=100 (848B) 2·U16 |

## Quick Reference

| Workload | Contrast | Count | Recommended N | Bytes | Tier |
|----------|----------|-------|---------------|-------|------|
| Low-rate API | 10² | 1000 | `Histogram<24>` | 240 | Medium |
| Typical API | 10³ | 10000 | `Histogram<72>` | 624 | Medium |
| High-rate API | 10³ | 100000 | `Histogram<122>` | 1024 | Medium |
| Wide-range API | 10⁵ | 10000 | `Histogram<80>` | 688 | Coarse |
| Embedded/IoT | 10² | 100 | `Histogram<4>` | 80 | Coarse |

Byte counts are approximate (N×8 + ~48 bytes struct overhead).
