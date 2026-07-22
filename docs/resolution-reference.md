# Exponential Histogram: Resolution Reference Table

Terminal scale, counter width, and relative error for variable-width
exponential histograms fed lognormal-distributed data, as a function
of data contrast (max/min ratio) and measurement count per interval.

Five histogram sizes are compared:

- **XS** — `Histogram<10>`: 128 bytes, 640 bits of counter storage
- **S** — `Histogram<26>`: 256 bytes, 1664 bits of counter storage
- **M** — `Histogram<58>`: 512 bytes, 3712 bits of counter storage
- **L** — `Histogram<122>`: 1024 bytes, 7808 bits of counter storage
- **XL** — `Histogram<250>`: 2048 bytes, 16000 bits of counter storage

Results are the **median of 441 independent runs** using the actual
histogram implementation. Data is drawn from a lognormal distribution
whose parameters are chosen so that the expected contrast (max/min of
all samples) matches the specified contrast for the given count.

## Resolution Table

Each cell shows: `scale / relative_error%`
Width abbreviations: B1(1-bit) B2(2) B4(4) U8(8) U16(16) U32(32) U64(64)

|      Contrast      | Size |       10       |      100       |       1K       |      10K       |      100K      |       1M       |
|:------------------:|:----:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|        10×         | XS (128B) |   7·B1 0.3%    |   4·B4 2.2%    |   2·U8 8.6%    |  1·U16 17.2%   |  1·U16 17.2%   |  -1·U32 60.0%  |
|                    | S (256B) |   9·B1 0.1%    |   7·B2 0.3%    |   4·U8 2.2%    |   2·U16 8.6%   |   2·U16 8.6%   |  0·U32 33.3%   |
|                    | M (512B) |   10·B1 0.0%   |   9·B2 0.1%    |   7·B4 0.3%    |   3·U16 4.3%   |   3·U16 4.3%   |  1·U32 17.2%   |
|                    | L (1024B) |   10·B1 0.0%   |   9·B2 0.1%    |   7·B4 0.3%    |   3·U16 4.3%   |   3·U16 4.3%   |  1·U32 17.2%   |
|                    | XL (2048B) |   10·B1 0.0%   |   9·B2 0.1%    |   7·B4 0.3%    |   3·U16 4.3%   |   3·U16 4.3%   |  1·U32 17.2%   |
|                    |      |                |                |                |                |                |                |
|        10²         | XS (128B) |   6·B1 0.5%    |   3·B4 4.3%    |   1·U8 17.2%   |  0·U16 33.3%   |  0·U16 33.3%   |  -2·U32 88.2%  |
|                    | S (256B) |   8·B1 0.1%    |   6·B2 0.5%    |   3·U8 4.3%    |  1·U16 17.2%   |  1·U16 17.2%   |  -1·U32 60.0%  |
|                    | M (512B) |   9·B1 0.1%    |   8·B2 0.1%    |   6·B4 0.5%    |   2·U16 8.6%   |   2·U16 8.6%   |  0·U32 33.3%   |
|                    | L (1024B) |   10·B1 0.0%   |   9·B2 0.1%    |   7·B4 0.3%    |   5·U8 1.1%    |   3·U16 4.3%   |  1·U32 17.2%   |
|                    | XL (2048B) |   10·B1 0.0%   |   9·B2 0.1%    |   7·B4 0.3%    |   5·U8 1.1%    |   3·U16 4.3%   |  1·U32 17.2%   |
|                    |      |                |                |                |                |                |                |
|        10³         | XS (128B) |   6·B1 0.5%    |   2·B4 8.6%    |   1·U8 17.2%   |  -1·U16 60.0%  |  -1·U16 60.0%  |  -3·U32 99.2%  |
|                    | S (256B) |   7·B1 0.3%    |   6·B2 0.5%    |   2·U8 8.6%    |  0·U16 33.3%   |  0·U16 33.3%   |  -1·U32 60.0%  |
|                    | M (512B) |   8·B1 0.1%    |   7·B2 0.3%    |   5·B4 1.1%    |  1·U16 17.2%   |   2·U16 8.6%   |  0·U32 33.3%   |
|                    | L (1024B) |   9·B1 0.1%    |   8·B2 0.1%    |   6·B4 0.5%    |   4·U8 2.2%    |   3·U16 4.3%   |   3·U16 4.3%   |
|                    | XL (2048B) |   10·B1 0.0%   |   10·B1 0.0%   |   7·B4 0.3%    |   5·U8 1.1%    |   3·U16 4.3%   |   3·U16 4.3%   |
|                    |      |                |                |                |                |                |                |
|        10⁴         | XS (128B) |   5·B1 1.1%    |   2·B4 8.6%    |   0·U8 33.3%   |  -1·U16 60.0%  |  -1·U16 60.0%  |  -3·U32 99.2%  |
|                    | S (256B) |   7·B1 0.3%    |   5·B2 1.1%    |   2·U8 8.6%    |  0·U16 33.3%   |  0·U16 33.3%   |  -2·U32 88.2%  |
|                    | M (512B) |   8·B1 0.1%    |   7·B2 0.3%    |   5·B4 1.1%    |  1·U16 17.2%   |  1·U16 17.2%   |  -1·U32 60.0%  |
|                    | L (1024B) |   9·B1 0.1%    |   8·B2 0.1%    |   6·B4 0.5%    |   4·U8 2.2%    |   2·U16 8.6%   |  0·U32 33.3%   |
|                    | XL (2048B) |   10·B1 0.0%   |   10·B1 0.0%   |   7·B4 0.3%    |   5·U8 1.1%    |   3·U16 4.3%   |   3·U16 4.3%   |
|                    |      |                |                |                |                |                |                |
|        10⁵         | XS (128B) |   5·B1 1.1%    |   2·B4 8.6%    |   0·U8 33.3%   |  -2·U16 88.2%  | -4·U32 100.0%  |  -3·U32 99.2%  |
|                    | S (256B) |   6·B1 0.5%    |   5·B2 1.1%    |   1·U8 17.2%   |  -1·U16 60.0%  |  0·U16 33.3%   |  -2·U32 88.2%  |
|                    | M (512B) |   7·B1 0.3%    |   6·B2 0.5%    |   2·U8 8.6%    |  1·U16 17.2%   |  1·U16 17.2%   |  -1·U32 60.0%  |
|                    | L (1024B) |   9·B1 0.1%    |   7·B2 0.3%    |   5·B4 1.1%    |   4·U8 2.2%    |   2·U16 8.6%   |   2·U16 8.6%   |
|                    | XL (2048B) |   10·B1 0.0%   |   9·B2 0.1%    |   6·B4 0.5%    |   5·U8 1.1%    |   3·U16 4.3%   |   3·U16 4.3%   |
|                    |      |                |                |                |                |                |                |
|        10⁶         | XS (128B) |   5·B1 1.1%    |   2·B4 8.6%    |   0·U8 33.3%   |  -2·U16 88.2%  |  -2·U16 88.2%  | -4·U32 100.0%  |
|                    | S (256B) |   6·B1 0.5%    |   4·B2 2.2%    |   1·U8 17.2%   |  -1·U16 60.0%  |  -1·U16 60.0%  |  -2·U32 88.2%  |
|                    | M (512B) |   7·B1 0.3%    |   6·B2 0.5%    |   4·B4 2.2%    |  0·U16 33.3%   |  1·U16 17.2%   |  -1·U32 60.0%  |
|                    | L (1024B) |   8·B1 0.1%    |   7·B2 0.3%    |   5·B4 1.1%    |   3·U8 4.3%    |  1·U16 17.2%   |   2·U16 8.6%   |
|                    | XL (2048B) |   9·B1 0.1%    |   9·B1 0.1%    |   6·B4 0.5%    |   4·U8 2.2%    |   3·U16 4.3%   |   3·U16 4.3%   |

## How to Read This Table

**Rows** are grouped by contrast — the ratio of the largest to
smallest observed value in your data:

| Contrast | Octaves | Example range |
|----------|---------|---------------|
| 10×      | 3.3     | 10ms – 100ms  |
| 10²      | 6.6     | 1ms – 100ms   |
| 10³      | 10      | 1ms – 1s      |
| 10⁴      | 13      | 1ms – 10s     |
| 10⁵      | 17      | 1ms – 100s    |
| 10⁶      | 20      | 1μs – 1s      |

**Columns** are the number of measurements per collection interval.
A typical OTel SDK collecting every 15–60 seconds at 100–1000 RPS
sees n ≈ 1,500–60,000 measurements per interval.

**Cell values** show `scale·width error%` where:
- **Scale** determines bucket resolution (higher = finer)
- **Width** is the counter bit-width (B4 = 4-bit, U16 = 16-bit, etc.)
- **Error%** is the worst-case relative error = (base−1)/(base+1)

## Relationship to OTel Specification Defaults

The OTel specification recommends `MaxSize = 160` fixed-width buckets.
The following table (from the specification) shows the ideal scale for
160 buckets as a function of input range:

| Input range | Contrast | Ideal Scale | Relative error |
|-------------|----------|-------------|----------------|
| 1ms – 4ms   | 4×       | 6           | 0.54%          |
| 1ms – 100ms | 10²      | 4           | 2.2%           |
| 1ms – 1s    | 10³      | 4           | 2.2%           |
| 1ms – 100s  | 10⁵      | 3           | 4.3%           |
| 1μs – 10s   | 10⁷      | 2           | 8.6%           |

With variable-width counters, the scale depends on both contrast
**and** count, because counter overflow reduces the number of
available slots. The reference table above shows this interaction.

Key observations:

- At low counts (n ≤ 100), variable-width histograms achieve
  **higher** scale than the fixed-width 160-bucket default because
  narrow counters (B1–B4) provide more slots than 160.

- At moderate counts (n ≈ 1K–10K), L (1024B) and XL (2048B)
  match or exceed the 160-bucket default's resolution.

- At high counts (n ≈ 100K–1M), counter pressure reduces scale
  below the range-only ideal. This is the price of compact storage.

## Sizing Guidance

| Size | Bytes | Best for |
|------|-------|----------|
| XS (`Histogram<10>`)  | 128   | Embedded, high-cardinality, `no_std` |
| S  (`Histogram<26>`)  | 256   | Constrained environments, many histograms |
| M  (`Histogram<58>`)  | 512   | General-purpose OTel metrics |
| L  (`Histogram<122>`) | 1024  | High-resolution, long intervals |
| XL (`Histogram<250>`) | 2048  | Maximum resolution, low-cardinality |

Choose the smallest size whose error% is acceptable for your
contrast and count. For most OTel workloads (contrast 10³–10⁵,
n ≈ 1K–100K), M (512B) or L (1024B) provides 2–9% error.
