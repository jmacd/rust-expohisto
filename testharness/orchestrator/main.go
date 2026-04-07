// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// orchestrator generates test batches, filters them through bignum
// verification, runs both the Rust and Go expohisto CLIs, and compares
// their output using the two-pass protocol.
//
// Usage:
//
//	orchestrator -rust-bin PATH -go-bin PATH [-duration 30s] [-scale 8] [-seed 0]
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/jmacd/rust-expohisto/testharness/orchestrator/bignum"
)

func main() {
	rustBin := flag.String("rust-bin", "", "path to Rust expohisto-cli binary")
	goBin := flag.String("go-bin", "", "path to Go go-cli binary")
	duration := flag.Duration("duration", 30*time.Second, "how long to run tests")
	scale := flag.Int("scale", 8, "initial max scale for Rust CLI")
	seed := flag.Int64("seed", 0, "random seed (0 = time-based)")
	flag.Parse()

	if *rustBin == "" || *goBin == "" {
		fmt.Fprintln(os.Stderr, "both -rust-bin and -go-bin are required")
		os.Exit(1)
	}

	if *seed == 0 {
		*seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(*seed))
	fmt.Printf("seed=%d scale=%d duration=%s\n\n", *seed, *scale, *duration)

	deadline := time.Now().Add(*duration)

	type generatorStats struct {
		name         string
		gen          func(rng *rand.Rand) []float64
		pn           bool // true = PN mode (positive + negative values)
		tests        int
		totalValues  int
		filtered     int
		emptyBatches int
	}

	generators := []*generatorStats{
		// NN generators (positive values only)
		{name: "random_uniform_small", gen: genRandomUniformSmall},
		{name: "random_uniform_wide", gen: genRandomUniformWide},
		{name: "random_ieee754_bits", gen: genRandomBits},
		{name: "powers_of_two", gen: genPowersOfTwo},
		{name: "near_one", gen: genNearOne},
		{name: "near_boundary", gen: genNearBoundary(int32(*scale))},
		{name: "near_min_normal", gen: genNearMinNormal},
		{name: "zeros_mixed", gen: genZerosMixed},
		{name: "single_value_repeated", gen: genSingleRepeated},
		{name: "ascending_wide", gen: genAscendingWide},
		{name: "descending_wide", gen: genDescendingWide},
		{name: "geometric_spread", gen: genGeometricSpread},
		{name: "boundary_values", gen: genBoundaryValues},
		// PN generators (positive + negative values)
		{name: "pn_symmetric", gen: genPNSymmetric, pn: true},
		{name: "pn_asymmetric", gen: genPNAsymmetric, pn: true},
		{name: "pn_scale_coupling", gen: genPNScaleCoupling, pn: true},
		{name: "pn_negative_only", gen: genPNNegativeOnly, pn: true},
		{name: "pn_zeros_mixed", gen: genPNZerosMixed, pn: true},
		{name: "pn_extreme_spread", gen: genPNExtremeSpread, pn: true},
		{name: "pn_near_boundary", gen: genPNNearBoundary(int32(*scale)), pn: true},
		{name: "pn_single_neg_repeated", gen: genPNSingleNegRepeated, pn: true},
	}

	var totalTests, totalPassed int

	for time.Now().Before(deadline) {
		for _, g := range generators {
			if time.Now().After(deadline) {
				break
			}

			values := g.gen(rng)
			scaleI32 := int32(*scale)

			g.totalValues += len(values)

			if g.pn {
				// For PN mode, filter positive and negative
				// values independently, then recombine with
				// zeros preserved.
				accurate := filterPNAccurate(values, scaleI32)
				g.filtered += len(values) - len(accurate)

				if len(accurate) == 0 {
					g.emptyBatches++
					continue
				}

				g.tests++
				totalTests++
				err := runPNComparison(*rustBin, *goBin, accurate, *scale)
				if err != nil {
					fmt.Printf("FAIL [%s]: %v\n", g.name, err)
					fmt.Printf("  values (%d): ", len(accurate))
					for i, v := range accurate {
						if i > 10 {
							fmt.Printf("... (%d more)", len(accurate)-i)
							break
						}
						fmt.Printf("%016x ", math.Float64bits(v))
					}
					fmt.Println()
					os.Exit(1)
				}
				totalPassed++
			} else {
				accurate, filtered := bignum.FilterAccurate(values, scaleI32)
				g.filtered += filtered

				if len(accurate) == 0 {
					g.emptyBatches++
					continue
				}

				g.tests++
				totalTests++
				err := runComparison(*rustBin, *goBin, accurate, *scale)
				if err != nil {
					fmt.Printf("FAIL [%s]: %v\n", g.name, err)
					fmt.Printf("  values (%d): ", len(accurate))
					for i, v := range accurate {
						if i > 10 {
							fmt.Printf("... (%d more)", len(accurate)-i)
							break
						}
						fmt.Printf("%016x ", math.Float64bits(v))
					}
					fmt.Println()
					os.Exit(1)
				}
				totalPassed++
			}
		}
	}

	// Per-generator summary
	fmt.Printf("%-25s %7s %10s %8s %8s\n", "generator", "tests", "values", "filtered", "rate")
	fmt.Printf("%-25s %7s %10s %8s %8s\n", strings.Repeat("-", 25), "-------", "----------", "--------", "--------")
	var grandValues, grandFiltered int
	for _, g := range generators {
		rate := ""
		if g.totalValues > 0 {
			rate = fmt.Sprintf("%.2f%%", 100*float64(g.filtered)/float64(g.totalValues))
		}
		fmt.Printf("%-25s %7d %10d %8d %8s\n", g.name, g.tests, g.totalValues, g.filtered, rate)
		grandValues += g.totalValues
		grandFiltered += g.filtered
	}
	rate := ""
	if grandValues > 0 {
		rate = fmt.Sprintf("%.2f%%", 100*float64(grandFiltered)/float64(grandValues))
	}
	fmt.Printf("%-25s %7s %10s %8s %8s\n", strings.Repeat("-", 25), "-------", "----------", "--------", "--------")
	fmt.Printf("%-25s %7d %10d %8d %8s\n", "TOTAL", totalPassed, grandValues, grandFiltered, rate)
	fmt.Printf("\nResult: %d tests PASSED, seed=%d\n", totalPassed, *seed)
}

// filterPNAccurate filters a mixed positive/negative value set.
// Positive and negative non-zero values are filtered independently
// through bignum; zeros are always kept.
func filterPNAccurate(values []float64, scale int32) []float64 {
	var positives, negatives, zeros []float64
	// Track original ordering with indices
	type entry struct {
		val float64
		idx int
	}
	var posEntries, negEntries []entry

	for i, v := range values {
		if v == 0.0 {
			zeros = append(zeros, v)
		} else if v > 0 {
			positives = append(positives, v)
			posEntries = append(posEntries, entry{v, i})
		} else {
			negatives = append(negatives, -v) // filter absolute values
			negEntries = append(negEntries, entry{v, i})
		}
	}

	// Filter positives
	accPos, _ := bignum.FilterAccurate(positives, scale)
	posSet := make(map[uint64]int)
	for _, v := range accPos {
		posSet[math.Float64bits(v)]++
	}

	// Filter negatives (as absolute values)
	absNegs := make([]float64, len(negatives))
	copy(absNegs, negatives)
	accNeg, _ := bignum.FilterAccurate(absNegs, scale)
	negSet := make(map[uint64]int)
	for _, v := range accNeg {
		negSet[math.Float64bits(v)]++
	}

	// Rebuild the result preserving original order
	var result []float64
	for _, v := range values {
		if v == 0.0 {
			result = append(result, v)
		} else if v > 0 {
			bits := math.Float64bits(v)
			if posSet[bits] > 0 {
				result = append(result, v)
				posSet[bits]--
			}
		} else {
			bits := math.Float64bits(-v)
			if negSet[bits] > 0 {
				result = append(result, v)
				negSet[bits]--
			}
		}
	}

	return result
}

// runComparison runs both CLIs in NN mode and compares semantically.
func runComparison(rustBin, goBin string, values []float64, scale int) error {
	input := valuesToHex(values)

	rustOut, err := runCLI(rustBin, []string{
		"--scale", strconv.Itoa(scale),
		"--size", "16000",
	}, input)
	if err != nil {
		return fmt.Errorf("rust CLI failed: %w", err)
	}

	goOut, err := runCLI(goBin, []string{
		"-size", "16384",
	}, input)
	if err != nil {
		return fmt.Errorf("go CLI failed: %w", err)
	}

	rustParsed, err := parseOutput(rustOut)
	if err != nil {
		return fmt.Errorf("parsing rust output: %w", err)
	}
	goParsed, err := parseOutput(goOut)
	if err != nil {
		return fmt.Errorf("parsing go output: %w", err)
	}

	if err := compareStats(rustParsed, goParsed); err != nil {
		return err
	}

	return compareBuckets(rustParsed, goParsed, "positive")
}

// runPNComparison runs both CLIs in PN mode and compares semantically.
func runPNComparison(rustBin, goBin string, values []float64, scale int) error {
	input := valuesToHex(values)

	rustOut, err := runCLI(rustBin, []string{
		"--scale", strconv.Itoa(scale),
		"--size", "16000",
		"--pn",
	}, input)
	if err != nil {
		return fmt.Errorf("rust CLI failed: %w", err)
	}

	goOut, err := runCLI(goBin, []string{
		"-size", "16384",
		"-pn",
	}, input)
	if err != nil {
		return fmt.Errorf("go CLI failed: %w", err)
	}

	rustParsed, err := parseOutput(rustOut)
	if err != nil {
		return fmt.Errorf("parsing rust output: %w", err)
	}
	goParsed, err := parseOutput(goOut)
	if err != nil {
		return fmt.Errorf("parsing go output: %w", err)
	}

	if err := compareStats(rustParsed, goParsed); err != nil {
		return err
	}

	if err := compareBuckets(rustParsed, goParsed, "positive"); err != nil {
		return err
	}

	return compareBuckets(rustParsed, goParsed, "negative")
}

func compareStats(rust, goP *parsedOutput) error {
	if rust.count != goP.count {
		return fmt.Errorf("count mismatch: rust=%d go=%d", rust.count, goP.count)
	}
	if rust.sumHex != goP.sumHex {
		return fmt.Errorf("sum mismatch: rust=%s go=%s", rust.sumHex, goP.sumHex)
	}
	if rust.minHex != goP.minHex {
		return fmt.Errorf("min mismatch: rust=%s go=%s", rust.minHex, goP.minHex)
	}
	if rust.maxHex != goP.maxHex {
		return fmt.Errorf("max mismatch: rust=%s go=%s", rust.maxHex, goP.maxHex)
	}
	if rust.zeroCount != goP.zeroCount {
		return fmt.Errorf("zero_count mismatch: rust=%d go=%d", rust.zeroCount, goP.zeroCount)
	}
	return nil
}

func compareBuckets(rust, goP *parsedOutput, side string) error {
	var rOffset, gOffset int
	var rCounts, gCounts []uint64

	if side == "positive" {
		rOffset = rust.positiveOffset
		gOffset = goP.positiveOffset
		rCounts = rust.positiveCounts
		gCounts = goP.positiveCounts
	} else {
		rOffset = rust.negativeOffset
		gOffset = goP.negativeOffset
		rCounts = rust.negativeCounts
		gCounts = goP.negativeCounts
	}

	// Normalize both to the lower scale
	targetScale := rust.scale
	if goP.scale < targetScale {
		targetScale = goP.scale
	}

	rustNorm := downscaleBuckets(rust.scale, rOffset, rCounts, targetScale)
	goNorm := downscaleBuckets(goP.scale, gOffset, gCounts, targetScale)

	if rustNorm.offset != goNorm.offset {
		return fmt.Errorf("%s_offset mismatch at scale %d: rust=%d go=%d (rust_orig_scale=%d go_orig_scale=%d)",
			side, targetScale, rustNorm.offset, goNorm.offset, rust.scale, goP.scale)
	}
	if len(rustNorm.counts) != len(goNorm.counts) {
		return fmt.Errorf("%s_counts length mismatch at scale %d: rust=%d go=%d",
			side, targetScale, len(rustNorm.counts), len(goNorm.counts))
	}
	for i := range rustNorm.counts {
		if rustNorm.counts[i] != goNorm.counts[i] {
			return fmt.Errorf("%s_counts[%d] mismatch at scale %d: rust=%d go=%d",
				side, i, targetScale, rustNorm.counts[i], goNorm.counts[i])
		}
	}

	return nil
}

type normalizedBuckets struct {
	offset int
	counts []uint64
}

// downscaleBuckets normalizes a bucket array to targetScale by merging
// adjacent buckets via arithmetic right shift.
func downscaleBuckets(scale, offset int, counts []uint64, targetScale int) normalizedBuckets {
	if scale <= targetScale || len(counts) == 0 {
		return normalizedBuckets{offset: offset, counts: counts}
	}
	shift := uint(scale - targetScale)

	newStart := offset >> shift
	newEnd := (offset + len(counts) - 1) >> shift
	newLen := newEnd - newStart + 1

	newCounts := make([]uint64, newLen)
	for i, c := range counts {
		oldIdx := offset + i
		newIdx := oldIdx >> shift
		newCounts[newIdx-newStart] += c
	}

	// Trim leading and trailing zeros
	first, last := 0, len(newCounts)-1
	for first < len(newCounts) && newCounts[first] == 0 {
		first++
	}
	for last > first && newCounts[last] == 0 {
		last--
	}
	if first > last {
		return normalizedBuckets{offset: 0, counts: nil}
	}

	return normalizedBuckets{
		offset: newStart + first,
		counts: newCounts[first : last+1],
	}
}

func runCLI(bin string, args []string, stdin string) (string, error) {
	cmd := exec.Command(bin, args...)
	cmd.Stdin = strings.NewReader(stdin)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%s: %s", err, stderr.String())
	}
	return stdout.String(), nil
}

func valuesToHex(values []float64) string {
	var sb strings.Builder
	for _, v := range values {
		fmt.Fprintf(&sb, "%016x\n", math.Float64bits(v))
	}
	return sb.String()
}

type parsedOutput struct {
	scale          int
	count          uint64
	sumHex         string
	minHex         string
	maxHex         string
	zeroCount      uint64
	positiveOffset int
	positiveCounts []uint64
	negativeOffset int
	negativeCounts []uint64
}

func parseOutput(s string) (*parsedOutput, error) {
	p := &parsedOutput{}
	scanner := bufio.NewScanner(strings.NewReader(s))
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key, val := parts[0], parts[1]
		switch key {
		case "scale":
			v, err := strconv.Atoi(val)
			if err != nil {
				return nil, err
			}
			p.scale = v
		case "count":
			v, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				return nil, err
			}
			p.count = v
		case "sum":
			p.sumHex = val
		case "min":
			p.minHex = val
		case "max":
			p.maxHex = val
		case "zero_count":
			v, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				return nil, err
			}
			p.zeroCount = v
		case "positive_offset":
			v, err := strconv.Atoi(val)
			if err != nil {
				return nil, err
			}
			p.positiveOffset = v
		case "positive_counts":
			counts, err := parseCounts(val)
			if err != nil {
				return nil, err
			}
			p.positiveCounts = counts
		case "negative_offset":
			v, err := strconv.Atoi(val)
			if err != nil {
				return nil, err
			}
			p.negativeOffset = v
		case "negative_counts":
			counts, err := parseCounts(val)
			if err != nil {
				return nil, err
			}
			p.negativeCounts = counts
		}
	}
	return p, nil
}

func parseCounts(val string) ([]uint64, error) {
	val = strings.TrimPrefix(val, "[")
	val = strings.TrimSuffix(val, "]")
	if val == "" {
		return nil, nil
	}
	var counts []uint64
	for _, cs := range strings.Split(val, ",") {
		c, err := strconv.ParseUint(cs, 10, 64)
		if err != nil {
			return nil, err
		}
		counts = append(counts, c)
	}
	return counts, nil
}

// ── NN test generators ──────────────────────────────────────────────

func genRandomUniformSmall(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(50)
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = rng.Float64()*100 + 0.01
	}
	return vals
}

func genRandomUniformWide(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(50)
	vals := make([]float64, n)
	for i := range vals {
		exp := rng.Float64()*20 - 10 // 2^-10 to 2^10
		vals[i] = math.Pow(2, exp)
	}
	return vals
}

// genRandomBits generates random positive normal f64 values by
// constructing IEEE-754 bits directly.
func genRandomBits(rng *rand.Rand) []float64 {
	n := 20 + rng.Intn(40)
	vals := make([]float64, n)
	for i := range vals {
		// Random biased exponent in [1, 2046] (normal range)
		exp := uint64(1 + rng.Intn(2046))
		// Random significand (52 bits)
		sig := rng.Uint64() & ((1 << 52) - 1)
		bits := (exp << 52) | sig
		vals[i] = math.Float64frombits(bits)
	}
	return vals
}

func genPowersOfTwo(rng *rand.Rand) []float64 {
	n := 5 + rng.Intn(20)
	vals := make([]float64, n)
	for i := range vals {
		exp := rng.Intn(40) - 20 // 2^-20 to 2^20
		vals[i] = math.Pow(2, float64(exp))
	}
	return vals
}

func genNearOne(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(30)
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = 1.0 + (rng.Float64()-0.5)*0.01
	}
	return vals
}

// genNearBoundary generates values that are near bucket boundaries at
// the given scale. These are the most likely to trigger log inaccuracy.
func genNearBoundary(scale int32) func(*rand.Rand) []float64 {
	return func(rng *rand.Rand) []float64 {
		n := 20 + rng.Intn(20)
		vals := make([]float64, n)
		for i := range vals {
			// Pick a random bucket index, compute its boundary,
			// then perturb by ±1 ULP
			exp := rng.Intn(20) - 10
			subBucket := rng.Intn(1 << scale)
			// Boundary = 2^((exp * 2^scale + subBucket) / 2^scale)
			//          = 2^(exp + subBucket / 2^scale)
			idx := float64(exp) + float64(subBucket)/float64(int(1)<<scale)
			boundary := math.Pow(2, idx)
			if math.IsInf(boundary, 0) || boundary == 0 {
				vals[i] = 1.0
				continue
			}
			// Perturb by a few ULPs in either direction
			ulps := rng.Intn(5) - 2
			v := boundary
			for j := 0; j < abs(ulps); j++ {
				if ulps > 0 {
					v = math.Nextafter(v, math.MaxFloat64)
				} else {
					v = math.Nextafter(v, 0)
				}
			}
			if v <= 0 {
				v = math.SmallestNonzeroFloat64
			}
			vals[i] = v
		}
		return vals
	}
}

// genNearMinNormal generates values near and below the smallest normal float.
// Both CLIs normalize subnormals to MIN_VALUE, so these test that path.
func genNearMinNormal(rng *rand.Rand) []float64 {
	minNormal := math.Float64frombits(0x0010000000000000) // 2^-1022
	vals := []float64{
		math.SmallestNonzeroFloat64,       // smallest subnormal
		math.SmallestNonzeroFloat64 * 2,   // small subnormal
		math.SmallestNonzeroFloat64 * 100, // larger subnormal
		5e-324,                            // smallest subnormal alt
		1e-320,                            // mid subnormal
		1e-310,                            // near-normal subnormal
		minNormal,                         // exact MIN_VALUE
		math.Nextafter(minNormal, math.MaxFloat64), // just above
		minNormal * 1.0001,
		minNormal * 1.5,
		minNormal * 2,
	}
	// Add some random values in the near-MIN_VALUE range
	for i := 0; i < 5; i++ {
		vals = append(vals, minNormal*(1.0+rng.Float64()*3.0))
	}
	return vals
}

func genZerosMixed(rng *rand.Rand) []float64 {
	n := 5 + rng.Intn(10)
	vals := make([]float64, n)
	for i := range vals {
		if rng.Float64() < 0.3 {
			if rng.Float64() < 0.5 {
				vals[i] = 0.0
			} else {
				vals[i] = math.Copysign(0, -1) // -0.0
			}
		} else {
			vals[i] = rng.Float64()*10 + 0.1
		}
	}
	return vals
}

func genSingleRepeated(rng *rand.Rand) []float64 {
	v := rng.Float64()*1000 + 0.001
	n := 10 + rng.Intn(50)
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = v
	}
	return vals
}

// genAscendingWide generates ascending values spanning many orders of magnitude.
func genAscendingWide(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(30)
	vals := make([]float64, n)
	startExp := rng.Float64()*10 - 5
	for i := range vals {
		exp := startExp + float64(i)*rng.Float64()*2
		vals[i] = math.Pow(2, exp)
	}
	return vals
}

func genDescendingWide(rng *rand.Rand) []float64 {
	vals := genAscendingWide(rng)
	for i, j := 0, len(vals)-1; i < j; i, j = i+1, j-1 {
		vals[i], vals[j] = vals[j], vals[i]
	}
	return vals
}

// genGeometricSpread generates values with exponentially growing gaps.
func genGeometricSpread(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(20)
	vals := make([]float64, n)
	base := rng.Float64()*10 + 0.001
	ratio := 1.1 + rng.Float64()*5 // growth ratio 1.1x to 6.1x
	v := base
	for i := range vals {
		vals[i] = v
		v *= ratio
		if math.IsInf(v, 0) {
			v = base // wrap around
		}
	}
	return vals
}

func genBoundaryValues(_ *rand.Rand) []float64 {
	minNormal := math.Float64frombits(0x0010000000000000) // 2^-1022
	return []float64{
		minNormal,
		math.Nextafter(minNormal, math.MaxFloat64),
		minNormal * 2,
		minNormal * 1.5,
		math.MaxFloat64,
		math.Nextafter(math.MaxFloat64, 0),
		1.0,
		2.0,
		0.5,
		math.Nextafter(1.0, 2.0),
		math.Nextafter(1.0, 0.0),
		math.Nextafter(2.0, 3.0),
		math.Nextafter(2.0, 1.0),
		math.Nextafter(0.5, 1.0),
		math.Nextafter(0.5, 0.0),
		3e-308, // normal, near MIN_VALUE
		1e308,
	}
}

// ── PN test generators (positive + negative values) ─────────────────

// genPNSymmetric generates equal numbers of positive and negative values
// with the same magnitude distribution. Both ranges see the same scale
// pressure, so sync_scales should be a no-op.
func genPNSymmetric(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(30)
	vals := make([]float64, 0, n*2)
	for i := 0; i < n; i++ {
		v := rng.Float64()*100 + 0.01
		vals = append(vals, v, -v)
	}
	return vals
}

// genPNAsymmetric generates many positive values with a narrow range
// and a few negative values with a wide range. This forces the positive
// side to stay at high scale while the negative side downscales, then
// sync_scales pulls the positive side down.
func genPNAsymmetric(rng *rand.Rand) []float64 {
	nPos := 20 + rng.Intn(30)
	nNeg := 3 + rng.Intn(5)
	vals := make([]float64, 0, nPos+nNeg)

	// Narrow positive range: values near 1.0
	for i := 0; i < nPos; i++ {
		vals = append(vals, 1.0+(rng.Float64()-0.5)*0.001)
	}
	// Wide negative range spanning many orders of magnitude
	for i := 0; i < nNeg; i++ {
		exp := rng.Float64()*20 - 10
		vals = append(vals, -math.Pow(2, exp))
	}
	return vals
}

// genPNScaleCoupling specifically stress-tests scale synchronization.
// It creates a scenario where the positive range needs a high scale
// (tight cluster) and the negative range forces a low scale (wide spread),
// or vice versa, in alternating batches.
func genPNScaleCoupling(rng *rand.Rand) []float64 {
	vals := make([]float64, 0, 40)

	// Phase 1: tight positive cluster (high scale needed)
	center := 100.0
	for i := 0; i < 10; i++ {
		vals = append(vals, center*(1.0+rng.Float64()*0.001))
	}

	// Phase 2: wide-spread negatives (low scale needed)
	// This should force sync_scales to pull the positive side down
	for i := 0; i < 10; i++ {
		exp := rng.Float64()*30 - 15 // 2^-15 to 2^15
		vals = append(vals, -math.Pow(2, exp))
	}

	// Phase 3: another tight positive cluster at different magnitude
	center2 := 0.001
	for i := 0; i < 10; i++ {
		vals = append(vals, center2*(1.0+rng.Float64()*0.001))
	}

	// Phase 4: tight negative cluster (now the neg side has both
	// wide-spread and tight values)
	for i := 0; i < 5; i++ {
		vals = append(vals, -(50.0 + rng.Float64()*0.01))
	}

	return vals
}

// genPNNegativeOnly generates exclusively negative values (plus maybe zeros).
// Tests that the positive bucket side stays empty.
func genPNNegativeOnly(rng *rand.Rand) []float64 {
	n := 10 + rng.Intn(30)
	vals := make([]float64, n)
	for i := range vals {
		if rng.Float64() < 0.1 {
			vals[i] = 0.0
		} else {
			vals[i] = -(rng.Float64()*1000 + 0.001)
		}
	}
	return vals
}

// genPNZerosMixed generates a mix of positive, negative, and zero values
// with zeros being common. Tests zero_count accounting.
func genPNZerosMixed(rng *rand.Rand) []float64 {
	n := 15 + rng.Intn(20)
	vals := make([]float64, n)
	for i := range vals {
		r := rng.Float64()
		if r < 0.3 {
			// Zero (positive or negative)
			if rng.Float64() < 0.5 {
				vals[i] = 0.0
			} else {
				vals[i] = math.Copysign(0, -1)
			}
		} else if r < 0.65 {
			vals[i] = rng.Float64()*10 + 0.1
		} else {
			vals[i] = -(rng.Float64()*10 + 0.1)
		}
	}
	return vals
}

// genPNExtremeSpread generates values that span the full representable
// range on both positive and negative sides. This maximally stresses
// downscaling and scale coupling.
func genPNExtremeSpread(rng *rand.Rand) []float64 {
	vals := make([]float64, 0, 20)

	// Extreme positive values
	vals = append(vals, math.MaxFloat64)
	vals = append(vals, math.SmallestNonzeroFloat64)
	vals = append(vals, 1.0)
	for i := 0; i < 5; i++ {
		exp := uint64(1 + rng.Intn(2046))
		sig := rng.Uint64() & ((1 << 52) - 1)
		bits := (exp << 52) | sig
		vals = append(vals, math.Float64frombits(bits))
	}

	// Extreme negative values
	vals = append(vals, -math.MaxFloat64)
	vals = append(vals, -math.SmallestNonzeroFloat64)
	vals = append(vals, -1.0)
	for i := 0; i < 5; i++ {
		exp := uint64(1 + rng.Intn(2046))
		sig := rng.Uint64() & ((1 << 52) - 1)
		bits := (1 << 63) | (exp << 52) | sig // sign bit set
		vals = append(vals, math.Float64frombits(bits))
	}

	// A few zeros
	vals = append(vals, 0.0, math.Copysign(0, -1))

	return vals
}

// genPNNearBoundary generates positive and negative values near bucket
// boundaries at the given scale. Tests that the index mapping agrees
// for both signs.
func genPNNearBoundary(scale int32) func(*rand.Rand) []float64 {
	return func(rng *rand.Rand) []float64 {
		n := 10 + rng.Intn(20)
		vals := make([]float64, 0, n*2)
		for i := 0; i < n; i++ {
			exp := rng.Intn(20) - 10
			subBucket := rng.Intn(1 << scale)
			idx := float64(exp) + float64(subBucket)/float64(int(1)<<scale)
			boundary := math.Pow(2, idx)
			if math.IsInf(boundary, 0) || boundary == 0 {
				boundary = 1.0
			}
			// Perturb
			ulps := rng.Intn(5) - 2
			v := boundary
			for j := 0; j < abs(ulps); j++ {
				if ulps > 0 {
					v = math.Nextafter(v, math.MaxFloat64)
				} else {
					v = math.Nextafter(v, 0)
				}
			}
			if v <= 0 {
				v = math.SmallestNonzeroFloat64
			}
			vals = append(vals, v)
			// Add the negative mirror for some values
			if rng.Float64() < 0.5 {
				vals = append(vals, -v)
			}
		}
		return vals
	}
}

// genPNSingleNegRepeated generates a single negative value repeated many
// times. Tests that negative bucket counting is correct.
func genPNSingleNegRepeated(rng *rand.Rand) []float64 {
	v := -(rng.Float64()*1000 + 0.001)
	n := 10 + rng.Intn(50)
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = v
	}
	// Add a few positive values to ensure both sides have data
	for i := 0; i < 3; i++ {
		vals = append(vals, rng.Float64()*100+0.01)
	}
	return vals
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
