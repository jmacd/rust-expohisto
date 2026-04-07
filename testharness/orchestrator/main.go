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
		name          string
		gen           func(rng *rand.Rand) []float64
		tests         int
		totalValues   int
		filtered      int
		emptyBatches  int
	}

	generators := []*generatorStats{
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

// runComparison runs both CLIs and compares semantically.
//
// Because Rust starts at table_scale (e.g. 8) and Go starts at 20,
// they may end up at different scales. The comparison normalizes both
// outputs to the lower scale before comparing.
func runComparison(rustBin, goBin string, values []float64, scale int) error {
	input := valuesToHex(values)

	// Run Rust with large capacity (B1 width → 16000 slots)
	rustOut, err := runCLI(rustBin, []string{
		"--scale", strconv.Itoa(scale),
		"--size", "16000",
	}, input)
	if err != nil {
		return fmt.Errorf("rust CLI failed: %w", err)
	}

	// Run Go with maximum capacity
	goOut, err := runCLI(goBin, []string{
		"-size", "16384",
	}, input)
	if err != nil {
		return fmt.Errorf("go CLI failed: %w", err)
	}

	// Parse both outputs
	rustParsed, err := parseOutput(rustOut)
	if err != nil {
		return fmt.Errorf("parsing rust output: %w", err)
	}
	goParsed, err := parseOutput(goOut)
	if err != nil {
		return fmt.Errorf("parsing go output: %w", err)
	}

	// Compare stats (these are scale-independent)
	if rustParsed.count != goParsed.count {
		return fmt.Errorf("count mismatch: rust=%d go=%d", rustParsed.count, goParsed.count)
	}
	if rustParsed.sumHex != goParsed.sumHex {
		return fmt.Errorf("sum mismatch: rust=%s go=%s", rustParsed.sumHex, goParsed.sumHex)
	}
	if rustParsed.minHex != goParsed.minHex {
		return fmt.Errorf("min mismatch: rust=%s go=%s", rustParsed.minHex, goParsed.minHex)
	}
	if rustParsed.maxHex != goParsed.maxHex {
		return fmt.Errorf("max mismatch: rust=%s go=%s", rustParsed.maxHex, goParsed.maxHex)
	}
	if rustParsed.zeroCount != goParsed.zeroCount {
		return fmt.Errorf("zero_count mismatch: rust=%d go=%d", rustParsed.zeroCount, goParsed.zeroCount)
	}

	// Normalize both to the lower scale
	targetScale := rustParsed.scale
	if goParsed.scale < targetScale {
		targetScale = goParsed.scale
	}

	rustNorm := downscaleOutput(rustParsed, targetScale)
	goNorm := downscaleOutput(goParsed, targetScale)

	// Compare normalized bucket data
	if rustNorm.positiveOffset != goNorm.positiveOffset {
		return fmt.Errorf("positive_offset mismatch at scale %d: rust=%d go=%d (rust_orig_scale=%d go_orig_scale=%d)",
			targetScale, rustNorm.positiveOffset, goNorm.positiveOffset, rustParsed.scale, goParsed.scale)
	}
	if len(rustNorm.positiveCounts) != len(goNorm.positiveCounts) {
		return fmt.Errorf("positive_counts length mismatch at scale %d: rust=%d go=%d",
			targetScale, len(rustNorm.positiveCounts), len(goNorm.positiveCounts))
	}
	for i := range rustNorm.positiveCounts {
		if rustNorm.positiveCounts[i] != goNorm.positiveCounts[i] {
			return fmt.Errorf("positive_counts[%d] mismatch at scale %d: rust=%d go=%d",
				i, targetScale, rustNorm.positiveCounts[i], goNorm.positiveCounts[i])
		}
	}

	return nil
}

// downscaleOutput normalizes a parsed histogram output to targetScale
// by merging adjacent buckets via arithmetic right shift.
func downscaleOutput(p *parsedOutput, targetScale int) *parsedOutput {
	if p.scale <= targetScale || len(p.positiveCounts) == 0 {
		return p
	}
	shift := uint(p.scale - targetScale)

	oldOffset := p.positiveOffset
	oldCounts := p.positiveCounts

	// Compute new index range after shifting
	newStart := oldOffset >> shift
	newEnd := (oldOffset + len(oldCounts) - 1) >> shift
	newLen := newEnd - newStart + 1

	newCounts := make([]uint64, newLen)
	for i, c := range oldCounts {
		oldIdx := oldOffset + i
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
		return &parsedOutput{
			scale:          targetScale,
			count:          p.count,
			zeroCount:      p.zeroCount,
			sumHex:         p.sumHex,
			minHex:         p.minHex,
			maxHex:         p.maxHex,
			positiveOffset: 0,
			positiveCounts: nil,
		}
	}

	return &parsedOutput{
		scale:          targetScale,
		count:          p.count,
		zeroCount:      p.zeroCount,
		sumHex:         p.sumHex,
		minHex:         p.minHex,
		maxHex:         p.maxHex,
		positiveOffset: newStart + first,
		positiveCounts: newCounts[first : last+1],
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
			val = strings.TrimPrefix(val, "[")
			val = strings.TrimSuffix(val, "]")
			if val == "" {
				break
			}
			for _, cs := range strings.Split(val, ",") {
				c, err := strconv.ParseUint(cs, 10, 64)
				if err != nil {
					return nil, err
				}
				p.positiveCounts = append(p.positiveCounts, c)
			}
		}
	}
	return p, nil
}

// Test generators

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

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
