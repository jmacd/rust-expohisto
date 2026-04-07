// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// CLI tool that reads hex-encoded f64 values from stdin, feeds them into
// an exponential histogram, and prints the result in a canonical text format
// for cross-implementation comparison.
//
// Usage: go-cli [-size N] [-pn]
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/open-telemetry/opentelemetry-collector-contrib/pkg/expohisto/structure"
)

func main() {
	size := flag.Int("size", 160, "maximum number of bucket indices (maxSize)")
	pnMode := flag.Bool("pn", false, "positive+negative mode (accept negative values)")
	flag.Parse()

	var hist structure.Float64
	hist.Init(structure.NewConfig(structure.WithMaxSize(int32(*size))))

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var bits uint64
		if _, err := fmt.Sscanf(line, "%x", &bits); err != nil {
			fmt.Fprintf(os.Stderr, "invalid hex: %s\n", line)
			os.Exit(1)
		}
		value := math.Float64frombits(bits)

		// Skip NaN and Inf always
		if math.IsNaN(value) || math.IsInf(value, 0) {
			continue
		}

		if !*pnMode {
			// NN mode: skip negative non-zero
			if math.Signbit(value) && value != 0 {
				continue
			}
		}

		// Normalize subnormals (positive or negative) to the smallest
		// normal so both implementations agree on bucket assignment.
		absVal := math.Abs(value)
		if absVal > 0 && absVal < 0x1p-1022 {
			value = math.Copysign(0x1p-1022, value)
		}

		hist.Update(value)
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "read error: %v\n", err)
		os.Exit(1)
	}

	scale := hist.Scale()
	count := hist.Count()
	sum := hist.Sum()
	min := hist.Min()
	max := hist.Max()
	zeroCount := hist.ZeroCount()

	positive := hist.Positive()
	positiveOffset := positive.Offset()
	positiveLen := positive.Len()

	// Normalize min/max: canonicalize ±0 to +0
	minBits := normalizeZero(min)
	maxBits := normalizeZero(max)
	sumBits := normalizeZero(sum)

	fmt.Printf("scale=%d\n", scale)
	fmt.Printf("count=%d\n", count)
	fmt.Printf("sum=%016x\n", sumBits)
	fmt.Printf("min=%016x\n", minBits)
	fmt.Printf("max=%016x\n", maxBits)
	fmt.Printf("zero_count=%d\n", zeroCount)
	fmt.Printf("positive_offset=%d\n", positiveOffset)

	posCounts := make([]string, positiveLen)
	for i := uint32(0); i < positiveLen; i++ {
		posCounts[i] = fmt.Sprintf("%d", positive.At(i))
	}
	fmt.Printf("positive_counts=[%s]\n", strings.Join(posCounts, ","))

	if *pnMode {
		negative := hist.Negative()
		negativeOffset := negative.Offset()
		negativeLen := negative.Len()

		fmt.Printf("negative_offset=%d\n", negativeOffset)
		negCounts := make([]string, negativeLen)
		for i := uint32(0); i < negativeLen; i++ {
			negCounts[i] = fmt.Sprintf("%d", negative.At(i))
		}
		fmt.Printf("negative_counts=[%s]\n", strings.Join(negCounts, ","))
	}
}

func normalizeZero(v float64) uint64 {
	if v == 0 {
		return 0
	}
	return math.Float64bits(v)
}
