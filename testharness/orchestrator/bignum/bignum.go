// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package bignum verifies that Go's logarithm-based MapToIndex produces
// correct results for a given f64 value at a given scale, by computing
// the exact bucket index with arbitrary-precision arithmetic.
//
// The exact index for a positive, normal value is:
//
//	floor(log2(value) * 2^scale)        for non-power-of-two
//	(exponent << scale) - 1             for exact powers of 2
//
// We compute log2(value) * 2^scale with 256-bit precision and compare
// with Go's float64-based result.
package bignum

import (
	"math"
	"math/big"

	"github.com/open-telemetry/opentelemetry-collector-contrib/pkg/expohisto/mapping/logarithm"
)

// Precision for big.Float calculations. 256 bits is more than enough
// for exact comparison with float64 (53-bit significand).
const bigPrec = 256

// CheckMapToIndex verifies that Go's logarithm mapper produces the
// correct bucket index for the given value at the given scale.
//
// Returns (goIndex, exactIndex, match) where:
//   - goIndex is what Go's MapToIndex returns
//   - exactIndex is the mathematically correct index
//   - match is true if they agree
//
// Only checks positive, normal, non-zero values at positive scales.
func CheckMapToIndex(value float64, scale int32) (goIndex, exactIndex int32, match bool) {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) || scale < 1 {
		return 0, 0, true // not applicable
	}

	m, err := logarithm.NewMapping(scale)
	if err != nil {
		return 0, 0, true
	}

	goIndex = m.MapToIndex(value)
	exactIndex = ExactMapToIndex(value, scale)
	return goIndex, exactIndex, goIndex == exactIndex
}

// ExactMapToIndex computes the mathematically correct bucket index for
// a positive f64 value at the given scale using arbitrary-precision math.
//
// For exact powers of 2: index = (exponent << scale) - 1
// For others: index = floor(log2(value) * 2^scale) when value is not
// on a boundary, or ceil(log2(value) * 2^scale) - 1 when exact.
func ExactMapToIndex(value float64, scale int32) int32 {
	bits := math.Float64bits(value)
	significand := bits & ((1 << 52) - 1)

	// Exact powers of two
	if significand == 0 {
		exp := int32((bits>>52)&0x7FF) - 1023
		return (exp << scale) - 1
	}

	// Subnormals
	if value <= math.SmallestNonzeroFloat64*2 {
		return (-1022 << scale) - 1
	}

	return computeExactIndex(value, scale)
}

// computeExactIndex uses big.Float to compute the exact bucket index.
//
// The bucket index i satisfies:
//
//	2^(i / 2^scale) < value <= 2^((i+1) / 2^scale)
//
// Equivalently: i < log2(value) * 2^scale <= i+1
// So: i = ceil(log2(value) * 2^scale) - 1
//
// If log2(value) * 2^scale is exactly an integer, value sits on a bucket
// boundary and belongs to the bucket below (index = integer - 1).
// Otherwise, index = floor(log2(value) * 2^scale).
func computeExactIndex(value float64, scale int32) int32 {
	bigValue := new(big.Float).SetPrec(bigPrec).SetFloat64(value)
	bigLog2 := bigLog2Float(bigValue)

	scaleFactor := new(big.Float).SetPrec(bigPrec).SetMantExp(big.NewFloat(1.0), int(scale))
	scaled := new(big.Float).SetPrec(bigPrec).Mul(bigLog2, scaleFactor)

	intPart, accuracy := scaled.Int(nil)

	if accuracy == big.Exact {
		// Value is on a bucket boundary: index = integer - 1
		intPart.Sub(intPart, big.NewInt(1))
		return int32(intPart.Int64())
	}

	// For negative scaled values, Int() truncates toward zero
	// (gives ceiling), but we want floor.
	if scaled.Sign() < 0 {
		intPart.Sub(intPart, big.NewInt(1))
	}

	return int32(intPart.Int64())
}

// bigLog2Float computes log2(x) with 256-bit precision.
//
// Decompose x = 2^exp * y where y in [1, 2), then
// log2(x) = exp + log2(y) = exp + ln(y)/ln(2).
func bigLog2Float(x *big.Float) *big.Float {
	mant := new(big.Float).SetPrec(bigPrec).Copy(x)
	exp := mant.MantExp(mant)
	// Now mant in [0.5, 1.0) and x = mant * 2^exp
	// y = mant * 2, so log2(x) = (exp-1) + log2(y)
	y := new(big.Float).SetPrec(bigPrec).Mul(mant, big.NewFloat(2.0))
	adjustedExp := exp - 1

	lnY := bigLn(y)
	ln2 := bigLn(new(big.Float).SetPrec(bigPrec).SetFloat64(2.0))
	log2Y := new(big.Float).SetPrec(bigPrec).Quo(lnY, ln2)

	result := new(big.Float).SetPrec(bigPrec).SetInt64(int64(adjustedExp))
	result.Add(result, log2Y)
	return result
}

// bigLn computes ln(x) for x > 0 using the identity:
//
//	ln(x) = 2 * atanh((x-1)/(x+1))
//	atanh(u) = u + u^3/3 + u^5/5 + ...
//
// This converges well for x in [1, 2) since u = (x-1)/(x+1) < 1/3.
func bigLn(x *big.Float) *big.Float {
	one := new(big.Float).SetPrec(bigPrec).SetFloat64(1.0)
	if x.Cmp(one) == 0 {
		return new(big.Float).SetPrec(bigPrec)
	}

	two := new(big.Float).SetPrec(bigPrec).SetFloat64(2.0)

	xm1 := new(big.Float).SetPrec(bigPrec).Sub(x, one)
	xp1 := new(big.Float).SetPrec(bigPrec).Add(x, one)
	u := new(big.Float).SetPrec(bigPrec).Quo(xm1, xp1)

	u2 := new(big.Float).SetPrec(bigPrec).Mul(u, u)

	sum := new(big.Float).SetPrec(bigPrec).Copy(u)
	term := new(big.Float).SetPrec(bigPrec).Copy(u)

	for k := int64(3); k < 300; k += 2 {
		term.Mul(term, u2)
		kf := new(big.Float).SetPrec(bigPrec).SetInt64(k)
		contrib := new(big.Float).SetPrec(bigPrec).Quo(term, kf)
		sum.Add(sum, contrib)

		if contrib.MantExp(nil) < -int(bigPrec)-10 {
			break
		}
	}

	return new(big.Float).SetPrec(bigPrec).Mul(two, sum)
}

// FilterAccurate returns only values for which Go's MapToIndex gives
// the correct result at the given scale. Zeros, NaNs, Infs, and
// negatives pass through unchanged. Subnormals pass through because
// both CLIs normalize them to MIN_VALUE at input time.
func FilterAccurate(values []float64, scale int32) (accurate []float64, inaccurateCount int) {
	for _, v := range values {
		if v == 0 || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			accurate = append(accurate, v)
			continue
		}
		// Subnormals: both CLIs normalize to MIN_VALUE, so the
		// bignum check would be against MIN_VALUE (a power of 2,
		// always exact). Just pass them through.
		if v < MinNormalFloat64 {
			accurate = append(accurate, v)
			continue
		}
		_, _, ok := CheckMapToIndex(v, scale)
		if ok {
			accurate = append(accurate, v)
		} else {
			inaccurateCount++
		}
	}
	return
}

// MinNormalFloat64 is the smallest positive normal float64: 2^-1022.
const MinNormalFloat64 = 0x1p-1022
