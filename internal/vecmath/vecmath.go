// Package vecmath holds the distance kernels.
//
// These functions are where a vector index spends nearly all of its time: a
// profile of a 20,000-vector build attributes 34% of samples to L2Squared
// alone, and that is the honest shape for an ANN index — everything else is
// bookkeeping around distance computations.
//
// So they are written to be easy for the compiler to keep in registers. Two
// details do the work. Re-slicing to a fixed eight-element window lets the
// compiler prove the indexes are in range and drop the bounds checks. Eight
// separate accumulators break the dependency chain: a single running sum makes
// every addition wait for the previous one, and floating-point addition has
// several cycles of latency, so the loop stalls on itself rather than on
// memory.
//
// Measured on an Apple M2, 128 dimensions: 97 ns naive, 33 ns unrolled four
// wide, 29 ns unrolled eight wide. Go has no portable SIMD intrinsics, so this
// is as close to the hardware as the standard library allows; a NEON assembly
// kernel would go further at the cost of one implementation per architecture.
package vecmath

import "math"

// L2Squared returns the squared Euclidean distance between a and b.
//
// The square root is deliberately not taken. Ranking by squared distance
// gives the same order as ranking by distance, so every comparison in the
// index avoids a sqrt, and callers that want the real distance take it once
// at the end rather than millions of times in the middle.
func L2Squared(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vecmath: mismatched vector lengths")
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i, n := 0, len(a)
	for ; i+8 <= n; i += 8 {
		x := a[i : i+8 : i+8]
		y := b[i : i+8 : i+8]
		d0 := x[0] - y[0]
		d1 := x[1] - y[1]
		d2 := x[2] - y[2]
		d3 := x[3] - y[3]
		d4 := x[4] - y[4]
		d5 := x[5] - y[5]
		d6 := x[6] - y[6]
		d7 := x[7] - y[7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	// Summed as a tree rather than left to right, for the same reason the
	// accumulators are separate.
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// Dot returns the inner product of a and b.
func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vecmath: mismatched vector lengths")
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i, n := 0, len(a)
	for ; i+8 <= n; i += 8 {
		x := a[i : i+8 : i+8]
		y := b[i : i+8 : i+8]
		s0 += x[0] * y[0]
		s1 += x[1] * y[1]
		s2 += x[2] * y[2]
		s3 += x[3] * y[3]
		s4 += x[4] * y[4]
		s5 += x[5] * y[5]
		s6 += x[6] * y[6]
		s7 += x[7] * y[7]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm returns the Euclidean length of a.
func Norm(a []float32) float32 {
	return float32(math.Sqrt(float64(Dot(a, a))))
}

// Normalize scales a to unit length in place and reports whether it could.
//
// Cosine similarity is the inner product of normalized vectors, so an index
// with the cosine metric normalizes once on insert and then never pays for
// the division again — every subsequent distance is a plain dot product.
func Normalize(a []float32) bool {
	n := Norm(a)
	if n == 0 || math.IsNaN(float64(n)) || math.IsInf(float64(n), 0) {
		return false
	}
	inv := 1 / n
	for i := range a {
		a[i] *= inv
	}
	return true
}

// Add accumulates b into a, used when averaging vectors during k-means.
func Add(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// Scale multiplies a by f in place.
func Scale(a []float32, f float32) {
	for i := range a {
		a[i] *= f
	}
}
