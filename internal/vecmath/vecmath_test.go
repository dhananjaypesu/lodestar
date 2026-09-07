package vecmath

import (
	"math"
	"math/rand"
	"testing"
)

// naive is the obvious implementation, kept as the oracle the unrolled kernels
// are checked against and as the baseline the benchmarks measure against.
func naive(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func naiveDot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Unrolling changes the order the partial sums are added, so the results are
// not bit-identical to the naive loop — floating-point addition is not
// associative. They must agree to within rounding, and every length must be
// handled, including the ones that do not divide evenly by the unroll width.
func TestKernelsMatchNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{0, 1, 3, 7, 8, 9, 15, 16, 31, 64, 128, 257, 768} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		if got, want := L2Squared(a, b), naive(a, b); !close(got, want) {
			t.Errorf("L2Squared at n=%d: got %v, want %v", n, got, want)
		}
		if got, want := Dot(a, b), naiveDot(a, b); !close(got, want) {
			t.Errorf("Dot at n=%d: got %v, want %v", n, got, want)
		}
	}
}

func close(a, b float32) bool {
	diff := math.Abs(float64(a - b))
	return diff <= 1e-4*math.Max(1, math.Abs(float64(b)))
}

func TestL2Properties(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	if got := L2Squared(a, a); got != 0 {
		t.Errorf("distance from a vector to itself is %v, want 0", got)
	}
	b := []float32{5, 4, 3, 2, 1}
	if L2Squared(a, b) != L2Squared(b, a) {
		t.Error("L2Squared is not symmetric")
	}
}

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	if !Normalize(v) {
		t.Fatal("Normalize refused a valid vector")
	}
	if !close(Norm(v), 1) {
		t.Errorf("norm after normalizing is %v", Norm(v))
	}
	if !close(v[0], 0.6) || !close(v[1], 0.8) {
		t.Errorf("normalized to %v, want [0.6 0.8]", v)
	}
	// A zero vector has no direction, so normalizing it must fail rather than
	// silently produce NaNs that poison every later distance.
	zero := []float32{0, 0, 0}
	if Normalize(zero) {
		t.Error("Normalize accepted a zero vector")
	}
}

func TestMismatchedLengthsPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("comparing vectors of different lengths did not panic")
		}
	}()
	L2Squared([]float32{1, 2}, []float32{1, 2, 3})
}

var (
	benchA, benchB []float32
	sink           float32
)

func init() {
	rng := rand.New(rand.NewSource(1))
	benchA = make([]float32, 128)
	benchB = make([]float32, 128)
	for i := range benchA {
		benchA[i] = rng.Float32()
		benchB[i] = rng.Float32()
	}
}

// The point of these three is the comparison, not the absolute number: it is
// what justifies the unrolled kernel being written the way it is.
func BenchmarkL2Naive(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sink = naive(benchA, benchB)
	}
}

func BenchmarkL2Unrolled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sink = L2Squared(benchA, benchB)
	}
}

func BenchmarkDotUnrolled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sink = Dot(benchA, benchB)
	}
}
