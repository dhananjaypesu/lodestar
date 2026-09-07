package pq

import (
	"math"
	"math/rand"
	"testing"

	"github.com/dhananjaypesu/lodestar/internal/vecmath"
)

// clustered data is what quantization is designed for: if vectors were
// uniformly random, no codebook of 256 centroids could describe an 8-
// dimensional subspace well, and the technique would have nothing to exploit.
func clustered(n, dim, clusters int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([]float32, clusters*dim)
	for i := range centers {
		centers[i] = float32(rng.NormFloat64()) * 3
	}
	out := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		c := rng.Intn(clusters)
		for j := 0; j < dim; j++ {
			out[i*dim+j] = centers[c*dim+j] + float32(rng.NormFloat64())
		}
	}
	return out
}

func TestTrainRejectsBadInput(t *testing.T) {
	data := clustered(500, 12, 4, 1)
	if _, err := Train(data, 12, 5, 10, 1); err == nil {
		t.Error("a dimension not divisible by the subspace count was accepted")
	}
	if _, err := Train(data[:12*100], 12, 4, 10, 1); err == nil {
		t.Error("training on fewer vectors than centroids was accepted")
	}
}

// Reconstruction error must fall as the codes get finer. This is the one
// property that says the codebooks are actually being learned rather than
// merely being written.
func TestReconstructionImprovesWithSubspaces(t *testing.T) {
	dim, n := 32, 4000
	data := clustered(n, dim, 10, 2)

	var last float64 = math.MaxFloat64
	for _, m := range []int{2, 4, 8, 16} {
		q, err := Train(data, dim, m, 20, 5)
		if err != nil {
			t.Fatal(err)
		}
		code := make([]byte, q.CodeLen())
		recon := make([]float32, dim)
		var total float64
		for i := 0; i < n; i++ {
			v := data[i*dim : (i+1)*dim]
			q.Encode(v, code)
			q.Decode(code, recon)
			total += float64(vecmath.L2Squared(v, recon))
		}
		mse := total / float64(n)
		t.Logf("subspaces=%-3d bytes=%-3d MSE=%.4f", m, q.CodeLen(), mse)
		if mse >= last {
			t.Errorf("MSE did not improve at m=%d: %.4f then %.4f", m, last, mse)
		}
		last = mse
	}
}

// Asymmetric distance computation must approximate the true distance, and the
// approximation must be good enough to rank by. Correlation is the honest test
// — the absolute values are allowed to be off, the ordering is not.
func TestAsymmetricDistanceRanksLikeTheTruth(t *testing.T) {
	dim, n := 32, 3000
	data := clustered(n, dim, 8, 4)
	q, err := Train(data, dim, 8, 25, 6)
	if err != nil {
		t.Fatal(err)
	}

	codes := make([]byte, n*q.CodeLen())
	for i := 0; i < n; i++ {
		q.Encode(data[i*dim:(i+1)*dim], codes[i*q.CodeLen():(i+1)*q.CodeLen()])
	}

	table := make([]float32, q.TableLen())
	rng := rand.New(rand.NewSource(9))
	agree, total := 0, 0
	for trial := 0; trial < 50; trial++ {
		query := data[rng.Intn(n)*dim:]
		query = query[:dim]
		q.BuildTable(query, ModeL2, table)

		// Over random pairs, the approximate distance must order them the same
		// way the true distance does, except when they are nearly tied.
		for p := 0; p < 200; p++ {
			i, j := rng.Intn(n), rng.Intn(n)
			trueI := vecmath.L2Squared(query, data[i*dim:(i+1)*dim])
			trueJ := vecmath.L2Squared(query, data[j*dim:(j+1)*dim])
			if math.Abs(float64(trueI-trueJ)) < 0.1*float64(trueI+trueJ) {
				continue // a near tie says nothing about the approximation
			}
			approxI := q.Lookup(table, codes[i*q.CodeLen():(i+1)*q.CodeLen()])
			approxJ := q.Lookup(table, codes[j*q.CodeLen():(j+1)*q.CodeLen()])
			total++
			if (trueI < trueJ) == (approxI < approxJ) {
				agree++
			}
		}
	}
	rate := float64(agree) / float64(total)
	t.Logf("approximate distances ordered %d of %d clearly-separated pairs correctly (%.4f)", agree, total, rate)
	if rate < 0.97 {
		t.Errorf("ordering agreement is only %.4f", rate)
	}
}

// The dot-product table must rank vectors the way the true inner product does,
// since cosine and inner-product indexes order results by it directly.
//
// The property is stated as ordering agreement rather than as relative error
// on purpose. Inner products cross zero, so a relative error is unbounded near
// the crossing and says nothing about whether the approximation is usable —
// what a ranking needs is that it puts pairs in the right order.
func TestDotTableRanksLikeTheTruth(t *testing.T) {
	dim, n := 32, 3000
	data := clustered(n, dim, 6, 7)
	q, err := Train(data, dim, 16, 25, 8)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([]byte, n*q.CodeLen())
	for i := 0; i < n; i++ {
		q.Encode(data[i*dim:(i+1)*dim], codes[i*q.CodeLen():(i+1)*q.CodeLen()])
	}

	table := make([]float32, q.TableLen())
	rng := rand.New(rand.NewSource(21))
	agree, total := 0, 0
	var absErr, scale float64
	for trial := 0; trial < 40; trial++ {
		query := data[rng.Intn(n)*dim:][:dim]
		q.BuildTable(query, ModeDot, table)
		for p := 0; p < 200; p++ {
			i, j := rng.Intn(n), rng.Intn(n)
			trueI := float64(vecmath.Dot(query, data[i*dim:(i+1)*dim]))
			trueJ := float64(vecmath.Dot(query, data[j*dim:(j+1)*dim]))
			approxI := float64(q.Lookup(table, codes[i*q.CodeLen():(i+1)*q.CodeLen()]))
			approxJ := float64(q.Lookup(table, codes[j*q.CodeLen():(j+1)*q.CodeLen()]))
			absErr += math.Abs(trueI - approxI)
			scale += math.Abs(trueI)
			if math.Abs(trueI-trueJ) < 0.05*(math.Abs(trueI)+math.Abs(trueJ)) {
				continue
			}
			total++
			if (trueI < trueJ) == (approxI < approxJ) {
				agree++
			}
		}
	}
	rate := float64(agree) / float64(total)
	t.Logf("ordering agreement %.4f over %d pairs; mean absolute error %.4f against a mean magnitude of %.4f",
		rate, total, absErr/float64(total), scale/float64(total))
	if rate < 0.97 {
		t.Errorf("ordering agreement is only %.4f", rate)
	}
}

// Training must be reproducible, or no measurement taken against a trained
// quantizer would be comparable to the next one.
func TestTrainingIsDeterministic(t *testing.T) {
	data := clustered(1500, 16, 5, 3)
	a, err := Train(data, 16, 4, 15, 42)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Train(data, 16, 4, 15, 42)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Centroids {
		if a.Centroids[i] != b.Centroids[i] {
			t.Fatalf("centroid %d differs between two runs with the same seed", i)
		}
	}
}

// Every code must name a real centroid, and encoding must pick the nearest
// one — that is the entire contract of Encode.
func TestEncodePicksTheNearestCentroid(t *testing.T) {
	dim := 16
	data := clustered(2000, dim, 6, 11)
	q, err := Train(data, dim, 4, 20, 13)
	if err != nil {
		t.Fatal(err)
	}
	code := make([]byte, q.CodeLen())
	for i := 0; i < 100; i++ {
		v := data[i*dim : (i+1)*dim]
		q.Encode(v, code)
		for s := 0; s < q.M; s++ {
			part := v[s*q.Sub : (s+1)*q.Sub]
			book := q.Centroids[s*K*q.Sub : (s+1)*K*q.Sub]
			chosen := vecmath.L2Squared(part, book[int(code[s])*q.Sub:(int(code[s])+1)*q.Sub])
			for c := 0; c < K; c++ {
				if d := vecmath.L2Squared(part, book[c*q.Sub:(c+1)*q.Sub]); d < chosen-1e-6 {
					t.Fatalf("vector %d subspace %d: chose centroid %d at %v, but %d is closer at %v",
						i, s, code[s], chosen, c, d)
				}
			}
		}
	}
}
