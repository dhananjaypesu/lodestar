package lodestar_test

import (
	"bytes"
	"testing"

	"github.com/dhananjaypesu/lodestar"
)

// Quantization is a trade, so the test asserts both sides of it: the index
// really does get smaller, and recall really does survive.
func TestQuantizationTradesMemoryForRecall(t *testing.T) {
	d := clusteredDataset(6000, 64, 16, 31, lodestar.L2)
	build := func() *lodestar.Index {
		return d.build(t, lodestar.Config{M: 16, EfConstruction: 200, Seed: 7})
	}

	exact := build()
	baseline := recall(t, exact, d, 100, 10, 100, nil)
	t.Logf("full precision:          recall@10 %.4f at %d bytes/vector", baseline, d.dim*4)
	if baseline < 0.97 {
		t.Fatalf("baseline recall %.4f is too low to draw a comparison from", baseline)
	}

	// Compressed with reranking: the graph walk uses 16-byte codes, then the
	// survivors are rescored exactly. This should cost almost nothing.
	reranked := build()
	if err := reranked.Quantize(lodestar.QuantizeOptions{Subspaces: 16, KeepVectors: true}); err != nil {
		t.Fatalf("quantize: %v", err)
	}
	withRerank := recall(t, reranked, d, 100, 10, 100, nil)
	mse, err := reranked.ReconstructionError()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("16 subspaces + rerank:   recall@10 %.4f, reconstruction MSE %.4f", withRerank, mse)
	if withRerank < baseline-0.10 {
		t.Errorf("reranked recall %.4f fell more than 10 points below the exact baseline %.4f", withRerank, baseline)
	}

	// Compressed without the vectors: 16 bytes per vector instead of 256, and
	// the ranking is approximate all the way through.
	compact := build()
	if err := compact.Quantize(lodestar.QuantizeOptions{Subspaces: 16, KeepVectors: false}); err != nil {
		t.Fatalf("quantize: %v", err)
	}
	quantized, bytesPer := compact.Quantized()
	if !quantized || bytesPer != 16 {
		t.Fatalf("Quantized() = %v, %d; want true, 16", quantized, bytesPer)
	}
	withoutRerank := recall(t, compact, d, 100, 10, 100, nil)
	t.Logf("16 subspaces, codes only: recall@10 %.4f at %d bytes/vector (%.0fx smaller)",
		withoutRerank, bytesPer, float64(d.dim*4)/float64(bytesPer))

	if withoutRerank < 0.35 {
		t.Errorf("recall without reranking collapsed to %.4f", withoutRerank)
	}
	// The whole argument for keeping vectors is that reranking recovers what
	// the codes lost, so it must actually be better.
	if withRerank <= withoutRerank {
		t.Errorf("reranking (%.4f) did not beat raw codes (%.4f)", withRerank, withoutRerank)
	}

	// An index whose vectors were dropped cannot accept new ones.
	if err := compact.Add(1<<40, d.vecs[0]); err == nil {
		t.Error("adding to an index with no raw vectors was accepted")
	}
	if _, err := compact.ReconstructionError(); err == nil {
		t.Error("reconstruction error was computed without the raw vectors")
	}
}

// More subspaces means finer codes, so error must fall and recall must rise.
// The direction is what is being asserted, not any particular value.
func TestMoreSubspacesReduceError(t *testing.T) {
	d := clusteredDataset(3000, 64, 10, 13, lodestar.L2)
	var lastMSE float64 = 1e18
	for _, m := range []int{4, 8, 16, 32} {
		ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 128, Seed: 3})
		if err := ix.Quantize(lodestar.QuantizeOptions{Subspaces: m, KeepVectors: true}); err != nil {
			t.Fatalf("m=%d: %v", m, err)
		}
		mse, err := ix.ReconstructionError()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("subspaces=%-3d bytes/vector=%-3d MSE=%.4f", m, m, mse)
		if mse >= lastMSE {
			t.Errorf("MSE did not fall when subspaces rose to %d: %.4f then %.4f", m, lastMSE, mse)
		}
		lastMSE = mse
	}
}

func TestQuantizeRejectsBadConfiguration(t *testing.T) {
	d := clusteredDataset(500, 30, 5, 2, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 8, EfConstruction: 64})

	// 30 is not divisible by 16.
	if err := ix.Quantize(lodestar.QuantizeOptions{Subspaces: 16}); err == nil {
		t.Error("an indivisible subspace count was accepted")
	}
	if err := ix.Quantize(lodestar.QuantizeOptions{Subspaces: 0}); err == nil {
		t.Error("zero subspaces was accepted")
	}
	// 500 vectors is fewer than the 256 centroids per subspace need... it is
	// not, but 200 would be, and that boundary is worth pinning.
	small, _ := lodestar.New(lodestar.Config{Dim: 8})
	for i := 0; i < 200; i++ {
		small.Add(uint64(i), d.vecs[i][:8])
	}
	if err := small.Quantize(lodestar.QuantizeOptions{Subspaces: 4}); err == nil {
		t.Error("training on fewer vectors than centroids was accepted")
	}

	empty, _ := lodestar.New(lodestar.Config{Dim: 8})
	if err := empty.Quantize(lodestar.QuantizeOptions{Subspaces: 4}); err != lodestar.ErrEmpty {
		t.Errorf("quantizing an empty index returned %v", err)
	}
	if _, err := empty.ReconstructionError(); err != lodestar.ErrNotTrained {
		t.Errorf("reconstruction error on an untrained index returned %v", err)
	}
}

func TestQuantizedIndexSurvivesSaveWithVectors(t *testing.T) {
	d := clusteredDataset(1500, 32, 6, 8, lodestar.Cosine)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 128})
	if err := ix.Quantize(lodestar.QuantizeOptions{Subspaces: 8, KeepVectors: true}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := lodestar.Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, bytesPer := loaded.Quantized(); bytesPer != 8+32*4 {
		t.Errorf("bytes per vector is %d, want %d", bytesPer, 8+32*4)
	}
	if got := recall(t, loaded, d, 50, 10, 100, nil); got < 0.90 {
		t.Errorf("recall after loading a quantized cosine index is %.4f", got)
	}
}

// The two sources of error in a quantized index are independent, and this is
// the cleanest way to see it.
//
// Raising ef makes the graph walk visit more nodes, so it can only help when
// the limit is which nodes were visited. When the index ranks by compressed
// codes, ef buys nothing at all — the ten smallest approximate distances are
// already found, and looking at more candidates does not change which ten they
// are. Recall is pinned by the codes, and the curve is flat.
//
// Reranking changes the question. The walk is still approximate, but the final
// ordering is exact, so a wider candidate list genuinely does contain better
// answers and recall climbs back to the exact baseline. That is the whole
// argument for keeping the vectors, stated as a measurement rather than an
// opinion.
func TestQuantizationErrorIsIndependentOfEf(t *testing.T) {
	d := clusteredDataset(4000, 64, 12, 31, lodestar.L2)

	codesOnly := d.build(t, lodestar.Config{M: 16, EfConstruction: 200, Seed: 7})
	if err := codesOnly.Quantize(lodestar.QuantizeOptions{Subspaces: 8, KeepVectors: false}); err != nil {
		t.Fatal(err)
	}
	reranked := d.build(t, lodestar.Config{M: 16, EfConstruction: 200, Seed: 7})
	if err := reranked.Quantize(lodestar.QuantizeOptions{Subspaces: 8, KeepVectors: true}); err != nil {
		t.Fatal(err)
	}

	efs := []int{50, 200, 800}
	flat := make([]float64, len(efs))
	rising := make([]float64, len(efs))
	for i, ef := range efs {
		flat[i] = recall(t, codesOnly, d, 60, 10, ef, nil)
		rising[i] = recall(t, reranked, d, 60, 10, ef, nil)
		t.Logf("ef=%-4d codes only %.4f   reranked %.4f", ef, flat[i], rising[i])
	}

	// Codes only: more search effort must not move recall meaningfully.
	spread := flat[len(flat)-1] - flat[0]
	if spread > 0.05 {
		t.Errorf("recall without reranking moved by %.4f across ef, expected it to be pinned by the codes", spread)
	}
	// Reranked: more search effort must pay.
	if rising[len(rising)-1] <= rising[0] {
		t.Errorf("reranked recall did not improve with ef: %.4f at ef=%d, %.4f at ef=%d",
			rising[0], efs[0], rising[len(rising)-1], efs[len(efs)-1])
	}
	if rising[len(rising)-1] < 0.95 {
		t.Errorf("reranked recall at ef=%d is only %.4f", efs[len(efs)-1], rising[len(rising)-1])
	}
}
