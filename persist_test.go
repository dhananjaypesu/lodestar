package lodestar_test

import (
	"bytes"
	"testing"

	"github.com/dhananjaypesu/lodestar"
)

// A reloaded index must answer identically, not merely similarly. Anything
// less means the links or the vectors did not survive the round trip, and the
// symptom would otherwise be a quiet drop in recall after a restart.
func TestSaveLoadRoundTrip(t *testing.T) {
	d := clusteredDataset(2000, 24, 8, 5, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 128, Seed: 999})
	ix.Delete(7)
	ix.Delete(11)

	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatalf("save: %v", err)
	}
	t.Logf("%d vectors, %d dims serialize to %d bytes (%.1f bytes/vector)",
		ix.Len(), d.dim, buf.Len(), float64(buf.Len())/float64(ix.Len()))

	loaded, err := lodestar.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Len() != ix.Len() {
		t.Fatalf("loaded index holds %d vectors, original held %d", loaded.Len(), ix.Len())
	}
	if loaded.Layers() != ix.Layers() {
		t.Fatalf("loaded index has %d layers, original had %d", loaded.Layers(), ix.Layers())
	}
	if loaded.Contains(7) {
		t.Error("a deleted id came back after loading")
	}

	for i := 0; i < 200; i++ {
		q := d.vecs[i*9%len(d.vecs)]
		want, err := ix.Search(q, 10, 100)
		if err != nil {
			t.Fatal(err)
		}
		got, err := loaded.Search(q, 10, 100)
		if err != nil {
			t.Fatal(err)
		}
		if len(want) != len(got) {
			t.Fatalf("query %d: %d results before, %d after", i, len(want), len(got))
		}
		for j := range want {
			if want[j] != got[j] {
				t.Fatalf("query %d position %d: %v before, %v after", i, j, want[j], got[j])
			}
		}
	}

	// The loaded index must still be usable, not just readable.
	if err := loaded.Add(999999, d.vecs[0]); err != nil {
		t.Fatalf("add to a loaded index: %v", err)
	}
	res, err := loaded.Search(d.vecs[0], 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("search after adding to a loaded index returned nothing")
	}
}

func TestLoadRejectsCorruption(t *testing.T) {
	d := randomDataset(600, 12, 4, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 8, EfConstruction: 64})
	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatal(err)
	}
	good := buf.Bytes()

	if _, err := lodestar.Load(bytes.NewReader([]byte("not an index at all"))); err == nil {
		t.Error("random bytes loaded as an index")
	}
	if _, err := lodestar.Load(bytes.NewReader(good[:len(good)/2])); err == nil {
		t.Error("a truncated index loaded")
	}

	// A single flipped bit in the middle of the payload must be caught. This
	// is the case the checksum exists for: it does not crash anything, it just
	// silently degrades the graph.
	corrupt := append([]byte(nil), good...)
	corrupt[len(corrupt)/2] ^= 0x01
	if _, err := lodestar.Load(bytes.NewReader(corrupt)); err == nil {
		t.Error("a flipped bit loaded without complaint")
	}
}

func TestSaveLoadPreservesQuantization(t *testing.T) {
	d := clusteredDataset(2000, 32, 8, 6, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 128})
	if err := ix.Quantize(lodestar.QuantizeOptions{Subspaces: 8, KeepVectors: false}); err != nil {
		t.Fatalf("quantize: %v", err)
	}

	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := lodestar.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if quantized, bytesPer := loaded.Quantized(); !quantized || bytesPer != 8 {
		t.Fatalf("loaded index reports quantized=%v at %d bytes/vector, want true at 8", quantized, bytesPer)
	}
	for i := 0; i < 100; i++ {
		q := d.vecs[i*13%len(d.vecs)]
		want, _ := ix.Search(q, 10, 100)
		got, _ := loaded.Search(q, 10, 100)
		for j := range want {
			if want[j].ID != got[j].ID {
				t.Fatalf("query %d position %d: id %d before, %d after", i, j, want[j].ID, got[j].ID)
			}
		}
	}
}
