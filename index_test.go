package lodestar_test

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"

	"github.com/dhananjaypesu/lodestar"
)

// dataset is a reproducible set of vectors plus the brute-force answers to
// compare an approximate index against. Every recall assertion in this file is
// made against exact results computed here, never against another run of the
// index — an index that is consistently wrong would pass that.
type dataset struct {
	vecs   [][]float32
	dim    int
	metric lodestar.Metric
}

func randomDataset(n, dim int, seed int64, metric lodestar.Metric) *dataset {
	rng := rand.New(rand.NewSource(seed))
	d := &dataset{vecs: make([][]float32, n), dim: dim, metric: metric}
	for i := range d.vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		d.vecs[i] = v
	}
	return d
}

// clusteredDataset is closer to real embedding data than uniform noise:
// vectors sit in a handful of gaussian clusters, which is the structure a
// navigable graph is supposed to exploit.
func clusteredDataset(n, dim, clusters int, seed int64, metric lodestar.Metric) *dataset {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for i := range centers {
		c := make([]float32, dim)
		for j := range c {
			c[j] = float32(rng.NormFloat64()) * 4
		}
		centers[i] = c
	}
	d := &dataset{vecs: make([][]float32, n), dim: dim, metric: metric}
	for i := range d.vecs {
		c := centers[rng.Intn(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64())
		}
		d.vecs[i] = v
	}
	return d
}

func (d *dataset) distance(a, b []float32) float64 {
	switch d.metric {
	case lodestar.Cosine:
		var dot, na, nb float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
	case lodestar.InnerProduct:
		var dot float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
		}
		return -dot
	default:
		var s float64
		for i := range a {
			diff := float64(a[i]) - float64(b[i])
			s += diff * diff
		}
		return math.Sqrt(s)
	}
}

// truth returns the exact k nearest ids, considering only ids for which keep
// returns true.
func (d *dataset) truth(q []float32, k int, keep func(uint64) bool) []uint64 {
	type hit struct {
		dist float64
		id   uint64
	}
	hits := make([]hit, 0, len(d.vecs))
	for i, v := range d.vecs {
		if keep != nil && !keep(uint64(i)) {
			continue
		}
		hits = append(hits, hit{d.distance(v, q), uint64(i)})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].dist != hits[j].dist {
			return hits[i].dist < hits[j].dist
		}
		return hits[i].id < hits[j].id
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	out := make([]uint64, len(hits))
	for i, h := range hits {
		out[i] = h.id
	}
	return out
}

func (d *dataset) build(t *testing.T, cfg lodestar.Config) *lodestar.Index {
	t.Helper()
	cfg.Dim = d.dim
	cfg.Metric = d.metric
	ix, err := lodestar.New(cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for i, v := range d.vecs {
		if err := ix.Add(uint64(i), v); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}
	return ix
}

// recall measures the fraction of true nearest neighbours the index returned,
// averaged over queries drawn from the dataset itself.
func recall(t *testing.T, ix *lodestar.Index, d *dataset, queries, k, ef int, keep func(uint64) bool) float64 {
	t.Helper()
	rng := rand.New(rand.NewSource(99))
	found, want := 0, 0
	for i := 0; i < queries; i++ {
		q := d.vecs[rng.Intn(len(d.vecs))]
		expected := d.truth(q, k, keep)
		got, err := ix.SearchFiltered(q, k, ef, keep)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		set := make(map[uint64]bool, len(got))
		for _, r := range got {
			set[r.ID] = true
		}
		for _, id := range expected {
			want++
			if set[id] {
				found++
			}
		}
	}
	return float64(found) / float64(want)
}

// Recall must rise with ef and must reach near-exactness at a large one. Both
// halves matter: an index that always returns the same wrong answers would
// still show high recall at one ef, and an index that ignores ef entirely
// would show a flat curve.
func TestRecallImprovesWithEf(t *testing.T) {
	d := clusteredDataset(5000, 32, 12, 42, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 200})

	var last float64
	for _, ef := range []int{10, 50, 200} {
		got := recall(t, ix, d, 100, 10, ef, nil)
		t.Logf("ef=%-3d recall@10 = %.4f", ef, got)
		if got < last-0.01 {
			t.Errorf("recall fell from %.4f to %.4f when ef rose to %d", last, got, ef)
		}
		last = got
	}
	if last < 0.98 {
		t.Errorf("recall@10 at ef=200 is %.4f, expected at least 0.98", last)
	}
	if first := recall(t, ix, d, 100, 10, 10, nil); first > last {
		t.Errorf("a small ef (%.4f) beat a large one (%.4f)", first, last)
	}
}

// A vector that is in the index must be its own nearest neighbour. This is the
// cheapest possible check on graph connectivity: a node that no search can
// reach fails it, however good the average recall looks.
func TestEveryVectorFindsItself(t *testing.T) {
	d := randomDataset(2000, 24, 7, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 100})

	misses := 0
	for i, v := range d.vecs {
		res, err := ix.Search(v, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != uint64(i) {
			misses++
		}
	}
	// Ties are possible in principle, so a handful of misses would be
	// tolerable; in practice on distinct random vectors there should be none.
	if misses > 0 {
		t.Errorf("%d of %d vectors did not retrieve themselves", misses, len(d.vecs))
	}
}

func TestDistancesAreCorrectAndOrdered(t *testing.T) {
	for _, metric := range []lodestar.Metric{lodestar.L2, lodestar.Cosine, lodestar.InnerProduct} {
		t.Run(metric.String(), func(t *testing.T) {
			d := clusteredDataset(1500, 16, 6, 11, metric)
			ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 200})

			q := d.vecs[3]
			res, err := ix.Search(q, 10, 200)
			if err != nil {
				t.Fatal(err)
			}
			if len(res) != 10 {
				t.Fatalf("got %d results, want 10", len(res))
			}
			for i := 1; i < len(res); i++ {
				if res[i].Distance < res[i-1].Distance {
					t.Fatalf("results are not sorted: %v then %v", res[i-1], res[i])
				}
			}
			// The reported distance must be the real distance under the
			// metric, not an internal proxy such as a squared value.
			for _, r := range res {
				want := d.distance(d.vecs[r.ID], q)
				if math.Abs(want-float64(r.Distance)) > 1e-3*math.Max(1, math.Abs(want)) {
					t.Errorf("id %d: reported %g, true %g", r.ID, r.Distance, want)
				}
			}
		})
	}
}

func TestDeleteRemovesFromResults(t *testing.T) {
	d := clusteredDataset(2000, 16, 5, 3, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 100})

	q := d.vecs[0]
	before, err := ix.Search(q, 10, 100)
	if err != nil {
		t.Fatal(err)
	}
	deleted := map[uint64]bool{}
	for _, r := range before[:5] {
		if !ix.Delete(r.ID) {
			t.Fatalf("delete of %d reported absent", r.ID)
		}
		deleted[r.ID] = true
	}
	if ix.Len() != len(d.vecs)-5 {
		t.Errorf("Len is %d after 5 deletes of %d", ix.Len(), len(d.vecs))
	}
	if ix.Delete(before[0].ID) {
		t.Error("deleting twice reported success")
	}

	after, err := ix.Search(q, 10, 100)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range after {
		if deleted[r.ID] {
			t.Errorf("deleted id %d came back in results", r.ID)
		}
	}
	// The graph must still work after deletion, not merely omit the rows.
	got := recall(t, ix, d, 50, 10, 100, func(id uint64) bool { return !deleted[id] })
	if got < 0.95 {
		t.Errorf("recall fell to %.4f after deletions", got)
	}
}

// Deleting the entry point is the one case tombstoning cannot ignore, because
// every search starts there.
func TestDeleteEntryPoint(t *testing.T) {
	d := randomDataset(500, 12, 5, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 8, EfConstruction: 64})

	// Delete a good fraction of the index, which is very likely to include
	// whichever node sits at the top.
	for i := 0; i < 100; i++ {
		ix.Delete(uint64(i))
	}
	res, err := ix.Search(d.vecs[400], 5, 64)
	if err != nil {
		t.Fatalf("search after deleting the top of the graph: %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("got %d results, want 5", len(res))
	}
	for _, r := range res {
		if r.ID < 100 {
			t.Errorf("deleted id %d returned", r.ID)
		}
	}
}

// A filter applied during the walk must return k matching results, not
// whatever survives filtering a fixed-size result set afterwards.
func TestFilteredSearchIsAppliedDuringTheWalk(t *testing.T) {
	d := clusteredDataset(4000, 24, 8, 17, lodestar.L2)
	ix := d.build(t, lodestar.Config{M: 16, EfConstruction: 200})

	// One id in twenty passes, so a post-filter over a k=10 result set would
	// usually return nothing at all.
	keep := func(id uint64) bool { return id%20 == 0 }
	res, err := ix.SearchFiltered(d.vecs[0], 10, 200, keep)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 10 {
		t.Fatalf("got %d results under a 5%% filter, want 10", len(res))
	}
	for _, r := range res {
		if !keep(r.ID) {
			t.Errorf("id %d does not match the filter", r.ID)
		}
	}
	if got := recall(t, ix, d, 50, 10, 400, keep); got < 0.90 {
		t.Errorf("filtered recall %.4f, expected at least 0.90", got)
	}
}

func TestErrors(t *testing.T) {
	if _, err := lodestar.New(lodestar.Config{Dim: 0}); err == nil {
		t.Error("a zero dimension was accepted")
	}
	ix, err := lodestar.New(lodestar.Config{Dim: 4})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ix.Search([]float32{1, 2, 3, 4}, 5, 10); err != lodestar.ErrEmpty {
		t.Errorf("searching an empty index returned %v, want ErrEmpty", err)
	}
	if err := ix.Add(1, []float32{1, 2, 3}); err == nil {
		t.Error("a short vector was accepted")
	}
	if err := ix.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add(1, []float32{5, 6, 7, 8}); err == nil {
		t.Error("a duplicate id was accepted")
	}
	if _, err := ix.Search([]float32{1, 2}, 5, 10); err == nil {
		t.Error("a short query was accepted")
	}
	if !ix.Contains(1) {
		t.Error("Contains says the only vector is absent")
	}
	if v, ok := ix.Vector(1); !ok || v[0] != 1 {
		t.Errorf("Vector returned %v, %v", v, ok)
	}
}

// A fixed seed must produce a fixed graph. Without this, no recall number in
// this file would be reproducible, and a regression would look like noise.
func TestBuildIsDeterministic(t *testing.T) {
	d := randomDataset(1000, 16, 21, lodestar.L2)
	cfg := lodestar.Config{M: 8, EfConstruction: 64, Seed: 12345}

	first := d.build(t, cfg)
	second := d.build(t, cfg)
	if first.Layers() != second.Layers() {
		t.Fatalf("layer counts differ: %d and %d", first.Layers(), second.Layers())
	}
	for i := 0; i < 100; i++ {
		q := d.vecs[i*7%len(d.vecs)]
		a, _ := first.Search(q, 10, 50)
		b, _ := second.Search(q, 10, 50)
		if len(a) != len(b) {
			t.Fatalf("query %d returned %d and %d results", i, len(a), len(b))
		}
		for j := range a {
			if a[j].ID != b[j].ID || a[j].Distance != b[j].Distance {
				t.Fatalf("query %d position %d: %v vs %v", i, j, a[j], b[j])
			}
		}
	}
}

// Concurrent inserts and searches must be race-free and must not corrupt the
// graph. The assertion at the end is the important half: -race proves there is
// no data race, but only recall proves the copy-on-write link updates did not
// lose edges.
func TestConcurrentAddAndSearch(t *testing.T) {
	d := randomDataset(3000, 16, 88, lodestar.L2)
	ix, err := lodestar.New(lodestar.Config{Dim: d.dim, M: 16, EfConstruction: 100})
	if err != nil {
		t.Fatal(err)
	}
	// Seed a few vectors so searches have something to walk.
	for i := 0; i < 50; i++ {
		if err := ix.Add(uint64(i), d.vecs[i]); err != nil {
			t.Fatal(err)
		}
	}

	const writers, readers = 4, 4
	var buildWG, readWG sync.WaitGroup
	errs := make(chan error, writers+readers)
	stop := make(chan struct{})

	remaining := d.vecs[50:]
	chunk := len(remaining) / writers
	for w := 0; w < writers; w++ {
		buildWG.Add(1)
		go func(w int) {
			defer buildWG.Done()
			for i := w * chunk; i < (w+1)*chunk; i++ {
				if err := ix.Add(uint64(50+i), remaining[i]); err != nil {
					errs <- fmt.Errorf("add: %w", err)
					return
				}
			}
		}(w)
	}
	for r := 0; r < readers; r++ {
		readWG.Add(1)
		go func(r int) {
			defer readWG.Done()
			rng := rand.New(rand.NewSource(int64(r)))
			for {
				select {
				case <-stop:
					return
				default:
				}
				q := d.vecs[rng.Intn(len(d.vecs))]
				if _, err := ix.Search(q, 10, 64); err != nil && err != lodestar.ErrEmpty {
					errs <- fmt.Errorf("search: %w", err)
					return
				}
			}
		}(r)
	}

	buildWG.Wait()
	close(stop)
	readWG.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}

	if got := ix.Len(); got != 50+writers*chunk {
		t.Fatalf("index holds %d vectors, expected %d", got, 50+writers*chunk)
	}
	// A graph damaged by a lost concurrent link update shows up here.
	present := &dataset{vecs: d.vecs[:50+writers*chunk], dim: d.dim, metric: d.metric}
	if got := recall(t, ix, present, 100, 10, 200, nil); got < 0.95 {
		t.Errorf("recall after concurrent build is %.4f, expected at least 0.95", got)
	}
}
