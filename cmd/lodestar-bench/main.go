// Command lodestar-bench measures the only thing that matters for an
// approximate index: what recall costs in throughput.
//
// A single number is meaningless here. An ANN index can be made arbitrarily
// fast by returning worse answers, so quoting queries per second without
// recall — or recall without the ef it was measured at — says nothing. What
// this prints is the curve, alongside the brute-force scan the whole exercise
// exists to avoid, so the speedup is stated against something real.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/dhananjaypesu/lodestar"
)

func main() {
	n := flag.Int("n", 100000, "vectors to index")
	dim := flag.Int("dim", 128, "dimension")
	clusters := flag.Int("clusters", 200, "gaussian clusters the data is drawn from")
	queries := flag.Int("queries", 1000, "queries per measurement")
	k := flag.Int("k", 10, "neighbours per query")
	m := flag.Int("m", 16, "graph degree")
	efc := flag.Int("ef-construction", 200, "candidate list size during build")
	subspaces := flag.Int("pq", 0, "if non-zero, also measure a product-quantized index with this many subspaces")
	seed := flag.Int64("seed", 1, "random seed")
	flag.Parse()

	fmt.Printf("%d vectors, %d dimensions, %d clusters, M=%d efConstruction=%d, %s\n\n",
		*n, *dim, *clusters, *m, *efc, runtime.GOARCH)

	data := generate(*n, *dim, *clusters, *seed)

	ix, err := lodestar.New(lodestar.Config{Dim: *dim, M: *m, EfConstruction: *efc, Seed: *seed})
	if err != nil {
		fatal(err)
	}
	start := time.Now()
	for i := 0; i < *n; i++ {
		if err := ix.Add(uint64(i), vec(data, *dim, i)); err != nil {
			fatal(err)
		}
	}
	build := time.Since(start)
	fmt.Printf("build          %8s   %9.0f vectors/s   %s/vector   %d layers\n",
		build.Round(time.Millisecond), float64(*n)/build.Seconds(),
		(build / time.Duration(*n)).Round(time.Nanosecond), ix.Layers())

	// Ground truth is computed once by brute force, which also gives the
	// baseline the index is compared against.
	rng := rand.New(rand.NewSource(*seed + 1))
	qs := make([][]float32, *queries)
	for i := range qs {
		qs[i] = vec(data, *dim, rng.Intn(*n))
	}
	start = time.Now()
	truth := make([][]uint64, *queries)
	for i, q := range qs {
		truth[i] = bruteForce(data, *dim, *n, q, *k)
	}
	scan := time.Since(start)
	fmt.Printf("brute force    %8s   %9.0f queries/s  recall 1.0000  (the baseline)\n\n",
		scan.Round(time.Millisecond), float64(*queries)/scan.Seconds())

	fmt.Printf("%-6s %10s %10s %12s\n", "ef", "recall@"+fmt.Sprint(*k), "queries/s", "vs brute force")
	for _, ef := range []int{10, 20, 40, 80, 160, 320} {
		r, qps := measure(ix, qs, truth, *k, ef)
		fmt.Printf("%-6d %10.4f %10.0f %11.0fx\n", ef, r, qps, qps/(float64(*queries)/scan.Seconds()))
	}

	if *subspaces > 0 {
		fmt.Printf("\nproduct quantization, %d subspaces (%d bytes/vector, %.0fx smaller than %d)\n",
			*subspaces, *subspaces, float64(*dim*4)/float64(*subspaces), *dim*4)
		for _, rerank := range []bool{false, true} {
			qix := rebuild(data, *n, *dim, *m, *efc, *seed)
			if err := qix.Quantize(lodestar.QuantizeOptions{Subspaces: *subspaces, KeepVectors: rerank}); err != nil {
				fatal(err)
			}
			label := "codes only    "
			if rerank {
				label = "codes + rerank"
			}
			fmt.Printf("%-6s %10s %10s\n", "ef", "recall@"+fmt.Sprint(*k), "queries/s")
			for _, ef := range []int{40, 160, 640} {
				r, qps := measure(qix, qs, truth, *k, ef)
				fmt.Printf("%-6d %10.4f %10.0f   %s\n", ef, r, qps, label)
			}
		}
	}
}

func rebuild(data []float32, n, dim, m, efc int, seed int64) *lodestar.Index {
	ix, err := lodestar.New(lodestar.Config{Dim: dim, M: m, EfConstruction: efc, Seed: seed})
	if err != nil {
		fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := ix.Add(uint64(i), vec(data, dim, i)); err != nil {
			fatal(err)
		}
	}
	return ix
}

func measure(ix *lodestar.Index, qs [][]float32, truth [][]uint64, k, ef int) (recall, qps float64) {
	found, want := 0, 0
	start := time.Now()
	results := make([][]lodestar.Result, len(qs))
	for i, q := range qs {
		res, err := ix.Search(q, k, ef)
		if err != nil {
			fatal(err)
		}
		results[i] = res
	}
	elapsed := time.Since(start)

	// Recall is scored after the timed loop so that map construction does not
	// count as query time.
	for i := range qs {
		set := make(map[uint64]bool, len(results[i]))
		for _, r := range results[i] {
			set[r.ID] = true
		}
		for _, id := range truth[i] {
			want++
			if set[id] {
				found++
			}
		}
	}
	return float64(found) / float64(want), float64(len(qs)) / elapsed.Seconds()
}

func generate(n, dim, clusters int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([]float32, clusters*dim)
	for i := range centers {
		centers[i] = float32(rng.NormFloat64()) * 5
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

func vec(data []float32, dim, i int) []float32 { return data[i*dim : (i+1)*dim] }

func bruteForce(data []float32, dim, n int, q []float32, k int) []uint64 {
	type hit struct {
		d  float32
		id uint64
	}
	hits := make([]hit, n)
	for i := 0; i < n; i++ {
		v := data[i*dim : (i+1)*dim]
		var s float32
		for j := range v {
			d := v[j] - q[j]
			s += d * d
		}
		hits[i] = hit{s, uint64(i)}
	}
	sort.Slice(hits, func(a, b int) bool { return hits[a].d < hits[b].d })
	out := make([]uint64, k)
	for i := 0; i < k; i++ {
		out[i] = hits[i].id
	}
	return out
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "lodestar-bench: %v\n", err)
	os.Exit(1)
}
