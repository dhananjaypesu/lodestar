package lodestar_test

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/dhananjaypesu/lodestar"
)

// These benchmarks exist to check one claim: that searches take no locks and
// therefore scale across cores, while inserts do not, because they serialize
// on the structural lock that guards the shared arrays. Both halves are worth
// measuring, and the second is the honest limitation of the design.
//
//	go test -run xxx -bench 'Search|Add' -benchtime 2s .

func benchIndex(b *testing.B, n, dim int) (*lodestar.Index, [][]float32) {
	b.Helper()
	rng := rand.New(rand.NewSource(1))
	data := make([][]float32, n)
	for i := range data {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		data[i] = v
	}
	ix, err := lodestar.New(lodestar.Config{Dim: dim, M: 16, EfConstruction: 200})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range data {
		if err := ix.Add(uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}
	return ix, data
}

func BenchmarkSearch(b *testing.B) {
	ix, data := benchIndex(b, 30000, 64)
	rng := rand.New(rand.NewSource(2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ix.Search(data[rng.Intn(len(data))], 10, 64); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSearchParallel should scale with GOMAXPROCS. If it stops scaling,
// something has started taking a lock on the read path.
func BenchmarkSearchParallel(b *testing.B) {
	ix, data := benchIndex(b, 30000, 64)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		rng := rand.New(rand.NewSource(3))
		for pb.Next() {
			if _, err := ix.Search(data[rng.Intn(len(data))], 10, 64); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkAddParallel is the measurement behind the limitation in the README:
// insert throughput is flat in the number of goroutines, because every insert
// needs the write lock briefly and a waiting writer blocks the readers behind
// it.
func BenchmarkAddParallel(b *testing.B) {
	const dim = 64
	rng := rand.New(rand.NewSource(4))
	data := make([][]float32, b.N)
	for i := range data {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		data[i] = v
	}
	ix, err := lodestar.New(lodestar.Config{Dim: dim, M: 16, EfConstruction: 200})
	if err != nil {
		b.Fatal(err)
	}

	workers := 8
	var wg sync.WaitGroup
	chunk := b.N / workers
	if chunk == 0 {
		chunk = 1
	}
	b.ResetTimer()
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if w == workers-1 {
			end = b.N
		}
		if start >= b.N {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				ix.Add(uint64(i), data[i])
			}
		}(start, end)
	}
	wg.Wait()
}
