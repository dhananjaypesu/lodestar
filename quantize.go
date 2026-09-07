package lodestar

import (
	"fmt"
	"sync/atomic"

	"github.com/dhananjaypesu/lodestar/internal/pq"
)

// atomicLinks is the published neighbour list for one node at one layer.
type atomicLinks = atomic.Pointer[linkList]

// quantized is the compressed representation of every vector in the index.
type quantized struct {
	q     *pq.Quantizer
	codes []byte // node i occupies [i*codeLen, (i+1)*codeLen)
	mode  pq.Mode
	// metric is kept so the sum a lookup produces can be converted into the
	// index's distance convention, which is the same conversion the exact path
	// applies to a dot product.
	metric      Metric
	keepVectors bool
}

func (z *quantized) code(i uint32) []byte {
	l := z.q.CodeLen()
	off := int(i) * l
	return z.codes[off : off+l : off+l]
}

// distanceTable precomputes the query's distance to every centroid, reusing
// the scratch buffer so a steady-state query allocates nothing.
func (z *quantized) distanceTable(query []float32, s *searchScratch) []float32 {
	if cap(s.query) < z.q.TableLen() {
		s.query = make([]float32, z.q.TableLen())
	}
	table := s.query[:z.q.TableLen()]
	z.q.BuildTable(query, z.mode, table)
	return table
}

// asymmetric returns the approximate distance from the query behind table to
// node i.
func (z *quantized) asymmetric(table []float32, i uint32) float32 {
	sum := z.q.Lookup(table, z.code(i))
	switch z.metric {
	case Cosine:
		return 1 - sum
	case InnerProduct:
		return -sum
	default:
		return sum
	}
}

// QuantizeOptions configures compression.
type QuantizeOptions struct {
	// Subspaces is how many pieces each vector is split into, and therefore
	// how many bytes a code occupies. Dim must be divisible by it. Compression
	// is 4*Dim/Subspaces to one, so at Dim=128, Subspaces=16 gives 32x.
	Subspaces int
	// Iterations bounds the k-means refinement per subspace. Defaults to 25,
	// and training stops early once assignments stop moving.
	Iterations int
	// TrainSize caps how many vectors the codebooks are learned from. The
	// codebooks describe the shape of the data, not its contents, so a sample
	// of a hundred thousand vectors trains as well as ten million and takes a
	// hundredth of the time. Defaults to 100,000.
	TrainSize int
	// KeepVectors retains the raw vectors so results can be reranked exactly.
	// This gives up the memory saving on the vectors themselves while still
	// getting the cache benefit during traversal, and recovers nearly all the
	// recall that quantization costs. Set it false to actually shrink the
	// index.
	KeepVectors bool
	// Seed makes training reproducible.
	Seed int64
}

// Quantize compresses the index in place.
//
// It runs after the graph is built, not during: construction compares stored
// vectors against each other many times, and doing that through approximate
// distances would bake the quantization error into the graph's structure,
// where no amount of reranking could undo it. Building at full precision and
// compressing afterwards keeps the error confined to the ranking, which
// reranking can then largely remove.
func (ix *Index) Quantize(opts QuantizeOptions) error {
	if opts.Subspaces <= 0 {
		return fmt.Errorf("lodestar: Subspaces must be positive")
	}
	if opts.TrainSize <= 0 {
		opts.TrainSize = 100000
	}
	if opts.Seed == 0 {
		opts.Seed = int64(ix.cfg.Seed)
	}

	ix.mu.Lock()
	defer ix.mu.Unlock()
	if len(ix.vectors) == 0 {
		return ErrEmpty
	}
	n := len(ix.nodes)

	train := ix.vectors
	if n > opts.TrainSize {
		train = ix.vectors[:opts.TrainSize*ix.cfg.Dim]
	}
	q, err := pq.Train(train, ix.cfg.Dim, opts.Subspaces, opts.Iterations, opts.Seed)
	if err != nil {
		return err
	}

	mode := pq.ModeL2
	if ix.cfg.Metric != L2 {
		mode = pq.ModeDot
	}
	z := &quantized{q: q, mode: mode, metric: ix.cfg.Metric, keepVectors: opts.KeepVectors,
		codes: make([]byte, n*q.CodeLen())}
	for i := 0; i < n; i++ {
		q.Encode(ix.vector(uint32(i)), z.code(uint32(i)))
	}
	ix.quant.Store(z)
	if !opts.KeepVectors {
		ix.vectors = nil
	}
	return nil
}

// Quantized reports whether the index is compressed, and the bytes per vector
// it now stores.
func (ix *Index) Quantized() (bool, int) {
	z := ix.quant.Load()
	if z == nil {
		return false, ix.cfg.Dim * 4
	}
	bytes := z.q.CodeLen()
	if z.keepVectors {
		bytes += ix.cfg.Dim * 4
	}
	return true, bytes
}

// ReconstructionError returns the mean squared error between the stored
// vectors and their quantized approximations, which is the direct measure of
// how much information the compression threw away. It requires KeepVectors.
func (ix *Index) ReconstructionError() (float64, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	z := ix.quant.Load()
	if z == nil {
		return 0, ErrNotTrained
	}
	if !z.keepVectors {
		return 0, fmt.Errorf("lodestar: reconstruction error needs the raw vectors")
	}
	buf := make([]float32, ix.cfg.Dim)
	var total float64
	for i := range ix.nodes {
		z.q.Decode(z.code(uint32(i)), buf)
		orig := ix.vector(uint32(i))
		var sum float32
		for j := range buf {
			d := buf[j] - orig[j]
			sum += d * d
		}
		total += float64(sum)
	}
	return total / float64(len(ix.nodes)), nil
}
