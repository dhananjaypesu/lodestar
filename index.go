// Package lodestar is an approximate nearest-neighbour index built on
// Hierarchical Navigable Small World graphs.
//
// The problem it solves is that exact nearest-neighbour search over a large
// set of high-dimensional vectors has no good answer: every space-partitioning
// structure that works in two or three dimensions — kd-trees, ball trees, R-
// trees — degenerates to a full scan as the dimension climbs, because in high
// dimensions almost every point is roughly equidistant from every other and
// there is nothing left to prune. That is the curse of dimensionality, and it
// is not an implementation problem.
//
// So the goal changes. Instead of the exact answer, find something that is
// almost always the exact answer, in logarithmic rather than linear time, and
// make the trade explicit: recall against speed, tunable at query time by a
// single parameter. HNSW achieves it by giving up on partitioning space and
// navigating a graph instead. Each vector is a node linked to its close
// neighbours; a search walks greedily downhill toward the query. A single flat
// graph would get stuck in local minima, so the graph is layered — sparse long
// links at the top for covering ground quickly, dense short links at the
// bottom for precision — and a search descends through them like a skip list
// over space rather than over a sorted key.
package lodestar

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/dhananjaypesu/lodestar/internal/vecmath"
)

// Metric selects how distance between two vectors is measured.
type Metric uint8

const (
	// L2 is Euclidean distance, the default.
	L2 Metric = iota
	// Cosine is angular distance. Vectors are normalized on insert, so a
	// cosine index costs exactly one inner product per comparison at query
	// time — the same as InnerProduct, with the normalization paid once.
	Cosine
	// InnerProduct ranks by negated dot product. Note that it is not a metric:
	// it violates the triangle inequality, and a point is not its own nearest
	// neighbour when its norm is small. HNSW still works well in practice,
	// which is why every production index offers it, but the graph has no
	// theoretical guarantee here that it has under L2.
	InnerProduct
)

func (m Metric) String() string {
	switch m {
	case L2:
		return "l2"
	case Cosine:
		return "cosine"
	case InnerProduct:
		return "inner_product"
	}
	return "unknown"
}

// Errors returned by the index.
var (
	ErrDimension   = errors.New("lodestar: vector has the wrong dimension")
	ErrDuplicateID = errors.New("lodestar: id is already in the index")
	ErrEmpty       = errors.New("lodestar: index is empty")
	ErrNotTrained  = errors.New("lodestar: quantizer has not been trained")
)

// Config describes an index. The zero value of every field except Dim is
// replaced by a default, so Config{Dim: 128} is a working configuration.
type Config struct {
	// Dim is the vector dimension. Required.
	Dim int
	// Metric selects the distance function. Defaults to L2.
	Metric Metric

	// M is the number of neighbours each node keeps per layer above the
	// bottom, and half the number it keeps at the bottom. It is the main
	// memory-versus-recall dial: 16 is a good default, 32-48 helps on hard,
	// high-dimensional data, and below 8 the graph starts to lose
	// connectivity. Defaults to 16.
	M int

	// EfConstruction is the size of the candidate list kept while inserting.
	// Larger values build a better graph and take longer to build, and unlike
	// EfSearch it cannot be changed after the fact. Defaults to 200.
	EfConstruction int

	// EfSearch is the default candidate-list size at query time, used when
	// Search is called with ef <= 0. This is the recall dial, and the one
	// worth tuning: it can be raised per query without rebuilding anything.
	// Defaults to 64.
	EfSearch int

	// Seed fixes the layer-assignment randomness so that a build is
	// reproducible. Defaults to a fixed value rather than to the clock,
	// because a search index that returns different results run to run is
	// nearly impossible to test.
	Seed int64
}

func (c *Config) applyDefaults() error {
	if c.Dim <= 0 {
		return fmt.Errorf("lodestar: Dim must be positive, got %d", c.Dim)
	}
	if c.M <= 0 {
		c.M = 16
	}
	if c.EfConstruction <= 0 {
		c.EfConstruction = 200
	}
	if c.EfSearch <= 0 {
		c.EfSearch = 64
	}
	if c.Seed == 0 {
		c.Seed = 0x5EED
	}
	if c.EfConstruction < c.M {
		// A candidate list smaller than the number of neighbours to choose
		// from cannot fill the node's links, which silently produces a sparse,
		// badly connected graph.
		c.EfConstruction = c.M
	}
	return nil
}

// Result is one hit, ordered nearest first.
type Result struct {
	ID uint64
	// Distance is in the units of the index's metric: Euclidean distance for
	// L2, one minus cosine similarity for Cosine, and the negated inner
	// product for InnerProduct. Smaller is always nearer.
	Distance float32
}

// linkList is a node's neighbours at one layer. It is immutable once
// published: a writer builds a new one and swaps the pointer, so readers never
// need a lock and never observe a half-updated list.
type linkList struct{ ids []uint32 }

// node is one vector's place in the graph. The vector itself lives in the
// index's flat array rather than here, so that walking neighbours touches
// contiguous memory instead of chasing per-node allocations.
type node struct {
	id      uint64
	deleted atomic.Bool
	// mu serializes writers to this node's links. Readers take no lock: they
	// load the pointer for the layer they are on and walk the slice it names.
	mu    sync.Mutex
	links []atomic.Pointer[linkList]
}

func (n *node) neighbours(layer int) []uint32 {
	if layer >= len(n.links) {
		return nil
	}
	l := n.links[layer].Load()
	if l == nil {
		return nil
	}
	return l.ids
}

// setNeighbours publishes a new neighbour list. The caller must hold n.mu.
func (n *node) setNeighbours(layer int, ids []uint32) {
	n.links[layer].Store(&linkList{ids: ids})
}

// Index is a searchable set of vectors. It is safe for concurrent use: many
// searches may run alongside many inserts.
//
// The locking has two levels, because the two things that change have very
// different shapes. Adding a vector appends to shared arrays and occasionally
// moves the entry point, which is rare and needs exclusion — that is what mu
// guards. Linking a node changes one neighbour list, which is common and
// needs to not block searches — that is what the per-node copy-on-write does.
// The result is that a search takes exactly one read lock and then runs
// lock-free over the graph.
type Index struct {
	cfg   Config
	mMax  int // neighbour cap above layer 0
	mMax0 int // neighbour cap at layer 0
	mL    float64

	mu       sync.RWMutex
	nodes    []*node
	vectors  []float32 // flat: node i occupies [i*dim, (i+1)*dim)
	byID     map[uint64]uint32
	entry    uint32
	maxLayer int
	live     int

	rngMu sync.Mutex
	rng   *rand.Rand

	scratch sync.Pool

	// quant is set when the index stores compressed codes instead of raw
	// vectors; see Quantize.
	quant atomic.Pointer[quantized]
}

// New creates an empty index.
func New(cfg Config) (*Index, error) {
	if err := cfg.applyDefaults(); err != nil {
		return nil, err
	}
	ix := &Index{
		cfg:   cfg,
		mMax:  cfg.M,
		mMax0: cfg.M * 2,
		// The paper's mL = 1/ln(M) makes the expected number of layers
		// logarithmic in the number of nodes, which is what gives the descent
		// its skip-list behaviour.
		mL:       1 / math.Log(float64(cfg.M)),
		byID:     make(map[uint64]uint32),
		entry:    invalidNode,
		maxLayer: -1,
		rng:      rand.New(rand.NewSource(cfg.Seed)),
	}
	ix.scratch.New = func() any { return newSearchScratch() }
	return ix, nil
}

const invalidNode = ^uint32(0)

// Config returns the configuration the index was built with.
func (ix *Index) Config() Config { return ix.cfg }

// Len returns the number of live vectors.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.live
}

// Layers returns the current height of the graph, which is a useful sanity
// check: it should grow like log_M of the number of vectors.
func (ix *Index) Layers() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.maxLayer + 1
}

// vector returns node i's stored vector. Callers must hold at least a read
// lock, and must not retain the slice past it.
func (ix *Index) vector(i uint32) []float32 {
	off := int(i) * ix.cfg.Dim
	return ix.vectors[off : off+ix.cfg.Dim : off+ix.cfg.Dim]
}

// distance is the metric's distance between two raw vectors.
func (ix *Index) distance(a, b []float32) float32 {
	switch ix.cfg.Metric {
	case Cosine:
		// Both operands are unit length, so the inner product is the cosine.
		return 1 - vecmath.Dot(a, b)
	case InnerProduct:
		return -vecmath.Dot(a, b)
	default:
		return vecmath.L2Squared(a, b)
	}
}

// prepareQuery normalizes the query when the metric requires it, without
// touching the caller's slice.
func (ix *Index) prepareQuery(q []float32) ([]float32, error) {
	if len(q) != ix.cfg.Dim {
		return nil, fmt.Errorf("%w: want %d, got %d", ErrDimension, ix.cfg.Dim, len(q))
	}
	if ix.cfg.Metric != Cosine {
		return q, nil
	}
	cp := make([]float32, len(q))
	copy(cp, q)
	vecmath.Normalize(cp)
	return cp, nil
}

// reportDistance converts an internal distance into the one documented on
// Result: L2 is stored squared to keep a square root out of the inner loop, so
// it is taken here, once per returned row.
func (ix *Index) reportDistance(d float32) float32 {
	if ix.cfg.Metric == L2 {
		return float32(math.Sqrt(float64(d)))
	}
	return d
}

// randomLayer draws a node's height.
//
// The distribution is geometric: a node reaches layer l with probability
// exp(-l/mL), so layer 0 holds everything, the layer above holds about 1/M of
// it, and so on. That is what makes the top layers sparse enough to cross the
// whole space in a few hops.
func (ix *Index) randomLayer() int {
	ix.rngMu.Lock()
	r := ix.rng.Float64()
	ix.rngMu.Unlock()
	if r <= 0 {
		r = math.SmallestNonzeroFloat64
	}
	return int(-math.Log(r) * ix.mL)
}

// searchScratch is the per-search working set, pooled so that a query in a
// steady-state server allocates nothing.
type searchScratch struct {
	visited *visitedSet
	cands   minHeap
	results maxHeap
	// picked backs neighbour selection during inserts, and prune backs the
	// re-selection that happens when a back-link overflows a node's budget.
	// Both are reused across the whole build rather than reallocated per link.
	picked []candidate
	prune  []candidate
	// query holds a normalized copy of the query for cosine indexes.
	query []float32
}

func newSearchScratch() *searchScratch {
	return &searchScratch{visited: newVisitedSet(0)}
}

func (ix *Index) getScratch(n int) *searchScratch {
	s := ix.scratch.Get().(*searchScratch)
	s.visited.grow(n)
	s.visited.reset()
	s.cands.Reset()
	s.results.Reset()
	s.picked = s.picked[:0]
	return s
}

func (ix *Index) putScratch(s *searchScratch) { ix.scratch.Put(s) }
