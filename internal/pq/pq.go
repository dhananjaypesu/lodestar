// Package pq implements product quantization: lossy compression of vectors
// that still supports distance computation without decompressing them.
//
// The idea is a good one. A 768-dimensional float32 vector costs 3 KB, so ten
// million of them cost 30 GB and no longer fit in memory — and an index that
// does not fit in memory is an index that reads from disk on every hop of
// every graph traversal. Product quantization splits each vector into m
// contiguous subvectors, runs k-means over each subspace independently, and
// stores only which centroid each subvector landed on. With 256 centroids per
// subspace a code is one byte, so m=96 turns that 3 KB vector into 96 bytes:
// 32x smaller, and the compression is learned from the data rather than
// applied uniformly.
//
// The part that makes it useful rather than merely small is asymmetric
// distance computation. The query stays uncompressed. Before the search, its
// distance to all 256 centroids of every subspace is computed once into a
// table of m*256 floats; after that, the approximate distance to any stored
// vector is m table lookups and m additions — no decompression, no
// multiplication, and the query's own precision is never thrown away.
package pq

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/dhananjaypesu/lodestar/internal/vecmath"
)

// Mode selects what the distance table holds.
type Mode uint8

const (
	// ModeL2 builds a table of squared Euclidean distances, which sum across
	// subspaces to the squared distance of the whole vector.
	ModeL2 Mode = iota
	// ModeDot builds a table of inner products, which sum to the inner product
	// of the whole vector. Cosine and inner-product indexes both use it.
	ModeDot
)

// Codebook count. 256 is not arbitrary: it is the largest k whose codes fit in
// a byte, and byte codes are what make the lookup table a cache-resident array
// rather than a pointer chase.
const K = 256

// Quantizer holds the learned codebooks.
type Quantizer struct {
	Dim int
	M   int // number of subspaces
	Sub int // dimensions per subspace

	// Centroids is laid out [subspace][centroid][dimension], flattened. The
	// order matters: the distance table is built subspace by subspace, so
	// keeping a subspace's 256 centroids contiguous means the training data
	// and the query both stream through cache linearly.
	Centroids []float32
}

var (
	// ErrDivisible is returned when the dimension does not split evenly.
	ErrDivisible = errors.New("pq: dimension must be divisible by the number of subspaces")
	// ErrTooFew is returned when there is not enough data to train.
	ErrTooFew = errors.New("pq: need at least as many training vectors as centroids")
)

// Train learns codebooks from a flat array of n vectors of the given
// dimension.
func Train(vectors []float32, dim, m, iters int, seed int64) (*Quantizer, error) {
	if dim <= 0 || m <= 0 || dim%m != 0 {
		return nil, fmt.Errorf("%w: dim %d, subspaces %d", ErrDivisible, dim, m)
	}
	n := len(vectors) / dim
	if n < K {
		return nil, fmt.Errorf("%w: have %d, need %d", ErrTooFew, n, K)
	}
	if iters <= 0 {
		iters = 25
	}
	sub := dim / m
	q := &Quantizer{Dim: dim, M: m, Sub: sub, Centroids: make([]float32, m*K*sub)}

	rng := rand.New(rand.NewSource(seed))
	// Each subspace is trained independently — that independence is the whole
	// trick, since it turns one intractable k-means over 256^m effective
	// centroids into m tractable ones over 256.
	points := make([]float32, n*sub)
	for s := 0; s < m; s++ {
		off := s * sub
		for i := 0; i < n; i++ {
			copy(points[i*sub:(i+1)*sub], vectors[i*dim+off:i*dim+off+sub])
		}
		centroids := q.Centroids[s*K*sub : (s+1)*K*sub]
		kmeans(points, centroids, sub, iters, rng)
	}
	return q, nil
}

// CodeLen is the number of bytes one encoded vector occupies.
func (q *Quantizer) CodeLen() int { return q.M }

// Encode writes the code for vec into out, which must be CodeLen bytes.
func (q *Quantizer) Encode(vec []float32, out []byte) {
	for s := 0; s < q.M; s++ {
		part := vec[s*q.Sub : (s+1)*q.Sub]
		book := q.Centroids[s*K*q.Sub : (s+1)*K*q.Sub]
		best, bestDist := 0, float32(math.MaxFloat32)
		for c := 0; c < K; c++ {
			d := vecmath.L2Squared(part, book[c*q.Sub:(c+1)*q.Sub])
			if d < bestDist {
				best, bestDist = c, d
			}
		}
		out[s] = byte(best)
	}
}

// Decode reconstructs an approximation of the original vector. It is used for
// measuring reconstruction error, not on any search path — the point of the
// scheme is that search never decodes.
func (q *Quantizer) Decode(code []byte, out []float32) {
	for s := 0; s < q.M; s++ {
		book := q.Centroids[s*K*q.Sub : (s+1)*K*q.Sub]
		c := int(code[s])
		copy(out[s*q.Sub:(s+1)*q.Sub], book[c*q.Sub:(c+1)*q.Sub])
	}
}

// TableLen is the size of the distance table for one query.
func (q *Quantizer) TableLen() int { return q.M * K }

// BuildTable fills table with the query's distance to every centroid. This is
// the once-per-query cost that makes every subsequent comparison m lookups.
func (q *Quantizer) BuildTable(query []float32, mode Mode, table []float32) {
	for s := 0; s < q.M; s++ {
		part := query[s*q.Sub : (s+1)*q.Sub]
		book := q.Centroids[s*K*q.Sub : (s+1)*K*q.Sub]
		row := table[s*K : (s+1)*K]
		for c := 0; c < K; c++ {
			centroid := book[c*q.Sub : (c+1)*q.Sub]
			if mode == ModeDot {
				row[c] = vecmath.Dot(part, centroid)
			} else {
				row[c] = vecmath.L2Squared(part, centroid)
			}
		}
	}
}

// Lookup sums the table entries a code selects, giving the approximate
// squared distance or inner product depending on how the table was built.
func (q *Quantizer) Lookup(table []float32, code []byte) float32 {
	var sum float32
	for s := 0; s < q.M; s++ {
		sum += table[s*K+int(code[s])]
	}
	return sum
}

// kmeans clusters points into len(centroids)/dim centroids, in place.
func kmeans(points, centroids []float32, dim, iters int, rng *rand.Rand) {
	n := len(points) / dim
	kmeansPlusPlusInit(points, centroids, dim, rng)

	assign := make([]int, n)
	counts := make([]int, K)
	sums := make([]float32, K*dim)

	for it := 0; it < iters; it++ {
		moved := 0
		for i := 0; i < n; i++ {
			p := points[i*dim : (i+1)*dim]
			best, bestDist := 0, float32(math.MaxFloat32)
			for c := 0; c < K; c++ {
				if d := vecmath.L2Squared(p, centroids[c*dim:(c+1)*dim]); d < bestDist {
					best, bestDist = c, d
				}
			}
			if assign[i] != best {
				moved++
			}
			assign[i] = best
		}
		// Convergence, not a fixed iteration count, is what stops the loop:
		// subspaces of low-entropy data settle in a handful of rounds and
		// there is no reason to keep paying for the rest.
		if it > 0 && moved == 0 {
			return
		}

		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := 0; i < n; i++ {
			c := assign[i]
			counts[c]++
			vecmath.Add(sums[c*dim:(c+1)*dim], points[i*dim:(i+1)*dim])
		}
		for c := 0; c < K; c++ {
			if counts[c] == 0 {
				// An empty cluster wastes a codebook entry, so it is moved to
				// the point currently worst served by its own centroid. That
				// is where the quantization error is largest and where the
				// spare centroid does the most good.
				worst := farthestPoint(points, centroids, assign, dim)
				copy(centroids[c*dim:(c+1)*dim], points[worst*dim:(worst+1)*dim])
				continue
			}
			copy(centroids[c*dim:(c+1)*dim], sums[c*dim:(c+1)*dim])
			vecmath.Scale(centroids[c*dim:(c+1)*dim], 1/float32(counts[c]))
		}
	}
}

// kmeansPlusPlusInit seeds the centroids by choosing each new one with
// probability proportional to its squared distance from the nearest existing
// centroid. Uniform seeding routinely puts several centroids inside one dense
// cluster and none in a sparse region, and Lloyd iterations rarely recover
// from it; this costs one extra pass and makes the result far less
// seed-dependent.
func kmeansPlusPlusInit(points, centroids []float32, dim int, rng *rand.Rand) {
	n := len(points) / dim
	first := rng.Intn(n)
	copy(centroids[:dim], points[first*dim:(first+1)*dim])

	closest := make([]float32, n)
	for i := range closest {
		closest[i] = vecmath.L2Squared(points[i*dim:(i+1)*dim], centroids[:dim])
	}
	for c := 1; c < K; c++ {
		var total float64
		for _, d := range closest {
			total += float64(d)
		}
		pick := n - 1
		if total > 0 {
			target := rng.Float64() * total
			var acc float64
			for i, d := range closest {
				acc += float64(d)
				if acc >= target {
					pick = i
					break
				}
			}
		} else {
			pick = rng.Intn(n)
		}
		copy(centroids[c*dim:(c+1)*dim], points[pick*dim:(pick+1)*dim])
		for i := 0; i < n; i++ {
			if d := vecmath.L2Squared(points[i*dim:(i+1)*dim], centroids[c*dim:(c+1)*dim]); d < closest[i] {
				closest[i] = d
			}
		}
	}
}

func farthestPoint(points, centroids []float32, assign []int, dim int) int {
	worst, worstDist := 0, float32(-1)
	for i := range assign {
		c := assign[i]
		d := vecmath.L2Squared(points[i*dim:(i+1)*dim], centroids[c*dim:(c+1)*dim])
		if d > worstDist {
			worst, worstDist = i, d
		}
	}
	return worst
}
