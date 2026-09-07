package lodestar

import (
	"fmt"
	"slices"

	"github.com/dhananjaypesu/lodestar/internal/vecmath"
)

// distanceFn measures a query against stored vectors.
//
// It exists as a struct rather than a closure because a search calls it a few
// thousand times, and because it is the seam where quantization plugs in: with
// a quantizer installed the distance comes from a lookup table over compressed
// codes instead of from the vectors themselves, and nothing else in the search
// has to know.
type distanceFn struct {
	ix    *Index
	query []float32
	q     *quantized
	table []float32 // asymmetric distance table, one row per subspace
}

func (ix *Index) newDistanceFn(query []float32, s *searchScratch) *distanceFn {
	d := &distanceFn{ix: ix, query: query}
	if q := ix.quant.Load(); q != nil {
		d.q = q
		d.table = q.distanceTable(query, s)
	}
	return d
}

// to returns the distance from the query to node i.
func (d *distanceFn) to(i uint32) float32 {
	if d.q != nil {
		return d.q.asymmetric(d.table, i)
	}
	return d.ix.distance(d.query, d.ix.vector(i))
}

// exact returns the true distance, ignoring any quantization. It is what
// reranking uses, and is only valid when raw vectors were kept.
func (d *distanceFn) exact(i uint32) float32 {
	return d.ix.distance(d.query, d.ix.vector(i))
}

// Add inserts a vector under the given id.
//
// The insert is the search algorithm run in reverse: descend the layers
// greedily to find where this vector belongs, then at each layer from its own
// height down to the bottom, find its nearest neighbours and link to them. The
// links are bidirectional, so every insert also edits the nodes it points at,
// which is what keeps the graph navigable as it grows.
func (ix *Index) Add(id uint64, vec []float32) error {
	if len(vec) != ix.cfg.Dim {
		return fmt.Errorf("%w: want %d, got %d", ErrDimension, ix.cfg.Dim, len(vec))
	}
	if q := ix.quant.Load(); q != nil && !q.keepVectors {
		return fmt.Errorf("lodestar: cannot add to an index whose raw vectors were dropped by Quantize")
	}

	stored := make([]float32, len(vec))
	copy(stored, vec)
	if ix.cfg.Metric == Cosine && !vecmath.Normalize(stored) {
		return fmt.Errorf("lodestar: cannot normalize a zero vector for a cosine index")
	}

	layer := ix.randomLayer()

	// Phase one, under the write lock: give the vector a home. This is short
	// on purpose — everything expensive happens afterwards, under a read lock.
	ix.mu.Lock()
	if _, exists := ix.byID[id]; exists {
		ix.mu.Unlock()
		return fmt.Errorf("%w: %d", ErrDuplicateID, id)
	}
	self := uint32(len(ix.nodes))
	n := &node{id: id, links: make([]atomicLinks, layer+1)}
	ix.nodes = append(ix.nodes, n)
	ix.vectors = append(ix.vectors, stored...)
	ix.byID[id] = self
	ix.live++
	entry, maxLayer := ix.entry, ix.maxLayer
	if entry == invalidNode {
		ix.entry, ix.maxLayer = self, layer
		ix.mu.Unlock()
		return nil
	}
	ix.mu.Unlock()

	// Phase two, under a read lock: no structural change, so searches proceed
	// alongside this. Link edits are published node by node.
	ix.mu.RLock()
	s := ix.getScratch(len(ix.nodes))
	d := ix.newDistanceFn(stored, s)

	ep := entry
	for l := maxLayer; l > layer; l-- {
		ep = ix.greedy(d, ep, l)
	}

	top := layer
	if maxLayer < top {
		top = maxLayer
	}
	for l := top; l >= 0; l-- {
		ix.searchLayer(d, []uint32{ep}, ix.cfg.EfConstruction, l, s, acceptAll)
		candidates := append(s.picked[:0], s.results.items...)
		sortCandidates(candidates)
		if len(candidates) == 0 {
			continue
		}
		ep = candidates[0].node

		selected := ix.selectNeighbours(candidates, ix.cfg.M, self)
		n.mu.Lock()
		ids := make([]uint32, len(selected))
		for i, c := range selected {
			ids[i] = c.node
		}
		n.setNeighbours(l, ids)
		n.mu.Unlock()

		cap := ix.mMax
		if l == 0 {
			cap = ix.mMax0
		}
		for _, c := range selected {
			ix.linkBack(c.node, self, l, cap, s)
		}
	}
	ix.putScratch(s)
	ix.mu.RUnlock()

	// Phase three: a taller node becomes the new entry point. Taking the write
	// lock again here rather than holding it throughout is what keeps inserts
	// from serializing on the expensive part.
	if layer > maxLayer {
		ix.mu.Lock()
		if layer > ix.maxLayer {
			ix.entry, ix.maxLayer = self, layer
		}
		ix.mu.Unlock()
	}
	return nil
}

// linkBack adds self to other's neighbour list, pruning if that would exceed
// the layer's cap.
//
// Pruning is where a naive implementation goes wrong. Dropping the farthest
// neighbour looks reasonable and slowly strangles the graph: every node ends
// up linked only to its immediate cluster, the long-range edges disappear, and
// searches get stuck in local minima. So the full list is re-selected with the
// same diversity heuristic used on insert, which deliberately keeps a distant
// neighbour when it reaches somewhere no closer neighbour covers.
func (ix *Index) linkBack(other, self uint32, layer, cap int, s *searchScratch) {
	n := ix.nodes[other]
	n.mu.Lock()
	defer n.mu.Unlock()

	existing := n.neighbours(layer)
	for _, id := range existing {
		if id == self {
			return
		}
	}
	if len(existing) < cap {
		ids := make([]uint32, len(existing)+1)
		copy(ids, existing)
		ids[len(existing)] = self
		n.setNeighbours(layer, ids)
		return
	}

	base := ix.vector(other)
	candidates := append(s.prune[:0], candidate{dist: ix.distance(base, ix.vector(self)), node: self})
	for _, id := range existing {
		candidates = append(candidates, candidate{dist: ix.distance(base, ix.vector(id)), node: id})
	}
	s.prune = candidates
	sortCandidates(candidates)
	selected := ix.selectNeighbours(candidates, cap, other)
	ids := make([]uint32, len(selected))
	for i, c := range selected {
		ids[i] = c.node
	}
	n.setNeighbours(layer, ids)
}

// selectNeighbours is the paper's heuristic (Algorithm 4).
//
// Taking the M closest candidates is the obvious choice and the wrong one: in
// a dense cluster all M would point back into the same cluster, and the node
// would have no edge leading out of it. The heuristic instead accepts a
// candidate only if it is closer to the new node than to any already-accepted
// neighbour. A candidate sitting behind one that is already selected — same
// direction, farther away — is rejected as redundant, and the budget is spent
// on a direction not yet covered. That is what produces the long-range links
// the upper layers need.
//
// candidates must be sorted nearest-first.
func (ix *Index) selectNeighbours(candidates []candidate, m int, exclude uint32) []candidate {
	if len(candidates) <= m {
		out := make([]candidate, 0, len(candidates))
		for _, c := range candidates {
			if c.node != exclude {
				out = append(out, c)
			}
		}
		return out
	}

	selected := make([]candidate, 0, m)
	var discarded []candidate
	for _, c := range candidates {
		if len(selected) >= m {
			break
		}
		if c.node == exclude {
			continue
		}
		good := true
		cv := ix.vector(c.node)
		for _, s := range selected {
			if ix.distance(cv, ix.vector(s.node)) < c.dist {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, c)
		} else {
			discarded = append(discarded, c)
		}
	}
	// Refilling from the rejects keeps the degree up when the heuristic was
	// strict. A node with too few links is worse than one with a redundant
	// link, because a search that reaches it has nowhere to go.
	for i := 0; len(selected) < m && i < len(discarded); i++ {
		selected = append(selected, discarded[i])
	}
	return selected
}

func acceptAll(uint32) bool { return true }

// sortCandidates orders nearest-first. slices.SortFunc rather than sort.Slice:
// the latter sorts through reflection-based swaps, and this runs on every
// layer of every insert.
func sortCandidates(c []candidate) {
	slices.SortFunc(c, func(a, b candidate) int {
		switch {
		case a.dist < b.dist:
			return -1
		case a.dist > b.dist:
			return 1
		}
		return 0
	})
}

// greedy walks one layer downhill from ep, returning the local minimum. It is
// the ef=1 special case of searchLayer, written separately because the upper
// layers run it once per layer per insert and per query, and it needs no heap
// at all.
func (ix *Index) greedy(d *distanceFn, ep uint32, layer int) uint32 {
	best := ep
	bestDist := d.to(ep)
	for {
		improved := false
		for _, n := range ix.nodes[best].neighbours(layer) {
			if dist := d.to(n); dist < bestDist {
				best, bestDist, improved = n, dist, true
			}
		}
		if !improved {
			return best
		}
	}
}

// searchLayer is the beam search at the heart of both insert and query. It
// leaves its output in s.results, a max-heap of at most ef entries.
//
// The loop maintains one invariant worth stating: the candidate queue is only
// worth expanding while its nearest entry is closer than the current worst
// result. Once that stops being true, no path onward can improve the answer,
// because every neighbour is at least as far as the node it was reached
// through was — that is the assumption the graph's construction is designed to
// make true, and the reason ef controls recall so directly.
func (ix *Index) searchLayer(d *distanceFn, entries []uint32, ef, layer int, s *searchScratch, accept func(uint32) bool) {
	s.visited.reset()
	s.cands.Reset()
	s.results.Reset()

	for _, ep := range entries {
		if !s.visited.visit(ep) {
			continue
		}
		dist := d.to(ep)
		s.cands.Push(candidate{dist: dist, node: ep})
		if accept(ep) {
			s.results.Push(candidate{dist: dist, node: ep})
		}
	}

	for s.cands.Len() > 0 {
		c := s.cands.Pop()
		if s.results.Len() >= ef && c.dist > s.results.Top().dist {
			break
		}
		// Neighbours are read without a lock: the slice a node publishes is
		// never mutated, only replaced, so the worst a concurrent insert can
		// do is make this search miss an edge that did not exist when it
		// started.
		for _, n := range ix.nodes[c.node].neighbours(layer) {
			if !s.visited.visit(n) {
				continue
			}
			dist := d.to(n)
			if s.results.Len() < ef || dist < s.results.Top().dist {
				s.cands.Push(candidate{dist: dist, node: n})
				// A node that fails the filter still gets expanded — it is a
				// stepping stone even when it cannot be an answer — but it
				// never enters the result set.
				if accept(n) {
					s.results.Push(candidate{dist: dist, node: n})
					if s.results.Len() > ef {
						s.results.Pop()
					}
				}
			}
		}
	}
}

// Search returns the k nearest vectors to q.
//
// ef is the size of the candidate list: larger means higher recall and a
// slower query, and it must be at least k. Pass 0 to use the index's
// configured default.
func (ix *Index) Search(q []float32, k, ef int) ([]Result, error) {
	return ix.SearchFiltered(q, k, ef, nil)
}

// SearchFiltered returns the k nearest vectors for which keep returns true.
//
// The filter is applied during the walk rather than to its output, so a query
// that matches few vectors still returns k of them instead of returning
// whatever survived a post-filter. The graph is still traversed through
// non-matching nodes, which is what keeps it connected; the cost is that a
// very selective filter makes the search do more work per result, and ef
// should be raised to match.
func (ix *Index) SearchFiltered(q []float32, k, ef int, keep func(id uint64) bool) ([]Result, error) {
	if k <= 0 {
		return nil, nil
	}
	query, err := ix.prepareQuery(q)
	if err != nil {
		return nil, err
	}
	if ef <= 0 {
		ef = ix.cfg.EfSearch
	}
	if ef < k {
		ef = k
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.entry == invalidNode || ix.live == 0 {
		return nil, ErrEmpty
	}

	s := ix.getScratch(len(ix.nodes))
	defer ix.putScratch(s)
	d := ix.newDistanceFn(query, s)

	accept := func(i uint32) bool {
		n := ix.nodes[i]
		if n.deleted.Load() {
			return false
		}
		return keep == nil || keep(n.id)
	}

	ep := ix.entry
	for l := ix.maxLayer; l > 0; l-- {
		ep = ix.greedy(d, ep, l)
	}
	ix.searchLayer(d, []uint32{ep}, ef, 0, s, accept)

	hits := append([]candidate(nil), s.results.items...)
	// With quantization the ranking so far is approximate twice over — once
	// from the graph, once from the codes. Recomputing exact distances on the
	// handful of survivors removes the second error for almost no cost, which
	// is why keeping the raw vectors is usually worth the memory.
	if qz := ix.quant.Load(); qz != nil && qz.keepVectors {
		for i := range hits {
			hits[i].dist = d.exact(hits[i].node)
		}
	}
	sortCandidates(hits)
	if len(hits) > k {
		hits = hits[:k]
	}

	out := make([]Result, len(hits))
	for i, h := range hits {
		out[i] = Result{ID: ix.nodes[h.node].id, Distance: ix.reportDistance(h.dist)}
	}
	return out, nil
}

// Delete removes an id from the results, reporting whether it was present.
//
// The node stays in the graph as a stepping stone. Physically removing it
// would mean repairing every node that links to it, and a node deleted from
// the middle of a cluster can be the only path between two regions — cutting
// it can disconnect the graph, which costs recall for every future query, not
// just for the deleted vector. Tombstoning trades a little memory and a little
// search work for a graph that stays navigable; rebuilding is how the space is
// actually reclaimed.
func (ix *Index) Delete(id uint64) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	i, ok := ix.byID[id]
	if !ok {
		return false
	}
	ix.nodes[i].deleted.Store(true)
	delete(ix.byID, id)
	ix.live--
	if ix.entry == i {
		ix.repointEntry()
	}
	return true
}

// repointEntry finds a new entry point after the old one was deleted. The
// caller must hold the write lock.
//
// A deleted entry point is the one case tombstoning cannot paper over: every
// search starts here, and starting at a node that no longer answers is fine,
// but the entry must still sit at the top of the graph or the descent skips
// layers. The scan is linear, which is acceptable because it happens only when
// the single highest node in the index is deleted.
func (ix *Index) repointEntry() {
	best, bestLayer := invalidNode, -1
	for i, n := range ix.nodes {
		if n.deleted.Load() {
			continue
		}
		if l := len(n.links) - 1; l > bestLayer {
			best, bestLayer = uint32(i), l
		}
	}
	ix.entry, ix.maxLayer = best, bestLayer
	if best == invalidNode {
		ix.maxLayer = -1
	}
}

// Contains reports whether an id is present and live.
func (ix *Index) Contains(id uint64) bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	_, ok := ix.byID[id]
	return ok
}

// Vector returns a copy of a stored vector, or false if the id is absent or
// its raw vector was dropped by Quantize.
func (ix *Index) Vector(id uint64) ([]float32, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	i, ok := ix.byID[id]
	if !ok || len(ix.vectors) == 0 {
		return nil, false
	}
	out := make([]float32, ix.cfg.Dim)
	copy(out, ix.vector(i))
	return out, true
}
