package lodestar

// A search keeps two priority queues: the candidate set, ordered so the
// closest unexplored node comes out first, and the result set, ordered so the
// farthest kept node comes out first and can be evicted when a better one
// arrives. They are the same data laid out under opposite comparisons, which
// is why both are open-coded here rather than expressed through container/heap.
//
// container/heap would do the job, but it dispatches Less and Swap through an
// interface on every sift step. These heaps are pushed and popped millions of
// times per build, so the indirection is worth removing: the specialized
// version is roughly twice as fast and, more importantly, keeps the candidate
// struct in registers instead of forcing it through an any-shaped Swap.

// candidate is a node paired with its distance to the current query.
type candidate struct {
	dist float32
	node uint32
}

// minHeap yields the smallest distance first. It is the frontier of a search:
// always expand the closest thing not yet expanded.
type minHeap struct{ items []candidate }

func (h *minHeap) Len() int       { return len(h.items) }
func (h *minHeap) Top() candidate { return h.items[0] }
func (h *minHeap) Reset()         { h.items = h.items[:0] }

func (h *minHeap) Push(c candidate) {
	h.items = append(h.items, c)
	i := len(h.items) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if h.items[parent].dist <= h.items[i].dist {
			break
		}
		h.items[parent], h.items[i] = h.items[i], h.items[parent]
		i = parent
	}
}

func (h *minHeap) Pop() candidate {
	top := h.items[0]
	last := len(h.items) - 1
	h.items[0] = h.items[last]
	h.items = h.items[:last]
	h.down(0)
	return top
}

func (h *minHeap) down(i int) {
	n := len(h.items)
	for {
		l, r := 2*i+1, 2*i+2
		smallest := i
		if l < n && h.items[l].dist < h.items[smallest].dist {
			smallest = l
		}
		if r < n && h.items[r].dist < h.items[smallest].dist {
			smallest = r
		}
		if smallest == i {
			return
		}
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
		i = smallest
	}
}

// maxHeap yields the largest distance first. It holds the current best
// results, so its top is the row that gets evicted when a closer node is
// found — and its top is also the radius that decides when the search can
// stop, since nothing farther than the current worst result can improve it.
type maxHeap struct{ items []candidate }

func (h *maxHeap) Len() int       { return len(h.items) }
func (h *maxHeap) Top() candidate { return h.items[0] }
func (h *maxHeap) Reset()         { h.items = h.items[:0] }

func (h *maxHeap) Push(c candidate) {
	h.items = append(h.items, c)
	i := len(h.items) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if h.items[parent].dist >= h.items[i].dist {
			break
		}
		h.items[parent], h.items[i] = h.items[i], h.items[parent]
		i = parent
	}
}

func (h *maxHeap) Pop() candidate {
	top := h.items[0]
	last := len(h.items) - 1
	h.items[0] = h.items[last]
	h.items = h.items[:last]
	h.down(0)
	return top
}

func (h *maxHeap) down(i int) {
	n := len(h.items)
	for {
		l, r := 2*i+1, 2*i+2
		largest := i
		if l < n && h.items[l].dist > h.items[largest].dist {
			largest = l
		}
		if r < n && h.items[r].dist > h.items[largest].dist {
			largest = r
		}
		if largest == i {
			return
		}
		h.items[i], h.items[largest] = h.items[largest], h.items[i]
		i = largest
	}
}
