package lodestar

// visitedSet marks which nodes a search has already seen.
//
// The obvious implementation is a bitset cleared between searches, but the
// clear is O(n) in the size of the index while a search touches only a few
// hundred nodes — at a million vectors, resetting the set would cost more
// than the search it serves. Stamping each slot with a generation number
// instead makes the reset a single increment.
//
// The stamp is a uint32, so it wraps after four billion searches. Wrapping is
// handled rather than ignored: on overflow the array is zeroed once and the
// counter restarts, which is the only O(n) reset the structure ever performs.
type visitedSet struct {
	stamps []uint32
	gen    uint32
}

func newVisitedSet(n int) *visitedSet {
	return &visitedSet{stamps: make([]uint32, n), gen: 1}
}

// grow extends the set to cover at least n nodes.
func (v *visitedSet) grow(n int) {
	if n <= len(v.stamps) {
		return
	}
	stamps := make([]uint32, n)
	copy(stamps, v.stamps)
	v.stamps = stamps
}

// reset begins a new search.
func (v *visitedSet) reset() {
	v.gen++
	if v.gen == 0 {
		for i := range v.stamps {
			v.stamps[i] = 0
		}
		v.gen = 1
	}
}

// visit marks a node and reports whether it was newly marked.
func (v *visitedSet) visit(node uint32) bool {
	if v.stamps[node] == v.gen {
		return false
	}
	v.stamps[node] = v.gen
	return true
}
