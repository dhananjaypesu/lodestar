package lodestar

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"
	"math/rand"

	"github.com/dhananjaypesu/lodestar/internal/pq"
)

// The on-disk format is deliberately plain: a header, the vectors, the
// quantizer if there is one, then the adjacency lists, with a CRC of
// everything at the end.
//
// Rebuilding an HNSW graph from vectors is expensive — it is the dominant cost
// of using one — so the links are what actually need persisting. Writing them
// verbatim means loading is a read and a few allocations rather than a rebuild,
// which is the difference between a process that starts in a second and one
// that starts in an hour.
//
// A checksum is here because the failure mode without one is silent: a
// corrupted link array does not crash, it quietly makes searches return worse
// answers, and nothing in the results says so.

const (
	magic         = "LODESTR1"
	formatVersion = uint32(1)
)

// castagnoli is the CRC polynomial with hardware support on both amd64 and
// arm64, so checksumming costs a fraction of what the IEEE table would.
var castagnoli = crc32.MakeTable(crc32.Castagnoli)

type writer struct {
	w   *bufio.Writer
	crc hash.Hash32
	buf [8]byte
	err error
}

func newWriter(w io.Writer) *writer {
	return &writer{w: bufio.NewWriterSize(w, 1<<16), crc: crc32.New(castagnoli)}
}

func (w *writer) bytes(b []byte) {
	if w.err != nil {
		return
	}
	if _, err := w.w.Write(b); err != nil {
		w.err = err
		return
	}
	w.crc.Write(b)
}

func (w *writer) u32(v uint32) {
	binary.LittleEndian.PutUint32(w.buf[:4], v)
	w.bytes(w.buf[:4])
}

func (w *writer) u64(v uint64) {
	binary.LittleEndian.PutUint64(w.buf[:8], v)
	w.bytes(w.buf[:8])
}

func (w *writer) u8(v uint8) {
	w.buf[0] = v
	w.bytes(w.buf[:1])
}

// floats writes a float32 slice as raw little-endian words. binary.Write would
// do this too, through reflection, one value at a time; a build of any size
// makes that visible.
func (w *writer) floats(f []float32) {
	w.u32(uint32(len(f)))
	const chunk = 4096
	buf := make([]byte, chunk*4)
	for start := 0; start < len(f); start += chunk {
		end := start + chunk
		if end > len(f) {
			end = len(f)
		}
		part := buf[:(end-start)*4]
		for i, v := range f[start:end] {
			binary.LittleEndian.PutUint32(part[i*4:], math.Float32bits(v))
		}
		w.bytes(part)
	}
}

func (w *writer) u32s(v []uint32) {
	w.u32(uint32(len(v)))
	const chunk = 4096
	buf := make([]byte, chunk*4)
	for start := 0; start < len(v); start += chunk {
		end := start + chunk
		if end > len(v) {
			end = len(v)
		}
		part := buf[:(end-start)*4]
		for i, x := range v[start:end] {
			binary.LittleEndian.PutUint32(part[i*4:], x)
		}
		w.bytes(part)
	}
}

type reader struct {
	r   *bufio.Reader
	crc hash.Hash32
	buf [8]byte
	err error
}

func newReader(r io.Reader) *reader {
	return &reader{r: bufio.NewReaderSize(r, 1<<16), crc: crc32.New(castagnoli)}
}

func (r *reader) bytes(b []byte) {
	if r.err != nil {
		return
	}
	if _, err := io.ReadFull(r.r, b); err != nil {
		r.err = err
		return
	}
	r.crc.Write(b)
}

func (r *reader) u32() uint32 {
	r.bytes(r.buf[:4])
	return binary.LittleEndian.Uint32(r.buf[:4])
}

func (r *reader) u64() uint64 {
	r.bytes(r.buf[:8])
	return binary.LittleEndian.Uint64(r.buf[:8])
}

func (r *reader) u8() uint8 {
	r.bytes(r.buf[:1])
	return r.buf[0]
}

func (r *reader) floats() []float32 {
	n := r.u32()
	if r.err != nil {
		return nil
	}
	out := make([]float32, n)
	const chunk = 4096
	buf := make([]byte, chunk*4)
	for start := 0; start < int(n); start += chunk {
		end := start + chunk
		if end > int(n) {
			end = int(n)
		}
		part := buf[:(end-start)*4]
		r.bytes(part)
		if r.err != nil {
			return nil
		}
		for i := range out[start:end] {
			out[start+i] = math.Float32frombits(binary.LittleEndian.Uint32(part[i*4:]))
		}
	}
	return out
}

func (r *reader) u32s() []uint32 {
	n := r.u32()
	if r.err != nil {
		return nil
	}
	out := make([]uint32, n)
	const chunk = 4096
	buf := make([]byte, chunk*4)
	for start := 0; start < int(n); start += chunk {
		end := start + chunk
		if end > int(n) {
			end = int(n)
		}
		part := buf[:(end-start)*4]
		r.bytes(part)
		if r.err != nil {
			return nil
		}
		for i := range out[start:end] {
			out[start+i] = binary.LittleEndian.Uint32(part[i*4:])
		}
	}
	return out
}

// Save writes the index to w. The index may be searched concurrently; it may
// not be written to.
func (ix *Index) Save(dst io.Writer) error {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	w := newWriter(dst)
	w.bytes([]byte(magic))
	w.u32(formatVersion)
	w.u32(uint32(ix.cfg.Dim))
	w.u8(uint8(ix.cfg.Metric))
	w.u32(uint32(ix.cfg.M))
	w.u32(uint32(ix.cfg.EfConstruction))
	w.u32(uint32(ix.cfg.EfSearch))
	w.u64(uint64(ix.cfg.Seed))
	w.u32(uint32(len(ix.nodes)))
	w.u32(ix.entry)
	w.u32(uint32(int32(ix.maxLayer)))
	w.u32(uint32(ix.live))

	if len(ix.vectors) > 0 {
		w.u8(1)
		w.floats(ix.vectors)
	} else {
		w.u8(0)
	}

	if z := ix.quant.Load(); z != nil {
		w.u8(1)
		w.u32(uint32(z.q.M))
		w.u32(uint32(z.q.Sub))
		w.u8(uint8(z.mode))
		if z.keepVectors {
			w.u8(1)
		} else {
			w.u8(0)
		}
		w.floats(z.q.Centroids)
		w.u32(uint32(len(z.codes)))
		w.bytes(z.codes)
	} else {
		w.u8(0)
	}

	for _, n := range ix.nodes {
		w.u64(n.id)
		if n.deleted.Load() {
			w.u8(1)
		} else {
			w.u8(0)
		}
		w.u32(uint32(len(n.links)))
		for l := range n.links {
			w.u32s(n.neighbours(l))
		}
	}

	sum := w.crc.Sum32()
	if w.err != nil {
		return w.err
	}
	var trailer [4]byte
	binary.LittleEndian.PutUint32(trailer[:], sum)
	if _, err := w.w.Write(trailer[:]); err != nil {
		return err
	}
	return w.w.Flush()
}

// Load reads an index written by Save.
func Load(src io.Reader) (*Index, error) {
	r := newReader(src)
	header := make([]byte, len(magic))
	r.bytes(header)
	if r.err != nil {
		return nil, r.err
	}
	if string(header) != magic {
		return nil, fmt.Errorf("lodestar: not a lodestar index (bad magic)")
	}
	if v := r.u32(); v != formatVersion {
		return nil, fmt.Errorf("lodestar: unsupported format version %d", v)
	}

	cfg := Config{Dim: int(r.u32())}
	cfg.Metric = Metric(r.u8())
	cfg.M = int(r.u32())
	cfg.EfConstruction = int(r.u32())
	cfg.EfSearch = int(r.u32())
	cfg.Seed = int64(r.u64())
	nodeCount := int(r.u32())
	entry := r.u32()
	maxLayer := int(int32(r.u32()))
	live := int(r.u32())
	if r.err != nil {
		return nil, r.err
	}

	ix, err := New(cfg)
	if err != nil {
		return nil, err
	}
	ix.rng = rand.New(rand.NewSource(cfg.Seed))
	ix.entry, ix.maxLayer, ix.live = entry, maxLayer, live

	if r.u8() == 1 {
		ix.vectors = r.floats()
	}

	var z *quantized
	if r.u8() == 1 {
		z = &quantized{metric: cfg.Metric}
		m := int(r.u32())
		sub := int(r.u32())
		z.mode = pq.Mode(r.u8())
		z.keepVectors = r.u8() == 1
		centroids := r.floats()
		z.q = &pq.Quantizer{Dim: cfg.Dim, M: m, Sub: sub, Centroids: centroids}
		n := int(r.u32())
		z.codes = make([]byte, n)
		r.bytes(z.codes)
	}
	if r.err != nil {
		return nil, r.err
	}

	ix.nodes = make([]*node, nodeCount)
	ix.byID = make(map[uint64]uint32, nodeCount)
	for i := 0; i < nodeCount; i++ {
		id := r.u64()
		deleted := r.u8() == 1
		layers := int(r.u32())
		if r.err != nil {
			return nil, r.err
		}
		n := &node{id: id, links: make([]atomicLinks, layers)}
		if deleted {
			n.deleted.Store(true)
		} else {
			ix.byID[id] = uint32(i)
		}
		for l := 0; l < layers; l++ {
			n.setNeighbours(l, r.u32s())
		}
		ix.nodes[i] = n
	}
	if r.err != nil {
		return nil, r.err
	}

	// The checksum covers everything above, so it is read outside the digest.
	want := r.crc.Sum32()
	var trailer [4]byte
	if _, err := io.ReadFull(r.r, trailer[:]); err != nil {
		return nil, fmt.Errorf("lodestar: truncated index: %w", err)
	}
	if got := binary.LittleEndian.Uint32(trailer[:]); got != want {
		return nil, fmt.Errorf("lodestar: checksum mismatch, index is corrupt (want %08x, got %08x)", want, got)
	}
	if z != nil {
		ix.quant.Store(z)
	}
	if err := ix.validate(); err != nil {
		return nil, err
	}
	return ix, nil
}

// validate rejects a structurally impossible index rather than letting a
// corrupt-but-checksum-valid file cause an out-of-range panic deep in a
// search. The checksum catches bit rot; this catches a file written by a
// different or broken implementation.
func (ix *Index) validate() error {
	n := uint32(len(ix.nodes))
	if n == 0 {
		if ix.entry != invalidNode {
			return fmt.Errorf("lodestar: empty index names an entry point")
		}
		return nil
	}
	if ix.entry != invalidNode && ix.entry >= n {
		return fmt.Errorf("lodestar: entry point %d is out of range", ix.entry)
	}
	if len(ix.vectors) != 0 && len(ix.vectors) != int(n)*ix.cfg.Dim {
		return fmt.Errorf("lodestar: vector array holds %d floats, expected %d", len(ix.vectors), int(n)*ix.cfg.Dim)
	}
	for i, node := range ix.nodes {
		for l := range node.links {
			for _, id := range node.neighbours(l) {
				if id >= n {
					return fmt.Errorf("lodestar: node %d layer %d links to %d, out of range", i, l, id)
				}
			}
		}
	}
	return nil
}
