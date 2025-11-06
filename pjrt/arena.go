package pjrt

/*
#include <string.h>
*/
import "C"
import (
	"fmt"
	"math/bits"
	"reflect"
	"sync"
	"unsafe"
)

// arenaContainer implements a trivial arena object to speed up allocations that will be used in CGO calls.
//
// The issue it is trying to solve is that individual CGO calls are slow, including C.malloc().
//
// It pre-allocates the given size in bytes in C -- so it does not need to be pinned when using CGO and allows
// for fast suballocations.
// It can only be freed all at once.
//
// If you don't call Free at the end, it will leak the C allocated space.
//
// The Plugin object also provides an arenaPool that improves things a bit.
type arenaContainer struct {
	buf           []byte
	size, current int
	poolIndex     int // index in the arenaPools, -1 if not from pool
}

// newArena creates a new Arena with the given fixed size.
//
// It provides fast sub-allocations, which can only be freed all at once.
//
// TODO: support memory-alignment.
//
// See arenaAlloc, arena.Free and arena.Reset.
func newArena(size int) *arenaContainer {
	buf := cMallocArray[byte](size)
	a := &arenaContainer{
		buf:       unsafe.Slice(buf, size),
		size:      size,
		poolIndex: -1,
	}
	return a
}

const arenaAlignBytes = 8

// arenaAlloc allocates a type T from the arena. It panics if the arena run out of memory.
func arenaAlloc[T any](a *arenaContainer) (ptr *T) {
	allocSize := cSizeOf[T]()
	if a.current+int(allocSize) > a.size {
		panic(fmt.Sprintf("Arena out of memory while allocating %d bytes for %q", allocSize, reflect.TypeOf(ptr).Elem()))
	}
	ptr = (*T)(unsafe.Pointer(&a.buf[a.current]))
	a.current += int(allocSize)
	a.current = (a.current + arenaAlignBytes - 1) &^ (arenaAlignBytes - 1)
	return
}

// arenaAllocSlice allocates an array of n elements of type T from the arena.
//
// It panics if the arena run out of memory.
func arenaAllocSlice[T any](a *arenaContainer, n int) (slice []T) {
	allocSize := C.size_t(n) * cSizeOf[T]()
	if a.current+int(allocSize) > a.size {
		panic(fmt.Sprintf("Arena out of memory while allocating %d bytes for [%d]%s", allocSize, n, reflect.TypeOf(slice).Elem()))
	}
	ptr := (*T)(unsafe.Pointer(&a.buf[a.current]))
	a.current += int(allocSize)
	a.current = (a.current + arenaAlignBytes - 1) &^ (arenaAlignBytes - 1)
	slice = unsafe.Slice(ptr, n)
	return
}

// Free invalidates all previous allocations of the arena and frees the C allocated area.
func (a *arenaContainer) Free() {
	cFree(&a.buf[0])
	a.buf = nil
	a.size = 0
	a.current = 0
}

// Reset invalidates all previous allocations with the arena, but does not free the C allocated area.
// This way the arena can be re-used.
func (a *arenaContainer) Reset() {
	// Zero the values used.
	if a.buf == nil || a.size == 0 {
		a.current = 0
		return
	}
	if a.current > 0 {
		clearSize := min(a.size, a.current)
		C.memset(unsafe.Pointer(&a.buf[0]), 0, C.size_t(clearSize))
	}
	a.current = 0
}

const (
	// minPooledArenaSize is the minimum size for pooled arenas.
	minPooledArenaSize = 2048
	// maxPooledArenaSize is the maximum size for pooled arenas (16MB).
	maxPooledArenaSize = 16 * 1024 * 1024
)

// arenaPools manages pools of arenaContainer objects with power-of-2 sizes.
// It provides fast, concurrent-safe allocation and reuse of arena objects.
type arenaPools struct {
	// pools[i] contains arenas of size 2^(i+11), where i=0 is DefaultArenaSize (2048 = 2^11)
	// and the maximum is 16MB (2^24).
	pools []sync.Pool
	// minShift is the bit position for DefaultArenaSize (11 for 2048)
	minShift int
	// maxShift is the bit position for maxPooledArenaSize (24 for 16MB)
	maxShift int
}

// newArenaPools creates a new arenaPools manager.
func newArenaPools() *arenaPools {
	minShift := bits.TrailingZeros(uint(minPooledArenaSize))
	maxShift := bits.TrailingZeros(uint(maxPooledArenaSize))
	numPools := maxShift - minShift + 1

	ap := &arenaPools{
		pools:    make([]sync.Pool, numPools),
		minShift: minShift,
		maxShift: maxShift,
	}

	return ap
}

// Get returns an arenaContainer of at least targetSize bytes.
// The actual size will be the next power-of-2 >= targetSize.
// The returned arena is reset and ready to use.
func (ap *arenaPools) Get(targetSize int) *arenaContainer {
	if targetSize <= 0 {
		targetSize = minPooledArenaSize
	}

	// Calculate the next power of 2 >= targetSize
	shift := bits.Len(uint(targetSize - 1))
	if shift < ap.minShift {
		shift = ap.minShift
	}

	// If the requested size is larger than max pooled size, allocate directly
	if shift > ap.maxShift {
		return newArena(targetSize)
	}

	// Calculate pool index and actual size
	poolIndex := shift - ap.minShift
	actualSize := 1 << shift

	// Try to get from the pool.
	if obj := ap.pools[poolIndex].Get(); obj != nil {
		arena := obj.(*arenaContainer)
		arena.Reset()
		return arena
	}

	// Create the new arena.
	buf := cMallocArray[byte](actualSize)
	arena := &arenaContainer{
		buf:       unsafe.Slice(buf, actualSize),
		size:      actualSize,
		poolIndex: poolIndex,
	}
	return arena
}

// Return returns an arenaContainer to the pool for reuse.
// The arena is reset before being returned to the pool.
// Arenas larger than maxPooledArenaSize are freed instead of pooled.
func (ap *arenaPools) Return(arena *arenaContainer) {
	if arena == nil {
		return
	}

	// If not from pool or too large, just free it
	if arena.poolIndex < 0 || arena.poolIndex >= len(ap.pools) {
		arena.Free()
		return
	}

	// Reset and return to the pool.
	arena.Reset()
	ap.pools[arena.poolIndex].Put(arena)
}
