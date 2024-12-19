package pjrt

/*
#include <string.h>
*/
import "C"
import (
	"fmt"
	"reflect"
	"sync"
	"unsafe"
)

// arenaContainer implements a trivial arena object to accelerate allocations that will be used in CGO calls.
//
// The issue it is trying to solve is that individual CGO calls are slow, including C.malloc().
//
// It pre-allocates the given size in bytes in C -- so it does not needs to be pinned when using CGO, and allow
// for fast sub-allocations.
// It can only be freed all at once.
//
// If you don't call Free at the end, it will leak the C allocated space.
//
// See newArena and arenaAlloc, and also arenaPool.
type arenaContainer struct {
	buf           []byte
	size, current int
}

const arenaDefaultSize = 2048

var arenaPool sync.Pool = sync.Pool{
	New: func() interface{} {
		return newArena(arenaDefaultSize)
	},
}

// getArenaFromPool gets an arena of the default size.
// Must be matched with a call returnArenaToPool when it's no longer used.
func getArenaFromPool() *arenaContainer {
	return arenaPool.Get().(*arenaContainer)
}

// returnArenaToPool returns an arena acquired with getArenaFromPool.
// It also resets the arena.
func returnArenaToPool(a *arenaContainer) {
	a.Reset()
	arenaPool.Put(a)
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
		buf:  unsafe.Slice(buf, size),
		size: size,
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
