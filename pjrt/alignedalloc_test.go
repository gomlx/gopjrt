package pjrt

import (
	"math/rand/v2"
	"testing"
	"unsafe"
)

func TestAlignedAlloc(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	numLivePointers := 1_000
	maxAllocSize := 1_000
	pointers := make([]unsafe.Pointer, numLivePointers)
	for _ = range 1_000_000 {
		idx := rng.IntN(numLivePointers)
		if pointers[idx] != nil {
			AlignedFree(pointers[idx])
		}
		size := uintptr(rng.IntN(maxAllocSize))
		pointers[idx] = AlignedAlloc(size, BufferAlignment)
	}
	for _, ptr := range pointers {
		AlignedFree(ptr)
	}
}
