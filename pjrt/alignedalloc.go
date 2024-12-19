package pjrt

// This file defines an alignedAlloc and alignedFree, modelled after mm_malloc.

/*
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// BufferAlignment is the default alignment required for memory shared with CPU PJRT.
// See AlignedAlloc and FreeAlloc.
const BufferAlignment = 64

// AlignedAlloc assumes that malloc/calloc already aligns to 8 bytes. And that alignment is a multiple of 8.
// The pointer returned must be freed with AlignedFree.
//
// The allocation is filled with 0s.
func AlignedAlloc(size, alignment uintptr) unsafe.Pointer {
	if alignment < 8 || alignment%8 != 0 {
		panic(fmt.Sprintf("alignedAlloc: alignment must be a multiple of 8, got %d", alignment))
	}

	// It uses a strategy of allocating extra to allow the alignment, and it stores the pointer to the
	// original allocation just before the alignedPtr.
	totalSize := size + alignment
	ptr := unsafe.Pointer(C.calloc(C.size_t(totalSize), C.size_t(1)))
	if ptr == nil {
		return nil
	}

	alignedPtr := ptr
	offset := uintptr(ptr) % alignment
	if offset != 0 {
		alignedPtr = unsafe.Pointer(uintptr(ptr) + (alignment - offset))
	} else {
		alignedPtr = unsafe.Pointer(uintptr(ptr) + alignment) // This way we have the space to save the original ptr.
	}

	originalPtrPtr := (*uintptr)(unsafe.Pointer(uintptr(alignedPtr) - unsafe.Sizeof(uintptr(0))))
	*originalPtrPtr = uintptr(ptr)

	return alignedPtr
}

// AlignedFree frees an allocation created with AlignedAlloc.
func AlignedFree(ptr unsafe.Pointer) {
	originalPtrPtr := (*uintptr)(unsafe.Pointer(uintptr(ptr) - unsafe.Sizeof(uintptr(0))))
	originalPtr := unsafe.Pointer(*originalPtrPtr)
	C.free(originalPtr)
}
