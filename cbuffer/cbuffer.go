// Package cbuffer provides a wrapper for a C/C++ buffer that can be used to transfer data in-between pjrt, xlabuilder and the user
// of the library.
//
// It is used to feed data: literals for xlabuilder, and actual values for pjrt, and as a holder of the StableHLO
// program built by xlabuilder.
package cbuffer

/*
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// CBuffer is a generic wrapper to a C/C++ data.
//
// It doesn't define the semantics of ownership: depends on how it is being used.
// But if needed it provides a convenient Free() method, that checks so that it is not freed more than once.
type CBuffer struct {
	Data unsafe.Pointer
	Size int
}

// Free underlying data.
// It sets the pointer to nil, so if it is called again it is a no-op.
func (b *CBuffer) Free() {
	if b.Data == nil || b.Size == 0 {
		return
	}
	C.free(b.Data)
	b.Data = nil
	b.Size = 0
}

// AsBytes create an unsafe buffer slice to the underlying data, if one wants to use it from Go.
func (b *CBuffer) AsBytes() []byte {
	return unsafe.Slice((*byte)(b.Data), b.Size)
}
