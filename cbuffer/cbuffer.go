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
	"k8s.io/klog/v2"
	"runtime"
	"unsafe"
)

// CBuffer is a generic wrapper to a C/C++ data, which is assumed to own the underlying data.
type CBuffer struct {
	data  unsafe.Pointer
	size  int
	stack []byte
}

// New returns a CBuffer object to manage the C/C++ data.
//
// If `withStack` is set to true, it also stores a stack of where it was created.
// This is used for debugging if it is garbage collected without being freed.
func New(data unsafe.Pointer, size int, withStack bool) *CBuffer {
	b := &CBuffer{data: data, size: size}
	if withStack {
		buf := make([]byte, 10*1024)
		n := runtime.Stack(buf, false)
		b.stack = buf[:n]
	}
	runtime.SetFinalizer(b, func(b *CBuffer) {
		if b.data == nil {
			return // Correctly freed.
		}

		// Notice that we cannot automatically free the underlying data here because it may still be in use: if the user
		// uses the output of CBuffer.Bytes() and no longer uses the CBuffer it may immediately be garbage
		// collected before the goroutine using the returned C++ pointer (as []byte) has had a chance to use it.
		// To avoid this the user would have to use a `runtime.KeepAlive(*CBuffer)` call after they use `CBuffer.Bytes()`.
		// But then it's better to have the user to call CBuffer.Free -- a memory leak with a warning is better than
		// some mysterious spurious errors due to reuse of a freed pointer.

		// Log about CBuffer not having been freed.
		if b.stack == nil {
			klog.Errorf("CBuffer of %d bytes garbage collected without the corresponding data being freed", b.size)
		} else {
			klog.Errorf("CBuffer of %d bytes garbage collected without the corresponding data being freed. Stack:\n%s\n", b.size, b.stack)
		}
	})
	return b
}

// Free underlying data.
// It sets the pointer to nil, so if it is called again it is a no-op.
func (b *CBuffer) Free() {
	if b.data == nil {
		return
	}
	C.free(b.data)
	b.data = nil
	b.size = 0
}

// Bytes returns the buffer as a byte slice.
//
// Ownership is not transferred: remember to free CBuffer afterwards.
func (b *CBuffer) Bytes() []byte {
	if b.data == nil {
		return nil
	}
	return unsafe.Slice((*byte)(b.data), b.size)
}
