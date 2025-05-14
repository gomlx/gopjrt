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
	wrapper *cBufferWrapper
}

type cBufferWrapper struct {
	size  int
	data  unsafe.Pointer
	stack []byte
}

// New returns a CBuffer object to manage the C/C++ data.
//
// If `withStack` is set to true, it also stores a stack of where it was created.
// This is used for debugging if it is garbage collected without being freed.
func New(data unsafe.Pointer, size int, withStack bool) *CBuffer {
	b := &CBuffer{&cBufferWrapper{data: data, size: size}}
	if withStack {
		buf := make([]byte, 10*1024)
		n := runtime.Stack(buf, false)
		b.wrapper.stack = buf[:n]
	}
	runtime.AddCleanup(b, func(wrapper *cBufferWrapper) {
		if wrapper.data == nil {
			return // Correctly freed.
		}

		// Notice that we cannot automatically free the underlying data here because it may still be in use: if the user
		// uses the output of CBuffer.Bytes() and no longer uses the CBuffer it may immediately be garbage
		// collected before the goroutine using the returned C++ pointer (as []byte) has had a chance to use it.
		// To avoid this the user would have to use a `runtime.KeepAlive(*CBuffer)` call after they use `CBuffer.Bytes()`.
		// But then it's better to have the user to call CBuffer.Destroy -- a memory leak with a warning is better than
		// some mysterious spurious errors due to reuse of a freed pointer.

		// Log about CBuffer not having been freed.
		if wrapper.stack == nil {
			klog.Errorf("CBuffer of %d bytes garbage collected without the corresponding data being freed", wrapper.size)
		} else {
			klog.Errorf("CBuffer of %d bytes garbage collected without the corresponding data being freed. Stack:\n%s\n", wrapper.size, wrapper.stack)
		}
	}, b.wrapper)
	return b
}

// NewFromString returns a CBuffer that holds a copy of the given string.
//
// Like a normal CBuffer, it needs to be freed.
func NewFromString(s string, withStack bool) *CBuffer {
	data := unsafe.Pointer(C.CString(s))
	// Notice we don't include the '\0' in the length, even thought it's likely
	// allocated along.
	return New(data, len(s), withStack)
}

func (wrapper *cBufferWrapper) Free() {
	if wrapper.data == nil {
		return
	}
	C.free(wrapper.data)
	wrapper.data = nil
	wrapper.size = 0
}

// Free the underlying data.
// It sets the pointer to nil, so if it is called again, it is a no-op.
func (b *CBuffer) Free() {
	b.wrapper.Free()
}

// Bytes returns the buffer as a byte slice.
//
// Ownership is not transferred: remember to free CBuffer afterward.
func (b *CBuffer) Bytes() []byte {
	if b.wrapper.data == nil {
		return nil
	}
	return unsafe.Slice((*byte)(b.wrapper.data), b.wrapper.size)
}
