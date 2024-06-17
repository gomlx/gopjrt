package xlabuilder

// #cgo LDFLAGS: -lgomlx_xlabuilder
/*
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"github.com/pkg/errors"
	"unsafe"
)

//go:generate go run ../cmd/xlabuilder_codegen

// Since CGO C types cannot cross boundaries of a package (see issue https://github.com/golang/go/issues/13467)
// We make a copy of chelper.go for every sub-directory that needs it.
//go:generate go run ../cmd/copy_go_code --original=chelper.go

// XlaBuilder is used to create a "StableHLO" program, that can then be compiled and executed by a PJRT plugin (see package pjrt) on
// accelerators.
//
// Once created (New), one can issue "operations" ("ops" for short), like "Add", "Mul", etc, which are recorded.
// When the computation definition is finalized, call "XlaBuilder.StableHLO" to get the program to use with PJRT.
//
// Once done (usually, just after StableHLO is called) deallocate the underlying C++ resources by calling Free.
//
// Some observations:
//
//   - The XlaBuilder is used by all ops creating functions (like "Add", "Mul", etc.). But since the input of most ops,
//     are other created ops, and they hold a link to the builder, there is no need to explicitly pass the XlaBuilder to
//     every op function.
type XlaBuilder struct {
	builder *C.XlaBuilder
}

// CBuffer defines an interface of something that returns a pointer to an area managed by C memory manager.
// The semantic of ownership is not defined by this interface.
type CBuffer interface {
	// Data returns an unsafe.Pointer to the underlying C data.
	Data() unsafe.Pointer

	// Size returns the size in bytes of the underlying data.
	Size() int

	// Free frees the underlying data.
	Free()
}

// cBuffer implements CBuffer.
type cBuffer struct {
	data unsafe.Pointer
	size int
}

func (b *cBuffer) Data() unsafe.Pointer { return b.data }
func (b *cBuffer) Size() int            { return b.size }
func (b *cBuffer) Free() {
	if b.data == nil || b.size == 0 {
		return
	}
	C.free(b.data)
	b.data = nil
	b.size = 0
}

// Assert cBuffer implements CBuffer
var _ CBuffer = (*cBuffer)(nil)

// StableHLO generates the StableHLO program as a <serialized HLOModule proto> (something that PJRT can consume).
//
// The returned CBuffer needs to be freed (CBuffer.Free) after being used (presumably by PJRT, or saved to a file).
//
// It takes as input outputOp that is returned by the program.
func (b *XlaBuilder) StableHLO(outputOp *Op) (CBuffer, error) {
	statusOr := C.XlaBuilderSerializedHLO(unsafe.Pointer(b.builder), unsafe.Pointer(outputOp.op))
	vectorData, err := pointerOrError[C.VectorData](statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "while converting the XlaBuilder ops to a StableHLO representation")
	}
	buf := &cBuffer{
		data: unsafe.Pointer(vectorData.data),
		size: int(vectorData.count),
	}
	return buf, nil
}

// Free must be called once the builder is done to free the underlying object.
// The garbage collector won't free it by itself.
// It can be called more than once -- once finalized the first time, it becomes a no-op.
func (b *XlaBuilder) Free() {
	if b.builder == nil {
		return
	}
	C.XlaBuilderDestroy(unsafe.Pointer(b.builder))
	b.builder = nil
}
