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
//     are other created ops, and they hold a link to the cBuilder, there is no need to explicitly pass the XlaBuilder to
//     every op function.
type XlaBuilder struct {
	cBuilder *C.XlaBuilder
}

// New create a new XlaBuilder with the given name, that can be used to create a new StableHLO program.
// See details on how to use it on XlaBuilder.
func New(name string) *XlaBuilder {
	var cBuilder *C.XlaBuilder
	cName := C.CString(name)
	defer cFree(cName)

	cBuilder = (*C.XlaBuilder)(C.NewXlaBuilder(cName))
	return &XlaBuilder{cBuilder: cBuilder}
}

// Free must be called once the cBuilder is done to free the underlying object.
// The garbage collector won't free it by itself.
// It can be called more than once -- once finalized the first time, it becomes a no-op.
func (b *XlaBuilder) Free() {
	if b.cBuilder == nil {
		return
	}
	C.XlaBuilderDestroy(unsafe.Pointer(b.cBuilder))
	b.cBuilder = nil
}

// addOp will add the operation described by op.
// If it succeeds it fills the fields Op.index and Op.op, with the C++ references.
func (b *XlaBuilder) addOp(op *Op) error {
	if b == nil {
		return errors.Errorf("trying to add op %s to a nil XlaBuilder", op.Type)
	}
	if op.builder != nil {
		return errors.Errorf("XlaBuilder.Op %s being added seems to have been already added to some cBuilder", op.Type)
	}
	op.builder = b
	serializedOp := serializeToC(op)
	err := errorFromStatus(C.XlaBuilderAddOp(unsafe.Pointer(b.cBuilder), serializedOp))
	if err != nil {
		return errors.Wrapf(err, "while trying to add op %s to XlaBuilder", op.Type)
	}
	op.cOp = (*C.XlaOp)(serializedOp.new_op)
	op.Shape = shapeFromCShape(serializedOp.new_shape)
	freeCSerializedOp(serializedOp)
	return nil
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

// CBufferToBytes create an unsafe buffer slice to the underlying data, if one wants to use it from Go.
func CBufferToBytes(cBuf CBuffer) []byte {
	return unsafe.Slice((*byte)(cBuf.Data()), cBuf.Size())
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
	statusOr := C.XlaBuilderSerializedHLO(unsafe.Pointer(b.cBuilder), unsafe.Pointer(outputOp.cOp))
	var err error
	var vectorData *C.VectorData
	vectorData, err = pointerOrError[C.VectorData](statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "while converting the XlaBuilder ops to a StableHLO representation")
	}
	buf := &cBuffer{
		data: unsafe.Pointer(vectorData.data),
		size: int(vectorData.count),
	}
	return buf, nil
}
