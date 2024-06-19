package xlabuilder

// #cgo LDFLAGS: -lgomlx_xlabuilder
/*
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"github.com/pkg/errors"
	"runtime"
	"unsafe"
)

//go:generate go run ../cmd/xlabuilder_codegen

// Since CGO C types cannot cross boundaries of a package (see issue https://github.com/golang/go/issues/13467)
// We make a copy of chelper.go for every sub-directory that needs it.
//go:generate go run ../cmd/copy_go_code --original=chelper.go

// XlaBuilder is used to create "computations" (XlaComputation), that are like "StableHLO" functions.
//
// In turn XlaComputation can be exported to a serialized `HloModuleProto` (a binary blob) and used by a PJRT plugin
// (see github.com/gomlx/gopjrt/pjrt package) to compile and execute on accelerators.
//
// Once created (New), one can issue "operations" ("ops" for short), like "Add", "Mul", etc, which are recorded.
// When the computation definition is finalized, call "XlaBuilder.Build" to get the XlaComputation representing
// the function built.
// The XlaComputation can then be used with PJRT (see XlaComputation.SerializedHLO), or pretty (+/-, relatively speaking)
// print (text, HTML, graphviz, etc). See XlaComputation documentation.
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
	b := &XlaBuilder{cBuilder: cBuilder}
	runtime.SetFinalizer(b, xlaBuilderFinalizer)
	return b
}

func xlaBuilderFinalizer(b *XlaBuilder) {
	b.Free()
}

// Free must be called once the cBuilder is done to free the underlying object.
// The garbage collector won't free it by itself.
// It can be called more than once -- once finalized the first time, it becomes a no-op.
func (b *XlaBuilder) Free() {
	if b == nil || b.cBuilder == nil {
		return
	}
	C.XlaBuilderDestroy(unsafe.Pointer(b.cBuilder))
	b.cBuilder = nil
}

// addOp will add the operation described by op.
// If it succeeds it fills the fields Op.index and Op.op, with the C++ references.
func (b *XlaBuilder) addOp(op *Op) error {
	if b == nil || b.cBuilder == nil {
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

// Build builds the computation (*XlaComputation) with the requested operations (the outputOp and all its dependencies)
// or returns a non-ok status.
//
// Note that all ops that have been enqueued will be moved to the computation being returned and will no longer be valid.
func (b *XlaBuilder) Build(outputOp *Op) (*XlaComputation, error) {
	statusOr := C.XlaBuilderBuildComp(unsafe.Pointer(b.cBuilder), unsafe.Pointer(outputOp.cOp))
	var err error
	var cComp *C.XlaComputation
	cComp, err = pointerOrError[C.XlaComputation](statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "while building the computation with the XlaBuilder (outputOp=%s)", outputOp.Type)
	}
	return newXlaComputation(cComp), nil
}
