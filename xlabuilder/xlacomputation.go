package xlabuilder

/*
#include <gomlx/xlabuilder/utils.h>
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"github.com/gomlx/gopjrt/internal/cbuffer"
	"github.com/pkg/errors"
	"runtime"
	"unsafe"
)

// XlaComputation represents a computation created with XlaBuilder.
//
// It can be used as is by pjrt.Client.Compile or serialized (to be saved) with XlaComputation.SerializedHLO.
//
// It is also used as a "subroutine" for other XlaBuilder ops, like Reduce, which takes the computation to use for
// reduction.
//
// To print the contents of the HloModuleProto, the github.com/openxla/xla repository offers a small utility called
// `run_hlo_module`. Follow the XLA build instructions and build the target `//xla/tools:run_hlo_module`.
// E.g.: if you saved your serialized HLO to a file name "my_hlo.ph", you can print it out as:
//
//	$ run_hlo_module --platform=cpu --xla_dump_hlo_as_text my_hlo.pb
//
// The `run_hlo_module` tool can also be used to run the program, export go HTML, graphviz, etc.
type XlaComputation struct {
	cXlaComputation *C.XlaComputation
	name            string
}

func newXlaComputation(cComp *C.XlaComputation) *XlaComputation {
	comp := &XlaComputation{
		cXlaComputation: cComp,
		name:            cStrFree(C.XlaComputationName(unsafe.Pointer(cComp))),
	}
	runtime.SetFinalizer(comp, func(comp *XlaComputation) { comp.Destroy() })
	return comp
}

// Destroy immediately the underlying (C/C++) XlaComputation.
// This is called automatically at garbage-collection.
func (comp *XlaComputation) Destroy() {
	if comp == nil || comp.cXlaComputation == nil {
		return
	}
	C.XlaComputationDestroy(unsafe.Pointer(comp.cXlaComputation))
	comp.cXlaComputation = nil
}

// IsNil returns whether the computation or the underlying C/C++ object are nil.
// It's true after it is destroyed.
func (comp *XlaComputation) IsNil() bool {
	return comp == nil || comp.cXlaComputation == nil
}

// Name returns the name assigned to the computation (given at the builder construction).
func (comp *XlaComputation) Name() string {
	return comp.name
}

// SerializedHLO generates the StableHLO program as a <serialized HLOModule proto> (something that PJRT can consume) for
// the given computation.
//
// The returned CBuffer needs to be freed (CBuffer.Destroy) after being used (presumably by PJRT, or saved to a file).
//
// See XlaComputation documentation on how to pretty-print the computation as text HLO.
func (comp *XlaComputation) SerializedHLO() *cbuffer.CBuffer {
	if comp.IsNil() {
		panicf("XlaComputation is nil, maybe it has already been destroyed?")
	}
	defer runtime.KeepAlive(comp)
	if comp == nil || comp.cXlaComputation == nil {
		return nil
	}
	var vectorData *C.VectorData
	vectorData = (*C.VectorData)(C.XlaComputationSerializedHLO(unsafe.Pointer(comp.cXlaComputation)))
	cBuf := cbuffer.New(unsafe.Pointer(vectorData.data), int(vectorData.count), true)
	C.free(unsafe.Pointer(vectorData))
	return cBuf
}

// HasStableHLO returns whether StableHLO support was included in the build -- it's very large, so by default it is not.
func HasStableHLO() bool {
	return bool(C.HasStableHLO)
}

// HasStableHLO returns whether StableHLO support was included in the build -- it's very large, so by default it is not.
func (comp *XlaComputation) HasStableHLO() bool {
	return HasStableHLO()
}

// SerializedStableHLO exports the computation as a StableHLO as an `mlir:ModuleOp`.
//
// It does that by converting the `HLOModule` proto to an `mlir:ModuleOp`.
//
// This functionality is not included by default -- linking StableHLO will include LLVM and make the XlaBuilder
// library literally 10 times larger. If not included, it will return an error.
func (comp *XlaComputation) SerializedStableHLO() (*cbuffer.CBuffer, error) {
	if comp.IsNil() {
		panicf("XlaComputation is nil, maybe it has already been destroyed?")
	}
	if !HasStableHLO() {
		return nil, errors.New("StableHLO support was not included in this build")
	}

	defer runtime.KeepAlive(comp)
	var vectorData *C.VectorData
	statusOr := C.XlaComputationSerializedStableHLO(unsafe.Pointer(comp.cXlaComputation))
	vectorData, err := pointerOrError[C.VectorData](statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "while converting XlaComputation to StableHLO")
	}
	cBuf := cbuffer.New(unsafe.Pointer(vectorData.data), int(vectorData.count), true)
	C.free(unsafe.Pointer(vectorData))
	return cBuf, nil
}

// TextHLO generates the HLO program as a <serialized HLOModule proto> and returns its text representation.
// It can be used for testing and debugging.
//
// Alternatively, see XlaComputation documentation on how to pretty-print the computation as text HLO.
func (comp *XlaComputation) TextHLO() string {
	if comp.IsNil() {
		panicf("XlaComputation is nil, maybe it has already been destroyed?")
	}
	defer runtime.KeepAlive(comp)
	if comp == nil || comp.cXlaComputation == nil {
		return ""
	}
	return cStrFree(C.XlaComputationTextHLO(unsafe.Pointer(comp.cXlaComputation)))
}

// TextStableHLO generates the StableHLO program.
//
// It returns an error if StableHLO code was not linked in (it's large). This can be checked with HasStableHLO.
func (comp *XlaComputation) TextStableHLO() (string, error) {
	if comp.IsNil() {
		return "", errors.New("XlaComputation is nil, maybe it has already been destroyed?")
	}
	if !HasStableHLO() {
		return "", errors.New("StableHLO support was not included in this build")
	}
	defer runtime.KeepAlive(comp)

	statusOr := C.XlaComputationStableHLOText(unsafe.Pointer(comp.cXlaComputation))
	var err error
	var cText *C.char
	cText, err = pointerOrError[C.char](statusOr)
	if err != nil {
		return "", errors.Wrapf(err, "while converting XlaComputation to StableHLO")
	}
	return cStrFree(cText), nil
}
