package xlabuilder

/*
#include <gomlx/xlabuilder/utils.h>
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gopjrt/cbuffer"
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

// SerializedHLO generates the StableHLO program as a <serialized HLOModule proto> (something that PJRT can consume) for
// the given computation.
//
// The returned CBuffer needs to be freed (CBuffer.Destroy) after being used (presumably by PJRT, or saved to a file).
//
// See XlaComputation documentation on how to pretty-print the computation as text HLO.
func (comp *XlaComputation) SerializedHLO() *cbuffer.CBuffer {
	if comp.IsNil() {
		exceptions.Panicf("XlaComputation is nil, maybe it has already been destroyed?")
	}
	defer runtime.KeepAlive(comp)
	if comp == nil || comp.cXlaComputation == nil {
		return nil
	}
	var vectorData *C.VectorData
	vectorData = (*C.VectorData)(C.XlaComputationSerializedHLO(unsafe.Pointer(comp.cXlaComputation)))
	return cbuffer.New(unsafe.Pointer(vectorData.data), int(vectorData.count), true)
}

// TextHLO generates the StableHLO program as a <serialized HLOModule proto> and returns its text representation.
// It can be used for testing and debugging.
//
// Alternatively, see XlaComputation documentation on how to pretty-print the computation as text HLO.
func (comp *XlaComputation) TextHLO() string {
	if comp.IsNil() {
		exceptions.Panicf("XlaComputation is nil, maybe it has already been destroyed?")
	}
	defer runtime.KeepAlive(comp)
	if comp == nil || comp.cXlaComputation == nil {
		return ""
	}
	return cStrFree(C.XlaComputationTextHLO(unsafe.Pointer(comp.cXlaComputation)))
}
