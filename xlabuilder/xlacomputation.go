package xlabuilder

/*
#include <gomlx/xlabuilder/utils.h>
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"gopjrt/cbuffer"
	"runtime"
	"unsafe"
)

// XlaComputation represents a computation created with XlaBuilder.
//
// It can be exported to PJRT with XlaComputation.SerializedHLO.
//
// To print the contents of the HloModuleProto, the github.com/openxla/xla repository offers a small utility called
// `run_hlo_module`. Follow the XLA build instructions and build the target `//xla/tools:run_hlo_module`.
// E.g.: if you saved your serialized HLO to a file name "my_hlo.ph", you can print it out as:
//
//	$ run_hlo_module --platform=cpu --xla_dump_hlo_as_text my_hlo.pb
//
// The `run_hlo_module` tool can also be used to run the program, export go HTML, graphviz, etc.
type XlaComputation struct {
	cComp *C.XlaComputation
}

func newXlaComputation(cComp *C.XlaComputation) *XlaComputation {
	comp := &XlaComputation{cComp: cComp}
	runtime.SetFinalizer(comp, xlaComputationFinalizer)
	return comp
}

func xlaComputationFinalizer(comp *XlaComputation) {
	comp.Free()
}

func (comp *XlaComputation) Free() {
	if comp == nil || comp.cComp == nil {
		return
	}
	C.XlaComputationDestroy(unsafe.Pointer(comp.cComp))
	comp.cComp = nil
}

// SerializedHLO generates the StableHLO program as a <serialized HLOModule proto> (something that PJRT can consume) for
// the given computation.
//
// The returned CBuffer needs to be freed (CBuffer.Free) after being used (presumably by PJRT, or saved to a file).
//
// See XlaComputation documentation on how to pretty-print the computation as text HLO.
func (comp *XlaComputation) SerializedHLO(outputOp *Op) *cbuffer.CBuffer {
	if comp == nil || comp.cComp == nil {
		return nil
	}
	var vectorData *C.VectorData
	vectorData = (*C.VectorData)(C.XlaComputationSerializedHLO(unsafe.Pointer(comp.cComp)))
	return &cbuffer.CBuffer{
		Data: unsafe.Pointer(vectorData.data),
		Size: int(vectorData.count),
	}
}

// TextHLO generates the StableHLO program as a <serialized HLOModule proto> and returns its text representation.
// It can be used for testing and debugging.
//
// Alternatively, see XlaComputation documentation on how to pretty-print the computation as text HLO.
func (comp *XlaComputation) TextHLO() string {
	if comp == nil || comp.cComp == nil {
		return ""
	}
	return cStrFree(C.XlaComputationTextHLO(unsafe.Pointer(comp.cComp)))
}
