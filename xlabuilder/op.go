package xlabuilder

/*
#include <gomlx/xlabuilder/literal.h>
#include <gomlx/xlabuilder/op.h>
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"runtime"
	"slices"
	"unsafe"
)

// Op holds information about an Op that is part of a computation being built with an XlaBuilder.
//
// Each operation (e.g: Add, Mul) will return an Op that represents both the operation itself and the output
// of that operation, which can be used as input of another.
//
// While the public fields can be introspected, they shouldn't be changed, except of UserPayload.
type Op struct {
	builder *XlaBuilder
	cOp     *C.XlaOp // Pointer to a reference to the underlying C++ object. This should be deleted once Op is garbage collected.

	// Type is an OpType enum.
	Type OpType

	// Shape of the result of this Op.
	Shape Shape

	// UserPayload allows the user to add any type of meta-data. XlaBuilder simply ignores it.
	// Typically, extensions like github.com/gomlx/autodiff will cast UserPayload to the interfaces that matter to
	// them.
	UserPayload any

	// ReduceType is informative only. For some ops (ReduceMax, ScatterSum, etc.) it informs what kind of
	// standard computations were used (set in ComputationArg).
	ReduceType ReduceOpType

	// Arguments used for the various special ops:
	// TODO: Re-write these into more readable arguments to the various operations.

	// OpInputs are the inputs that are generated by other ops, these are the edges on the computation graph.
	// Other inputs are "static", meaning they are independent of the values during the calculation.
	OpInputs []*Op // Index to other nodes that are used as inputs.

	LiteralArg                           *Literal        // If a LiteralArg (constant) is involved in the operation.
	IntArg                               int             // Used for any static integer inputs.
	StrArg                               string          // Used for any static string argument.
	IntsArg                              []int           // List of integer numbers.
	FloatArg                             float32         // For a float parameter.
	ShapeArg                             Shape           // For Ops that require a shape parameter.
	ComputationArg, SecondComputationArg *XlaComputation // For Ops that require a sub-computation(s).
}

// newOp creates the Op of the given type with the given Op inputs and sets the correct finalizer.
//
// After this Op is created, other static arguments (if any) need to be set, and finally
// it needs to be added to the computation with XlaBuilder.addOp.
func newOp(opType OpType, opInputs ...*Op) *Op {
	op := &Op{
		Type:     opType,
		OpInputs: slices.Clone(opInputs),
	}
	runtime.SetFinalizer(op, opFinalizer)
	return op
}

// Builder returns the XlaBuilder associated with this Op.
func (op *Op) Builder() *XlaBuilder {
	return op.builder
}

// opFinalizer by freeing the underlying C++ resources.
func opFinalizer(op *Op) {
	if op.cOp == nil {
		return
	}
	if op.Type != IdentityOp {
		// IdentityOp borrows the cOp from its parent.
		C.XlaOpDestroy(unsafe.Pointer(op.cOp))
	}
	op.cOp = nil
}

// serializeToC convert an Op not yet added to XlaBuilder to a C.SerializedOp that can be used by the C-wrapper library.
func serializeToC(op *Op) *C.SerializedOp {
	// Allocate and set C.SerializedOps struct using Go memory. It can be discarded when this function exit.
	numInputs := len(op.OpInputs)
	var sOp *C.SerializedOp
	sOp = &C.SerializedOp{
		op_type:       C.int32_t(op.Type),
		num_op_inputs: C.int32_t(numInputs),
		integer:       C.int64_t(op.IntArg),
		float_v:       C.float(op.FloatArg),
	}
	if numInputs > 0 {
		// Create the `inputs` array.
		sOp.op_inputs = cMallocArrayAndSet[C.XlaOpPtr](numInputs, func(ii int) C.XlaOpPtr {
			return (C.XlaOpPtr)(unsafe.Pointer(op.OpInputs[ii].cOp))
		})
	}
	sOp.shape = cShapeFromShape(op.ShapeArg)
	if op.StrArg != "" {
		sOp.string = C.CString(op.StrArg)
	}
	if !op.LiteralArg.IsNil() {
		sOp.literal = op.LiteralArg.cLiteral
	}
	if len(op.IntsArg) > 0 {
		sOp.integer_array_size = C.int32_t(len(op.IntsArg))
		sOp.integer_array = cMallocArrayAndSet[C.int64_t](len(op.IntsArg), func(ii int) C.int64_t { return C.int64_t(op.IntsArg[ii]) })
	}
	if op.ComputationArg != nil {
		sOp.computation = unsafe.Pointer(op.ComputationArg.cXlaComputation)
	}
	if op.SecondComputationArg != nil {
		sOp.second_computation = unsafe.Pointer(op.SecondComputationArg.cXlaComputation)
	}
	return sOp
}

// destroyCSerializedOp destroys cOp, freeing all contained C objects.
// Note that cOp itself is assumed to be allocated in Go space, hence it is (and should be) automatically garbage collected.
func destroyCSerializedOp(cOp *C.SerializedOp) {
	if cOp.op_inputs != nil {
		cFree(cOp.op_inputs)
		cOp.op_inputs = nil
		cOp.num_op_inputs = 0
	}
	if cOp.shape != nil {
		C.DeleteShape(cOp.shape)
		cOp.shape = nil
	}
	if cOp.new_shape != nil {
		C.DeleteShape(cOp.new_shape)
		cOp.new_shape = nil
	}
	if cOp.string != nil {
		cFree(cOp.string)
		cOp.string = nil
	}
	if cOp.integer_array != nil {
		C.free(unsafe.Pointer(cOp.integer_array))
		cOp.integer_array = nil
		cOp.integer_array_size = 0
	}
}
