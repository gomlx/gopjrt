package xlabuilder

import (
	"github.com/pkg/errors"
	"slices"
)

// Manual implementation of the special ops.

// Parameter creates a "retrieves a parameter value" op in builder.
//
// The name is cosmetic, but should be unique among the parameters.
//
// The paramIndex must be carefully set to match the parameters fed to the computation during execution and after
// it is compiled (see package pjrt for that).
//
// The shape of the parameter must be given -- and match the value given during execution.
func Parameter(builder *XlaBuilder, name string, paramIndex int, shape Shape) (*Op, error) {
	paramOp := newOp(ParameterOp)

	// Convert to the compact Op parameters form.
	paramOp.Int = paramIndex
	paramOp.Str = name
	paramOp.ShapeArg = shape

	err := builder.addOp(paramOp)
	if err != nil {
		return nil, err
	}
	return paramOp, nil
}

// DecodeParameter extracts the arguments to the Parameter call that created the op.
func DecodeParameter(paramOp *Op) (name string, paramIndex int, shape Shape) {
	return paramOp.Str, paramOp.Int, paramOp.ShapeArg
}

// Tuple organizes multiple nodes in one tuple-node.
//
// This is particularly useful to get multiple outputs to a computation.
func Tuple(inputs ...*Op) (*Op, error) {
	builder := inputs[0].builder
	for ii, input := range inputs {
		if ii == 0 {
			continue
		}
		if input.builder != builder {
			return nil, errors.Errorf("arguments 0 and %d of Tuple(inputs...) come from different XlaBuilder objects (or nil)", ii)
		}
	}
	tupleOp := newOp(TupleOp)
	tupleOp.OpInputs = slices.Clone(inputs)
	err := builder.addOp(tupleOp)
	if err != nil {
		return nil, err
	}
	return tupleOp, nil
}
