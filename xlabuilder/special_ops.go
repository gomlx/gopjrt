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
	paramOp.IntArg = paramIndex
	paramOp.StrArg = name
	paramOp.ShapeArg = shape

	err := builder.addOp(paramOp)
	if err != nil {
		return nil, err
	}
	return paramOp, nil
}

// DecodeParameter extracts the arguments to the Parameter call that created the op.
func DecodeParameter(paramOp *Op) (name string, paramIndex int, shape Shape) {
	return paramOp.StrArg, paramOp.IntArg, paramOp.ShapeArg
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

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func Iota(builder *XlaBuilder, shape Shape, iotaAxis int) (*Op, error) {
	if shape.IsScalar() {
		return nil, errors.Errorf("cannot Iota a scalar shape, shape=%s", shape)
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("invalid axis #%d for Iota, when shape is rank %d", iotaAxis, shape.Rank())
	}
	op := newOp(IotaOp)
	op.ShapeArg = shape
	op.IntArg = iotaAxis
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeIota retrieves the arguments of an IotaOp.
func DecodeIota(op *Op) (shape Shape, iotaAxis int) {
	return op.ShapeArg, op.IntArg
}

// Identity returns an Op whose output is the same as its input.
//
// It's a no-op that is not registered with the C++ XlaBuilder, it's simply serves as a place-holder
// for some arbitrary meta-data the user may want to include in the UserPayload field.
func Identity(input *Op) *Op {
	builder := input.builder
	op := newOp(IdentityOp)
	op.OpInputs = []*Op{input}
	_ = builder.addOp(op) // addOp doesn't return any errors for the identity op.
	return op
}

// Constant introduces an Op
func Constant(builder *XlaBuilder, x *Literal) (*Op, error) {
	if x == nil || x.cLiteralPtr == nil {
		return nil, errors.New("Constant() needs a non-nil literal value")
	}
	op := newOp(ConstantOp)
	op.LiteralArg = x
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}
