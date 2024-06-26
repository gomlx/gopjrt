package xlabuilder

import (
	"github.com/gomlx/gopjrt/dtypes"
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

// DecodeIota retrieves the arguments of an Iota op.
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
	if x == nil || x.IsNil() {
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

// ConvertDType of x to dtype.
func ConvertDType(x *Op, dtype dtypes.DType) (*Op, error) {
	op := newOp(ConvertDTypeOp, x)
	op.IntArg = int(dtype)
	err := x.builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeConvertDType retrieves the arguments for a ConvertDType op.
func DecodeConvertDType(op *Op) (dtype dtypes.DType) { return dtypes.DType(op.IntArg) }

// Where takes element-wise values from onTrue or onFalse depending on the value of condition (expected to be boolean).
func Where(condition, onTrue, onFalse *Op) (*Op, error) {
	builder := condition.builder
	op := newOp(WhereOp, condition, onTrue, onFalse)
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// Reshape reshapes x to the new dimensions.
// Total size cannot change, it's just a "reinterpretation" of the same flat data.
//
// The dtype remains the same, see ConvertDType to actually convert the values.
func Reshape(x *Op, dimensions ...int) (*Op, error) {
	newSize := 1
	for _, dim := range dimensions {
		newSize *= dim
	}
	if newSize != x.Shape.Size() {
		return nil, errors.Errorf("trying to Reshape(x, %v), where x size (%d elements) doesn't match new size of %d",
			dimensions, x.Shape.Size(), newSize)
	}
	op := newOp(ReshapeOp, x)
	op.ShapeArg = MakeShape(x.Shape.DType, dimensions...)
	err := x.builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeReshape retrieves the arguments for a Reshape op.
func DecodeReshape(op *Op) (dimensions []int) { return op.ShapeArg.Dimensions }

// Broadcast prefixes dimensions to an array by duplicating the data in the array.
// See BroadcastInDim for a broadcast in between the axes.
//
// The new dimensions dims are inserted on the left, i.e., if
// prefixDims has values `{a0, ..., aN}` and the operand shape
// has dimensions {b0, ..., bM} then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
//
// The new dimensions id into copies of the operand, i.e.
//
//	output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
func Broadcast(x *Op, prefixDims ...int) (*Op, error) {
	op := newOp(BroadcastOp, x)
	op.IntsArg = slices.Clone(prefixDims)
	err := x.builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeBroadcast retrieves the arguments for a Broadcast op.
func DecodeBroadcast(op *Op) (prefixDims []int) { return op.IntsArg }

// BroadcastInDim broadcasts x to an output with the given shape.
// broadcastAxes has an output axes value for each x axes (len(broadcastAxes) == x.Shape.Rank()).
// The i-th axis of x is mapped to the broadcastDim[i]-th dimension of the output.
// broadcastAxes must be also increasing: this operation cannot be used to transpose axes, it will only
// broadcast and introduce new axes in-between.
//
// This also requires that the i-th input dimension is either 1 or is the same as the
// output dimension it's broadcasting into.
//
// For example, say operand `x = (s32)[2]{1, 2}`; outputShape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcast_dimension will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcast_dimension
//     will generate output
//     {{1 , 1},
//     {2 , 2}}
func BroadcastInDim(x *Op, outputShape Shape, broadcastAxes []int) (*Op, error) {
	if x.Shape.DType != outputShape.DType {
		return nil, errors.Errorf("BroadcastInDim(x.shape=%s, outputShape=%s, broadcastAxes=%v): cannot change the DType of the shape", x.Shape, outputShape, broadcastAxes)
	}
	for _, dim := range outputShape.Dimensions {
		if dim <= 0 {
			return nil, errors.Errorf("BroadcastInDim(x.shape=%s, outputShape=%s, broadcastAxes=%v): cannot create a shape with an axis with dimension <= 0", x.Shape, outputShape, broadcastAxes)
		}
	}
	if x.Shape.Rank() != len(broadcastAxes) {
		return nil, errors.Errorf("BroadcastInDim(x.shape=%s, outputShape=%s, broadcastAxes=%v): there must be one broadcastAxes value for each axis of x", x.Shape, outputShape, broadcastAxes)
	}
	for xAxis, outputAxis := range broadcastAxes {
		if xAxis > 0 {
			if broadcastAxes[xAxis-1] >= outputAxis {
				return nil, errors.Errorf("BroadcastInDim(x.shape=%s, outputShape=%s, broadcastAxes=%v): broadcastAxes[%d] is out-of-order, the values must be strictly increasing", x.Shape, outputShape, broadcastAxes, xAxis)
			}
		}
		if outputAxis < 0 || outputAxis >= outputShape.Rank() {
			return nil, errors.Errorf("BroadcastInDim(x.shape=%s, outputShape=%s, broadcastAxes=%v): broadcastAxes values should be 0 <= axis < outputShape.Rank(), got axis=%d instead", x.Shape, outputShape, broadcastAxes, outputAxis)
		}
		if x.Shape.Dimensions[xAxis] != outputShape.Dimensions[outputAxis] && x.Shape.Dimensions[xAxis] != 1 {
			return nil, errors.Errorf("BroadcastInDim(x.shape=%s, outputShape=%s, broadcastAxes=%v): x axis %d (dimension=%d) is being broadcast to axis %d (dimension=%d) of the output -- x axis is changing dimension but it is not originally 1: only axes of dimension 1 can be broadcast",
				x.Shape, outputShape, broadcastAxes, xAxis, x.Shape.Dimensions[xAxis], outputAxis, outputShape.Dimensions[outputAxis])
		}
	}
	op := newOp(BroadcastInDimOp, x)
	op.ShapeArg = outputShape
	op.IntsArg = slices.Clone(broadcastAxes)
	err := x.builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeBroadcastInDim retrieves the arguments for a BroadcastInDim op.
func DecodeBroadcastInDim(op *Op) (outputShape Shape, broadcastAxes []int) {
	return op.ShapeArg, op.IntsArg
}

// Transpose axes of x.
// There should be one value in permutations for each axis in x.
// The output will have: output.Shape.Dimension[permutation[i]] = x.Shape.Dimension[i].
func Transpose(x *Op, permutations ...int) (*Op, error) {
	rank := x.Shape.Rank()
	if len(permutations) != rank {
		return nil, errors.Errorf("in TransposeAllDims(x=%s, %v), there must be one permutation per axis in x, but x rank is %d",
			x.Shape, permutations, rank)
	}
	used := make([]bool, rank)
	for xAxis, outputAxis := range permutations {
		if outputAxis >= rank || outputAxis < 0 {
			return nil, errors.Errorf("in TransposeAllDims(x=%s, %v), the permutations[%d]=%d is out-of-range",
				x.Shape, permutations, xAxis, outputAxis)
		}
		if used[outputAxis] {
			return nil, errors.Errorf("in TransposeAllDims(x=%s, %v), the output axis %d appears more than once",
				x.Shape, permutations, outputAxis)
		}
	}
	op := newOp(TransposeOp, x)
	op.IntsArg = slices.Clone(permutations)
	err := x.builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeTranspose retrieves the arguments for a Transpose op.
func DecodeTranspose(op *Op) (permutations []int) { return op.IntsArg }
