package xlabuilder

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/proto"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
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
	tupleOp := newOp(TupleOp, inputs...)
	err := builder.addOp(tupleOp)
	if err != nil {
		return nil, err
	}
	return tupleOp, nil
}

// GetTupleElement extracts one element from a Tuple.
func GetTupleElement(input *Op, elementIdx int) (*Op, error) {
	builder := input.builder
	op := newOp(GetTupleElementOp, input)
	op.IntArg = elementIdx
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeGetTupleElement retrieves the arguments of an GetTupleElement op.
func DecodeGetTupleElement(op *Op) (elementIdx int) {
	return op.IntArg
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
	if builder.IsNil() {
		exceptions.Panicf("trying to access XlaBuilder that is nil or already destroyed")
	}
	op := newOp(IdentityOp)
	op.OpInputs = []*Op{input}
	_ = builder.addOp(op) // addOp doesn't return any errors for the identity op.
	return op
}

// Constant introduces an Op
func Constant(builder *XlaBuilder, x *Literal) (*Op, error) {
	if builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
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
	if x.builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
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
	if condition.builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
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
	if x.builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
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
	if x.builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
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

// Call will evaluate a subComputation with the given operands.
// The given subComputation must have been created with a sub-builder (see XlaBuilder.CreateSubBuilder) of the given
// builder.
func Call(builder *XlaBuilder, subComputation *XlaComputation, operands ...*Op) (*Op, error) {
	if builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
	op := newOp(CallOp, operands...)
	op.ComputationArg = subComputation
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// Concatenate results on the given axis.
//
// All axes that are not being concatenated must match dimensions.
// It doesn't work with scalars -- use ExpandDims.
//
// If there is only one operand, it is returned and this is a no-op.
func Concatenate(axis int, operands ...*Op) (*Op, error) {
	if len(operands) == 0 {
		return nil, errors.New("cannot Concatenate with 0 operands")
	}
	if len(operands) == 1 {
		// Trivial solution.
		return operands[0], nil
	}
	builder := operands[0].builder
	op := newOp(ConcatenateOp, operands...)
	op.IntArg = axis
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeConcatenate retrieves the arguments for a Concatenate op.
func DecodeConcatenate(op *Op) (axis int) { return op.IntArg }

// Slice extracts a sub-array from the input array.
// The sub-array is of the same rank as the input and contains the values inside a bounding box within the input array
// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
//
// The strides set the input stride of the slice in each axis and must be >= 1.
// It is optional, and if missing it is assumed to be 1 for every dimension.
//
// Examples:
//
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
func Slice(x *Op, starts, limits, strides []int) (*Op, error) {
	builder := x.builder
	rank := x.Shape.Rank()
	if len(strides) == 0 {
		strides = make([]int, rank)
		for ii := range strides {
			strides[ii] = 1
		}
	}
	if len(starts) != rank || len(limits) != rank || len(strides) != rank {
		return nil, errors.Errorf("in SliceWithStridesXLA(x, starts, limits, strides) passed %d start values, %d limits values and %d stride values, but x has rank %d -- they must all match.", len(starts), len(limits), len(strides), rank)
	}

	// Encode starts, limits and strides sequentially, since their size are the same,
	// it will be easy to separate them in Const++.
	op := newOp(SliceOp, x)
	op.IntsArg = make([]int, 0, 3*rank)
	op.IntsArg = append(op.IntsArg, starts...)
	op.IntsArg = append(op.IntsArg, limits...)
	op.IntsArg = append(op.IntsArg, strides...)
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeSlice retrieves the arguments for a Slice op.
func DecodeSlice(op *Op) (starts, limits, strides []int) {
	rank := op.OpInputs[0].Shape.Rank()
	if len(op.IntsArg) != 3*rank {
		exceptions.Panicf("DecodeSlice() has input of rank %d, but arguments don't have 3*%d elements, instead got %d",
			rank, 3*rank, len(op.IntsArg))
	}
	starts = op.IntsArg[0:rank]
	limits = op.IntsArg[rank : 2*rank]
	strides = op.IntsArg[2*rank:]
	return
}

func boolToInt(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}

// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
//
// It's a form of reduction on the given axis, and that axis goes away. So the rank of the result is one less than
// the rank of x.
//
// Examples:
//
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it choose the 2, 4 and 7)
func ArgMinMax(x *Op, axis int, outputDType dtypes.DType, isMin bool) (*Op, error) {
	builder := x.builder
	rank := x.Shape.Rank()
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("in ArgMinMax(): axis=%d must be between 0 and x.Shape.Rank()=%d", axis, rank)
	}
	op := newOp(ArgMinMaxOp, x)
	op.IntArg = axis
	op.IntsArg = []int{boolToInt(isMin), int(outputDType.PrimitiveType())}
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeArgMinMax retrieves the arguments for a ArgMinMax op.
func DecodeArgMinMax(op *Op) (axis int, outputDType dtypes.DType, isMin bool) {
	axis = op.IntArg
	isMin = op.IntsArg[0] != 0
	outputDType = dtypes.FromPrimitiveType(proto.PrimitiveType(op.IntsArg[1]))
	return
}

// PadAxis defines the amount of padding preceding one axis (Start), at the end of axis (End)
// or in between the inputs (Interior).
// This is used as a parameter for the Pad operation.
type PadAxis struct {
	Start, End, Interior int
}

// Pad injects padding on the start, end or interior (in between each element) of the given operand.
// There must be at most `operand.Rank()` axesConfig values. Missing PadAxis are assumed to be zeros,
// that is, no padding for those axes.
func Pad(x, fillValue *Op, axesConfig ...PadAxis) (*Op, error) {
	builder := x.builder
	rank := x.Shape.Rank()
	if rank == 0 {
		return nil, errors.New("cannot use Pad() with scalar values")
	}
	if len(axesConfig) != rank {
		return nil, errors.Errorf("Pad() requires one axis configuration per x axis: x rank is %d, and %d PadAxis configurations were given",
			rank, len(axesConfig))
	}
	op := newOp(PadOp, x, fillValue)
	op.IntsArg = make([]int, 0, 3*rank)
	for axis := 0; axis < rank; axis++ {
		var padding PadAxis
		if axis < len(axesConfig) {
			padding = axesConfig[axis]
		}
		op.IntsArg = append(op.IntsArg, padding.Start, padding.End, padding.Interior)
	}
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodePad retrieves the arguments for a Pad op.
func DecodePad(op *Op) (axesConfig []PadAxis) {
	rank := op.OpInputs[0].Shape.Rank()
	axesConfig = make([]PadAxis, rank)
	for axisIdx := range axesConfig {
		ii := 3 * axisIdx
		axesConfig[axisIdx] = PadAxis{
			Start:    op.IntsArg[ii],
			End:      op.IntsArg[ii+1],
			Interior: op.IntsArg[ii+2],
		}
	}
	return
}

// Gather is a powerful but cumbersome Gather operation offered by XLA.
// Full details in https://www.tensorflow.org/xla/operation_semantics#gather.
// (Warning: it's poorly described, with many undefined terms)
//
// Arguments:
//   - startIndices: are the indices we want to gather. There will be one axis with which enumerates the indices
//     in the operand array, typically the last one. All other axes are "batch dimensions" and they will have
//     equivalent axes in the output.
//   - indexVectorAxis: typically the last axis of startIndices, so startIndices.Shape.Rank()-1.
//     Usually, one has the dimension of the indexVectorAxis equal to the full rank of the operand.
//     That is: startIndices.Shape.Dimensions[indexVectorAxis] = operand.Shape.Rank()
//     Lets call "one index vector" a value of startIndices formed by a slice across indexVectorAxis.
//   - startIndexMap: for each "index vector" from startIndices, this maps each element of the vector goes to
//     which axes of the operand. Typically, this is [0, 1, 2, ..., operand.Shape.Rank()-1], that is, each
//     "index vector" fully defines an element on the operand. If one is gathering slices of the operand (as
//     opposed to individual values), one can skip some of those axes from startIndexMap, and the index for those
//     axis is considered 0, and set sliceSizes to take the slice one wants (typically the full slice).
//   - sliceSizes: the "index vector" described above points to the data in the operand to be gathered. Then sliceSizes
//     indicates how much data to gather. One value per axis of the operand must be set. For gathering individual
//     values, set these all to 1.
//   - collapsedSliceAxes: the slice gathered for each "index vector" (with sizes sliceSizes), often has dimension one
//     for most (or all, in case of gathering individual items) axes. collapsedSliceAxes allows one to collapse those
//     axes, so they don't show up in the output. Usually, collapse all axes that are size one.
//     These are axes within the rank of operand (from 0 to operand.Shape.Rank()-1).
//   - offsetAxes: for those gathered slices not collapsed (with collapsedSliceAxes), this maps them to a position in
//     the output array. Typically, these will be consecutive numbers starting with indexVectorAxis. So, the output
//     will have the same prefix shape (the "batch dimensions") as the startIndices array, and the suffix shape will
//     be the gathered slices mapped to these `offsetAxes`. There must be one value per axis not collapsed with
//     collapsedSliceAxes -- the value itself is an axis in the output shape.
func Gather(operand, startIndices *Op, indexVectorAxis int, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int, indicesAreSorted bool) (*Op, error) {
	builder := operand.builder
	rank := operand.Shape.Rank()
	if rank == 0 {
		return nil, errors.New("cannot use Gather() with scalar values")
	}
	op := newOp(GatherOp, operand, startIndices)

	if klog.V(2).Enabled() {
		klog.Infof("\tGather(operand=%s, start=%s, indexVectorAxis=%d, offsetAxes=%v, collapsedSliceAxes=%v, startIndexMap=%v, sliceSizes=%v\n",
			operand.Shape, startIndices.Shape, indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes)
	}

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/xlabuilder/xlabuilder.cpp, in function XlaBuilderAddOp, under GatherOp case,
	// and with DecodeGather below.
	//
	//  * 6 first elements store the various parameters and lengths:
	op.IntsArg = make([]int, 6+len(offsetAxes)+len(collapsedSliceAxes)+len(startIndexMap)+len(sliceSizes))
	op.IntsArg[0] = indexVectorAxis
	op.IntsArg[1] = len(offsetAxes)
	op.IntsArg[2] = len(collapsedSliceAxes)
	op.IntsArg[3] = len(startIndexMap)
	op.IntsArg[4] = len(sliceSizes)
	op.IntsArg[5] = boolToInt(indicesAreSorted)

	//  * Copy sequentially the contents of the 3 int arrays:
	pos := 6
	for _, slice := range [][]int{offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes} {
		if len(slice) > 0 {
			copy(op.IntsArg[pos:], slice)
			pos += len(slice)
		}
	}

	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeGather retrieves the arguments for a Gather op.
func DecodeGather(op *Op) (indexVectorAxis int, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int, indicesAreSorted bool) {
	indexVectorAxis = op.IntsArg[0]
	if op.IntsArg[1] > 0 {
		offsetAxes = make([]int, op.IntsArg[1])
	}
	if op.IntsArg[2] > 0 {
		collapsedSliceAxes = make([]int, op.IntsArg[2])
	}
	if op.IntsArg[3] > 0 {
		startIndexMap = make([]int, op.IntsArg[3])
	}
	if op.IntsArg[4] > 0 {
		sliceSizes = make([]int, op.IntsArg[4])
	}
	indicesAreSorted = op.IntsArg[5] != 0

	//  * Copy sequentially the contents of the 3 int arrays:
	pos := 6
	for _, slice := range [][]int{offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes} {
		if len(slice) > 0 {
			copy(slice, op.IntsArg[pos:])
			pos += len(slice)
		}
	}
	return
}

// ScatterCustom is a powerful but cumbersome Scatter operation offered by XLA.
// Full details in https://www.tensorflow.org/xla/operation_semantics#scatter.
//
// It takes a custom updateComputation used when scattering values.
// See ScatterAdd for a version that adds the values when scattering.
func ScatterCustom(operand, scatterIndices, updates *Op,
	updateComputation *XlaComputation,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (*Op, error) {
	builder := operand.builder
	if operand.Shape.Rank() == 0 {
		return nil, errors.New("cannot use ScatterCustom() with scalar operand")
	}

	if klog.V(0).Enabled() {
		klog.Infof("\tScatterCustom: operand=%s, scatterIndices=%s, updates=%s, indexVectorAxis=%d, updateWindowAxes=%v, insertedWindowAxes=%v, scatterAxesToOperandAxes=%v, indicesAreSorted=%v, uniqueIndices=%v\n",
			operand.Shape, scatterIndices.Shape, updates.Shape, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
	}

	op := newOp(ScatterOp, operand, scatterIndices, updates)
	op.ComputationArg = updateComputation

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/xlabuilder/xlabuilder.cpp, in function XlaBuilderAddOp, under ScatterOp case.
	// And with DecodeScatterCustom bellow.
	//
	//  * 6 first elements store the various parameters and lengths:
	op.IntsArg = make([]int, 0, 6+len(updateWindowAxes)+len(insertedWindowAxes)+len(scatterAxesToOperandAxes))
	op.IntsArg = append(op.IntsArg, indexVectorAxis)
	op.IntsArg = append(op.IntsArg, boolToInt(indicesAreSorted))
	op.IntsArg = append(op.IntsArg, boolToInt(uniqueIndices))
	op.IntsArg = append(op.IntsArg, len(updateWindowAxes))
	op.IntsArg = append(op.IntsArg, len(insertedWindowAxes))
	op.IntsArg = append(op.IntsArg, len(scatterAxesToOperandAxes))
	for _, slice := range [][]int{updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes} {
		op.IntsArg = append(op.IntsArg, slice...)
	}

	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// ScatterAdd values from updates pointed by scatterIndices to operand.
// Details in ScatterCustom, which is used with the updateComputation set to Sum.
func ScatterAdd(operand, scatterIndices, updates *Op,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (*Op, error) {
	return scatterImpl(operand, scatterIndices, updates, ReduceSumType, indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterMax scatter values from updates pointed by scatterIndices to operand, by taking the Max.
// Details in ScatterCustom, which is used with the updateComputation set to Max.
func ScatterMax(operand, scatterIndices, updates *Op,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (*Op, error) {
	return scatterImpl(operand, scatterIndices, updates, ReduceMaxType, indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterMin scatter values from updates pointed by scatterIndices to operand, by taking the Min.
// Details in ScatterCustom, which is used with the updateComputation set to Min.
func ScatterMin(operand, scatterIndices, updates *Op,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (*Op, error) {
	return scatterImpl(operand, scatterIndices, updates, ReduceMinType, indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// scatterImpl is a helper function for ScatterAdd, ScatterMax, ScatterMin.
func scatterImpl(operand, scatterIndices, updates *Op,
	reduceType ReduceOpType,
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) (*Op, error) {
	builder := operand.builder
	reduceComputation, _, err := builder.GetReduceComputationAndInitialValue(reduceType, operand.Shape.DType)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to get update computation %s for scatter", reduceType)
	}
	return ScatterCustom(operand, scatterIndices, updates, reduceComputation, indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// DecodeScatter retrieves the arguments for a Scatter (ScatterCustom or ScatterAdd) op.
func DecodeScatter(op *Op) (
	indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool) {
	indexVectorAxis = op.IntsArg[0]
	indicesAreSorted = op.IntsArg[1] != 0
	uniqueIndices = op.IntsArg[2] != 0
	if op.IntsArg[3] > 0 {
		updateWindowAxes = make([]int, op.IntsArg[3])
	}
	if op.IntsArg[4] > 0 {
		insertedWindowAxes = make([]int, op.IntsArg[4])
	}
	if op.IntsArg[5] > 0 {
		fmt.Printf("> len(scatterAxesToOperandAxes)=%d\n", op.IntsArg[5])
		scatterAxesToOperandAxes = make([]int, op.IntsArg[5])
	}
	//  * Copy sequentially the contents of the 3 int arrays:
	pos := 6
	for _, slice := range [][]int{updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes} {
		if len(slice) > 0 {
			copy(slice, op.IntsArg[pos:])
			pos += len(slice)
		}
	}
	return
}
