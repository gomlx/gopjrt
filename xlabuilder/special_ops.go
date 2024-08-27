package xlabuilder

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/protos/xla_data"
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

// SplitTuple is a convenience wrapper around GetTupleElement, it will return an array with all the nodes.
func SplitTuple(tuple *Op) ([]*Op, error) {
	numElements := tuple.Shape.TupleSize()
	if numElements == 0 {
		return nil, errors.Errorf("value passed to SplitTuple is not a tuple, shape=%s", tuple.Shape)
	}
	split := make([]*Op, numElements)
	var err error
	for ii := 0; ii < numElements; ii++ {
		split[ii], err = GetTupleElement(tuple, ii)
		if err != nil {
			return nil, err
		}
	}
	return split, nil
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
		panicf("trying to access XlaBuilder that is nil or already destroyed")
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

// ScalarZero returns a zero constant for the given dtype.
// It caches the constant, so it doesn't get defined multiple times.
func ScalarZero(builder *XlaBuilder, dtype dtypes.DType) (*Op, error) {
	cacheKey := fmt.Sprintf("#_zero_%s", dtype)
	value := builder.cachedStandardConstants[cacheKey]
	if value != nil {
		return value, nil
	}
	literal, err := NewScalarLiteralFromFloat64(0, dtype)
	if err != nil {
		return nil, errors.WithMessagef(err, "while trying to create a %s zero constant", dtype)
	}
	value, err = Constant(builder, literal)
	if err != nil {
		return nil, errors.WithMessagef(err, "while trying to create a %s zero constant", dtype)
	}
	builder.cachedStandardConstants[cacheKey] = value
	return value, nil
}

// ScalarOne returns a one (1) constant for the given dtype.
// It caches the constant, so it doesn't get defined multiple times.
func ScalarOne(builder *XlaBuilder, dtype dtypes.DType) (*Op, error) {
	cacheKey := fmt.Sprintf("#_one_%s", dtype)
	value := builder.cachedStandardConstants[cacheKey]
	if value != nil {
		return value, nil
	}
	literal, err := NewScalarLiteralFromFloat64(1, dtype)
	if err != nil {
		return nil, errors.WithMessagef(err, "while trying to create a %s one constant", dtype)
	}
	value, err = Constant(builder, literal)
	if err != nil {
		return nil, errors.WithMessagef(err, "while trying to create a %s one constant", dtype)
	}
	builder.cachedStandardConstants[cacheKey] = value
	return value, nil
}

// ConvertDType of x to dtype.
func ConvertDType(x *Op, dtype dtypes.DType) (*Op, error) {
	if x.builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
	op := newOp(ConvertDTypeOp, x)
	op.IntArg = int(dtype.PrimitiveType())
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
	if onTrue.Shape.DType != onFalse.Shape.DType {
		return nil, errors.Errorf("dtype of onTrue (%s) and onFalse (%s) don't match", onTrue.Shape.DType, onFalse.Shape.DType)
	}
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
// The i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
// broadcastAxes must be also increasing: this operation cannot be used to transpose axes, it will only
// broadcast and introduce new axes in-between.
//
// This also requires that the i-th input axis is either 1 or is the same as the
// output dimension it's broadcasting into.
//
// For example, say operand `x = (s32)[2]{1, 2}`; outputShape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcastAxes will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcastAxes
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
// The output will have: output.Shape.Dimension[ii] = x.Shape.Dimension[permutations[i]].
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
	dtype := operands[0].Shape.DType
	for ii, op := range operands {
		if op.Shape.DType != dtype {
			return nil, errors.Errorf("Concatenate operand 0 has dtype %s, by operand %d has dtype %s: dtypes must match",
				dtype, ii, op.Shape.DType)
		}
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
		panicf("DecodeSlice() has input of rank %d, but arguments don't have 3*%d elements, instead got %d -- probably the op as not a SliceOp?",
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
	outputDType = dtypes.FromPrimitiveType(xla_data.PrimitiveType(op.IntsArg[1]))
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
	if x.Shape.DType != fillValue.Shape.DType {
		return nil, errors.Errorf("operand and fillValue dtypes (%s and %s) don't match for Pad()", x.Shape.DType, fillValue.Shape.DType)
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
	if operand.Shape.DType != updates.Shape.DType {
		return nil, errors.Errorf("Scatter operand (dtype=%s) and updates (dtype=%s) have different dtypes",
			operand.Shape.DType, updates.Shape.DType)
	}

	if klog.V(2).Enabled() {
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
	op, err := ScatterCustom(operand, scatterIndices, updates, reduceComputation, indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
	if err != nil {
		return nil, err
	}
	op.ReduceType = reduceType
	return op, nil
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

// SelectAndScatterCustom runs windows (similar to ReduceWindow) over the operand, selects values (selectComputation) to updates the output (like Scatter)
// using the scatterComputation with values from source. The output is initialized with defaultValue.
// See details in https://openxla.org/xla/operation_semantics#selectandscatter
func SelectAndScatterCustom(operand, source, defaultValue *Op, selectComputation, scatterComputation *XlaComputation,
	windowDimensions, windowStrides []int, paddings [][2]int) (*Op, error) {
	builder := operand.builder
	dtype := operand.Shape.DType
	rank := operand.Shape.Rank()
	if operand.Shape.Rank() == 0 {
		return nil, errors.New("cannot use SelectAndScatterCustom() with scalar operand")
	}
	if source.Shape.DType != dtype || defaultValue.Shape.DType != dtype {
		return nil, errors.Errorf("SelectAndScatter operand (dtype=%s), source (dtype=%s) and defaultValue (dtype=%s) must all have the same dtype",
			operand.Shape.DType, source.Shape.DType, defaultValue.Shape.DType)
	}
	if len(windowDimensions) != rank {
		return nil, errors.Errorf("SelectAndScatter windowSizes (length %d) must have same length as the rank of the operand (rank %d)",
			len(windowDimensions), rank)
	}
	if len(windowStrides) != rank {
		return nil, errors.Errorf("SelectAndScatter windowStrides (length %d) must have same length as the rank of the operand (rank %d)",
			len(windowStrides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		return nil, errors.Errorf("SelectAndScatter paddings (length %d) must either be empty or have same length as the rank of the operand (rank %d)",
			len(paddings), rank)
	}

	if klog.V(2).Enabled() {
		klog.Infof("SelectAndScatterCustom(operand=%s, source=%s, defaultValue=%s, selectComputation=%s, scatterComputation=%s, "+
			"windowDimensions=%v, windowStrides=%v, paddings=%v)",
			operand.Shape, source.Shape, defaultValue.Shape,
			selectComputation.Name(), scatterComputation.Name(),
			windowDimensions, windowStrides, paddings)
	}

	op := newOp(SelectAndScatterOp, operand, source, defaultValue)
	op.ComputationArg = selectComputation
	op.SecondComputationArg = scatterComputation

	op.IntsArg = make([]int, 0, 2+2*rank+2*len(paddings))
	encode := func(values ...int) {
		op.IntsArg = append(op.IntsArg, values...)
	}
	encode(rank, len(paddings))
	encode(windowDimensions...)
	encode(windowStrides...)
	for _, pair := range paddings {
		encode(pair[0], pair[1])
	}

	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// SelectAndScatterMax calls SelectAndScatterCustom with a zero defaultValue, sum for updateComputation and an appropriate selectComputation
// to implement a SelectAndScatter that updates the max value in the windows.
// Details in SelectAndScatterCustom.
func SelectAndScatterMax(operand, source *Op,
	windowDimensions, windowStrides []int, paddings [][2]int) (*Op, error) {
	reduceType := ReduceMaxType
	return selectAndScatterImpl(operand, source, reduceType, windowDimensions, windowStrides, paddings)
}

// SelectAndScatterMin calls SelectAndScatterCustom with a zero defaultValue, sum for updateComputation and an appropriate selectComputation
// to implement a SelectAndScatter that updates the max value in the windows.
// Details in SelectAndScatterCustom.
func SelectAndScatterMin(operand, source *Op,
	windowDimensions, windowStrides []int, paddings [][2]int) (*Op, error) {
	reduceType := ReduceMinType
	return selectAndScatterImpl(operand, source, reduceType, windowDimensions, windowStrides, paddings)
}

// SelectAndScatterSum calls SelectAndScatterCustom with a zero defaultValue, sum for updateComputation and a selectComputation that always selects.
// to implement a SelectAndScatter that updates the max value in the windows.
// Details in SelectAndScatterCustom.
func SelectAndScatterSum(operand, source *Op,
	windowDimensions, windowStrides []int, paddings [][2]int) (*Op, error) {
	reduceType := ReduceSumType
	return selectAndScatterImpl(operand, source, reduceType, windowDimensions, windowStrides, paddings)
}

// selectAndScatterImpl calls SelectAndScatterCustom with initialValue, selectComp and scatterComp specialized
// for a reduceType (ReduceMaxType, ReduceMinType, ReduceAddType).
func selectAndScatterImpl(operand, source *Op, reduceType ReduceOpType,
	windowDimensions, windowStrides []int, paddings [][2]int) (*Op, error) {
	builder := operand.builder
	dtype := operand.Shape.DType
	selectComputation, scatterComputation, err := builder.GetSelectAndScatterComputation(reduceType, dtype)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to get select and scatter computations %q for SelectAnScatter operation", reduceType)
	}
	zero, err := ScalarZero(builder, dtype)
	if err != nil {
		return nil, err
	}
	op, err := SelectAndScatterCustom(operand, source, zero, selectComputation, scatterComputation,
		windowDimensions, windowStrides, paddings)
	if err != nil {
		return nil, err
	}
	op.ReduceType = reduceType
	return op, nil
}

// GetSelectAndScatterComputation builds or returns a cached computation that implements a select and scatter functions with one
// of the standard ReduceOpType: sum, multiply, max or min.
// This is used for SelectAndScatter family of operations.
func (b *XlaBuilder) GetSelectAndScatterComputation(reduction ReduceOpType, dtype dtypes.DType) (selectComputation, scatterComputation *XlaComputation, err error) {
	if b.IsNil() {
		err = errors.New("trying to access XlaBuilder that is nil or already destroyed")
		return
	}
	if dtype == dtypes.InvalidDType {
		err = errors.Errorf("invalid dtype (%s) for select operation", dtype)
		return
	}

	selectName := fmt.Sprintf("#_select_%s_%s", reduction, dtype)
	scatterName := fmt.Sprintf("#_scatter_%s", dtype)
	selectComputation = b.cachedStandardComputations[selectName]
	scatterComputation = b.cachedStandardComputations[scatterName]
	if selectComputation != nil && scatterComputation != nil {
		return
	}

	if selectComputation == nil {
		// Generate new computation for selection.
		subBuilder := b.CreateSubBuilder(selectName)
		// lhs -> left-hand-side, rhs -> right-hand-side
		var lhs, rhs *Op
		lhs, err = Parameter(subBuilder, "lhs", 0, MakeShape(dtype))
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a select computation %s", reduction)
			return
		}
		rhs, err = Parameter(subBuilder, "rhs", 1, MakeShape(dtype))
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a select computation for %q", reduction)
			return
		}
		var output *Op
		switch reduction {
		case ReduceSumType, ReduceProductType:
			// All values are selected, since they all affect the result.
			output, err = Constant(subBuilder, NewScalarLiteral(true))
		case ReduceMaxType:
			output, err = GreaterOrEqual(lhs, rhs)
		case ReduceMinType:
			output, err = LessOrEqual(lhs, rhs)
		default:
			err = errors.Errorf("unknown select computation type: %s (%d)", reduction, reduction)
			return
		}
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a select computation for %q", reduction)
			return
		}
		selectComputation, err = subBuilder.Build(output)
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a select computation for %q", reduction)
			return
		}
		subBuilder.Destroy()
		b.cachedStandardComputations[selectName] = selectComputation
	}

	if scatterComputation == nil {
		// Generate new computation for scatter function.
		subBuilder := b.CreateSubBuilder(scatterName)
		// lhs -> left-hand-side, rhs -> right-hand-side
		var lhs, rhs *Op
		lhs, err = Parameter(subBuilder, "lhs", 0, MakeShape(dtype))
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a scatter computation %s", reduction)
			return
		}
		rhs, err = Parameter(subBuilder, "rhs", 1, MakeShape(dtype))
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a scatter computation %s", reduction)
			return
		}
		var output *Op
		output, err = Add(lhs, rhs)
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a scatter computation %s", reduction)
			return
		}
		scatterComputation, err = subBuilder.Build(output)
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a scatter computation %s", reduction)
			return
		}
		subBuilder.Destroy()
		b.cachedStandardComputations[scatterName] = scatterComputation
	}
	return
}

// DecodeSelectAndScatter retrieves the arguments for a SelectAndScatter (ScatterAndScatterCustom or ScatterAndScatterMax) op.
func DecodeSelectAndScatter(op *Op) (
	operand, source, defaultValue *Op,
	selectComputation, scatterComputation *XlaComputation,
	windowDimensions, windowStrides []int, paddings [][2]int) {
	operand = op.OpInputs[0]
	source = op.OpInputs[1]
	defaultValue = op.OpInputs[2]
	selectComputation = op.ComputationArg
	scatterComputation = op.SecondComputationArg

	rank := op.IntsArg[0]
	if op.IntsArg[1] > 0 {
		paddings = make([][2]int, op.IntsArg[1])
	}
	windowDimensions = make([]int, rank)
	windowStrides = make([]int, rank)
	pos := 2
	for _, slice := range [][]int{windowDimensions, windowStrides} {
		copy(slice, op.IntsArg[pos:])
		pos += len(slice)
	}
	for ii := range paddings {
		paddings[ii][0] = op.IntsArg[pos]
		paddings[ii][1] = op.IntsArg[pos+1]
		pos += 2
	}
	return
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
//
// It provides the basic means of implementing Einsum.
func DotGeneral(lhs *Op, lhsContractingAxes, lhsBatchAxes []int,
	rhs *Op, rhsContractingAxes, rhsBatchAxes []int) (*Op, error) {
	builder := lhs.builder
	dtype := lhs.Shape.DType
	if lhs.Shape.IsScalar() || rhs.Shape.IsScalar() {
		return nil, errors.Errorf("cannot use DotGeneral() with scalar values (lhs.shape=%s, rhs.shape=%s)", lhs.Shape, rhs.Shape)
	}
	if rhs.Shape.DType != dtype {
		return nil, errors.Errorf("DotGeneral lhs (dtype=%s) and rhs (dtype=%s) must match",
			lhs.Shape.DType, rhs.Shape.DType)
	}
	if lhs.Shape.Rank() < len(lhsBatchAxes)+len(lhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral lhs (shape=%s) doesn't have enough axes to contract (%d) and batch (%d), something is wrong",
			lhs.Shape, len(lhsContractingAxes), len(lhsBatchAxes))
	}
	if rhs.Shape.Rank() < len(rhsBatchAxes)+len(rhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral rhs (shape=%s) doesn't have enough axes to contract (%d) and batch (%d), something is wrong",
			rhs.Shape, len(rhsContractingAxes), len(rhsBatchAxes))
	}

	if klog.V(2).Enabled() {
		klog.Infof("DotGeneral(lhs=%s, lhsContractingAxes=%v, lhsBatchAxes=%v, rhs=%s, rhsContractingAxes=%v, rhsBatchAxes=%v",
			lhs.Shape, lhsContractingAxes, lhsBatchAxes, rhs.Shape, rhsContractingAxes, rhsBatchAxes)
	}

	op := newOp(DotGeneralOp, lhs, rhs)
	var lists = [][]int{lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes}
	intsLen := len(lists) // One value
	for _, list := range lists {
		intsLen += len(list)
	}
	op.IntsArg = make([]int, 0, intsLen)
	for _, list := range lists {
		op.IntsArg = append(op.IntsArg, len(list))
	}
	for _, list := range lists {
		op.IntsArg = append(op.IntsArg, list...)
	}
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeDotGeneral retrieves the arguments for a DotGeneral op.
func DecodeDotGeneral(op *Op) (lhs *Op, lhsContractingAxes, lhsBatchAxes []int,
	rhs *Op, rhsContractingAxes, rhsBatchAxes []int) {
	lhs = op.OpInputs[0]
	rhs = op.OpInputs[1]
	var lists = []*[]int{&lhsContractingAxes, &lhsBatchAxes, &rhsContractingAxes, &rhsBatchAxes}
	pos := 0
	for _, listRef := range lists {
		*listRef = make([]int, op.IntsArg[pos])
		pos++
	}
	for _, listRef := range lists {
		copy(*listRef, op.IntsArg[pos:])
		pos += len(*listRef)
	}
	return
}

// Reverse returns x with the values for the given dimensions reversed, that is,
// the value indexed at `i` will be swapped with the value at indexed `(dimension_size - 1 - i)`.
// The shape remains the same.
func Reverse(x *Op, axes ...int) (*Op, error) {
	builder := x.builder
	if x.Shape.IsScalar() {
		return nil, errors.Errorf("cannot use Reverse() with scalar value (x.shape=%s)", x.Shape)
	}
	op := newOp(ReverseOp, x)
	op.IntsArg = axes
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeReverse retrieves the arguments for the Reverse op.
func DecodeReverse(op *Op) (x *Op, axes []int) {
	x = op.OpInputs[0]
	axes = op.IntsArg
	return
}

// BatchNormForInference implements Batch Norm for inference. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnorminference
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormForInference(operand, scale, offset, mean, variance *Op, epsilon float32, axis int) (*Op, error) {
	builder := operand.builder
	op := newOp(BatchNormInferenceOp, operand, scale, offset, mean, variance)
	op.IntArg = axis
	op.FloatArg = epsilon
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeBatchNormForInference retrieves the arguments for the BatchNormForInference op.
func DecodeBatchNormForInference(op *Op) (operand, scale, offset, mean, variance *Op, epsilon float32, axis int) {
	operand = op.OpInputs[0]
	scale = op.OpInputs[1]
	offset = op.OpInputs[2]
	mean = op.OpInputs[3]
	variance = op.OpInputs[4]
	epsilon = op.FloatArg
	axis = op.IntArg
	return
}

// BatchNormForTraining implements Batch Norm for training. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnormtraining
//
// It returns the normalized tensor, the batchMean and the batchVariance.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormForTraining(operand, scale, offset *Op, epsilon float32, axis int) (normalized, batchMean, batchVariance *Op, err error) {
	builder := operand.builder
	op := newOp(BatchNormTrainingOp, operand, scale, offset)
	op.IntArg = axis
	op.FloatArg = epsilon
	err = builder.addOp(op)
	if err != nil {
		return
	}
	var parts []*Op
	parts, err = SplitTuple(op)
	if err != nil {
		err = errors.WithMessage(err, "failed to split results of BatchNormForTraining")
		return
	}
	if len(parts) != 3 {
		err = errors.Errorf("BatchNormForTraining should have returned a tuple with 3 parts, got %s instead", op.Shape)
		return
	}
	normalized, batchMean, batchVariance = parts[0], parts[1], parts[2]
	return
}

// DecodeBatchNormForTraining retrieves the arguments for the BatchNormForTraining op.
func DecodeBatchNormForTraining(op *Op) (operand, scale, offset *Op, epsilon float32, axis int) {
	operand = op.OpInputs[0]
	scale = op.OpInputs[1]
	offset = op.OpInputs[2]
	epsilon = op.FloatArg
	axis = op.IntArg
	return
}

// BatchNormGradient calculates the BatchNorm gradient. See details in
// https://openxla.org/xla/operation_semantics#batchnormgrad
//
// The gradOutput is the adjoint gradient, that is, the gradient with respect to the output of the
// batch normalization.
//
// It returns  as a tuple with the 3 elements.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormGradient(operand, scale, mean, variance, gradOutput *Op, epsilon float32, axis int) (gradOperand, gradScale, gradOffset *Op, err error) {
	builder := operand.builder
	op := newOp(BatchNormGradOp, operand, scale, mean, variance, gradOutput)
	op.IntArg = axis
	op.FloatArg = epsilon
	err = builder.addOp(op)
	if err != nil {
		return
	}
	var parts []*Op
	parts, err = SplitTuple(op)
	if err != nil {
		err = errors.WithMessage(err, "failed to split results of BatchNormGradient")
		return
	}
	if len(parts) != 3 {
		err = errors.Errorf("BatchNormGradient should have returned a tuple with 3 parts, got %s instead", op.Shape)
		return
	}
	gradOperand, gradScale, gradOffset = parts[0], parts[1], parts[2]
	return
}

// DecodeBatchNormGrad retrieves the arguments for the BatchNormGradient op.
func DecodeBatchNormGrad(op *Op) (operand, scale, mean, variance, gradOutput *Op, epsilon float32, axis int) {
	operand = op.OpInputs[0]
	scale = op.OpInputs[1]
	mean = op.OpInputs[2]
	variance = op.OpInputs[3]
	gradOutput = op.OpInputs[4]
	epsilon = op.FloatArg
	axis = op.IntArg
	return
}

// FFT calls the XLA FFT operation, which implements {Forward, Inverse} x {Complex, Real} versions.
//
// See documentation in https://www.tensorflow.org/xla/operation_semantics.
// Underlying, CPU FFT is backed by Eigen's TensorFFT and GPU FFT uses cuFFT.
func FFT(operand *Op, fftType xla_data.FftType, fftLength []int) (*Op, error) {
	builder := operand.builder
	op := newOp(FftOp, operand)
	op.IntArg = int(fftType)
	op.IntsArg = slices.Clone(fftLength)
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeFFT retrieves the arguments for the FFT op.
func DecodeFFT(op *Op) (operand *Op, fftType xla_data.FftType, fftLength []int) {
	operand = op.OpInputs[0]
	fftType = xla_data.FftType(op.IntArg)
	fftLength = op.IntsArg
	return
}

// RngBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RngState or RngStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func RngBitGenerator(state *Op, shape Shape) (newState, values *Op, err error) {
	builder := state.builder
	op := newOp(RngBitGeneratorOp, state)
	op.ShapeArg = shape
	err = builder.addOp(op)
	if err != nil {
		return
	}
	var parts []*Op
	parts, err = SplitTuple(op)
	if err != nil {
		err = errors.WithMessage(err, "failed to split results of RngBitGenerator")
		return
	}
	if len(parts) != 2 {
		err = errors.Errorf("RngBitGenerator should have returned a tuple with 2 parts, got %s instead", op.Shape)
		return
	}
	newState, values = parts[0], parts[1]
	return
}

// DecodeRngBitGenerator retrieves the arguments for the FFT op.
func DecodeRngBitGenerator(op *Op) (state *Op, shape Shape) {
	state = op.OpInputs[0]
	shape = op.ShapeArg
	return
}

// While executes a loop in the computation.
//
// It takes as input:
//
//   - initialState: usually a tuple, that includes all variables used by condition and body.
//   - condition: a sub-computation (see XlaBuilder.CreateSubBuilder) takes the current state as input and outputs
//     a bool (dtypes.PRED) whether the loop should keep iterating.
//   - body: a sub-computation (see XlaBuilder.CreateSubBuilder) takes the current state as input and outputs
//     an updated state.
//
// See details in https://openxla.org/xla/operation_semantics#while
func While(initialState *Op, condition, body *XlaComputation) (*Op, error) {
	builder := initialState.builder
	op := newOp(WhileOp, initialState)
	op.ComputationArg = condition
	op.SecondComputationArg = body
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeWhile retrieves the arguments for the While op.
func DecodeWhile(op *Op) (initialState *Op, condition, body *XlaComputation) {
	initialState = op.OpInputs[0]
	condition = op.ComputationArg
	body = op.SecondComputationArg
	return
}

// DynamicSlice extracts a sub-array from the input array at dynamic start_indices.
// The size of the slice in each axis is passed in sliceDims, which specify the slice
// intervals for each axis: [start, start + size).
// The shape of startIndices must be rank == 1, with dimension size equal to the rank of operand.
//
// See description in https://openxla.org/xla/operation_semantics#dynamicslice
func DynamicSlice(operand *Op, startIndices []*Op, sliceDims []int) (*Op, error) {
	builder := operand.builder
	allOps := append([]*Op{operand}, startIndices...)
	op := newOp(DynamicSliceOp, allOps...)
	op.IntsArg = sliceDims
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeDynamicSlice retrieves the arguments for the DynamicSlice op.
func DecodeDynamicSlice(op *Op) (operand *Op, startIndices []*Op, sliceDims []int) {
	operand = op.OpInputs[0]
	startIndices = op.OpInputs[1:]
	sliceDims = op.IntsArg
	return
}

// DynamicUpdateSlice generates a result which is the value of the input array operand, with a slice update overwritten
// at startIndices.
// The shape of update determines the shape of the sub-array of the result which is updated.
// The shape of startIndices must be rank == 1, with dimension size equal to the rank of operand.
//
// See description in https://openxla.org/xla/operation_semantics#dynamicupdateslice
func DynamicUpdateSlice(operand, update *Op, startIndices []*Op) (*Op, error) {
	builder := operand.builder
	if operand.Shape.DType != update.Shape.DType {
		return nil, errors.Errorf("operand and update dtypes (%s and %s) don't match for DynamicUpdateSlice", operand.Shape.DType, update.Shape.DType)
	}
	allOps := append([]*Op{operand, update}, startIndices...)
	op := newOp(DynamicUpdateSliceOp, allOps...)
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeDynamicUpdateSlice retrieves the arguments for the DynamicUpdateSlice op.
func DecodeDynamicUpdateSlice(op *Op) (operand, update *Op, startIndices []*Op) {
	operand = op.OpInputs[0]
	update = op.OpInputs[1]
	startIndices = op.OpInputs[2:]
	return
}
