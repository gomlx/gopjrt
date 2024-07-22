package xlabuilder_test

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
)

func TestTuple(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// f(x) = [x^2, sqrt(x)]
	x := capture(Parameter(builder, "x", 0, MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	x2 := capture(Mul(x, x)).Test(t)
	sqrtX := capture(Sqrt(x)).Test(t)
	fX := capture(Tuple(x2, sqrtX)).Test(t)

	// Take program and compile.
	comp := capture(builder.Build(fX)).Test(t)
	exec := compile(t, client, comp)
	require.Equal(t, 2, exec.NumOutputs)
}

func TestGetTupleElement(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	x0 := capture(Constant(builder, NewScalarLiteral(int32(7)))).Test(t)
	x1 := capture(Constant(builder, NewArrayLiteral([]complex64{11, 15}))).Test(t)
	x2 := capture(Constant(builder, NewScalarLiteral(1.0))).Test(t)
	tuple := capture(Tuple(x0, x1, x2)).Test(t)
	output := capture(GetTupleElement(tuple, 1)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	want := []complex64{11, 15}
	got, dims := execArrayOutput[complex64](t, client, exec)
	require.Equal(t, want, got)
	require.Equal(t, []int{2}, dims)
}

func TestConstants(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// f(x)=x+1
	x := capture(Parameter(builder, "x", 0, MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	one := capture(Constant(builder, NewScalarLiteral(float32(1)))).Test(t)
	fX := capture(Add(x, one)).Test(t)
	comp := capture(builder.Build(fX)).Test(t)

	// Check values.
	addOne := compile(t, client, comp)
	require.InDelta(t, float32(2), execWithScalars(t, client, addOne, float32(1)), 1e-3)
	require.InDelta(t, float32(8), execWithScalars(t, client, addOne, float32(7)), 1e-3)

	// f(x)=x+1 with broadcast
	x = capture(Parameter(builder, "x", 0, MakeShape(dtypes.Int64, 3))).Test(t) // Scalar float32.
	one = capture(Constant(builder, NewScalarLiteral(int64(1)))).Test(t)
	fX = capture(Add(x, one)).Test(t)
	comp = capture(builder.Build(fX)).Test(t)

	// Check values.
	addOne = compile(t, client, comp)
	got, dims := execWithSlices(t, client, addOne, []int64{1, 7, 13})
	require.Equal(t, []int64{1 + 1, 7 + 1, 13 + 1}, got)
	require.Equal(t, []int{3}, dims)
}

func TestIota(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	iotaOp := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	exec := compile(t, client, capture(builder.Build(iotaOp)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{3, 2}, dims)

	iotaOp = capture(Iota(builder, MakeShape(dtypes.F64, 2, 3), 1)).Test(t)
	exec = compile(t, client, capture(builder.Build(iotaOp)).Test(t))
	got, dims = execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 1, 2, 0, 1, 2}, got)
	require.Equal(t, []int{2, 3}, dims)

}

func TestIdentity(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Exact same test as iota, just with an identity op in between.
	iotaOp := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	identityOp := Identity(iotaOp)
	exec := compile(t, client, capture(builder.Build(identityOp)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{3, 2}, dims)
}

func TestConvertDType(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Exact same test as iota, but change the dtype of the result.
	iotaOp := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	output := capture(ConvertDType(iotaOp, dtypes.Int64)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[int64](t, client, exec)
	require.Equal(t, []int64{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{3, 2}, dims)
}

func TestWhere(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Exact same test as iota, but change the dtype of the result.
	shape := MakeShape(dtypes.Float32, 5, 3)
	zeros := capture(Constant(builder, NewLiteralFromShape(shape))).Test(t)
	one := capture(Constant(builder, NewScalarLiteral(float32(1)))).Test(t)
	ones := capture(Add(zeros, one)).Test(t)
	values := capture(Iota(builder, shape, 0)).Test(t)
	two := capture(Constant(builder, NewScalarLiteral(float32(2)))).Test(t)
	greaterThanTwo := capture(GreaterThan(values, two)).Test(t)
	output := capture(Where(greaterThanTwo, ones, zeros)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float32](t, client, exec)
	require.Equal(t, []float32{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		1, 1, 1,
		1, 1, 1,
	}, got)
	require.Equal(t, shape.Dimensions, dims)
}

func TestReshape(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	input := capture(Iota(builder, MakeShape(dtypes.Int8, 3, 2), 0)).Test(t)
	_, err := Reshape(input, 7) // Bad reshape.
	require.Error(t, err)
	output := capture(Reshape(input, 6, 1, 1)).Test(t)
	require.Equal(t, []int{6, 1, 1}, DecodeReshape(output)) // Check decoding.
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, []int8{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{6, 1, 1}, dims)
}

func TestBroadcast(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	input := capture(Iota(builder, MakeShape(dtypes.Float32, 3, 2), 0)).Test(t)
	output := capture(Broadcast(input, 2)).Test(t)
	require.Equal(t, []int{2}, DecodeBroadcast(output)) // Check decoding.
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float32](t, client, exec)
	require.Equal(t, []float32{0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{2, 3, 2}, dims)
}

func TestBroadcastInDim(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	input := capture(Iota(builder, MakeShape(dtypes.Float32, 3, 1), 0)).Test(t)
	outputShape := MakeShape(dtypes.Float32, 2, 3, 3)
	broadcastAxes := []int{1, 2}
	output := capture(BroadcastInDim(input, outputShape, broadcastAxes)).Test(t)

	// Check decoding.
	gotShape, gotAxes := DecodeBroadcastInDim(output)
	require.Equal(t, outputShape.Dimensions, gotShape.Dimensions)
	require.Equal(t, broadcastAxes, gotAxes)

	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float32](t, client, exec)
	require.Equal(t, outputShape.Dimensions, dims)
	require.Equal(t, []float32{
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2}, got)
}

func TestTranspose(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	input := capture(Iota(builder, MakeShape(dtypes.Float32, 3, 1, 2), 0)).Test(t)
	output := capture(Transpose(input, 2, 0, 1)).Test(t)
	require.Equal(t, []int{2, 0, 1}, DecodeTranspose(output)) // Check decoding.
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float32](t, client, exec)
	require.Equal(t, []float32{0, 1, 2, 0, 1, 2}, got)
	require.Equal(t, []int{2, 3, 1}, dims)
}

func TestCall(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Create a sub-computation.
	subBuilder := builder.CreateSubBuilder("x_plus_one")
	fmt.Printf("\tSubBuilder %q:\n", subBuilder.Name())
	p2X := capture(Parameter(subBuilder, "x", 0, MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	p2One := capture(Constant(subBuilder, NewScalarLiteral(float32(1)))).Test(t)
	p2Output := capture(Add(p2X, p2One)).Test(t)
	xPlusOneComp := capture(subBuilder.Build(p2Output)).Test(t)

	x := capture(Parameter(builder, "x", 0, MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	xOne := capture(Call(builder, xPlusOneComp, x)).Test(t)
	xTwo := capture(Call(builder, xPlusOneComp, xOne)).Test(t)
	output := capture(Mul(xOne, xTwo)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))

	inputs := []float32{1, 3, 7}
	fmt.Printf("f(x) = (x+1) * (x+2), using %q:\n", xPlusOneComp.Name())
	for _, input := range inputs {
		want := (input + 1) * (input + 2)
		got := execWithScalars(t, client, exec, input)
		fmt.Printf("\tf(%g) = %g, got %g\n", input, want, got)
		require.InDelta(t, want, got, 1e-3)
	}
}

func TestConcatenate(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Try 2 different axes to concatenate the arrays:
	x0 := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	x1 := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 1)).Test(t)
	output := capture(Concatenate(0, x0, x1)).Test(t)
	require.Equal(t, 0, DecodeConcatenate(output))
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1}, got)
	require.Equal(t, []int{6, 2}, dims)

	x0 = capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	x1 = capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 1)).Test(t)
	output = capture(Concatenate(1, x0, x1)).Test(t)
	require.Equal(t, 1, DecodeConcatenate(output))
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims = execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 0, 1}, got)
	require.Equal(t, []int{3, 4}, dims)
}

func TestSlice(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Try 2 different axes to concatenate the arrays:
	x := capture(Iota(builder, MakeShape(dtypes.F64, 7), 0)).Test(t)
	output := capture(Slice(x, []int{2}, []int{5}, nil)).Test(t)
	starts, limits, strides := DecodeSlice(output)
	require.Equal(t, []int{2}, starts)
	require.Equal(t, []int{5}, limits)
	require.Equal(t, []int{1}, strides) // Default should be 1.
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{2, 3, 4}, got)
	require.Equal(t, []int{3}, dims)

	x = capture(Iota(builder, MakeShape(dtypes.F64, 7), 0)).Test(t)
	output = capture(Slice(x, []int{2}, []int{5}, []int{2})).Test(t)
	starts, limits, strides = DecodeSlice(output)
	require.Equal(t, []int{2}, starts)
	require.Equal(t, []int{5}, limits)
	require.Equal(t, []int{2}, strides)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims = execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{2, 4}, got)
	require.Equal(t, []int{2}, dims)
}

func TestArgMinMax(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	x := capture(Constant(builder, NewArrayLiteral([]int64{2, 0, 7, -3, 4, 2}, 2, 3))).Test(t)
	output := capture(ArgMinMax(x, 1, dtypes.Int8, true)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, []int8{1, 0}, got)
	require.Equal(t, []int{2}, dims)

	x = capture(Constant(builder, NewArrayLiteral([]int64{2, 0, 7, -3, 4, 2}, 2, 3))).Test(t)
	output = capture(ArgMinMax(x, 0, dtypes.Int8, false)).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims = execArrayOutput[int8](t, client, exec)
	require.Equal(t, []int8{0, 1, 0}, got)
	require.Equal(t, []int{3}, dims)
}

func TestPad(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	x := capture(Iota(builder, MakeShape(dtypes.F64, 6), 0)).Test(t)
	x = capture(Reshape(x, 2, 3)).Test(t)
	fillValue := capture(Constant(builder, NewScalarLiteral(-1.0))).Test(t)
	padRows := PadAxis{Start: 1, End: 2} // One row at the start, two rows at the end.
	padColumns := PadAxis{Interior: 1}   // One value in between every column.
	output := capture(Pad(x, fillValue, padRows, padColumns)).Test(t)

	axesConfig := DecodePad(output)
	require.Equal(t, padRows, axesConfig[0])
	require.Equal(t, padColumns, axesConfig[1])

	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{
		-1, -1, -1, -1, -1,
		0, -1, 1, -1, 2,
		3, -1, 4, -1, 5,
		-1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1,
	}, got)
	require.Equal(t, []int{5, 5}, dims)
}

func TestGather(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	{
		shape := MakeShape(dtypes.Float64 /* batch */, 5, 3)
		x := capture(Iota(builder, MakeShape(shape.DType, shape.Size()), 0)).Test(t)
		x = capture(Reshape(x, shape.Dimensions...)).Test(t)
		indices := capture(Constant(builder, NewArrayLiteral([]int32{2, 0}, 2, 1))).Test(t)
		indexVectorAxis := 1
		offsetAxes := []int{1}
		collapsedSliceAxes := []int{0}
		startIndexMap := []int{0}
		sliceSizes := []int{1, 3}
		indicesAreSorted := false
		output := capture(Gather(x, indices, indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)).Test(t)

		gotIndexVectorAxis, gotOffsetAxes, gotCollapsedSliceAxes, gotStartIndexMap, gotSliceSizes, gotIndicesAreSorted := DecodeGather(output)
		require.Equal(t, indexVectorAxis, gotIndexVectorAxis)
		require.Equal(t, offsetAxes, gotOffsetAxes)
		require.Equal(t, collapsedSliceAxes, gotCollapsedSliceAxes)
		require.Equal(t, startIndexMap, gotStartIndexMap)
		require.Equal(t, sliceSizes, gotSliceSizes)
		require.Equal(t, indicesAreSorted, gotIndicesAreSorted)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float64](t, client, exec)
		require.Equal(t, []float64{6, 7, 8, 0, 1, 2}, got)
		require.Equal(t, []int{2, 3}, dims)
	}

	{
		shape := MakeShape(dtypes.Float32 /* batch */, 3, 2, 6)
		x := capture(Iota(builder, MakeShape(shape.DType, shape.Size()), 0)).Test(t)
		x = capture(Reshape(x, shape.Dimensions...)).Test(t)
		indices := capture(Constant(builder, NewArrayLiteral([]int32{
			0, 0, 0,
			2, 1, 5}, 2, 3))).Test(t)
		indexVectorAxis := 1
		offsetAxes := []int{1}
		collapsedSliceAxes := []int{0, 1}
		startIndexMap := []int{0, 1, 2}
		sliceSizes := []int{1, 1, 1}
		indicesAreSorted := false
		output := capture(Gather(x, indices, indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)).Test(t)

		gotIndexVectorAxis, gotOffsetAxes, gotCollapsedAxes, gotStartIndexMap, gotSliceSizes, gotIndicesAreSorted := DecodeGather(output)
		require.Equal(t, indexVectorAxis, gotIndexVectorAxis)
		require.Equal(t, offsetAxes, gotOffsetAxes)
		require.Equal(t, collapsedSliceAxes, gotCollapsedAxes)
		require.Equal(t, startIndexMap, gotStartIndexMap)
		require.Equal(t, sliceSizes, gotSliceSizes)
		require.Equal(t, indicesAreSorted, gotIndicesAreSorted)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.Equal(t, []float32{0, 2*12 + 1*6 + 5}, got)
		require.Equal(t, []int{2, 1}, dims)
	}
}

func TestScatter(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	{
		dtype := dtypes.Float64
		shape := MakeShape(dtype /* batch */, 5, 3)
		operand := capture(Constant(builder, NewLiteralFromShape(shape))).Test(t) // 5*3 zeros
		indices := capture(Constant(builder, NewArrayLiteral([]int32{
			// 5 scatter updates:
			0, 1,
			0, 2,
			2, 1,
			4, 2,
			4, 2,
		}, 5, 2))).Test(t)
		updates := capture(Iota(builder, MakeShape(dtype, 5, 1), 0)).Test(t)
		one := capture(Constant(builder, NewScalarLiteral(1.0))).Test(t)
		updates = capture(Add(updates, one)).Test(t)

		indexVectorAxis := 1
		updateWindowAxes := []int{1}
		insertedWindowAxes := []int{0}
		scatterAxesToOperandAxes := []int{0, 1}
		indicesAreSorted := true
		uniqueIndices := false // We scatter twice to index (4, 2)
		output := capture(ScatterAdd(operand, indices, updates, indexVectorAxis, updateWindowAxes,
			insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)).Test(t)

		gotIndexVectorAxis, gotUpdateWindowAxes, gotInsertedWindowAxes, gotScatterAxesToOperandAxes, gotIndicesAreSorted, gotUniqueIndices := DecodeScatter(output)
		require.Equal(t, indexVectorAxis, gotIndexVectorAxis)
		require.Equal(t, updateWindowAxes, gotUpdateWindowAxes)
		require.Equal(t, insertedWindowAxes, gotInsertedWindowAxes)
		require.Equal(t, scatterAxesToOperandAxes, gotScatterAxesToOperandAxes)
		require.Equal(t, indicesAreSorted, gotIndicesAreSorted)
		require.Equal(t, uniqueIndices, gotUniqueIndices)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float64](t, client, exec)
		require.Equal(t, []float64{
			// Values with scattered: notice the index (4, 2), the last one, is scattered twice, and values are added.
			0, 1, 2,
			0, 0, 0,
			0, 3, 0,
			0, 0, 0,
			0, 0, 9}, got)
		require.Equal(t, []int{5, 3}, dims)
	}
}

func TestSelectAndScatter(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	{
		dtype := dtypes.Float64
		operand := capture(Iota(builder, MakeShape(dtype, 1, 6, 1), 1)).Test(t)
		source := capture(Iota(builder, MakeShape(dtype, 1, 2, 1), 1)).Test(t)
		one := capture(ScalarOne(builder, dtype)).Test(t)
		source = capture(Add(source, one)).Test(t)
		windowDimensions := []int{1, 3, 1}
		windowStrides := []int{1, 3, 1}
		output := capture(SelectAndScatterMax(operand, source, windowDimensions, windowStrides, nil)).Test(t)

		gotOperand, gotSource, _, gotSelectComp, gotScatterComp, gotWindowDimensions, gotWindowStrides, gotPaddings := DecodeSelectAndScatter(output)
		require.Same(t, operand, gotOperand)
		require.Same(t, source, gotSource)
		require.True(t, strings.HasPrefix(gotSelectComp.Name(), "#_select_ReduceMaxType_Float64"))
		require.True(t, strings.HasPrefix(gotScatterComp.Name(), "#_scatter_Float64"))
		require.Equal(t, windowDimensions, gotWindowDimensions)
		require.Equal(t, windowStrides, gotWindowStrides)
		require.Empty(t, gotPaddings)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float64](t, client, exec)
		require.Equal(t, []float64{0, 0, 1, 0, 0, 2}, got)
		require.Equal(t, []int{1, 6, 1}, dims)
	}
}

func TestDotGeneral(t *testing.T) {
	client := getPJRTClient(t)

	{
		builder := New(t.Name() + ": [3, 4] x [3, 4]")
		dtype := dtypes.Float32
		lhs := capture(Iota(builder, MakeShape(dtype, 3*4), 0)).Test(t)
		lhs = capture(Reshape(lhs, 3, 4)).Test(t)
		one := capture(ScalarOne(builder, dtype)).Test(t)
		lhs = capture(Add(lhs, one)).Test(t)
		rhs := capture(Constant(builder, NewScalarLiteral(float32(0.1)))).Test(t)
		rhs = capture(Broadcast(rhs, 3, 4)).Test(t)

		lhsContractingAxes, lhsBatchAxes := []int{1}, []int{0}
		rhsContractingAxes, rhsBatchAxes := []int{1}, []int{0}
		output := capture(DotGeneral(
			lhs, lhsContractingAxes, lhsBatchAxes,
			rhs, rhsContractingAxes, rhsBatchAxes,
		)).Test(t)
		gotLhs, gotLhsContractingAxes, gotLhsBatchAxes, gotRhs, gotRhsContractingAxes, gotRhsBatchAxes := DecodeDotGeneral(output)
		require.Same(t, lhs, gotLhs)
		require.Equal(t, lhsContractingAxes, gotLhsContractingAxes)
		require.Equal(t, lhsBatchAxes, gotLhsBatchAxes)
		require.Same(t, rhs, gotRhs)
		require.Equal(t, rhsContractingAxes, gotRhsContractingAxes)
		require.Equal(t, rhsBatchAxes, gotRhsBatchAxes)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.Equal(t, []float32{1, 2.6, 4.2}, got)
		require.Equal(t, []int{3}, dims)
		builder.Destroy()
	}

	{
		builder := New(t.Name() + ": [3, 2, 4] x [3, 5, 4] -> [3, 2, 5]")
		dtype := dtypes.Float32
		lhs := capture(Iota(builder, MakeShape(dtype, 3*2*4), 0)).Test(t)
		lhs = capture(Reshape(lhs, 3, 2, 4)).Test(t)
		one := capture(ScalarOne(builder, dtype)).Test(t)
		lhs = capture(Add(lhs, one)).Test(t)
		rhs := capture(Constant(builder, NewScalarLiteral(float32(0.1)))).Test(t)
		rhs = capture(Broadcast(rhs, 3, 5, 4)).Test(t)

		lhsContractingAxes, lhsBatchAxes := []int{2}, []int{0}
		rhsContractingAxes, rhsBatchAxes := []int{2}, []int{0}
		output := capture(DotGeneral(
			lhs, lhsContractingAxes, lhsBatchAxes,
			rhs, rhsContractingAxes, rhsBatchAxes,
		)).Test(t)
		gotLhs, gotLhsContractingAxes, gotLhsBatchAxes, gotRhs, gotRhsContractingAxes, gotRhsBatchAxes := DecodeDotGeneral(output)
		require.Same(t, lhs, gotLhs)
		require.Equal(t, lhsContractingAxes, gotLhsContractingAxes)
		require.Equal(t, lhsBatchAxes, gotLhsBatchAxes)
		require.Same(t, rhs, gotRhs)
		require.Equal(t, rhsContractingAxes, gotRhsContractingAxes)
		require.Equal(t, rhsBatchAxes, gotRhsBatchAxes)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.InDeltaSlice(t, []float32{
			1, 1, 1, 1, 1,
			2.6, 2.6, 2.6, 2.6, 2.6,
			4.2, 4.2, 4.2, 4.2, 4.2,
			5.8, 5.8, 5.8, 5.8, 5.8,
			7.4, 7.4, 7.4, 7.4, 7.4,
			9, 9, 9, 9, 9,
			1, 2.6, 4.2,
		}, got, 0.001)
		require.Equal(t, []int{3, 2, 5}, dims)
	}
}

func TestReverse(t *testing.T) {
	client := getPJRTClient(t)

	{
		builder := New(t.Name() + ": [3, 4]")
		dtype := dtypes.Float32
		x := capture(Iota(builder, MakeShape(dtype, 3*4), 0)).Test(t)
		x = capture(Reshape(x, 3, 4)).Test(t)
		axes := []int{0}
		output := capture(Reverse(x, axes...)).Test(t)

		gotX, gotAxes := DecodeReverse(output)
		require.Same(t, x, gotX)
		require.Equal(t, axes, gotAxes)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.Equal(t, []float32{8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3}, got)
		require.Equal(t, []int{3, 4}, dims)
		builder.Destroy()
	}
}

func TestWhile(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())
	dtype := dtypes.Int64
	x := capture(Parameter(builder, "x", 0, MakeShape(dtype))).Test(t)
	value := capture(ScalarOne(builder, dtype)).Test(t)

	// Calculate factorial using a While loop.
	// While loop:
	// - initialState: (x, 1)
	initialState := capture(Tuple(x, value)).Test(t)
	var cond, body *XlaComputation
	// - condition: x > 0
	{
		condBuilder := builder.CreateSubBuilder(t.Name() + "_condition")
		tuple := capture(Parameter(condBuilder, "tuple", 0, initialState.Shape.Clone())).Test(t)
		loopX := capture(GetTupleElement(tuple, 0)).Test(t)
		zero := capture(ScalarZero(condBuilder, dtype)).Test(t)
		output := capture(GreaterThan(loopX, zero)).Test(t)
		cond = capture(condBuilder.Build(output)).Test(t)
	}
	// - body: value = value * x; x = x-1;
	{
		bodyBuilder := builder.CreateSubBuilder(t.Name() + "_body")
		tuple := capture(Parameter(bodyBuilder, "tuple", 0, initialState.Shape.Clone())).Test(t)
		loopX := capture(GetTupleElement(tuple, 0)).Test(t)
		loopValue := capture(GetTupleElement(tuple, 1)).Test(t)
		loopValue = capture(Mul(loopValue, loopX)).Test(t)
		one := capture(ScalarOne(bodyBuilder, dtype)).Test(t)
		loopX = capture(Sub(loopX, one)).Test(t)
		output := capture(Tuple(loopX, loopValue)).Test(t)
		body = capture(bodyBuilder.Build(output)).Test(t)
	}
	state := capture(While(initialState, cond, body)).Test(t)

	gotInitialState, gotCond, gotBody := DecodeWhile(state)
	require.Same(t, initialState, gotInitialState)
	require.Same(t, cond, gotCond)
	require.Same(t, body, gotBody)

	output := capture(GetTupleElement(state, 1)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))

	// 5! = 120
	got := int(execWithScalars(t, client, exec, int64(5)))
	require.Equal(t, 120, got)

	// 7! = 5040
	got = int(execWithScalars(t, client, exec, int64(7)))
	require.Equal(t, 5040, got)
	builder.Destroy()
}

func TestDynamicSlice(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())
	dtype := dtypes.Float64
	operand := capture(Iota(builder, MakeShape(dtype, 3*4), 0)).Test(t)
	operand = capture(Reshape(operand, 3, 4)).Test(t)
	startIndices := capture(Constant(builder, NewArrayLiteral([]int32{1, 1}, 2))).Test(t)
	output := capture(DynamicSlice(operand, []*Op{startIndices}, []int{2, 2})).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	gotFlat, gotDims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []int{2, 2}, gotDims)
	require.Equal(t, []float64{5, 6, 9, 10}, gotFlat)
}

func TestDynamicUpdateSlice(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())
	dtype := dtypes.Float64
	operand := capture(Iota(builder, MakeShape(dtype, 3*4), 0)).Test(t)
	operand = capture(Reshape(operand, 3, 4)).Test(t)
	update := capture(Constant(builder, NewArrayLiteral([]float64{-1, -1, -1, -1}, 2, 2))).Test(t)
	startIndices := capture(Constant(builder, NewArrayLiteral([]int32{1, 1}, 2))).Test(t)
	output := capture(DynamicUpdateSlice(operand, update, []*Op{startIndices})).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	gotFlat, gotDims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []int{3, 4}, gotDims)
	require.Equal(t, []float64{
		0, 1, 2, 3,
		4, -1, -1, 7,
		8, -1, -1, 11}, gotFlat)
}
