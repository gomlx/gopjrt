package xlabuilder_test

import (
	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestReduce(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Test with ReduceMax:
	{
		input := capture(Iota(builder, MakeShape(dtypes.Float32, 9), 0)).Test(t)
		input = capture(Reshape(input, 3, 3)).Test(t)
		maxColumns := capture(ReduceMax(input, 0)).Test(t)
		maxRows := capture(ReduceMax(input, 1)).Test(t)
		// Check that cache worked: computation and initial value used by both ReduceMax operations must be the same.
		require.Equal(t, maxColumns.ComputationArg, maxRows.ComputationArg)
		require.Equal(t, maxColumns.OpInputs[1], maxRows.OpInputs[1])
		pointOne := capture(Constant(builder, NewScalarLiteralFromFloat64(0.1, dtypes.Float32))).Test(t)
		maxRows = capture(Mul(maxRows, pointOne)).Test(t)
		output := capture(Add(maxColumns, maxRows)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.Equal(t, []float32{6.2, 7.5, 8.8}, got)
		require.Equal(t, []int{3}, dims)
	}

	// Test with ReduceSum and ReduceProduct
	{
		input := capture(Iota(builder, MakeShape(dtypes.Float64, 9), 0)).Test(t)
		input = capture(Reshape(input, 3, 3)).Test(t)
		columnsProduct := capture(ReduceProduct(input, 0)).Test(t)
		rowsSum := capture(ReduceSum(input, 1)).Test(t)
		shift := capture(Constant(builder, NewScalarLiteralFromFloat64(0.001, dtypes.Float64))).Test(t)
		rowsSum = capture(Mul(rowsSum, shift)).Test(t)
		output := capture(Add(columnsProduct, rowsSum)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float64](t, client, exec)
		require.Equal(t, []float64{0.003, 28.012, 80.021}, got)
		require.Equal(t, []int{3}, dims)
	}

	// Test ReduceMin on all dimensions.
	{
		input := capture(Iota(builder, MakeShape(dtypes.Int64, 9), 0)).Test(t)
		input = capture(Reshape(input, 3, 3)).Test(t)
		five := capture(Constant(builder, NewScalarLiteral(int64(5)))).Test(t)
		input = capture(Add(input, five)).Test(t)
		output := capture(ReduceMin(input)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got := execScalarOutput[int64](t, client, exec)
		require.Equal(t, int64(5), got)
	}
}

func TestReduceWindow(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Test with a 2x2 pooling of a 4x6 matrix with ReduceMax:
	{
		input := capture(Iota(builder, MakeShape(dtypes.Float32, 24), 0)).Test(t)
		input = capture(Reshape(input, 4, 6)).Test(t)
		output := capture(ReduceWindow(input, []int{2, 2}).Max().Done()).Test(t)

		reduceType, _, _, windowDimensions, strides, baseDilations, windowDilations, paddings := DecodeReduceWindow(output)
		require.Equal(t, ReduceMaxType, reduceType)
		require.Equal(t, []int{2, 2}, windowDimensions)
		require.Equal(t, []int{2, 2}, strides)
		require.Equal(t, []int{1, 1}, baseDilations)
		require.Equal(t, []int{1, 1}, windowDilations)
		require.Equal(t, [][2]int{{0, 0}, {0, 0}}, paddings)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.Equal(t, []float32{7, 9, 11, 19, 21, 23}, got)
		require.Equal(t, []int{2, 3}, dims)
	}
}
