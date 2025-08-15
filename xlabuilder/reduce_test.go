package xlabuilder_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMax tests the Max function, as part of the ReduceMax test.
// See https://github.com/openxla/xla/issues/21461
func TestMax(t *testing.T) {
	client := getPJRTClient(t)
	{
		builder := New(fmt.Sprintf("%s-Max(NaN, 1) as Constant", t.Name()))
		input0 := capture(Constant(builder, NewScalarLiteral(math.NaN()))).Test(t)
		input1 := capture(Constant(builder, NewScalarLiteral(1.0))).Test(t)
		output := capture(Max(input0, input1)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got := execScalarOutput[float64](t, client, exec)
		require.True(t, math.IsNaN(got))
		builder.Destroy()
	}
	{
		builder := New(fmt.Sprintf("%s-Max(NaN, 1) as Parameter", t.Name()))
		input0 := capture(Parameter(builder, "x", 0, MakeShape(dtypes.Float64))).Test(t)
		input1 := capture(Parameter(builder, "y", 1, MakeShape(dtypes.Float64))).Test(t)
		input0 = capture(Sqrt(input0)).Test(t)
		input1 = capture(Sqrt(input1)).Test(t)
		output := capture(Max(input0, input1)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got := execWithScalars(t, client, exec, -1.0, 1.0)
		require.True(t, math.IsNaN(got))
		builder.Destroy()
	}
}

func TestReduce(t *testing.T) {
	client := getPJRTClient(t)

	// Test with ReduceMax:
	t.Run("with normal values", func(t *testing.T) {
		builder := New(t.Name())
		input := capture(Iota(builder, MakeShape(dtypes.Float32, 9), 0)).Test(t)
		input = capture(Reshape(input, 3, 3)).Test(t)
		maxColumns := capture(ReduceMax(input, 0)).Test(t)
		maxRows := capture(ReduceMax(input, 1)).Test(t)
		// Check that cache worked: computation and initial value used by both ReduceMax operations must be the same.
		require.Equal(t, maxColumns.ComputationArg, maxRows.ComputationArg)
		require.Equal(t, maxColumns.OpInputs[1], maxRows.OpInputs[1])
		l, err := NewScalarLiteralFromFloat64(0.1, dtypes.Float32)
		require.NoError(t, err)
		pointOne := capture(Constant(builder, l)).Test(t)
		maxRows = capture(Mul(maxRows, pointOne)).Test(t)
		output := capture(Add(maxColumns, maxRows)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.Equal(t, []float32{6.2, 7.5, 8.8}, got)
		require.Equal(t, []int{3}, dims)
		builder.Destroy()
	})

	t.Run("with 0-dimension tensor", func(t *testing.T) {
		builder := New(t.Name())
		literal := capture(NewArrayLiteral([]float32{}, 2, 0, 2)).Test(t)
		input := capture(Constant(builder, literal)).Test(t)
		output := capture(ReduceMax(input, 2)).Test(t)
		comp := capture(builder.Build(output)).Test(t)
		//fmt.Printf("HLO:\n%s\n", comp.TextHLO())
		exec := compile(t, client, comp)
		results, err := exec.Execute().Done()
		require.NoError(t, err)
		require.Len(t, results, 1)
		dims, err := results[0].Dimensions()
		require.NoError(t, err)
		require.Equal(t, []int{2, 0}, dims)
		s, err := results[0].Size()
		require.NoError(t, err)
		require.Equal(t, 0, s)
		builder.Destroy()
	})

	t.Run("with NaN as constant", func(t *testing.T) {
		builder := New(t.Name())
		literal := capture(NewArrayLiteral([]float32{float32(math.NaN()), 1}, 2)).Test(t)
		input := capture(Constant(builder, literal)).Test(t)
		output := capture(ReduceMax(input, 0)).Test(t)
		comp := capture(builder.Build(output)).Test(t)
		//fmt.Printf("HLO:\n%s\n", comp.TextHLO())
		exec := compile(t, client, comp)
		got := execWithScalars[float32](t, client, exec)
		require.True(t, math.IsNaN(float64(got)))
		builder.Destroy()
	})

	t.Run("with NaN as parameter", func(t *testing.T) {
		builder := New(t.Name())
		input := capture(Parameter(builder, "x", 0, MakeShape(dtypes.Float32, 2))).Test(t)
		output := capture(ReduceMax(input, 0)).Test(t)
		comp := capture(builder.Build(output)).Test(t)
		//fmt.Printf("HLO:\n%s\n", comp.TextHLO())
		exec := compile(t, client, comp)
		got, dims := execWithSlices(t, client, exec, []float32{float32(math.NaN()), 1})
		require.Empty(t, dims)
		fmt.Printf("got: %f -- Should be NAN, but with CPU PJRT it's not\n", got[0])
		// TODO: re-enable this test when bug is fixed.
		// See https://github.com/openxla/xla/issues/21461
		// require.True(t, math.IsNaN(float64(got[0])))
		builder.Destroy()
	})

	// Test with ReduceSum and ReduceProduct
	t.Run("with ReduceSum and ReduceProduct", func(t *testing.T) {
		builder := New(t.Name())
		input := capture(Iota(builder, MakeShape(dtypes.Float64, 9), 0)).Test(t)
		input = capture(Reshape(input, 3, 3)).Test(t)
		columnsProduct := capture(ReduceProduct(input, 0)).Test(t)
		rowsSum := capture(ReduceSum(input, 1)).Test(t)
		l, err := NewScalarLiteralFromFloat64(0.001, dtypes.Float64)
		require.NoError(t, err)
		shift := capture(Constant(builder, l)).Test(t)
		rowsSum = capture(Mul(rowsSum, shift)).Test(t)
		output := capture(Add(columnsProduct, rowsSum)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float64](t, client, exec)
		require.Equal(t, []float64{0.003, 28.012, 80.021}, got)
		require.Equal(t, []int{3}, dims)
		builder.Destroy()
	})

	// Test ReduceMin on all dimensions.
	t.Run("with ReduceMin on all dimensions", func(t *testing.T) {
		builder := New(t.Name())
		input := capture(Iota(builder, MakeShape(dtypes.Int64, 9), 0)).Test(t)
		input = capture(Reshape(input, 3, 3)).Test(t)
		five := capture(Constant(builder, NewScalarLiteral(int64(5)))).Test(t)
		input = capture(Add(input, five)).Test(t)
		output := capture(ReduceMin(input)).Test(t)
		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got := execScalarOutput[int64](t, client, exec)
		require.Equal(t, int64(5), got)
		builder.Destroy()
	})
}

func TestReduceLogicalOps(t *testing.T) {
	client := getPJRTClient(t)

	// Test ReduceLogicalAnd, ReduceLogicalOr and ReduceLogicalXor on all dimensions.
	{
		testValues := []*Literal{
			capture(NewArrayLiteral([]bool{false, false, false}, 3)).Test(t),
			capture(NewArrayLiteral([]bool{false, true, false}, 3)).Test(t),
			capture(NewArrayLiteral([]bool{true, false, true}, 3)).Test(t),
			capture(NewArrayLiteral([]bool{true, true, true}, 3)).Test(t),
		}

		ops := []func(x *Op, axes ...int) (*Op, error){ReduceLogicalAnd, ReduceLogicalOr, ReduceLogicalXor}
		opsNames := []string{"LogicalAnd", "LogicalOr", "LogicalXor"}
		wantForOps := [][]any{
			// ReduceLogicalAnd
			{false, false, false, true},
			// ReduceLogicalOr
			{false, true, true, true},
			// ReduceLogicalXor
			{false, true, false, true},
		}

		for opIdx, op := range ops {
			fmt.Printf("Test %s\n", opsNames[opIdx])
			for testIdx, testInput := range testValues {
				builder := New(fmt.Sprintf("%s-Reduce%s-%d", t.Name(), opsNames[opIdx], testIdx))
				input := capture(Constant(builder, testInput)).Test(t)
				output := capture(op(input)).Test(t)
				exec := compile(t, client, capture(builder.Build(output)).Test(t))

				wantAny := wantForOps[opIdx][testIdx]
				switch want := wantAny.(type) {
				case bool:
					got := execScalarOutput[bool](t, client, exec)
					require.Equal(t, want, got)
				case int8:
					got := execScalarOutput[int8](t, client, exec)
					require.Equal(t, want, got)
				}
				builder.Destroy()
			}
		}
	}
}

func TestReduceBitwiseOps(t *testing.T) {
	client := getPJRTClient(t)

	// Test ReduceBitwiseAnd, ReduceBitwiseOr and ReduceBitwiseXor on all dimensions.
	{
		testValues := []*Literal{
			capture(NewArrayLiteral([]bool{false, false, false}, 3)).Test(t),
			capture(NewArrayLiteral([]bool{false, true, false}, 3)).Test(t),
			capture(NewArrayLiteral([]bool{true, false, true}, 3)).Test(t),
			capture(NewArrayLiteral([]bool{true, true, true}, 3)).Test(t),
			capture(NewArrayLiteral([]int8{15, 6, 2}, 3)).Test(t),
			capture(NewArrayLiteral([]int8{-1, -2, -4}, 3)).Test(t),
		}

		ops := []func(x *Op, axes ...int) (*Op, error){ReduceBitwiseAnd, ReduceBitwiseOr, ReduceBitwiseXor}
		opsNames := []string{"ReduceBitwiseAnd", "ReduceBitwiseOr", "ReduceBitwiseXor"}
		wantForOps := [][]any{
			// ReduceBitwiseAnd
			{false, false, false, true, int8(2), int8(-4)},
			// ReduceBitwiseOr
			{false, true, true, true, int8(15), int8(-1)},
			// ReduceBitwiseXor
			{false, true, false, true, int8(11), int8(-3)},
		}

		for opIdx, op := range ops {
			fmt.Printf("Test %s\n", opsNames[opIdx])
			for testIdx, testInput := range testValues {
				builder := New(fmt.Sprintf("%s-Reduce%s-%d", t.Name(), opsNames[opIdx], testIdx))
				input := capture(Constant(builder, testInput)).Test(t)
				output := capture(op(input)).Test(t)
				exec := compile(t, client, capture(builder.Build(output)).Test(t))

				wantAny := wantForOps[opIdx][testIdx]
				switch want := wantAny.(type) {
				case bool:
					got := execScalarOutput[bool](t, client, exec)
					assert.Equal(t, want, got)
				case int8:
					got := execScalarOutput[int8](t, client, exec)
					assert.Equal(t, want, got)
				}
				builder.Destroy()
			}
		}
	}
}

func TestReduceMaxBuggy(t *testing.T) {
	client := getPJRTClient(t)
	{
		builder := New(fmt.Sprintf("%s-ReduceMax with NaN as parameter", t.Name()))
		input := capture(Parameter(builder, "x", 0, MakeShape(dtypes.Float32, 2))).Test(t)
		output := capture(ReduceMax(input, 0)).Test(t)
		comp := capture(builder.Build(output)).Test(t)
		fmt.Printf("HLO:\n%s\n", comp.TextHLO())
		exec := compile(t, client, comp)
		got, dims := execWithSlices(t, client, exec, []float32{float32(math.NaN()), 1})
		require.Empty(t, dims)
		fmt.Printf("got: %f -- Should be NAN, but with CPU PJRT it's not\n", got[0])
		// TODO: re-enable this test when bug is fixed.
		// See https://github.com/openxla/xla/issues/21461
		//require.True(t, math.IsNaN(float64(got[0])))
		builder.Destroy()
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
