package xlabuilder_test

import (
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestLiterals(t *testing.T) {
	// Creates and destroys some of the created literals. Check it compiles and doesn't crash during execution.
	//
	// In gopjrt package there are some tests that check that the values loaded into the literals are actually correct.
	l, err := NewLiteralFromShape(MakeShape(dtypes.Float64, 1000, 4, 2))
	require.NoError(t, err)
	require.NotPanics(t, func() { l.Destroy() })
	require.NotPanics(t, func() { _ = NewScalarLiteral[float32](0) })
	require.NotPanics(t, func() { _ = NewScalarLiteral[complex128](complex(1.0, 0.0)) })
	require.NotPanics(t, func() { NewScalarLiteral[int8](0).Destroy() })

	l, err = NewArrayLiteral([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	require.NoError(t, err)
	l.Destroy()

	l, err = NewArrayLiteralFromAny([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	require.NoError(t, err)
	l.Destroy()

	// Error expected:
	// 1. Creating scalar with more than one value.
	l, err = NewArrayLiteralFromAny([]float64{1, 2}) // len(dimensions)==0 -> scalar
	require.Error(t, err)
	// 2. Wrong number of elements.
	l, err = NewArrayLiteralFromAny([]float64{1, 2}, 3) // len(dimensions)==0 -> scalar
	require.Error(t, err)

	// Check that various literals get correctly interpreted in PRJT.
	client := getPJRTClient(t)
	builder := New(t.Name())
	output := capture(Constant(builder, NewScalarLiteral(int16(3)))).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	require.Equal(t, int16(3), execScalarOutput[int16](t, client, exec))

	builder = New(t.Name())
	l, err = NewScalarLiteralFromFloat64(7, dtypes.Complex128)
	require.NoError(t, err)
	output = capture(Constant(builder, l)).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	require.Equal(t, complex128(7), execScalarOutput[complex128](t, client, exec))

	builder = New(t.Name())
	l, err = NewScalarLiteralFromAny(float16.Fromfloat32(15e-3))
	require.NoError(t, err)
	output = capture(Constant(builder, l)).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	require.Equal(t, float16.Fromfloat32(15e-3), execScalarOutput[float16.Float16](t, client, exec))

	builder = New(t.Name())
	l, err = NewArrayLiteralFromAny([]float64{1, 3, 5, 7, 11, 13}, 3, 2)
	require.NoError(t, err)
	output = capture(Constant(builder, l)).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	gotFlat, gotDims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []int{3, 2}, gotDims)
	require.Equal(t, []float64{1, 3, 5, 7, 11, 13}, gotFlat)
}

func TestZeroLiteral(t *testing.T) {
	// NewLIteralFromShape
	l, err := NewLiteralFromShape(MakeShape(dtypes.Float64, 1, 1, 0, 1))
	require.NoError(t, err)
	require.NotPanics(t, func() { l.Destroy() })
	require.NotPanics(t, func() { _ = NewScalarLiteral[float32](0) })
	require.NotPanics(t, func() { _ = NewScalarLiteral[complex128](complex(1.0, 0.0)) })
	require.NotPanics(t, func() { NewScalarLiteral[int8](0).Destroy() })

	// NewArrayLiteral
	l, err = NewArrayLiteral([]int32{}, 1, 1, 0, 256)
	require.NoError(t, err)
	l.Destroy()

	// no way right now to pass in nil like NewArrayLiteral(nil, 1, 1, 0), I think might have to change function signature? not too sure...
	var flat []float32
	l, err = NewArrayLiteral(flat, 1, 1, 0, 256)
	require.NoError(t, err)
	l.Destroy()
	// MakeShape
	dims := []int{1, 0, 0, 3}
	newShape := MakeShape(dtypes.Float64, dims...)
	require.Equal(t, newShape.Dimensions, dims)

	//MakeShapeOrError
	newShape, _ = MakeShapeOrError(dtypes.F32, dims...)
	require.Equal(t, newShape.Dimensions, dims)
	badDims := []int{1, -1, 4, 0}
	_, err = MakeShapeOrError(dtypes.Int64, badDims...)
	require.NotEqual(t, nil, err)
	badFunc := func() {
		MakeShape(dtypes.Float32, badDims...)
	}
	require.Panics(t, badFunc)

	// TODO: Literal.Data modifications, NewArrayLiteral modifications
}
