package xlabuilder_test

import (
	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
	"testing"
)

func TestLiterals(t *testing.T) {
	// Creates and destroys some of the created literals. Check it compiles and doesn't crash during execution.
	//
	// In gopjrt package there are some tests that check that the values loaded into the literals are actually correct.
	require.NotPanics(t, func() { NewLiteralFromShape(MakeShape(dtypes.Float64, 1000, 4, 2)).Destroy() })
	require.NotPanics(t, func() { _ = NewScalarLiteral[float32](0) })
	require.NotPanics(t, func() { _ = NewScalarLiteral[complex128](complex(1.0, 0.0)) })
	require.NotPanics(t, func() { NewScalarLiteral[int8](0).Destroy() })
	require.NotPanics(t, func() { NewArrayLiteral([]float32{1, 2, 3, 4, 5, 6}, 2, 3).Destroy() })
	require.NotPanics(t, func() { NewArrayLiteralFromAny([]float64{1, 2, 3, 4, 5, 6}, 2, 3).Destroy() })

	// Check that various literals get correcly interpreted in PRJT.
	client := getPJRTClient(t)
	builder := New(t.Name())
	output := capture(Constant(builder, NewScalarLiteral(int16(3)))).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	require.Equal(t, int16(3), execScalarOutput[int16](t, client, exec))

	builder = New(t.Name())
	output = capture(Constant(builder, NewScalarLiteralFromFloat64(7, dtypes.Complex128))).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	require.Equal(t, complex128(7), execScalarOutput[complex128](t, client, exec))

	builder = New(t.Name())
	output = capture(Constant(builder, NewScalarLiteralFromAny(float16.Fromfloat32(15e-3)))).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	require.Equal(t, float16.Fromfloat32(15e-3), execScalarOutput[float16.Float16](t, client, exec))

	builder = New(t.Name())
	output = capture(Constant(builder, NewArrayLiteralFromAny([]float64{1, 3, 5, 7, 11, 13}, 3, 2))).Test(t)
	exec = compile(t, client, capture(builder.Build(output)).Test(t))
	gotFlat, gotDims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []int{3, 2}, gotDims)
	require.Equal(t, []float64{1, 3, 5, 7, 11, 13}, gotFlat)
}
