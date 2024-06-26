package xlabuilder

import (
	"github.com/stretchr/testify/require"
	"gopjrt/dtypes"
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
	require.NotPanics(t, func() { NewLiteralFromFlatData([]float32{1, 2, 3, 4, 5, 6}, 2, 3).Destroy() })
}
