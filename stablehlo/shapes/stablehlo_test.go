package shapes

import (
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestToStableHLO(t *testing.T) {
	shape := Make(dtypes.Float32, 1, 10)
	require.Equal(t, "tensor<1x10xf32>", shape.ToStableHLO())

	// Test scalar.
	shape = Make(dtypes.Int32)
	require.Equal(t, "tensor<si32>", shape.ToStableHLO())
}
