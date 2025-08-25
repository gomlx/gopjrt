package stablehlo

import (
	"testing"

	"github.com/gomlx/gopjrt/stablehlo/optypes"
	"github.com/gomlx/gopjrt/stablehlo/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuilder_Simple(t *testing.T) {
	b := New("test_program")
	fn := b.NewFunction("main", true, nil, nil)

	// Create a constant.
	c1, err := fn.NewConstant(1.0)
	require.NoError(t, err)

	// Create another constant.
	c2, err := fn.NewConstant(2.0)
	require.NoError(t, err)

	// Add them.
	sum, err := fn.AddOp(optypes.Add, c1, c2)
	require.NoError(t, err)
	fn.Outputs = []shapes.Shape{sum.shape}

	comp, err := b.Build()
	require.NoError(t, err)

	// The attribute formatting for floating point values is a bit tricky.
	// Let's check for the parts we care about.
	assert.Contains(t, comp.StableHLO, `func.func public @main() -> (tensor<f64>)`)
	assert.Contains(t, comp.StableHLO, `%0 = "stablehlo.constant"`)
	assert.Contains(t, comp.StableHLO, `dense<1`)
	assert.Contains(t, comp.StableHLO, `%1 = "stablehlo.constant"`)
	assert.Contains(t, comp.StableHLO, `dense<2`)
	assert.Contains(t, comp.StableHLO, `%2 = "stablehlo.add"(%0, %1) : (tensor<f64>, tensor<f64>) -> tensor<f64>`)
	assert.Contains(t, comp.StableHLO, `"func.return"(%2) : (tensor<f64>) -> ()`)
}

func TestBuilder_NoMain(t *testing.T) {
	b := New("test_program")
	b.NewFunction("not_main", true, nil, nil)
	_, err := b.Build()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "program must have a main function")
}
