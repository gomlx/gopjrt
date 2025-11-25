package pjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
)

func TestError(t *testing.T) {
	client := getPJRTClient(t)
	builder := stablehlo.New(t.Name())
	mainFn := builder.Main()

	// f(x, y) = x+y
	scalarF32 := shapes.Make(dtypes.F32)
	x := must1(mainFn.NamedInput("x", scalarF32)) // Scalar float32.
	y := must1(mainFn.NamedInput("y", scalarF32)) // Scalar float32.
	fXY := capture(stablehlo.Add(x, y)).Test(t)

	// Take program and compile.
	err := mainFn.Return(fXY)
	require.NoError(t, err, "Failed to set return value")
	compBytes := capture(builder.Build()).Test(t)
	exec, err := client.Compile().WithStableHLO(compBytes).Done()
	require.NoErrorf(t, err, "Failed to compile program")

	// Call with no arguments: should return an error.
	_, err = exec.Execute().Done()
	require.ErrorContains(t, err, "PJRT error")
	require.ErrorContains(t, err, "Execution supplied 0 buffers but compiled program expected 2 buffers")
	fmt.Printf("Received expected error: %s", err)
}
