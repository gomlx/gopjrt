package pjrt

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestError(t *testing.T) {
	client := getPJRTClient(t)
	builder := xlabuilder.New(t.Name())

	// f(x, y) = x+y
	x := capture(xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	y := capture(xlabuilder.Parameter(builder, "y", 1, xlabuilder.MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	fXY := capture(xlabuilder.Add(x, y)).Test(t)

	// Take program and compile.
	comp := capture(builder.Build(fXY)).Test(t)
	exec, err := client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile program")

	// Call with no arguments: should return an error.
	_, err = exec.Execute().Done()
	require.ErrorContains(t, err, "PJRT error")
	require.ErrorContains(t, err, "Execution supplied 0 buffers but compiled program expected 2 buffers")
	fmt.Printf("Received expected error: %s", err)
}
