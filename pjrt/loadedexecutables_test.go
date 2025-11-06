package pjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
)

func TestDonatableConfig(t *testing.T) {
	client := getPJRTClient(t)
	builder := stablehlo.New(t.Name())
	mainFn := builder.Main()

	// f(x, y, z) = x*y + z
	scalarF32 := shapes.Make(dtypes.F32)
	x := mainFn.NamedInput("x", scalarF32) // Scalar float32.
	y := mainFn.NamedInput("y", scalarF32) // Scalar float32.
	z := mainFn.NamedInput("z", scalarF32) // Scalar float32.
	fX := capture(stablehlo.Multiply(x, y)).Test(t)
	fX = capture(stablehlo.Add(fX, z)).Test(t)

	// Take program and compile.
	err := mainFn.Return(fX)
	require.NoError(t, err, "Failed to set return value")
	compBytes := capture(builder.Build()).Test(t)
	exec, err := client.Compile().WithStableHLO(compBytes).Done()
	require.NoErrorf(t, err, "Failed to compile program")

	fmt.Println("Memory usage:")
	fmt.Printf("OnDevice: %+v\n", exec.OnDeviceMemoryUsageStats)
	fmt.Printf("OnHost: %+v\n", exec.OnHostMemoryUsageStats)

	// Test the ExecutionConfig:
	c := exec.Execute(nil, nil, nil)                       // nil values, we are not going to actually execute it.
	require.Equal(t, []int{0, 1, 2}, c.nonDonatableInputs) // None of the inputs to be donated by default.
	c = c.Donate(1)                                        // Donate 1.
	require.Equal(t, []int{0, 2}, c.nonDonatableInputs)
	c = c.Donate(0) // Donate 0.
	require.Equal(t, []int{2}, c.nonDonatableInputs)
	c = c.Donate(0) // Donate 0 again.
	require.Equal(t, []int{2}, c.nonDonatableInputs)
}
