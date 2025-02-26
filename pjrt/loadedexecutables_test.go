package pjrt

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestDonatableConfig(t *testing.T) {
	client := getPJRTClient(t)
	builder := xlabuilder.New(t.Name())

	// f(x, y, z) = x*y + z
	x := capture(xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	y := capture(xlabuilder.Parameter(builder, "y", 1, xlabuilder.MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	z := capture(xlabuilder.Parameter(builder, "y", 2, xlabuilder.MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	fX := capture(xlabuilder.Mul(x, y)).Test(t)
	fX = capture(xlabuilder.Add(fX, z)).Test(t)

	// Take program and compile.
	comp := capture(builder.Build(fX)).Test(t)
	exec, err := client.Compile().WithComputation(comp).Done()
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
