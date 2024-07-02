package gopjrt

import (
	"flag"
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"testing"
)

var flagPluginName = flag.String("plugin", "cpu", "PRJT plugin name or full path")

// TestEndToEnd builds, compiles and executes a minimal computation f(x) = x^2 using xlabuilder to build the computation,
// and pjrt to compile and execute it.
func TestEndToEnd(t *testing.T) {
	// f(x) = x^2+1
	builder := xlabuilder.New("x*x+1")
	x, err := xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err, "Failed to create Parameter")
	fX, err := xlabuilder.Mul(x, x)
	require.NoError(t, err, "Failed operation Mul")
	one, err := xlabuilder.Constant(builder, xlabuilder.NewScalarLiteral(float32(1)))
	require.NoError(t, err, "Failed to create constant of 1")
	fX, err = xlabuilder.Add(fX, one)
	require.NoError(t, err, "Failed operation Add")

	// Get computation created.
	comp, err := builder.Build(fX)
	require.NoError(t, err, "Failed to build XlaComputation from ops.")
	//fmt.Printf("HloModule proto:\n%s\n\n", comp.TextHLO())

	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *flagPluginName)
	fmt.Printf("Loaded %s\n", plugin)
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("	client: %s\n", client)

	// Compile program.
	loadedExec, err := client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile our x^2 HLO program")
	fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

	// Test values:
	inputs := []float32{0.1, 1, 3, 4, 5}
	wants := []float32{1.01, 2, 10, 17, 26}
	fmt.Printf("f(x) = x^2 + 1:\n")
	for ii, input := range inputs {
		// Transfer input to a on-device buffer.
		inputBuffer, err := pjrt.ScalarToBuffer(client, input)
		require.NoErrorf(t, err, "Failed to create on-device buffer for input %d", input)

		// Execute: it returns the output on-device buffer(s).
		outputBuffers, err := loadedExec.Execute(inputBuffer).Done()
		require.NoErrorf(t, err, "Failed to execute on input %d", input)

		// Transfer output on-device buffer to a "host" value (in Go).
		output, err := pjrt.BufferToScalar[float32](outputBuffers[0])
		require.NoErrorf(t, err, "Failed to transfer results of execution on input %d", input)

		// Print an check value is what we wanted.
		fmt.Printf("\tf(x=%g) = %g\n", input, output)
		require.InDelta(t, output, wants[ii], 0.001)
	}

	// Destroy the client and leave.
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
