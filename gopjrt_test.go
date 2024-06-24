package gopjrt

import (
	"flag"
	"fmt"
	"github.com/stretchr/testify/require"
	"gopjrt/dtypes"
	"gopjrt/pjrt"
	"gopjrt/xlabuilder"
	"testing"
)

var flagPluginName = flag.String("plugin", "cpu", "PRJT plugin name or full path")

func TestEndToEnd(t *testing.T) {
	// f(x) = x^2
	builder := xlabuilder.New("x*x")
	x, err := xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err, "Failed to create Parameter")
	fX, err := xlabuilder.Mul(x, x)
	require.NoError(t, err, "Failed operation Mul")

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

	// Sanity check: verify that there are addressable devices -- not needed usually, an proper error will be returned
	// if there isn't any.
	devices, err := client.AddressableDevices()
	require.NoErrorf(t, err, "Failed to fetch AddressableDevices() from client on %s", plugin)
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	// Compile program.
	loadedExec, err := client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile our x^2 HLO program")
	fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

	// Test values:
	inputs := []float32{0.1, 1, 3, 4, 5}
	wants := []float32{0.01, 1, 9, 16, 25}
	fmt.Printf("f(x) = x^2 :\n")
	for ii, input := range inputs {
		// Transfer input to a on-device buffer.
		inputBuffer, err := pjrt.BufferFromScalar(client, input)
		require.NoErrorf(t, err, "Failed to create on-device buffer for input %d", input)

		// Execute: it returns the output on-device buffer(s).
		outputBuffers, err := loadedExec.Execute(inputBuffer)
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
