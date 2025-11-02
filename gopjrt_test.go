package gopjrt

import (
	"flag"
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var (
	flagPluginName     = flag.String("plugin", "cpu", "PRJT plugin name or full path")
	flagTestAllDevices = flag.Bool("alldevices", false, "Test all devices. Defaults to false "+
		"because CPU PJRT adversive 4 devices (to help test multi-device setup)")
)

func init() {
	klog.InitFlags(nil)
}

// TestEndToEnd builds, compiles, and executes a minimal computation f(x) = x^2 using stablehlo to build the computation,
// and pjrt to compile and execute it.
func TestEndToEnd(t *testing.T) {
	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *flagPluginName)
	fmt.Printf("Loaded %s\n", plugin)
	fmt.Printf("\t- Attributes=%+v\n", plugin.Attributes())
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("	client: %s\n", client)

	// List devices.
	addressableDevices := client.AddressableDevices()
	fmt.Println("Addressable devices:")
	for deviceNum, device := range addressableDevices {
		hardwareId := device.LocalHardwareId()
		addressable, err := device.IsAddressable()
		require.NoError(t, err)
		desc, err := device.GetDescription()
		require.NoError(t, err)
		fmt.Printf("\tDevice #%d: hardwareId=%d, addressable=%v, description=%s\n", deviceNum, hardwareId, addressable, desc.DebugString())
	}

	// Default device assignment for SPMD:
	fmt.Println()
	fmt.Printf("Default device assignment for SPMD:\n")
	for i := range client.NumDevices() {
		numReplicas := i + 1
		spmdDefaultAssignment, err := client.DefaultDeviceAssignment(numReplicas, 1)
		require.NoError(t, err, "Failed to get default device assignment")
		fmt.Printf("\tWith %d devices: %v\n", numReplicas, spmdDefaultAssignment)
	}
	fmt.Println()

	// f(x) = x^2+1
	builder := stablehlo.New("x_times_x_plus_1") // Use valid identifier for module name
	scalarF32 := shapes.Make(dtypes.F32)         // Scalar float32 shape

	// Create main function and define its inputs.
	mainFn := builder.Main()
	x := mainFn.NamedInput("x", scalarF32)

	// Build computation graph
	fX, err := stablehlo.Multiply(x, x)
	require.NoError(t, err, "Failed operation Mul")

	one, err := mainFn.ConstantFromScalar(float32(1))
	require.NoError(t, err, "Failed to create a constant for 1")

	fX, err = stablehlo.Add(fX, one)
	require.NoError(t, err, "Failed operation Add")
	err = mainFn.Return(fX) // Set the return value for the main function
	require.NoError(t, err, "Failed to set return value")

	// Get computation created.
	compBytes, err := builder.Build()
	require.NoError(t, err, "Failed to build StableHLO from ops.")
	fmt.Printf("StableHLO:\n%s\n", string(compBytes))

	// Compile program.
	var loadedExec *pjrt.LoadedExecutable
	loadedExec, err = client.Compile().WithStableHLO(compBytes).Done()
	require.NoErrorf(t, err, "Failed to compile program")
	fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

	// Test values:
	inputs := []float32{0.1, 1, 3, 4, 5}
	wants := []float32{1.01, 2, 10, 17, 26}
	fmt.Printf("f(x) = x^2 + 1:\n")
	for ii, input := range inputs {
		for deviceNum := range addressableDevices {
			if deviceNum > 0 {
				if !*flagTestAllDevices {
					break
				}
			}
			// Transfer input to an on-device buffer.
			inputBuffer, err := pjrt.ScalarToBufferOnDeviceNum(client, deviceNum, input)
			require.NoErrorf(t, err, "Failed to create on-device buffer for input %v, deviceNum=%d", input, deviceNum)

			// Execute: it returns the output on-device buffer(s).
			outputBuffers, err := loadedExec.Execute(inputBuffer).OnDevicesByNum(deviceNum).Done()
			require.NoErrorf(t, err, "Failed to execute on input %d, deviceNum=%d", input, deviceNum)

			// Transfer output on-device buffer to a "host" value (in Go).
			output, err := pjrt.BufferToScalar[float32](outputBuffers[0])
			require.NoErrorf(t, err, "Failed to transfer results of execution on input %d", input)

			// Print and check value is what we wanted.
			fmt.Printf("\t[device#%d] f(x=%g) = %g\n", deviceNum, input, output)
			require.InDelta(t, output, wants[ii], 0.001)

			// Release inputBuffer -- and don't wait for the GC.
			require.NoError(t, inputBuffer.Destroy())
		}
	}

	// Destroy the client and leave.
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
