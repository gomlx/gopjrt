package gopjrt

import (
	"flag"
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var (
	flagPluginName     = flag.String("plugin", "cpu", "PRJT plugin name or full path")
	flagTestAllDevices = flag.Bool("alldevices", false, "Test all devices. Defaults to false because CPU PJRT seems to falsely advertise more than one device")
	flagSaveHLO        = flag.String("savehlo", "", "Save HLO to file. If empty, do not save.")
	flagLoadHLO        = flag.String("loadhlo", "", "Load HLO to test from file (as opposed to the one created with XlaBuilder.")
)

func init() {
	klog.InitFlags(nil)
}

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
	if *flagSaveHLO != "" {
		cBuffer := comp.SerializedHLO()
		err := os.WriteFile(*flagSaveHLO, cBuffer.Bytes(), 0644)
		require.NoError(t, err)
	}

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
	fmt.Println()

	// Compile program.
	var loadedExec *pjrt.LoadedExecutable
	if *flagLoadHLO == "" {
		loadedExec, err = client.Compile().WithComputation(comp).Done()
		require.NoErrorf(t, err, "Failed to compile our x^2 HLO program")
	} else {
		hloProgram, err := os.ReadFile(*flagLoadHLO)
		require.NoErrorf(t, err, "Failed to load HLO from file %q", *flagLoadHLO)
		loadedExec, err = client.Compile().WithHLO(hloProgram).Done()
		require.NoErrorf(t, err, "Failed to compile HLO program loaded from %q", *flagLoadHLO)
	}
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
				fmt.Printf("\t*** Warning: if using a CPU PJRT, it is known to fail on devices > 0: it seems to falsely advertise it has more than one device\n")
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
