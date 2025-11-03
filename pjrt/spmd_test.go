package pjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
)

// TestSPMD builds, compiles, and executes a minimal distributed (SPMD = Single Program Multiple Data) computation,
// and uses PJRT to compile and execute it.
func TestSPMD(t *testing.T) {
	// PJRT plugin and create a client.
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *flagPluginName)
	fmt.Printf("Loaded %s\n", plugin)
	fmt.Printf("\t- Attributes=%+v\n", plugin.Attributes())
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("	client: %s\n", client)

	// List devices.
	numReplicas := client.NumDevices()
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

	// Create replicaGroups [numPartitions=1][numReplicas=numDevices] according to the device assignment.
	fmt.Println()
	fmt.Printf("Device assignment for SPMD:\n")
	spmdDefaultAssignment, err := client.DefaultDeviceAssignment(numReplicas, 1)
	require.NoError(t, err, "Failed to get default device assignment")
	fmt.Printf("\tWith %d devices: %v\n", numReplicas, spmdDefaultAssignment)
	replicaGroups := [][]int{spmdDefaultAssignment}

	t.Run("CollectiveAllReduce_sum", func(t *testing.T) {
		// f(x_r) = Reduce_sum(CollectiveAllReduce_sum(x_r))
		builder := stablehlo.New("sum_x0")
		mainFn := builder.Main()
		argShape := shapes.Make(dtypes.F32, 2)
		x := mainFn.NamedInput("x", argShape)
		reductionFn := mainFn.Closure()
		lhs := reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32))
		rhs := reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32))
		must(reductionFn.Return(must1(stablehlo.Add(lhs, rhs))))
		reducedReplicas, err := stablehlo.CollectiveAllReduce(x, replicaGroups, reductionFn)
		require.NoError(t, err, "Failed operation CollectiveAllReduce")
		zero := must1(mainFn.ConstantFromScalar(float32(0)))
		sum, err := stablehlo.Reduce(reducedReplicas, zero, reductionFn, 0)
		require.NoError(t, err, "Failed operation Reduce")
		err = mainFn.Return(sum)
		require.NoError(t, err, "Failed operation Return")

		// Get computation created.
		compBytes, err := builder.Build()
		require.NoError(t, err, "Failed to build StableHLO from ops.")
		fmt.Printf("\nStableHLO:\n%s\n", string(compBytes))

		// Compile program.
		var loadedExec *LoadedExecutable
		loadedExec, err = client.Compile().
			WithStableHLO(compBytes).
			WithSPMD(numReplicas).
			Done()
		require.NoErrorf(t, err, "Failed to compile program")
		_, _, deviceAssignments, err := loadedExec.GetDeviceAssignment()
		require.NoError(t, err, "Failed to get device assignment for execution")

		// Test values:
		fmt.Printf("f(x_r) = Reduce_sum(CollectiveAllReduce_sum(x_r)):\n")
		inputBuffers := make([]*Buffer, numReplicas)
		for ii := range numReplicas {
			input := []float32{1.0 * float32(ii+1), 0.1 * float32(ii+1)}
			// Transfer input to an on-device buffer.
			fmt.Printf("\tInput #%d for device #%d is = %v\n", ii, deviceAssignments[ii], input)
			inputBuffers[ii], err = client.BufferFromHost().
				FromFlatDataWithDimensions(input, []int{2}).
				ToDeviceNum(deviceAssignments[ii]).
				Done()
			require.NoErrorf(t, err, "Failed to create on-device buffer for input %v, deviceNum=%d", input, ii)
		}

		// Execute: it returns the output on-device buffer(s).
		outputBuffers, err := loadedExec.Execute(inputBuffers...).Done()
		require.NoErrorf(t, err, "Failed to execute SPMD computation")
		require.Lenf(t, outputBuffers, numReplicas, "Expected %d outputs, got %d", numReplicas, len(outputBuffers))

		// Transfer output on-device buffer to a "host" value (in Go).
		output, err := BufferToScalar[float32](outputBuffers[0])
		require.NoErrorf(t, err, "Failed to transfer results of execution")

		// Print and check value is what we wanted.
		fmt.Printf("\tResult for %d replicas is = %g\n", numReplicas, output)
		want := float32(numReplicas)
		want = (want * (want + 1) / 2) * 1.1
		require.InDelta(t, output, want, 0.001)

		// Release inputBuffers -- and don't wait for the GC.
		for _, inputBuffer := range inputBuffers {
			require.NoError(t, inputBuffer.Destroy())
		}
		for _, outputBuffer := range outputBuffers {
			require.NoError(t, outputBuffer.Destroy())
		}
	})

	// Destroy the client and leave.
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
