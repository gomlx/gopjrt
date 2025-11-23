package pjrt_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
)

func panicf(format string, args ...any) {
	panic(errors.Errorf(format, args...))
}

func must(err error) {
	if err != nil {
		panicf("Failed: %+v", errors.WithStack(err))
	}
}

func must1[T any](t T, err error) T {
	must(err)
	return t
}

var (
	allReduceProgramFail = []byte(
		`
module @TestDistributedAllReduce_multiple_values__different_dtype attributes {stablehlo.num_replicas = 2 } {
  func.func @main(%x: tensor<f32>, %y: tensor<2xf32>, %z: tensor<3xf64>) -> (tensor<f32>, tensor<2xf32>, tensor<3xf64>) {
    %1 = "stablehlo.all_reduce"(%z) ({
      ^computation(%lhs: tensor<f64>, %rhs: tensor<f64>) :
          %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
          "stablehlo.return"(%0) : (tensor<f64>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<3xf64>) -> tensor<3xf64>
    %3, %4 = "stablehlo.all_reduce"(%x, %y) ({
      ^computation(%lhs: tensor<f32>, %rhs: tensor<f32>) :
          %2 = "stablehlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<f32>, tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>)
    "stablehlo.return"(%3, %4, %1) : (tensor<f32>, tensor<2xf32>, tensor<3xf64>) -> ()
  }
}
`)

	allReduceProgram = []byte(
		`
module @TestDistributedAllReduce_multiple_values__different_dtype attributes {stablehlo.num_replicas = 2 } {
  func.func @main(%x: tensor<f32>, %y: tensor<2xf32>, %z: tensor<3xf64>) -> (tensor<f32>, tensor<2xf32>, tensor<3xf64>) {
    %1 = "stablehlo.all_reduce"(%z) ({
      ^computation(%lhs: tensor<f64>, %rhs: tensor<f64>) :
          %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
          "stablehlo.return"(%0) : (tensor<f64>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_id = 1
    } : (tensor<3xf64>) -> tensor<3xf64>
    %3, %4 = "stablehlo.all_reduce"(%x, %y) ({
      ^computation(%lhs: tensor<f32>, %rhs: tensor<f32>) :
          %2 = "stablehlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_id = 2
    } : (tensor<f32>, tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>)
    "stablehlo.return"(%3, %4, %1) : (tensor<f32>, tensor<2xf32>, tensor<3xf64>) -> ()
  }
}
`)

	_ = allReduceProgram
	_ = allReduceProgramFail
)

func TestCollectiveAllReduce(t *testing.T) {
	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*pjrt.FlagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *pjrt.FlagPluginName)
	fmt.Printf("Loaded %s\n", plugin)
	fmt.Printf("\t- Attributes=%+v\n", plugin.Attributes())
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("	client: %s\n", client)

	// Verify that we have enough devices.
	devices := client.AddressableDevices()
	if len(devices) < 2 {
		t.Skipf("TestCollectiveAllReduce requires at least 2 devices, only %d available", len(devices))
	}

	// Compile program: the default compilation is "portable", meaning it can be executed by any device.
	var loadedExec *pjrt.LoadedExecutable
	loadedExec, err = client.Compile().
		WithStableHLO(allReduceProgram).
		WithSPMD(2).
		Done()
	require.NoErrorf(t, err, "Failed to compile program")
	fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)
}

// TestSPMD builds, compiles, and executes a minimal distributed (SPMD = Single Program Multiple Data) computation,
// and uses PJRT to compile and execute it.
func TestSPMD(t *testing.T) {
	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*pjrt.FlagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *pjrt.FlagPluginName)
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
		hardwareId := device.LocalHardwareID()
		addressable, err := device.IsAddressable()
		require.NoError(t, err)
		desc, err := device.GetDescription()
		require.NoError(t, err)
		fmt.Printf("\tDevice #%d: hardwareId=%d, addressable=%v, description=%s\n",
			deviceNum, hardwareId, addressable, desc.DebugString())
	}

	// Create replicaGroups [numPartitions=1][numReplicas=numDevices] according to the device assignment.
	fmt.Println()
	fmt.Printf("Device assignment for SPMD:\n")
	spmdDefaultAssignment, err := client.DefaultDeviceAssignment(numReplicas, 1)
	require.NoError(t, err, "Failed to get default device assignment")
	fmt.Printf("\tWith %d devices: %v\n", numReplicas, spmdDefaultAssignment)
	replicaGroups := [][]int{spmdDefaultAssignment}

	t.Run("AllReduce_sum", func(t *testing.T) {
		// f(x_r) = Reduce_sum(CollectiveAllReduce_sum(x_r))
		builder := stablehlo.New("sum_x0")
		mainFn := builder.Main()
		argShape := shapes.Make(dtypes.F32, 2)
		x := must1(mainFn.NamedInput("x", argShape))
		reductionFn := mainFn.Closure()
		lhs := must1(reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
		rhs := must1(reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
		must(reductionFn.Return(must1(stablehlo.Add(lhs, rhs))))
		reducedReplicas, err := stablehlo.AllReduce([]*stablehlo.Value{x}, replicaGroups, reductionFn)
		require.NoError(t, err, "Failed operation CollectiveAllReduce")
		zero := must1(mainFn.ConstantFromScalar(float32(0)))
		sum, err := stablehlo.Reduce(reducedReplicas[0], zero, reductionFn, 0)
		require.NoError(t, err, "Failed operation Reduce")
		err = mainFn.Return(sum)
		require.NoError(t, err, "Failed operation Return")

		// Get computation created.
		compBytes, err := builder.Build()
		require.NoError(t, err, "Failed to build StableHLO from ops.")
		fmt.Printf("\nStableHLO:\n%s\n", string(compBytes))

		// Compile program.
		var loadedExec *pjrt.LoadedExecutable
		loadedExec, err = client.Compile().
			WithStableHLO(compBytes).
			WithSPMD(numReplicas).
			Done()
		require.NoErrorf(t, err, "Failed to compile program")
		_, _, deviceAssignments, err := loadedExec.GetDeviceAssignment()
		require.NoError(t, err, "Failed to get device assignment for execution")

		// Test values:
		fmt.Printf("f(x_r) = Reduce_sum(CollectiveAllReduce_sum(x_r)):\n")
		inputBuffers := make([]*pjrt.Buffer, numReplicas)
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
		output, err := pjrt.BufferToScalar[float32](outputBuffers[0])
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
