package pjrt_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/gomlx/stablehlo/types/shardy"
	"github.com/stretchr/testify/require"
)

func TestShardy(t *testing.T) {
	plugin, err := pjrt.GetPlugin(*pjrt.FlagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *pjrt.FlagPluginName)
	fmt.Printf("Loaded %s\n", plugin)
	fmt.Printf("\t- Attributes=%+v\n", plugin.Attributes())
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("	client: %s\n", client)

	// We will test it with 2 devices.
	const numReplicas = 2
	numDevices := client.NumDevices()
	if numDevices < numReplicas {
		t.Skipf("Skipping test: not enough devices: %d < %d", numDevices, numReplicas)
		return
	}

	t.Run("input-data-sharding", func(t *testing.T) {
		mesh := must1(shardy.NewDeviceMesh("data_mesh", []int{2}, []string{"data"}))
		program := []byte(`module @TestShardy_input_data_sharding attributes {mhlo.num_replicas = 2:i32,  mhlo.num_partitions = 1:i32} {
		 sdy.mesh @data_mesh = <["data"=2]>
		 func.func @main(%arg0: tensor<2x3xf32> { sdy.sharding = #sdy.sharding<@data_mesh, [{"data"}, {}]> }) -> tensor<f32> {
		   %1 = "stablehlo.constant"() { value = dense<0.0> : tensor<f32> } : () -> tensor<f32>
		   %2 = "stablehlo.reduce"(%arg0, %1) ({
		     ^reductionFn(%lhs: tensor<f32>, %rhs: tensor<f32>) :
		         %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
		         "stablehlo.return"(%0) : (tensor<f32>) -> ()
		   }) { dimensions = array<i64: 0, 1> } : (tensor<2x3xf32>, tensor<f32>) -> tensor<f32>
		   "stablehlo.return"(%2) : (tensor<f32>) -> ()
		 }
		}`)
		x0 := must1(client.BufferFromHost().ToDeviceNum(0).FromFlatDataWithDimensions(
			[]float32{0, 1, 2}, []int{1, 3}).Done())
		x1 := must1(client.BufferFromHost().ToDeviceNum(1).FromFlatDataWithDimensions(
			[]float32{0, 0.1, 0.2}, []int{1, 3}).Done())
		outputs := shardyCompileAndExecute(t, client, program, mesh, x0, x1)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3.3}, nil},
			{[]float32{3.3}, nil},
		}, outputs)
	})
}

// compileAndExecute program with PJRT. All inputs are donated.
func shardyCompileAndExecute(t *testing.T, client *pjrt.Client, program []byte,
	mesh *shardy.DeviceMesh, inputs ...*pjrt.Buffer) []*pjrt.Buffer {
	loadedExec, err := client.Compile().
		WithStableHLO(program).
		WithShardy(mesh.NumDevices()).
		WithDeviceAssignment(mesh.DeviceAssignment()).
		Done()
	require.NoErrorf(t, err, "failed to compile program: \n%s", program)
	defer func() {
		err := loadedExec.Destroy()
		if err != nil {
			t.Errorf("failed to destroy loaded exec: %+v", err)
		}
	}()
	outputBuffers, err := loadedExec.Execute(inputs...).DonateAll().Done()
	require.NoErrorf(t, err, "failed to execute program: \n%s", program)
	return outputBuffers
}

type FlatAndDims struct {
	Flat any
	Dims []int
}

// requireBuffersEqual checks that the actual buffers contents match the expected flat values.
// It destroys the buffers.
func requireBuffersEqual(t *testing.T, expected []FlatAndDims, got []*pjrt.Buffer) {
	defer func() {
		for _, b := range got {
			err := b.Destroy()
			if err != nil {
				t.Errorf("failed to destroy buffer: %+v", err)
			}
		}
	}()
	require.Len(t, got, len(expected))
	for i, b := range got {
		gotFlat, gotDims, err := b.ToFlatDataAndDimensions()
		expectedShape, err := shapes.FromAnyValue(expected[i].Flat)
		require.NoErrorf(t, err, "failed to get shape for output #%d: %v", i, expected[i].Flat)
		dtype := expectedShape.DType
		fmt.Printf("\t - output #%d:\n\t   - Got: dims=%v, flat_values=%v\n", i, gotDims, gotFlat)
		fmt.Printf("\t   - Want(%s): dims=%v, flat_values=%v\n", dtype, expected[i].Dims, expected[i].Flat)
		require.NoErrorf(t, err, "failed to get buffer contents for output #%d, expected flat value %v", i, expected[i].Flat)
		require.Equalf(t, expected[i].Dims, gotDims, "output #%d dims don't match", i)
		switch dtype {
		case dtypes.Float64, dtypes.Float32:
			require.InDeltaSlicef(t, expected[i].Flat, gotFlat, 1e-4, "output #%d flat values don't match", i)
		default:
			require.Equalf(t, expected[i].Flat, gotFlat, "output #%d flat values don't match", i)
		}
	}
}
