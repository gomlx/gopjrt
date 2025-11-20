package pjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestZeroDim(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices := client.AddressableDevices()
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	// Test different zero-dimension buffer shapes
	testCases := []struct {
		name       string
		dtype      dtypes.DType
		dimensions []int
		expectSize int
	}{
		{
			name:       "rank_1_zero_dim",
			dtype:      dtypes.Float32,
			dimensions: []int{0},
			expectSize: 0,
		},
		{
			name:       "rank_2_zero_first",
			dtype:      dtypes.Int32,
			dimensions: []int{0, 5},
			expectSize: 0,
		},
		{
			name:       "rank_2_zero_second",
			dtype:      dtypes.Int64,
			dimensions: []int{3, 0},
			expectSize: 0,
		},
		{
			name:       "rank_3_zero_middle",
			dtype:      dtypes.Float32,
			dimensions: []int{2, 0, 4},
			expectSize: 0,
		},
		{
			name:       "rank_4_multiple_zeros",
			dtype:      dtypes.Int32,
			dimensions: []int{1, 0, 0, 3},
			expectSize: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fmt.Printf("Testing %s with dimensions %v\n", tc.name, tc.dimensions)

			// Test 1: Create zero-dimension buffer using BufferFromHost
			t.Run("BufferFromHost", func(t *testing.T) {
				fmt.Println("testing BufferFromHost")
				testZeroDimBufferFromHost(t, client, tc.dtype, tc.dimensions, tc.expectSize)
			})

			// Test 2: Create zero-dimension buffer using NewSharedBuffer (CPU only)
			if *FlagPluginName == "cpu" || *flagForceSharedBuffer {
				t.Run("NewSharedBuffer", func(t *testing.T) {
					fmt.Println("testing NewSharedBuffer")
					testZeroDimNewSharedBuffer(t, client, tc.dtype, tc.dimensions, tc.expectSize)
				})
			}

			// Test 3: Use zero-dimension buffer as input to computation
			t.Run("AsComputationInput", func(t *testing.T) {
				fmt.Println("testing AsComputationInput")
				testZeroDimAsInput(t, client, tc.dtype, tc.dimensions)
			})

			// Test 4: Create computation that outputs zero-dimension buffer
			t.Run("AsComputationOutput", func(t *testing.T) {
				fmt.Println("testing AsComputationOutput")
				testZeroDimAsOutput(t, client, tc.dtype, tc.dimensions)
			})
			fmt.Println()
			fmt.Println()
			fmt.Println()
		})
	}

	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}

func testZeroDimBufferFromHost(t *testing.T, client *Client, dtype dtypes.DType, dimensions []int, expectSize int) {
	// Calculate the required raw data size (should be 0 for zero-dimension buffers)
	dataSize := dtype.Size()
	for _, dim := range dimensions {
		dataSize *= (dim)
	}

	// should be zero
	rawData := make([]byte, dataSize)
	fmt.Printf("Creating buffer with raw data size: %d bytes\n", len(rawData))

	buffer, err := client.BufferFromHost().FromRawData(rawData, dtype, dimensions).Done()
	require.NoError(t, err, "Failed to create zero-dimension buffer from host")
	defer func() {
		require.NoError(t, buffer.Destroy(), "Failed to destroy buffer")
	}()

	// Verify buffer properties
	bufferDims, err := buffer.Dimensions()
	require.NoError(t, err, "Failed to get buffer dimensions")
	require.Equal(t, dimensions, bufferDims, "Buffer dimensions don't match")

	bufferDType, err := buffer.DType()
	require.NoError(t, err, "Failed to get buffer dtype")
	require.Equal(t, dtype, bufferDType, "Buffer dtype doesn't match")

	// Verify buffer size
	size, err := buffer.Size()
	require.NoError(t, err, "Failed to get buffer size")
	require.Equal(t, expectSize, size, "Buffer size doesn't match expected")

	fmt.Printf("BufferFromHost: dimensions=%v, dtype=%v, size=%d\n", bufferDims, bufferDType, size)
}

func testZeroDimNewSharedBuffer(t *testing.T, client *Client, dtype dtypes.DType, dimensions []int, expectSize int) {
	buffer, flatData, err := client.NewSharedBuffer(dtype, dimensions)
	require.NoError(t, err, "Failed to create zero-dimension shared buffer")
	defer func() {
		require.NoError(t, buffer.Destroy(), "Failed to destroy shared buffer")
	}()
	require.True(t, buffer.IsShared(), "Buffer should be marked as shared")

	// Verify buffer properties
	bufferDims, err := buffer.Dimensions()
	require.NoError(t, err, "Failed to get buffer dimensions")
	require.Equal(t, dimensions, bufferDims, "Buffer dimensions don't match")

	bufferDType, err := buffer.DType()
	require.NoError(t, err, "Failed to get buffer dtype")
	require.Equal(t, dtype, bufferDType, "Buffer dtype doesn't match")

	// Verify buffer size
	size, err := buffer.Size()
	require.NoError(t, err, "Failed to get buffer size")
	require.Equal(t, expectSize, size, "Buffer size doesn't match expected")

	switch dtype {
	case dtypes.Bool:
		flat := flatData.([]bool)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Float16:
		flat := flatData.([]float16.Float16)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.BFloat16:
		flat := flatData.([]bfloat16.BFloat16)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Float32:
		flat := flatData.([]float32)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Float64:
		flat := flatData.([]float64)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Int8:
		flat := flatData.([]int8)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Int16:
		flat := flatData.([]int16)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Int32:
		flat := flatData.([]int32)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Int64:
		flat := flatData.([]int64)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Uint8:
		flat := flatData.([]uint8)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Uint16:
		flat := flatData.([]uint16)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Uint32:
		flat := flatData.([]uint32)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Uint64:
		flat := flatData.([]uint64)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Complex64:
		flat := flatData.([]complex64)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	case dtypes.Complex128:
		flat := flatData.([]complex128)
		require.Equal(t, 0, len(flat), "Flat data should have zero length for zero-dimension buffer")
	default:
		t.Fatalf("Unsupported dtype: %v", dtype)
	}

	fmt.Printf("NewSharedBuffer: dimensions=%v, dtype=%v, size=%d, flat_len=%d\n",
		bufferDims, bufferDType, size, 0)
}

func testZeroDimAsInput(t *testing.T, client *Client, dtype dtypes.DType, dimensions []int) {
	// Create a simple computation that takes input and returns it unchanged (identity function)
	builder := stablehlo.New("ZeroDimIdentity")
	shape := shapes.Make(dtype, dimensions...)
	mainFn := builder.Main()
	param := must1(mainFn.NamedInput("input", shape))

	err := mainFn.Return(param)
	require.NoError(t, err, "Failed to set return value")

	compBytes, err := builder.Build()
	require.NoError(t, err, "Failed to build computation")

	exec, err := client.Compile().WithStableHLO(compBytes).Done()
	require.NoError(t, err, "Failed to compile computation")
	defer func() {
		require.NoError(t, exec.Destroy(), "Failed to destroy executable")
	}()

	// Create zero-dimension input buffer
	dataSize := dtype.Size()
	for _, dim := range dimensions {
		dataSize *= (dim)
	}
	rawData := make([]byte, dataSize)

	inputBuffer, err := client.BufferFromHost().FromRawData(rawData, dtype, dimensions).Done()
	require.NoError(t, err, "Failed to create input buffer")
	defer func() {
		require.NoError(t, inputBuffer.Destroy(), "Failed to destroy input buffer")
	}()

	// Execute computation
	outputs, err := exec.Execute(inputBuffer).Done()
	require.NoError(t, err, "Failed to execute computation with zero-dimension input")
	require.Len(t, outputs, 1, "Expected one output")
	defer func() {
		require.NoError(t, outputs[0].Destroy(), "Failed to destroy output buffer")
	}()

	// Verify output has same dimensions as input
	outputDims, err := outputs[0].Dimensions()
	require.NoError(t, err, "Failed to get output dimensions")
	require.Equal(t, dimensions, outputDims, "Output dimensions should match input dimensions")

	outputSize, err := outputs[0].Size()
	require.NoError(t, err, "Failed to get output size")
	require.Equal(t, 0, outputSize, "Output size should be 0 for zero-dimension buffer")

	fmt.Printf("AsInput: executed identity function with zero-dimension input, output_dims=%v\n", outputDims)
}

func testZeroDimAsOutput(t *testing.T, client *Client, dtype dtypes.DType, dimensions []int) {
	// Create a computation that outputs a zero-dimension constant
	builder := stablehlo.New("ZeroDimConstant")
	shape := shapes.Make(dtype, dimensions...)
	mainFn := builder.Main()

	var scalar *stablehlo.Value
	var err error

	switch dtype {
	case dtypes.Bool:
		scalar, err = mainFn.ConstantFromScalar(true)
	case dtypes.Float16:
		scalar, err = mainFn.ConstantFromScalar(float16.Float16(0))
	case dtypes.BFloat16:
		scalar, err = mainFn.ConstantFromScalar(bfloat16.BFloat16(0))
	case dtypes.Float32:
		scalar, err = mainFn.ConstantFromScalar(float32(0))
	case dtypes.Float64:
		scalar, err = mainFn.ConstantFromScalar(float64(0))
	case dtypes.Int8:
		scalar, err = mainFn.ConstantFromScalar(int8(0))
	case dtypes.Int16:
		scalar, err = mainFn.ConstantFromScalar(int16(0))
	case dtypes.Int32:
		scalar, err = mainFn.ConstantFromScalar(int32(0))
	case dtypes.Int64:
		scalar, err = mainFn.ConstantFromScalar(int64(0))
	case dtypes.Uint8:
		scalar, err = mainFn.ConstantFromScalar(uint8(0))
	case dtypes.Uint16:
		scalar, err = mainFn.ConstantFromScalar(uint16(0))
	case dtypes.Uint32:
		scalar, err = mainFn.ConstantFromScalar(uint32(0))
	case dtypes.Uint64:
		scalar, err = mainFn.ConstantFromScalar(uint64(0))
	case dtypes.Complex64:
		scalar, err = mainFn.ConstantFromScalar(complex64(0))
	case dtypes.Complex128:
		scalar, err = mainFn.ConstantFromScalar(complex128(0))
	default:
		t.Fatalf("Unsupported dtype: %v", dtype)
	}

	require.NoError(t, err, "Failed to create scalar constant")

	broadcast, err := stablehlo.BroadcastInDim(scalar, shape, nil)
	require.NoError(t, err, "Failed to broadcast to zero dimensions")

	err = mainFn.Return(broadcast)
	require.NoError(t, err, "Failed to set return value")

	compBytes, err := builder.Build()
	require.NoError(t, err, "Failed to build computation")

	exec, err := client.Compile().WithStableHLO(compBytes).Done()
	require.NoError(t, err, "Failed to compile computation")
	defer func() {
		require.NoError(t, exec.Destroy(), "Failed to destroy executable")
	}()

	// Execute computation (no inputs needed)
	outputs, err := exec.Execute().Done()
	require.NoError(t, err, "Failed to execute computation producing zero-dimension output")
	require.Len(t, outputs, 1, "Expected one output")
	defer func() {
		require.NoError(t, outputs[0].Destroy(), "Failed to destroy output buffer")
	}()

	// Verify output has expected zero dimensions
	outputDims, err := outputs[0].Dimensions()
	require.NoError(t, err, "Failed to get output dimensions")
	require.Equal(t, dimensions, outputDims, "Output dimensions should match expected zero dimensions")

	outputSize, err := outputs[0].Size()
	require.NoError(t, err, "Failed to get output size")
	require.Equal(t, 0, outputSize, "Output size should be 0 for zero-dimension buffer")

	outputDType, err := outputs[0].DType()
	require.NoError(t, err, "Failed to get output dtype")
	require.Equal(t, dtype, outputDType, "Output dtype should match expected")

	fmt.Printf("AsOutput: created zero-dimension constant, output_dims=%v, dtype=%v\n", outputDims, outputDType)
}
