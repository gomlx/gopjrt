package pjrt

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"gopjrt/dtypes"
	"testing"
	"unsafe"
)

func TestFlatDataToRaw(t *testing.T) {
	data := []float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
	rawData, dtype, dimensions := FlatDataToRawWithDimensions(data, 2, 3)
	require.Equal(t, unsafe.Pointer(unsafe.SliceData(data)), unsafe.Pointer(unsafe.SliceData(rawData)))
	require.Equal(t, dtypes.Float64, dtype)
	require.EqualValues(t, []int{2, 3}, dimensions)

	// Wrong size.
	require.Panics(t, func() {
		_, _, _ = FlatDataToRawWithDimensions(data, 7, 3)
	})

	// Zero dimension.
	require.Panics(t, func() {
		_, _, _ = FlatDataToRawWithDimensions([]float64{}, 7, 0)
	})
}

func TestScalarDataToRaw(t *testing.T) {
	rawData, dtype, dimensions := ScalarToRaw(uint32(3))
	require.Equal(t, dtype, dtypes.Uint32)
	require.Empty(t, dimensions)
	require.Equal(t, uint32(3), *(*uint32)(unsafe.Pointer(unsafe.SliceData(rawData))))
}

func testTransfersImpl[T interface {
	float64 | float32 | int64 | int8
}](t *testing.T, client *Client) {
	// Transfer arrays.
	input := []T{1, 2, 3}
	fmt.Printf("From %#v\n", input)
	buffer, err := ArrayToBuffer(client, input, 3, 1)
	require.NoError(t, err)

	output, outputDims, err := BufferToArray[T](buffer)
	require.NoError(t, err)
	fmt.Printf("\t> output=%#v\n", output)
	require.Equal(t, input, output)
	require.Equal(t, []int{3, 1}, outputDims)

	// Try an invalid transfer: it should complain about the invalid dtype.
	_, _, err = BufferToArray[complex128](buffer)
	fmt.Printf("\t> expected wrong dtype error: %v\n", err)
	require.Error(t, err)

	// Transfer scalars.
	from := T(13)
	fmt.Printf("From %T(%v)\n", from, from)
	buffer, err = ScalarToBuffer(client, from)
	require.NoError(t, err)
	to, err := BufferToScalar[T](buffer)
	require.NoError(t, err)
	fmt.Printf("\t> got %v\n", to)
	require.Equal(t, from, to)
}

func TestTransfers(t *testing.T) {
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices, err := client.AddressableDevices()
	require.NoErrorf(t, err, "Failed to fetch AddressableDevices() from client on %s", plugin)
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	testTransfersImpl[float64](t, client)
	testTransfersImpl[float32](t, client)
	testTransfersImpl[int64](t, client)
	testTransfersImpl[int8](t, client)

	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}

func TestBufferProperties(t *testing.T) {
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	{ // float32[3,4]
		dims := []int{3, 4}
		data := make([]float32, dims[0]*dims[1])

		buf, err := ArrayToBuffer(client, data, dims...)
		require.NoError(t, err)
		bufDims, err := buf.Dimensions()
		require.NoError(t, err)
		require.Equal(t, dims, bufDims)
		dtype, err := buf.DType()
		require.NoError(t, err)
		require.Equal(t, dtypes.Float32, dtype)
	}

	{ // Scalar uint8
		buf, err := ScalarToBuffer(client, uint8(3))
		require.NoError(t, err)
		bufDims, err := buf.Dimensions()
		require.NoError(t, err)
		require.Zero(t, len(bufDims))
		dtype, err := buf.DType()
		require.NoError(t, err)
		require.Equal(t, dtypes.Uint8, dtype)
	}
}
