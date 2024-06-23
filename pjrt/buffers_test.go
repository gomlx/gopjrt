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
	rawData, dtype, dimensions := FlatDataToRaw(data, 2, 3)
	require.Equal(t, unsafe.Pointer(unsafe.SliceData(data)), unsafe.Pointer(unsafe.SliceData(rawData)))
	require.Equal(t, dtypes.Float64, dtype)
	require.EqualValues(t, []int{2, 3}, dimensions)

	// Wrong size.
	require.Panics(t, func() {
		_, _, _ = FlatDataToRaw(data, 7, 3)
	})

	// Zero dimension.
	require.Panics(t, func() {
		_, _, _ = FlatDataToRaw([]float64{}, 7, 0)
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
	input := []T{1, 2, 3}
	fmt.Printf("From %#v\n", input)
	buffer, err := client.BufferFromHost().FromRawData(FlatDataToRaw(input, 3, 1)).Done()
	require.NoError(t, err)

	output := make([]T, 3)
	outputBytes, dtype, _ := FlatDataToRaw(output, 3, 1)
	size, err := buffer.Size()
	fmt.Printf("\t> buffer.dtype=%s\n", dtype)
	fmt.Printf("\t> buffer.size=%d\n", size)
	require.NoError(t, err)
	require.Equal(t, len(outputBytes), size, "Wrong buffer size in bytes")

	err = buffer.ToHost(outputBytes)
	require.NoError(t, err)
	fmt.Printf("\t> output=%#v\n", output)
	require.Equal(t, input, output)
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
