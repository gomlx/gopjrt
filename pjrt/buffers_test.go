package pjrt

import (
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
