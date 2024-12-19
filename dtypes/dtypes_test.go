package dtypes

import (
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
	"math"
	"testing"
)

func TestDType_HighestLowestSmallestValues(t *testing.T) {
	require.True(t, math.IsInf(Float64.HighestValue().(float64), 1))
	require.True(t, math.IsInf(float64(Float32.LowestValue().(float32)), -1))
	_, ok := Float16.SmallestNonZeroValueForDType().(float16.Float16)
	require.True(t, ok)
	_, ok = BFloat16.SmallestNonZeroValueForDType().(bfloat16.BFloat16)
	require.True(t, ok)

	// Complex numbers don't define Highest of Lowest, and instead return 0
	require.Equal(t, complex64(0), Complex64.HighestValue().(complex64))
	require.Equal(t, complex128(0), Complex128.LowestValue().(complex128))
	require.Equal(t, complex64(0), Complex64.SmallestNonZeroValueForDType().(complex64))
}

func TestMapOfNames(t *testing.T) {
	require.Equal(t, Float16, MapOfNames["Float16"])
	require.Equal(t, Float16, MapOfNames["float16"])
	require.Equal(t, Float16, MapOfNames["F16"])
	require.Equal(t, Float16, MapOfNames["f16"])

	require.Equal(t, BFloat16, MapOfNames["BFloat16"])
	require.Equal(t, BFloat16, MapOfNames["bfloat16"])
	require.Equal(t, BFloat16, MapOfNames["BF16"])
	require.Equal(t, BFloat16, MapOfNames["bf16"])
}

func TestFromAny(t *testing.T) {
	require.Equal(t, Int64, FromAny(int64(7)))
	require.Equal(t, Float32, FromAny(float32(13)))
	require.Equal(t, BFloat16, FromAny(bfloat16.FromFloat32(1.0)))
	require.Equal(t, Float16, FromAny(float16.Fromfloat32(3.0)))
}

func TestSize(t *testing.T) {
	require.Equal(t, 8, Int64.Size())
	require.Equal(t, 4, Float32.Size())
	require.Equal(t, 2, BFloat16.Size())
}

func TestSizeForDimensions(t *testing.T) {
	require.Equal(t, 2*3*8, Int64.SizeForDimensions(2, 3))
	require.Equal(t, 4, Float32.SizeForDimensions())
	require.Equal(t, 2, BFloat16.SizeForDimensions(1, 1, 1))
}
