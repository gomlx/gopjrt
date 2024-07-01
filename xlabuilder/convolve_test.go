package xlabuilder_test

import (
	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestConvGeneralDilated(t *testing.T) {
	client := getPJRTClient(t)

	{
		builder := New(t.Name())
		dtype := dtypes.Float32
		channelA := capture(Iota(builder, MakeShape(dtype, 1, 1, 3, 3), 2)).Test(t)
		pointOne := capture(Constant(builder, NewScalarLiteral(float32(0.1)))).Test(t)
		channelB := capture(Mul(channelA, pointOne)).Test(t)
		operand := capture(Concatenate(1, channelA, channelB)).Test(t)
		filter := capture(ScalarOne(builder, dtype)).Test(t)
		filter = capture(Broadcast(filter, 2, 3, 3, 1)).Test(t)

		axesConfig := ConvolveAxesConfig{
			InputBatch:          0,
			InputChannel:        1,
			InputSpatial:        []int{2, 3},
			KernelInputChannel:  0,
			KernelOutputChannel: 3,
			KernelSpatial:       []int{1, 2},
			OutputBatch:         0,
			OutputChannel:       3,
			OutputSpatial:       []int{1, 2},
		}
		strides := []int{1, 1}
		padding := [][2]int(nil)
		inputDilation := []int(nil)
		filterDilation := []int(nil)
		filterGroupCount := 1
		batchGroupCount := 1
		output := capture(ConvGeneralDilated(operand, filter, axesConfig,
			strides, padding, inputDilation, filterDilation,
			filterGroupCount, batchGroupCount)).Test(t)

		gotOperand, gotFilter, gotAxesConfig,
			gotStrides, gotPadding, gotInputDilation, gotFilterDilation,
			gotFilterGroupCount, gotBatchGroupCount := DecodeConvGeneralDilated(output)
		require.Same(t, operand, gotOperand)
		require.Same(t, filter, gotFilter)
		require.Equal(t, axesConfig, gotAxesConfig)
		require.Equal(t, strides, gotStrides)
		require.Equal(t, padding, gotPadding)
		require.Equal(t, inputDilation, gotInputDilation)
		require.Equal(t, filterDilation, gotFilterDilation)
		require.Equal(t, filterGroupCount, gotFilterGroupCount)
		require.Equal(t, batchGroupCount, gotBatchGroupCount)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.InDeltaSlice(t, []float32{9.9}, got, 0.001)
		require.Equal(t, []int{1, 1, 1, 1}, dims)
		builder.Destroy()
	}
}
