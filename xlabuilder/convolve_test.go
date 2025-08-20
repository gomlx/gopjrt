package xlabuilder_test

import (
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
)

func TestConvGeneralDilated(t *testing.T) {
	client := getPJRTClient(t)

	{
		builder := New(t.Name())
		dtype := dtypes.Float32
		channelA := capture(Iota(builder, MakeShape(dtype, 1, 1, 3, 3), 2)).Test(t)
		pointOne := capture(Constant(builder, NewScalarLiteral(float32(0.1)))).Test(t)
		channelB := capture(Mul(channelA, pointOne)).Test(t)
		input := capture(Concatenate(1, channelA, channelB)).Test(t)
		kernel := capture(ScalarOne(builder, dtype)).Test(t)
		kernel = capture(Broadcast(kernel, 2, 3, 3, 1)).Test(t)

		axesConfig := ConvolveAxesConfig{
			InputBatch:           0,
			InputChannels:        1,
			InputSpatial:         []int{2, 3},
			KernelInputChannels:  0,
			KernelOutputChannels: 3,
			KernelSpatial:        []int{1, 2},
			OutputBatch:          0,
			OutputChannels:       3,
			OutputSpatial:        []int{1, 2},
		}
		strides := []int{1, 1}
		padding := [][2]int(nil)
		inputDilations := []int(nil)
		kernelDilations := []int(nil)
		featureGroupCount := 1
		batchGroupCount := 1
		output := capture(ConvGeneral(input, kernel, axesConfig,
			strides, padding, inputDilations, kernelDilations,
			featureGroupCount, batchGroupCount)).Test(t)

		gotOperand, gotFilter, gotAxesConfig,
			gotStrides, gotPadding, gotInputDilation, gotFilterDilation,
			gotFilterGroupCount, gotBatchGroupCount := DecodeConvGeneral(output)
		require.Same(t, input, gotOperand)
		require.Same(t, kernel, gotFilter)
		require.Equal(t, axesConfig, gotAxesConfig)
		require.Equal(t, strides, gotStrides)
		require.Equal(t, padding, gotPadding)
		require.Equal(t, inputDilations, gotInputDilation)
		require.Equal(t, kernelDilations, gotFilterDilation)
		require.Equal(t, featureGroupCount, gotFilterGroupCount)
		require.Equal(t, batchGroupCount, gotBatchGroupCount)

		exec := compile(t, client, capture(builder.Build(output)).Test(t))
		got, dims := execArrayOutput[float32](t, client, exec)
		require.InDeltaSlice(t, []float32{9.9}, got, 0.001)
		require.Equal(t, []int{1, 1, 1, 1}, dims)
		builder.Destroy()
	}
}
