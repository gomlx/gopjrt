package xlabuilder

import "github.com/pkg/errors"

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// The input and output have batch and channels axes.
// The Kernel has an input feature and output channels axes.
//
// Note another common term for "channels" is "features".
type ConvolveAxesConfig struct {
	InputBatch, InputChannels int
	InputSpatial              []int

	KernelInputChannels, KernelOutputChannels int
	KernelSpatial                             []int

	OutputBatch, OutputChannels int
	OutputSpatial               []int
}

// ConvGeneral is a generic Convolution operation with support for:
//
// - Arbitrary number of spatial axes.
// - Arbitrary transposition of axes.
// - Strides and padding.
// - Dilations of the input.
// - Dilations of the kernel, aka. atrous convolution.
// - Channels grouping (on the input channels).
// - Batch grouping.
//
// Some details in https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution.
// There operand and filter are called lhs and rhs.
// (XLA documentation is unfortunately poor, much is guess-work).
// Also useful, https://arxiv.org/pdf/1603.07285v1.pdf.
//
// Note:
//   - Another common term for "channels" is "features".
//   - "Kernel" is also commonly called "weights" or "filters".
func ConvGeneral(input, kernel *Op, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int) (*Op, error) {
	builder := input.builder
	op := newOp(ConvGeneralDilatedOp, input, kernel)
	numSpatialDims := input.Shape.Rank() - 2
	if len(axes.InputSpatial) != numSpatialDims || len(axes.OutputSpatial) != numSpatialDims || len(axes.KernelSpatial) != numSpatialDims {
		return nil, errors.Errorf("ConvGeneralDilated: input has %d spatial dimensions, but axes configuration has InputSpatial=%d, KernelSpatial=%d, OutputSpatial=%d spatial axes configured "+
			"for input/kernel/output", numSpatialDims, len(axes.InputSpatial), len(axes.KernelSpatial), len(axes.OutputSpatial))
	}

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/xlabuilder/xlabuilder.cpp, in function XlaBuilderAddOp, under ConvGeneralDilatedOp case.
	//  * 8 first elements store the various parameters and lengths:
	axesConfigLen := 3 * (numSpatialDims + 2)
	op.IntsArg = make([]int, 0, 7+axesConfigLen+len(strides)+2*len(paddings)+len(inputDilations)+len(kernelDilations))
	encode := func(values ...int) {
		op.IntsArg = append(op.IntsArg, values...)
	}

	// Encode dimensions.
	encode(numSpatialDims, channelGroupCount, batchGroupCount)
	encode(len(strides), len(paddings), len(inputDilations), len(kernelDilations))

	// Append axes configuration.
	encode(axes.InputBatch, axes.InputChannels)
	encode(axes.InputSpatial...)
	encode(axes.KernelInputChannels, axes.KernelOutputChannels)
	encode(axes.KernelSpatial...)
	encode(axes.OutputBatch, axes.OutputChannels)
	encode(axes.OutputSpatial...)

	// Append arrays of ints.
	encode(strides...)
	for _, pair := range paddings {
		encode(pair[0], pair[1])
	}
	encode(inputDilations...)
	encode(kernelDilations...)

	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeConvGeneral retrieves the arguments for the ConvGeneral op.
func DecodeConvGeneral(op *Op) (input, kernel *Op, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int) {
	input = op.OpInputs[0]
	kernel = op.OpInputs[1]
	pos := 0
	decode := func() int {
		res := op.IntsArg[pos]
		pos++
		return res
	}
	decodeN := func(n int) []int {
		if n == 0 {
			return nil
		}
		res := op.IntsArg[pos : pos+n]
		pos += n
		return res
	}

	// Encode dimensions.
	numSpatialDims := decode()
	channelGroupCount = decode()
	batchGroupCount = decode()

	stridesLen := decode()
	paddingsLen := decode()
	inputDilationLen := decode()
	filterDilationLen := decode()

	// Append axes configuration.
	axes.InputBatch = decode()
	axes.InputChannels = decode()
	axes.InputSpatial = decodeN(numSpatialDims)
	axes.KernelInputChannels = decode()
	axes.KernelOutputChannels = decode()
	axes.KernelSpatial = decodeN(numSpatialDims)
	axes.OutputBatch = decode()
	axes.OutputChannels = decode()
	axes.OutputSpatial = decodeN(numSpatialDims)

	// Append arrays of ints.
	strides = decodeN(stridesLen)
	if paddingsLen > 0 {
		paddings = make([][2]int, paddingsLen)
		for ii := range paddings {
			paddings[ii][0] = decode()
			paddings[ii][1] = decode()
		}
	}
	inputDilations = decodeN(inputDilationLen)
	kernelDilations = decodeN(filterDilationLen)
	return
}
