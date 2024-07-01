package xlabuilder

import "github.com/pkg/errors"

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output has batch and channel axes. Kernel has inputChannel and outputChannel axes.
type ConvolveAxesConfig struct {
	InputBatch, InputChannel int
	InputSpatial             []int

	KernelInputChannel, KernelOutputChannel int
	KernelSpatial                           []int

	OutputBatch, OutputChannel int
	OutputSpatial              []int
}

// ConvGeneralDilated is a generic Convolution operation offered by XLA.
// featureAxisAfter defines whether the features (aka. channels or depth) axis comes after the
// spatial dimension. Example: a 2D input can be one of the two:
//
//   - featureAxisAfter=false: input=[batch_size, features, height, width], filter=[output_features, input_features, height, width]
//   - featureAxisAfter=true:  input=[batch_size, height, width, features], filter=[output_features, height, width, input_features]
//
// Some details in https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution.
// There operand and filter are called lhs and rhs.
// (XLA documentation is unfortunately poor, much is guess-work).
// Also useful, https://arxiv.org/pdf/1603.07285v1.pdf.
func ConvGeneralDilated(operand, filter *Op, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilation, filterDilation []int,
	filterGroupCount, batchGroupCount int) (*Op, error) {
	builder := operand.builder
	op := newOp(ConvGeneralDilatedOp, operand, filter)
	numSpatialDims := operand.Shape.Rank() - 2
	if len(axes.InputSpatial) != numSpatialDims || len(axes.OutputSpatial) != numSpatialDims || len(axes.KernelSpatial) != numSpatialDims {
		return nil, errors.Errorf("ConvGeneralDilated: operand has %d spatial dimensions, but axes configuration has InputSpatial=%d, KernelSpatial=%d, OutputSpatial=%d spatial axes configured "+
			"for operand/kernel/output", numSpatialDims, len(axes.InputSpatial), len(axes.KernelSpatial), len(axes.OutputSpatial))
	}

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/xlabuilder/xlabuilder.cpp, in function XlaBuilderAddOp, under ConvGeneralDilatedOp case.
	//  * 8 first elements store the various parameters and lengths:
	axesConfigLen := 3 * (numSpatialDims + 2)
	op.IntsArg = make([]int, 0, 7+axesConfigLen+len(strides)+2*len(paddings)+len(inputDilation)+len(filterDilation))
	encode := func(values ...int) {
		op.IntsArg = append(op.IntsArg, values...)
	}

	// Encode dimensions.
	encode(numSpatialDims, filterGroupCount, batchGroupCount)
	encode(len(strides), len(paddings), len(inputDilation), len(filterDilation))

	// Append axes configuration.
	encode(axes.InputBatch, axes.InputChannel)
	encode(axes.InputSpatial...)
	encode(axes.KernelInputChannel, axes.KernelOutputChannel)
	encode(axes.KernelSpatial...)
	encode(axes.OutputBatch, axes.OutputChannel)
	encode(axes.OutputSpatial...)

	// Append arrays of ints.
	encode(strides...)
	for _, pair := range paddings {
		encode(pair[0], pair[1])
	}
	encode(inputDilation...)
	encode(filterDilation...)

	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

// DecodeConvGeneralDilated retrieves the arguments for the ConvGeneralDilated op.
func DecodeConvGeneralDilated(op *Op) (operand, filter *Op, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilation, filterDilation []int,
	filterGroupCount, batchGroupCount int) {
	operand = op.OpInputs[0]
	filter = op.OpInputs[1]
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
	filterGroupCount = decode()
	batchGroupCount = decode()

	stridesLen := decode()
	paddingsLen := decode()
	inputDilationLen := decode()
	filterDilationLen := decode()

	// Append axes configuration.
	axes.InputBatch = decode()
	axes.InputChannel = decode()
	axes.InputSpatial = decodeN(numSpatialDims)
	axes.KernelInputChannel = decode()
	axes.KernelOutputChannel = decode()
	axes.KernelSpatial = decodeN(numSpatialDims)
	axes.OutputBatch = decode()
	axes.OutputChannel = decode()
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
	inputDilation = decodeN(inputDilationLen)
	filterDilation = decodeN(filterDilationLen)
	return
}
