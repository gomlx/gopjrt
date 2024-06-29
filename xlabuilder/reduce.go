package xlabuilder

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"slices"
)

// This file implement the various types of reduce operations.

// ReduceOpType select among the basic types of reduction supported, see XlaBuilder.ReduceComputation.
type ReduceOpType int

const (
	// UndefinedReduceType is an undefined value.
	UndefinedReduceType ReduceOpType = iota

	// ReduceSumType reduces by summing all elements being reduced.
	ReduceSumType

	// ReduceProductType reduces by multiplying all elements being reduced.
	ReduceProductType

	// ReduceMaxType reduces by taking the maximum value.
	ReduceMaxType

	// ReduceMinType reduces by taking the minimum value.
	ReduceMinType
)

//go:generate stringer -type ReduceOpType reduce.go

// GetReduceComputationAndInitialValue builds or returns a cached computation that implements a reduction function with one
// of the standard ReduceOpType: sum, multiply, max or min.
func (b *XlaBuilder) GetReduceComputationAndInitialValue(reduction ReduceOpType, dtype dtypes.DType) (comp *XlaComputation, initialValue *Op, err error) {
	if b.IsNil() {
		err = errors.New("trying to access XlaBuilder that is nil or already destroyed")
		return
	}
	if dtype == dtypes.InvalidDType {
		err = errors.Errorf("invalid dtype (%s) for reduce operation", dtype)
		return
	}

	reductionName := fmt.Sprintf("#_%s_%s", reduction, dtype)
	comp = b.cachedStandardComputations[reductionName]
	if comp == nil {
		// Generate new computation for reduction.
		subBuilder := b.CreateSubBuilder(reductionName)
		// lhs -> left-hand-side, rhs -> right-hand-side
		var lhs, rhs *Op
		lhs, err = Parameter(subBuilder, "lhs", 0, MakeShape(dtype))
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a reduce computation %s", reduction)
			return
		}
		rhs, err = Parameter(subBuilder, "rhs", 1, MakeShape(dtype))
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a reduce computation %s", reduction)
			return
		}
		var output *Op
		switch reduction {
		case ReduceSumType:
			output, err = Add(lhs, rhs)
		case ReduceProductType:
			output, err = Mul(lhs, rhs)
		case ReduceMaxType:
			output, err = Max(lhs, rhs)
		case ReduceMinType:
			output, err = Min(lhs, rhs)
		default:
			err = errors.Errorf("unknown reduce computation type: %s (%d)", reduction, reduction)
			return
		}
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a reduce computation %s", reduction)
			return
		}
		comp, err = subBuilder.Build(output)
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a reduce computation %s", reduction)
			return
		}
		subBuilder.Destroy()
		b.cachedStandardComputations[reductionName] = comp
	}

	initialValue = b.cachedStandardConstants[reductionName]
	if initialValue == nil {
		var literal *Literal
		switch reduction {
		case ReduceSumType:
			literal = NewScalarLiteralFromFloat64(0, dtype)
		case ReduceProductType:
			literal = NewScalarLiteralFromFloat64(1, dtype)
		case ReduceMaxType:
			literal = NewScalarLiteralFromAny(dtype.LowestValue())
		case ReduceMinType:
			literal = NewScalarLiteralFromAny(dtype.HighestValue())
		default:
			err = errors.Errorf("unknown reduce computation type: %s (%d)", reduction, reduction)
			return
		}
		initialValue, err = Constant(b, literal)
		if err != nil {
			err = errors.WithMessagef(err, "while trying to create a reduce computation %s", reduction)
			return
		}
		b.cachedStandardConstants[reductionName] = initialValue
	}
	return
}

// Reduce the selected axes of the input array, using the given custom reduceComputation.
// The initialValue should be a scalar value that the reduction starts with.
//
// If no axes are given, it reduces the full array.
//
// Consider instead using one of the standard ReduceSum, ReduceProduct, ReduceMax and ReduceMin. They use
// cached values for both the corresponding reduceComputation and initialValue, per dtype of x.
func Reduce(x *Op, reduceComputation *XlaComputation, initialValue *Op, axes ...int) (*Op, error) {
	builder := x.builder
	if builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}

	op := newOp(ReduceOp, x, initialValue)
	op.ComputationArg = reduceComputation
	op.IntsArg = slices.Clone(axes)
	err := builder.addOp(op)
	if err != nil {
		return nil, err
	}
	return op, nil
}

func simpleReduceImpl(reduceType ReduceOpType, x *Op, axes ...int) (*Op, error) {
	builder := x.builder
	if builder.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
	comp, initialValue, err := builder.GetReduceComputationAndInitialValue(reduceType, x.Shape.DType)
	if err != nil {
		return nil, errors.WithMessagef(err, "while creating %s sub-computation", ReduceMaxType)
	}

	return Reduce(x, comp, initialValue, axes...)
}

// ReduceMax is a shortcut for Reduce with the proper computation and initial value to reduce x on the given axes, by taking the max value.
//
// If no axes are given, it reduces the full array.
func ReduceMax(x *Op, axes ...int) (*Op, error) {
	return simpleReduceImpl(ReduceMaxType, x, axes...)
}

// ReduceMin is a shortcut for Reduce with the proper computation and initial value to reduce x on the given axes, by taking the min value.
//
// If no axes are given, it reduces the full array.
func ReduceMin(x *Op, axes ...int) (*Op, error) {
	return simpleReduceImpl(ReduceMinType, x, axes...)
}

// ReduceSum is a shortcut for Reduce with the proper computation and initial value to reduce x on the given axes, by taking the sum of the reduced axes.
//
// If no axes are given, it reduces the full array.
func ReduceSum(x *Op, axes ...int) (*Op, error) {
	return simpleReduceImpl(ReduceSumType, x, axes...)
}

// ReduceProduct is a shortcut for Reduce with the proper computation and initial value to reduce x on the given axes, by taking the product of the reduced axes.
//
// If no axes are given, it reduces the full array.
func ReduceProduct(x *Op, axes ...int) (*Op, error) {
	return simpleReduceImpl(ReduceProductType, x, axes...)
}

type ReduceWindowConfig struct {
	builder           *XlaBuilder
	x, initialValue   *Op
	rank              int
	dtype             dtypes.DType
	reduceType        ReduceOpType
	reduceComputation *XlaComputation
	windowDimensions  []int
	strides           []int
	baseDilations     []int
	windowDilations   []int
	paddings          [][2]int
	err               error
}

// ReduceWindow applies a reduction function to all elements in each window of x, producing an N multidimensional array as output.
// The output array has the same number of elements as the number of valid positions of the window.
//
// A pooling layer (typical in image processing) can be expressed as a ReduceWindow.
//
// x is the array to reduce, it cannot be a scalar. And windowDimensions is the size of the windows on which to reduce:
// they must be set for each axis of x.
//
// There are other options, so this uses the "builder pattern": it returns a ReduceWindowConfig object that can be further
// configured. When finished call the ReduceWindowConfig.Done to trigger its execution.
//
// More details and examples can be takes from OpenXLA site: https://openxla.org/xla/operation_semantics#reducewindow
func ReduceWindow(x *Op, windowDimensions []int) *ReduceWindowConfig {
	r := &ReduceWindowConfig{
		builder:          x.builder,
		x:                x,
		rank:             x.Shape.Rank(),
		dtype:            x.Shape.DType,
		windowDimensions: slices.Clone(windowDimensions),
	}
	if r.builder.IsNil() {
		r.err = errors.New("trying to access XlaBuilder that is nil or already destroyed")
		return r
	}
	if r.rank == 0 {
		r.err = errors.Errorf("cannot run a ReduceWindow on a scalar -- x.Shape=%s", x.Shape)
		return r
	}
	if len(windowDimensions) != r.rank {
		r.err = errors.Errorf("ReduceWindow requires a window dimension for each axis of x, but x has rank %d, and %d windowDimensions were given.", r.rank, len(windowDimensions))
		return r
	}
	return r
}

func (r *ReduceWindowConfig) standardReduction(t ReduceOpType) *ReduceWindowConfig {
	if r.err != nil {
		// Already in an invalid state.
		return r
	}
	if r.reduceComputation != nil {
		r.err = errors.Errorf("trying to configure ReduceWindow(...) with %s, but reduction type has already been configured", t)
		return r
	}
	r.reduceType = t
	r.reduceComputation, r.initialValue, r.err = r.builder.GetReduceComputationAndInitialValue(t, r.dtype)
	return r
}

// Max configures the reduction type.
//
// There is no defaults for the type of reduction: one has to either configure Max, Min, Sum, Product or
// some arbitrary computation with UseComputation.
func (r *ReduceWindowConfig) Max() *ReduceWindowConfig {
	return r.standardReduction(ReduceMaxType)
}

// Min configures the reduction type.
//
// There is no defaults for the type of reduction: one has to either configure Max, Min, Sum, Product or
// some arbitrary computation with UseComputation.
func (r *ReduceWindowConfig) Min() *ReduceWindowConfig {
	return r.standardReduction(ReduceMinType)
}

// Sum configures the reduction type.
//
// There is no defaults for the type of reduction: one has to either configure Max, Min, Sum, Product or
// some arbitrary computation with UseComputation.
func (r *ReduceWindowConfig) Sum() *ReduceWindowConfig {
	return r.standardReduction(ReduceSumType)
}

// Product configures the reduction type.
//
// There is no defaults for the type of reduction: one has to either configure Max, Min, Sum, Product or
// some arbitrary computation with UseComputation.
func (r *ReduceWindowConfig) Product() *ReduceWindowConfig {
	return r.standardReduction(ReduceProductType)
}

// UseComputation configures a custom reduction function and initial value.
//
// reduceComputation must take two scalars of x dtype as input, and return a scalar as the output.
// The initialValue must be a scalar of the same dtype (typically a Constant, but it can be the result of another
// operation).
//
// There is no defaults for the type of reduction: one has to either configure Max, Min, Sum, Product or
// some arbitrary computation with UseComputation.
func (r *ReduceWindowConfig) UseComputation(reduceComputation *XlaComputation, initialValue *Op) *ReduceWindowConfig {
	if r.err != nil {
		// Already in an invalid state.
		return r
	}
	if r.reduceComputation != nil {
		r.err = errors.Errorf("trying to configure ReduceWindow().UseComputation(...), but reduction type has already been configured")
		return r
	}
	r.reduceType = UndefinedReduceType
	r.reduceComputation = reduceComputation
	r.initialValue = initialValue
	return r
}

// WithStrides provides the stride size for each axis of x. One value per axis of x must be given.
//
// The default is same value as windowDimensions.
func (r *ReduceWindowConfig) WithStrides(strides []int) *ReduceWindowConfig {
	if r.err != nil {
		return r
	}
	if len(strides) != r.rank {
		r.err = errors.Errorf("ReduceWindow requires a stride for each axis of x, but x has rank %d, and %d strides were passed to WithStrides().", r.rank, len(strides))
		return r
	}
	r.strides = strides
	return r
}

// WithBaseDilations provides the base dilation for each axis of x. One value per axis of x must be given.
//
// The default is 0 for every axis.
func (r *ReduceWindowConfig) WithBaseDilations(baseDilations []int) *ReduceWindowConfig {
	if r.err != nil {
		return r
	}
	if len(baseDilations) != r.rank {
		r.err = errors.Errorf("ReduceWindow requires a stride for each axis of x, but x has rank %d, and %d baseDilations were passed to WithBaseDilations().", r.rank, len(baseDilations))
		return r
	}
	r.baseDilations = baseDilations
	return r
}

// WithWindowDilations provides the window dilation for each axis of x. One value per axis of x must be given.
//
// The default is 0 for every axis.
func (r *ReduceWindowConfig) WithWindowDilations(windowDilations []int) *ReduceWindowConfig {
	if r.err != nil {
		return r
	}
	if len(windowDilations) != r.rank {
		r.err = errors.Errorf("ReduceWindow requires a stride for each axis of x, but x has rank %d, and %d windowDilations were passed to WithWindowDilations().", r.rank, len(windowDilations))
		return r
	}
	r.windowDilations = windowDilations
	return r
}

// WithPadding provides the amount of padding on the start and end of each axis of x. One value per axis of x must be given.
//
// The default is (0, 0) for every axis.
func (r *ReduceWindowConfig) WithPadding(paddings [][2]int) *ReduceWindowConfig {
	if r.err != nil {
		return r
	}
	if len(paddings) != r.rank {
		r.err = errors.Errorf("ReduceWindow requires a padding definition for each axis of x, but x has rank %d, and %d paddings were passed to WithWindowDilations().", r.rank, len(paddings))
		return r
	}
	r.paddings = paddings
	return r
}

// sliceWithValue creates a slice of given size filled with given value.
func sliceWithValue[T any](size int, value T) []T {
	s := make([]T, size)
	for ii := range s {
		s[ii] = value
	}
	return s
}

// Done executes the ReduceWindow and returns the corresponding Op, or an error.
func (r *ReduceWindowConfig) Done() (*Op, error) {
	if r.err != nil {
		return nil, r.err
	}
	if r.reduceComputation == nil || r.initialValue == nil {
		return nil, errors.New("ReduceWindow(...) didnt specify the type of reduction, use Max, Min, Sum, Product or a custom one with UseComputation")
	}
	if r.strides == nil {
		r.strides = r.windowDimensions
	}
	if r.baseDilations == nil {
		r.baseDilations = sliceWithValue(r.rank, 1)
	}
	if r.windowDilations == nil {
		r.windowDilations = sliceWithValue(r.rank, 1)
	}
	if r.paddings == nil {
		r.paddings = make([][2]int, r.rank)
	}

	op := newOp(ReduceWindowOp, r.x, r.initialValue)
	op.ComputationArg = r.reduceComputation
	op.IntArg = int(r.reduceType)

	// Encode parameters in ints. We need for each axis:
	// - one value for windowDimensions;
	// - one value for strides;
	// - one value for baseDilations;
	// - one value for windowDilations;
	// - two values for paddings;
	op.IntsArg = make([]int, 0, 6*r.rank)
	encode := func(values ...int) {
		op.IntsArg = append(op.IntsArg, values...)
	}
	encode(r.windowDimensions...) // rank elements.
	encode(r.strides...)          // rank elements.
	encode(r.baseDilations...)
	encode(r.windowDilations...)
	for _, pair := range r.paddings {
		encode(pair[0], pair[1])
	}

	if klog.V(2).Enabled() {
		klog.Infof("ReduceWindow(%s):\n", r.reduceType)
		klog.Infof("    x.shape=%s\n", r.x.Shape)
		klog.Infof("    windowDimensions=%v\n", r.windowDimensions)
		klog.Infof("    strides=%v\n", r.strides)
		klog.Infof("    baseDilations=%v\n", r.baseDilations)
		klog.Infof("    windowDilations=%v\n", r.windowDilations)
		klog.Infof("    paddings=%v\n", r.paddings)
		klog.Infof("    op.IntArgs=%v\n", op.IntsArg)
	}
	err := r.builder.addOp(op)
	if err != nil {
		return nil, err
	}
	if klog.V(2).Enabled() {
		klog.Infof("    output shape=%s\n", op.Shape)
	}
	return op, nil
}

// DecodeReduceWindow retrieves the arguments for a ReduceWindow op.
func DecodeReduceWindow(op *Op) (reduceType ReduceOpType, reduceComputation *XlaComputation, initialValue *Op, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) {
	rank := op.OpInputs[0].Shape.Rank()
	reduceType = ReduceOpType(op.IntArg)
	reduceComputation = op.ComputationArg
	initialValue = op.OpInputs[1]
	windowDimensions = op.IntsArg[0:rank]
	strides = op.IntsArg[rank : 2*rank]
	baseDilations = op.IntsArg[2*rank : 3*rank]
	windowDilations = op.IntsArg[3*rank : 4*rank]
	paddings = make([][2]int, rank)
	for ii := range paddings {
		paddings[ii][0] = op.IntsArg[4*rank+ii*2]
		paddings[ii][1] = op.IntsArg[4*rank+ii*2+1]
	}
	return
}
