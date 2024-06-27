package xlabuilder

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"slices"
)

// This file implement the various types of reduce operations.

// ReduceOpType select among the basic types of reduction supported, see XlaBuilder.ReduceComputation.
type ReduceOpType int

const (
	// ReduceSumType reduces by summing all elements being reduced.
	ReduceSumType ReduceOpType = iota

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
