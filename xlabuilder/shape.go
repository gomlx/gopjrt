package xlabuilder

import "C"
import (
	"gopjrt/dtypes"
	"unsafe"
)

// Shape is a minimalistic shape representation of a tensor.
// It is used to describe the output of an Op, or as an input for operations that change the Shape of another Op,
// or part of a Literal value.
//
// It is defined as a DType (the underlying data type, e.g.: Float32, Int64, etc.) and the dimenions on each axis
// of the tensor. If len(Dimensions) is 0, it represents a scalar.
//
// Alternatively, in XLA, a value can represent a "tuple" of sub-values.
// In this case Shape.TupleShapes is defined with the shapes of its sub-values -- it is a recursive structure.
// In this case DType is set to Invalid, and the shape doesn't have a value of itself.
type Shape struct {
	DType      dtypes.DType
	Dimensions []int

	TupleShapes []Shape // Shapes of the tuple, if this is a tuple.
}

// Rank of a shape is the number of axes. A shortcut to len(Shape.Dimensions).
// Scalar values have rank 0.
func (s Shape) Rank() int {
	return len(s.Dimensions)
}

// TupleSize is an alias to len(Shape.TupleShapes).
func (s Shape) TupleSize() int {
	return len(s.TupleShapes)
}

// cShapeFromShape allocates int the C-heap a new C-struct representing the shape.
func cShapeFromShape(shape Shape) *C.Shape {
	var cShape *C.Shape
	cShape = cMalloc[C.Shape]()
	cShape.dtype = C.int32_t(shape.DType)
	rank := shape.Rank()
	cShape.rank = C.int64_t(rank)
	if rank > 0 {
		cShape.dimensions = cMallocArrayAndSet[C.int64_t](rank, func(ii int) C.int64_t { return C.int64_t(shape.Dimensions[ii]) })
	}
	cShape.tuple_size = C.int32_t(shape.TupleSize())
	if shape.TupleSize() > 0 {
		cShape.tuple_shapes = cMallocArrayAndSet[*C.Shape](shape.TupleSize(), func(ii int) *C.Shape {
			return cShapeFromShape(shape.TupleShapes[ii])
		})
	}
	return cShape
}

// shapeFromCShape converts a shape provided in C struct (cShape) into a shapes.Shape. cShape memory is NOT freed.
func shapeFromCShape(cShape *C.Shape) (shape Shape) {
	if cShape == nil {
		return
	}
	shape.DType = dtypes.DType(cShape.dtype)
	rank := int(cShape.rank)
	if rank > 0 {
		shape.Dimensions = make([]int, cShape.rank)
		dimensions := unsafe.Slice(cShape.dimensions, rank)
		for ii, dim := range dimensions {
			shape.Dimensions[ii] = int(dim)
		}
	}
	if cShape.tuple_size > 0 {
		shape.TupleShapes = make([]Shape, int(cShape.tuple_size))
		subShapes := unsafe.Slice(cShape.tuple_shapes, cShape.tuple_size)
		for ii, subShape := range subShapes {
			shape.TupleShapes[ii] = shapeFromCShape(subShape)
		}
	}
	return
}
