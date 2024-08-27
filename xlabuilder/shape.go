package xlabuilder

/*
#include <gomlx/xlabuilder/shape.h>
*/
import "C"
import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/protos/xla_data"
	"github.com/pkg/errors"
	"slices"
	"strings"
	"unsafe"
)

// Shape is a minimalistic shape representation of a tensor.
// It is used to describe the output of an Op, or as an input for operations that change the Shape of another Op,
// or part of a Literal value.
//
// It is defined as a DType (the underlying data type, e.g.: Float32, Int64, etc.) and the dimensions on each axis
// of the tensor. If len(Dimensions) is 0, it represents a scalar.
//
// Alternatively, in XLA, a value can represent a "tuple" of sub-values.
// In this case Shape.TupleShapes is defined with the shapes of its sub-values -- it is a recursive structure.
// In this case DType is set to InvalidDType, and the shape doesn't have a value of itself.
type Shape struct {
	DType      dtypes.DType
	Dimensions []int

	TupleShapes []Shape // Shapes of the tuple, if this is a tuple.
}

// MakeShape filled with the values given.
//
// The dimensions must be >= 1, and it doesn't work for tuple shapes.
func MakeShape(dtype dtypes.DType, dimensions ...int) Shape {
	s := Shape{Dimensions: slices.Clone(dimensions), DType: dtype}
	for _, dim := range dimensions {
		if dim <= 0 {
			exceptions.Panicf("shapes.Make(%+v): cannot create a shape with an axis with dimension <= 0", s)
		}
	}
	return s
}

// MakeShapeOrError is the same as MakeShape, but it returns an error instead if the dimensions are <= 0.
func MakeShapeOrError(dtype dtypes.DType, dimensions ...int) (Shape, error) {
	s := Shape{Dimensions: slices.Clone(dimensions), DType: dtype}
	for _, dim := range dimensions {
		if dim <= 0 {
			return Shape{}, errors.Errorf("shapes.Make(%+v): cannot create a shape with an axis with dimension <= 0", s)
		}
	}
	return s, nil
}

// IsScalar returns whether the Shape is a scalar, i.e. its len(Shape.Dimensions) == 0.
func (s Shape) IsScalar() bool { return s.Rank() == 0 }

// Rank of a shape is the number of axes. A shortcut to len(Shape.Dimensions).
// Scalar values have rank 0.
func (s Shape) Rank() int {
	return len(s.Dimensions)
}

// Size returns the total size of the shape. E.g.: a Shape of dimensions [3, 5] has size 15. A scalar has size 1.
func (s Shape) Size() int {
	size := 1
	for _, dim := range s.Dimensions {
		size *= dim
	}
	return size
}

// Memory returns the memory used to store an array of the given shape, the same as the size in bytes.
// Careful, so far all types in Go and on device seem to use the same sizes, but future type this is not guaranteed.
func (s Shape) Memory() uintptr {
	return s.DType.Memory() * uintptr(s.Size())
}

// Clone makes a deep copy (including dimensions and tuples) of the given shape.
func (s Shape) Clone() (newS Shape) {
	newS.DType = s.DType
	if len(s.Dimensions) > 0 {
		newS.Dimensions = slices.Clone(s.Dimensions)
	}
	if len(s.TupleShapes) > 0 {
		newS.TupleShapes = make([]Shape, len(s.TupleShapes))
		for ii, subS := range s.TupleShapes {
			newS.TupleShapes[ii] = subS.Clone()
		}
	}
	return newS
}

// TupleSize is an alias to len(Shape.TupleShapes).
func (s Shape) TupleSize() int {
	return len(s.TupleShapes)
}

// String implements fmt.Stringer and pretty-print the shape.
func (s Shape) String() string {
	if s.TupleSize() > 0 {
		parts := make([]string, 0, s.TupleSize())
		for _, tuple := range s.TupleShapes {
			parts = append(parts, tuple.String())
		}
		return fmt.Sprintf("Tuple<%s>", strings.Join(parts, ", "))
	}
	if s.Rank() == 0 {
		return fmt.Sprintf("(%s)[]", s.DType)
	}
	return fmt.Sprintf("(%s)%v", s.DType, s.Dimensions)
}

// cShapeFromShape allocates int the C-heap a new C-struct representing the shape.
// If shape is undefined (not used) it returns nil.
//
// Notice the dtype is converted to PrimitiveType, since in shape we are still using the PJRT's dtype.
func cShapeFromShape(shape Shape) *C.Shape {
	if shape.DType == dtypes.InvalidDType && shape.TupleSize() == 0 {
		return nil
	}

	var cShape *C.Shape
	cShape = cMalloc[C.Shape]()
	cShape.dtype = C.int32_t(shape.DType.PrimitiveType())
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
	shape.DType = dtypes.FromPrimitiveType(xla_data.PrimitiveType(cShape.dtype))
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
