package xlabuilder

/*
#include <gomlx/xlabuilder/literal.h>
*/
import "C"
import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/x448/float16"
	"reflect"
	"runtime"
	"unsafe"
)

// Literal defines a constant value for the graph, and is treated as immutable.
//
// Since it's only used to feed the graph value, it doesn't provide any way to inspecting its current values.
type Literal struct {
	cLiteral *C.Literal // C-Wrapper around the XlaLiteral.
	shape    Shape
}

// NewLiteralFromShape creates a zero-initialized literal with the given shape.
// It cannot be used to create Literal tuples.
func NewLiteralFromShape(shape Shape) *Literal {
	if shape.TupleSize() > 0 {
		exceptions.Panicf("NewLiteralFromShape cannot be used to create tuple literals, shape given was %s", shape)
	}
	cShape := cShapeFromShape(shape) // Ownership is given to the new Literal structure.
	cLiteral := C.MakeLiteralFromShape(cShape)
	return newLiteral(cLiteral, shape)
}

// NewArrayLiteral creates a Literal initialized from the array flat data (a slice) and the dimensions of the array.
//
// If dimensions is omitted, it is assumed to represent a 1D-array of the length given.
func NewArrayLiteral[T dtypes.Supported](flat []T, dimensions ...int) *Literal {
	if len(dimensions) == 0 {
		dimensions = []int{len(flat)}
	}
	shape := MakeShape(dtypes.FromGoType[T](), dimensions...)
	if shape.Size() != len(flat) {
		exceptions.Panicf("NewArrayLiteral got a slice of length %d, but the shape %s given has %d elements",
			len(flat), shape, shape.Size())
	}
	l := NewLiteralFromShape(shape)
	lData := unsafe.Slice((*T)(unsafe.Pointer(l.cLiteral.data)), int(l.cLiteral.size))
	copy(lData, flat)
	return l
}

// NewScalarLiteral creates a scalar Literal initialized with the given value.
func NewScalarLiteral[T dtypes.Supported](value T) *Literal {
	shape := MakeShape(dtypes.FromGoType[T]())
	l := NewLiteralFromShape(shape)
	*(*T)(unsafe.Pointer(l.cLiteral.data)) = value
	return l
}

// NewScalarLiteralFromFloat64 creates a scalar Literal with the given dtype initialized from the given value as float64.
// This can be used to create common constants for arbitrary dtypes.
//
// It panics if the value cannot be converted to the dtype.
func NewScalarLiteralFromFloat64(value float64, dtype dtypes.DType) *Literal {
	// Scalar value.
	if dtype == dtypes.Bool {
		return NewScalarLiteral(value != 0)
	}
	if dtype == dtypes.Complex64 {
		return NewScalarLiteral(complex(float32(value), 0))
	}
	if dtype == dtypes.Complex128 {
		return NewScalarLiteral(complex(value, 0))
	}
	if dtype == dtypes.Float16 {
		return NewScalarLiteral(float16.Fromfloat32(float32(value)))
	}

	convertedValue := reflect.ValueOf(value).Convert(dtype.GoType())
	l := NewLiteralFromShape(MakeShape(dtype))
	lValueOf := reflect.NewAt(dtype.GoType(), unsafe.Pointer(l.cLiteral.data)).Elem()
	lValueOf.Set(convertedValue)
	return l
}

// NewScalarLiteralFromAny creates a scalar Literal with the given dynamically typed value.
// It uses reflection to inspect the type.
func NewScalarLiteralFromAny(value any) *Literal {
	valueOf := reflect.ValueOf(value)
	dtype := dtypes.FromType(valueOf.Type())
	l := NewLiteralFromShape(MakeShape(dtype))
	lValueOf := reflect.NewAt(dtype.GoType(), unsafe.Pointer(l.cLiteral.data)).Elem()
	lValueOf.Set(valueOf)
	return l
}

// newLiteral creates the literal and registers the finalizer.
func newLiteral(cLiteral *C.Literal, shape Shape) *Literal {
	l := &Literal{cLiteral: cLiteral, shape: shape}
	if int(cLiteral.size) != shape.Size() {
		exceptions.Panicf("new literal being created has shape=%s (size %d), but internal size reported is %d",
			shape, shape.Size(), cLiteral.size)
	}
	runtime.SetFinalizer(l, func(l *Literal) { l.Destroy() })
	return l
}

// Destroy the Literal, release resources, and the Literal is no longer valid.
// This is automatically called if the Literal is garbage collected.
func (l *Literal) Destroy() {
	if l.cLiteral == nil {
		return
	}
	C.LiteralDestroy(l.cLiteral)
	l.cLiteral = nil
	l.shape = Shape{}
}

// IsNil returns true is either l is nil, or its underlying C pointer is nil.
func (l *Literal) IsNil() bool {
	return l == nil || l.cLiteral == nil
}

// Shape of the literal.
func (l *Literal) Shape() Shape {
	return l.shape
}
