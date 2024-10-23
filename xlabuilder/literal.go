package xlabuilder

/*
#include <gomlx/xlabuilder/literal.h>
*/
import "C"
import (
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
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
func NewLiteralFromShape(shape Shape) (*Literal, error) {
	if shape.TupleSize() > 0 {
		return nil, errors.Errorf("NewLiteralFromShape cannot be used to create tuple literals, shape given was %s", shape)
	}
	if shape.DType == dtypes.InvalidDType {
		return nil, errors.Errorf("cannot create literal of invalid dtype, shape=%s", shape)
	}
	if shape.Size() <= 0 {
		return nil, errors.Errorf("cannot create literal of size <= 0, shape=%s", shape)
	}
	cShape := cShapeFromShape(shape) // Ownership is given to the new Literal structure.
	cLiteral := C.MakeLiteralFromShape(cShape)
	l, err := newLiteral(cLiteral, shape)
	if err != nil {
		return nil, err
	}
	return l, nil
}

// NewArrayLiteral creates a Literal initialized from the array flat data (a slice) and the dimensions of the array.
//
// If dimensions is omitted, it is assumed to represent a 1D-array of the length given.
func NewArrayLiteral[T dtypes.Supported](flat []T, dimensions ...int) (*Literal, error) {
	if len(dimensions) == 0 {
		dimensions = []int{len(flat)}
	}
	shape := MakeShape(dtypes.FromGenericsType[T](), dimensions...)
	if shape.Size() != len(flat) {
		return nil, errors.Errorf("NewArrayLiteral got a slice of length %d, but the shape %s given has %d elements",
			len(flat), shape, shape.Size())
	}
	l, err := NewLiteralFromShape(shape)
	if err != nil {
		return nil, err
	}
	lData := unsafe.Slice((*T)(unsafe.Pointer(l.cLiteral.data)), int(l.cLiteral.size))
	copy(lData, flat)
	return l, nil
}

// Data calls accessFn with data pointing to the bytes (C++ allocated) of the literal.
//
// The ownership of the data is maintained by the Literal, and it is guaranteed to be live (not GC'ed) until
// the end of the current function call, at least.
func (l *Literal) Data(accessFn func(data []byte)) {
	defer runtime.KeepAlive(l)
	accessFn(unsafe.Slice((*byte)(unsafe.Pointer(l.cLiteral.data)), int(l.cLiteral.size_bytes)))
}

// NewScalarLiteral creates a scalar Literal initialized with the given value.
func NewScalarLiteral[T dtypes.Supported](value T) *Literal {
	shape := MakeShape(dtypes.FromGenericsType[T]())
	l, err := NewLiteralFromShape(shape)
	if err != nil {
		// Notice this should never happen, since dtypes.Support should always lead to valid shapes.
		panic(err)
	}
	*(*T)(unsafe.Pointer(l.cLiteral.data)) = value
	return l
}

// NewScalarLiteralFromFloat64 creates a scalar Literal with the given dtype initialized from the given value as float64.
// This can be used to create common constants for arbitrary dtypes.
//
// It returns an error if dtype cannot be converted.
func NewScalarLiteralFromFloat64(value float64, dtype dtypes.DType) (*Literal, error) {
	// Scalar values.
	switch dtype {
	case dtypes.InvalidDType:
		return nil, errors.Errorf("cannot create scalar literal of InvalidDtype, value=%g", value)
	case dtypes.Bool:
		return NewScalarLiteral(value != 0), nil
	case dtypes.Complex64:
		return NewScalarLiteral(complex(float32(value), 0)), nil
	case dtypes.Complex128:
		return NewScalarLiteral(complex(value, 0)), nil
	case dtypes.Float16:
		return NewScalarLiteral(float16.Fromfloat32(float32(value))), nil
	case dtypes.BFloat16:
		return NewScalarLiteral(bfloat16.FromFloat32(float32(value))), nil
	default:
		var convertedValue reflect.Value
		convertedValue = reflect.ValueOf(value)
		convertedValue = convertedValue.Convert(dtype.GoType())
		l, err := NewLiteralFromShape(MakeShape(dtype))
		if err != nil {
			return nil, err
		}
		lValueOf := reflect.NewAt(dtype.GoType(), unsafe.Pointer(l.cLiteral.data)).Elem()
		lValueOf.Set(convertedValue)
		return l, nil
	}
}

// NewScalarLiteralFromAny creates a scalar Literal with the given dynamically typed value.
// It uses reflection to inspect the type.
func NewScalarLiteralFromAny(value any) (*Literal, error) {
	valueOf := reflect.ValueOf(value)
	dtype := dtypes.FromGoType(valueOf.Type())
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("Go type %T has no equivalent dtype", value)
	}
	l, err := NewLiteralFromShape(MakeShape(dtype))
	if err != nil {
		return nil, err
	}
	lValueOf := reflect.NewAt(dtype.GoType(), unsafe.Pointer(l.cLiteral.data)).Elem()
	lValueOf.Set(valueOf)
	return l, nil
}

// NewArrayLiteralFromAny creates a scalar Literal with the given dynamically typed flat values and its underlying dimensions.
// It uses reflection to inspect the type.
func NewArrayLiteralFromAny(flatAny any, dimensions ...int) (*Literal, error) {
	flatV := reflect.ValueOf(flatAny)
	if flatV.Kind() != reflect.Slice {
		return nil, errors.Errorf("NewArrayLiteralFromAny expects a slice, got %T instead", flatAny)
	}
	dtype := dtypes.FromGoType(flatV.Type().Elem())
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("NewArrayLiteralFromAny expects a slice of valid DTypes, got %T instead", flatAny)
	}
	shape, err := MakeShapeOrError(dtype, dimensions...)
	if err != nil {
		return nil, err
	}
	if shape.Size() != flatV.Len() {
		return nil, errors.Errorf("NewArrayLiteralFromAny got a slice of length %d, but the shape %s given has %d elements",
			flatV.Len(), shape, shape.Size())
	}

	// Copy data as bytes -- to avoid using complex reflected slices.
	flatPtr := flatV.Index(0).Addr().UnsafePointer()
	var pinner runtime.Pinner
	pinner.Pin(flatPtr)
	defer pinner.Unpin()
	flatData := unsafe.Slice((*byte)(flatPtr), shape.Memory())

	l, err := NewLiteralFromShape(shape)
	if err != nil {
		return nil, err
	}
	lData := unsafe.Slice((*byte)(unsafe.Pointer(l.cLiteral.data)), shape.Memory())
	copy(lData, flatData)
	return l, nil
}

// newLiteral creates the literal and registers the finalizer.
func newLiteral(cLiteral *C.Literal, shape Shape) (*Literal, error) {
	l := &Literal{cLiteral: cLiteral, shape: shape}
	if int(cLiteral.size) != shape.Size() {
		return nil, errors.Errorf("new literal being created has shape=%s (size %d), but internal size reported is %d",
			shape, shape.Size(), cLiteral.size)
	}
	runtime.SetFinalizer(l, func(l *Literal) { l.Destroy() })
	return l, nil
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
