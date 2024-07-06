package dtypes

import (
	"github.com/gomlx/exceptions"
	"github.com/pkg/errors"
	"github.com/x448/float16"
	"math"
	"reflect"
	"strconv"
)

// Generate automatic C-to-Go boilerplate code for pjrt_c_api.h.
//go:generate go run ../cmd/dtypes_codegen

// FromGenericsType returns the DType enum for the given type that this package knows about.
func FromGenericsType[T Supported]() DType {
	var t T
	switch (any(t)).(type) {
	case float64:
		return Float64
	case float32:
		return Float32
	case float16.Float16:
		return Float16
	case int:
		switch strconv.IntSize {
		case 32:
			return Int32
		case 64:
			return Int64
		default:
			exceptions.Panicf("Cannot use int of %d bits with gopjrt -- try using int32 or int64", strconv.IntSize)
		}
	case int64:
		return Int64
	case int32:
		return Int32
	case int16:
		return Int16
	case int8:
		return Int8
	case bool:
		return Bool
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint64:
		return Uint64
	case complex64:
		return Complex64
	case complex128:
		return Complex128
	}
	return InvalidDType
}

// FromGoType returns the DType for the given [reflect.Type].
// It panics for unknown DType values.
func FromGoType(t reflect.Type) DType {
	if t == float16Type {
		return Float16
	}
	switch t.Kind() {
	case reflect.Int:
		switch strconv.IntSize {
		case 32:
			return Int32
		case 64:
			return Int64
		default:
			exceptions.Panicf("cannot use int of %d bits with GoMLX -- try using int32 or int64", strconv.IntSize)
		}
	case reflect.Int64:
		return Int64
	case reflect.Int32:
		return Int32
	case reflect.Int16:
		return Int16
	case reflect.Int8:
		return Int8

	case reflect.Uint64:
		return Uint64
	case reflect.Uint32:
		return Uint32
	case reflect.Uint16:
		return Uint16
	case reflect.Uint8:
		return Uint8

	case reflect.Bool:
		return Bool

	case reflect.Float32:
		return Float32
	case reflect.Float64:
		return Float64

	case reflect.Complex64:
		return Complex64
	case reflect.Complex128:
		return Complex128
	default:
		return InvalidDType
	}
	return InvalidDType
}

// FromAny introspects the underlying type of any and return the corresponding DType.
// Non-scalar types, or not supported types returns a InvalidType.
func FromAny(value any) DType {
	return FromGoType(reflect.TypeOf(value))
}

// Size returns the number of bytes for the given DType.
func (dtype DType) Size() int {
	return int(dtype.GoType().Size())
}

// Pre-generate constant reflect.TypeOf for convenience.
var (
	float32Type = reflect.TypeOf(float32(0))
	float64Type = reflect.TypeOf(float64(0))
	float16Type = reflect.TypeOf(float16.Float16(0))
)

// GoType returns the Go `reflect.Type` corresponding to the tensor DType.
func (dtype DType) GoType() reflect.Type {
	switch dtype {
	case Int64:
		return reflect.TypeOf(int64(0))
	case Int32:
		return reflect.TypeOf(int32(0))
	case Int16:
		return reflect.TypeOf(int16(0))
	case Int8:
		return reflect.TypeOf(int8(0))

	case Uint64:
		return reflect.TypeOf(uint64(0))
	case Uint32:
		return reflect.TypeOf(uint32(0))
	case Uint16:
		return reflect.TypeOf(uint16(0))
	case Uint8:
		return reflect.TypeOf(uint8(0))

	case Bool:
		return reflect.TypeOf(true)

	case Float16:
		return float16Type
	case Float32:
		return float32Type
	case Float64:
		return float64Type

	case Complex64:
		return reflect.TypeOf(complex64(0))
	case Complex128:
		return reflect.TypeOf(complex128(0))

	default:
		exceptions.Panicf("unknown dtype %q (%d) in DType.GoType", dtype, dtype)
		panic(nil) // Quiet lint warning.
	}
}

// GoStr converts dtype to the corresponding Go type and convert that to string.
// Notice the names are different from the Dtype (so `Int64` dtype is simply `int` in Go).
func (dtype DType) GoStr() string {
	return dtype.GoType().Name()
}

// LowestValue for dtype converted to the corresponding Go type.
// For float values it will return negative infinite.
// There is no lowest value for complex numbers, since they are not ordered.
func (dtype DType) LowestValue() any {
	switch dtype {
	case Int64:
		return int64(math.MinInt64)
	case Int32:
		return int32(math.MinInt32)
	case Int16:
		return int16(math.MinInt16)
	case Int8:
		return int16(math.MinInt8)

	case Uint64:
		return uint64(0)
	case Uint32:
		return uint32(0)
	case Uint16:
		return uint16(0)
	case Uint8:
		return uint8(0)

	case Bool:
		return false

	case Float32:
		return float32(math.Inf(-1))
	case Float64:
		return math.Inf(-1)
	case Float16:
		return float16.Inf(-1)

	default:
		exceptions.Panicf("LowestValue for dtype %s not defined", dtype)
	}
	return 0 // Never reaches here.
}

// HighestValue for dtype converted to the corresponding Go type.
// For float values it will return infinite.
// There is no lowest value for complex numbers, since they are not ordered.
func (dtype DType) HighestValue() any {
	switch dtype {
	case Int64:
		return int64(math.MaxInt64)
	case Int32:
		return int32(math.MaxInt32)
	case Int16:
		return int16(math.MaxInt16)
	case Int8:
		return int8(math.MaxInt8)

	case Uint64:
		return uint64(math.MaxUint64)
	case Uint32:
		return uint32(math.MaxUint32)
	case Uint16:
		return uint16(math.MaxUint16)
	case Uint8:
		return uint8(math.MaxUint8)

	case Bool:
		return true

	case Float32:
		return float32(math.Inf(1))
	case Float64:
		return math.Inf(1)
	case Float16:
		return float16.Inf(1)

	default:
		exceptions.Panicf("LowestValue for dtype %s not defined", dtype)
	}
	return 0 // Never reaches here.
}

// SmallestNonZeroValueForDType is the smallest non-zero value dtypes.
// Only useful for float types.
// The return value is converted to the corresponding Go type.
// There is no smallest non-zero value for complex numbers, since they are not ordered.
func (dtype DType) SmallestNonZeroValueForDType() any {
	switch dtype {
	case Int64:
		return int64(1)
	case Int32:
		return int32(1)
	case Int16:
		return int16(1)
	case Int8:
		return int8(1)

	case Uint64:
		return uint64(1)
	case Uint32:
		return uint32(1)
	case Uint16:
		return uint16(1)
	case Uint8:
		return uint8(1)

	case Bool:
		return true

	case Float32:
		return float32(math.SmallestNonzeroFloat32)
	case Float64:
		return math.SmallestNonzeroFloat64
	case Float16:
		return float16.Float16(0x0001) // 1p-24, see discussion in https://github.com/x448/float16/pull/46

	default:
		panic(errors.Errorf("SmallestNonZeroValueForDType not defined for dtype %s", dtype))
	}
}

// IsFloat returns whether dtype is a supported float -- float types not yet supported will return false.
// It returns false for complex numbers.
func (dtype DType) IsFloat() bool {
	return dtype == Float32 || dtype == Float64 || dtype == Float16 || dtype == BFloat16
}

// IsFloat16 returns whether dtype is a supported float with 16 bits: [Float16] or [BFloat16].
func (dtype DType) IsFloat16() bool {
	return dtype == Float16 || dtype == BFloat16
}

// IsComplex returns whether dtype is a supported complex number type.
func (dtype DType) IsComplex() bool {
	return dtype == Complex64 || dtype == Complex128
}

// RealDType returns the real component of complex dtypes.
// For float dtypes, it returns itself.
//
// It returns InvalidDType for other non-(complex or float) dtypes.
func (dtype DType) RealDType() DType {
	if dtype.IsFloat() {
		return dtype
	}
	switch dtype {
	case Complex64:
		return Float32
	case Complex128:
		return Float64
	default:
		// RealDType is not defined for other dtypes.
		return InvalidDType
	}
}

// IsInt returns whether dtype is a supported integer type -- float types not yet supported will return false.
func (dtype DType) IsInt() bool {
	return dtype == Int64 || dtype == Int32 || dtype == Int16 || dtype == Int8 ||
		dtype == Uint8 || dtype == Uint16 || dtype == Uint32 || dtype == Uint64
}

// IsSupported returns whether dtype is supported by `gopjrt`.
func (dtype DType) IsSupported() bool {
	return dtype == Bool || dtype == Float16 || dtype == Float32 || dtype == Float64 || dtype == Int64 || dtype == Int32 || dtype == Int16 || dtype == Int8 || dtype == Uint32 || dtype == Uint16 || dtype == Uint8 || dtype == Complex64 || dtype == Complex128
}

// Supported lists the Go types that `gopjrt` knows how to convert -- there are more types that can be manually
// converted.
// Used as traits for generics.
//
// Notice Go's `int` type is not portable, since it may translate to dtypes Int32 or Int64 depending
// on the platform.
type Supported interface {
	bool | float32 | float64 | float16.Float16 | int | int32 | int64 | uint8 | uint32 | uint64 | complex64 | complex128
}

// Number represents the Go numeric types that are supported by graph package.
// Used as traits for generics.
//
// Notice that "int" becomes int64 in the implementation.
// Since it needs a 1:1 mapping, it gets converted back to int64.
// It includes complex numbers.
type Number interface {
	float32 | float64 | int | int32 | int64 | uint8 | uint32 | uint64 | complex64 | complex128
}

// NumberNotComplex represents the Go numeric types that are supported by graph package except the complex numbers.
// Used as a Generics constraint.
// See Number for details.
type NumberNotComplex interface {
	float32 | float64 | int | int32 | int64 | uint8 | uint32 | uint64
}

// GoFloat represent a continuous Go numeric type, supported by GoMLX.
type GoFloat interface {
	float32 | float64
}
