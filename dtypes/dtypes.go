package dtypes

import (
	"github.com/gomlx/exceptions"
	"strconv"
)

// Generate automatic C-to-Go boilerplate code for pjrt_c_api.h.
//go:generate go run ../cmd/dtypes_codegen

// Aliases to the dtypes defined in pjrt_c_api.h
const (
	// InvalidDType (an alias for INVALID) represents an invalid (or not set) dtype.
	InvalidDType = INVALID

	// Bool (an alias for PRED) is used as the output and input of logic operations.
	Bool = PRED

	Int8  = S8
	Int16 = S16
	Int32 = S32
	Int64 = S64

	Uint8  = U8
	Uint16 = U16
	Uint32 = U32
	Uint64 = U64

	Float32 = F32
	Float64 = F64

	Complex64  = C64
	Complex128 = C128
)

// Supported lists the Go types that `gopjrt` knows how to convert -- there are more types that can be manually
// converted.
// Used as traits for generics.
type Supported interface {
	bool | float32 | float64 | int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | complex64 | complex128
}

// DTypeGeneric returns the DType enum for the given type that this package knows about.
func DTypeGeneric[T Supported]() DType {
	var t T
	switch (any(t)).(type) {
	case float64:
		return Float64
	case float32:
		return Float32
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
