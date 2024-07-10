package pjrt

/*
#include "pjrt_c_api.h"
#include "common.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"fmt"
	"github.com/pkg/errors"
	"unsafe"
)

// NamedValuesMap map names to any of the supported named values types defined by PJRT_NamedValue_Type.
type NamedValuesMap map[string]any

// pjrtNamedValuesToMap converts a slice of C.PJRT_NamedValue to a Go map of the name to the values (any).
// The values supported by pjrtNamedValues are the ones defined by PJRT_NamedValue_Type.
func pjrtNamedValuesToMap(namedValues []C.PJRT_NamedValue) NamedValuesMap {
	m := make(NamedValuesMap)
	for _, pair := range namedValues {
		name := cCharArray(pair.name, pair.name_size)
		value := C.Extract_PJRT_NamedValue_Union(&pair)

		// Note from CGO: Within the Go file, C's struct field names that are keywords in Go can be accessed by prefixing
		// them with an underscore: if x points at a C struct with a field named "type", x._type accesses the field.
		switch PJRT_NamedValue_Type(pair._type) {
		case PJRT_NamedValue_kString:
			m[name] = cCharArray(value.string_value, pair.value_size)
		case PJRT_NamedValue_kInt64:
			m[name] = int64(value.int64_value)
		case PJRT_NamedValue_kInt64List:
			m[name] = cDataToSlice[int64](unsafe.Pointer(value.int64_array_value), int(pair.value_size))
		case PJRT_NamedValue_kFloat:
			m[name] = float32(value.float_value)
		case PJRT_NamedValue_kBool:
			m[name] = bool(value.bool_value)
		default:
			m[name] = fmt.Sprintf("uknown_type_%d", int(pair._type))
		}
	}
	return m
}

// mallocArrayPJRT_NamedValue return a pointer to a C allocated array of C.PJRT_NamedValue and the number
// of elements in that array (same as len(NamedValuesMap)).
//
// If the NamedValuesMap is empty (or nil) it returns (nil, 0).
func (m NamedValuesMap) mallocArrayPJRT_NamedValue() (*C.PJRT_NamedValue, C.size_t, error) {
	if len(m) == 0 {
		return nil, 0, nil
	}

	// Fill struct_size for each element.
	rawData := cMallocArray[C.PJRT_NamedValue](len(m))
	placeHolder := C.new_PJRT_NamedValue() // Just so we get struct_size.
	sliceData := cDataToSlice[C.PJRT_NamedValue](unsafe.Pointer(rawData), len(m))
	ii := 0
	for key, anyValue := range m {
		// Initialize struct_size.
		cValue := &sliceData[ii]
		cValue.struct_size = placeHolder.struct_size
		cValue.name, cValue.name_size = C.CString(key), C.size_t(len(key))

		var splitValue C.PJRT_NamedValueUnion
		switch value := anyValue.(type) {
		case string:
			cValue._type = C.PJRT_NamedValue_Type(PJRT_NamedValue_kString)
			splitValue.string_value = C.CString(value)
			cValue.value_size = C.size_t(len(value))
		case int64:
			cValue._type = C.PJRT_NamedValue_Type(PJRT_NamedValue_kInt64)
			splitValue.int64_value = C.int64_t(value)
		case []int64:
			cValue._type = C.PJRT_NamedValue_Type(PJRT_NamedValue_kInt64List)
			splitValue.int64_array_value = cMallocArrayAndSet(len(value), func(jj int) C.int64_t {
				return C.int64_t(value[jj])
			})
			cValue.value_size = C.size_t(len(value))
		case float32:
			cValue._type = C.PJRT_NamedValue_Type(PJRT_NamedValue_kFloat)
			splitValue.float_value = C.float(value)
		case bool:
			cValue._type = C.PJRT_NamedValue_Type(PJRT_NamedValue_kBool)
			splitValue.bool_value = C.bool(value)
		default:
			destroyPJRT_NamedValue(rawData)
			return nil, 0, errors.Errorf("option (NamedValueMap) %q was set to unsupported type %T (value=%v). "+
				"Only values of type string, int64, []int64, float32 and bool are supported.",
				key, value, value)
		}
		C.Set_PJRT_NamedValue_Union(cValue, splitValue)

		// Next entry.
		ii++
	}
	return rawData, C.size_t(len(m)), nil
}

func destroyPJRT_NamedValue(cValue *C.PJRT_NamedValue) {
	// TODO: destroy union fields, if array.
	if cValue != nil {
		cFree(cValue)
	}
}
