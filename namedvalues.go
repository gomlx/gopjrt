package gopjrt

/*
#include "pjrt_c_api.h"
#include "common.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// NamedValuesMap map names to any of the supported named values types defined by PJRT_NamedValue_Type.
type NamedValuesMap map[string]any

// pjrtNamedValuesToMap convert an slice of C.PJRT_NamedValue to a Go map of the name to the values (any).
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
