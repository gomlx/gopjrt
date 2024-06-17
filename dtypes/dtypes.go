package dtypes

// Generate automatic C-to-Go boilerplate code for pjrt_c_api.h.
//go:generate go run ../cmd/dtypes_codegen

// Aliases to the dtypes defined in pjrt_c_api.h
const (
	// Invalid (an alias for INVALID) represents an invalid (or not set) dtype.
	Invalid = INVALID

	// Bool (an alias for PRED) is used as the output and input of logic operations.
	Bool = PRED

	Int8  = S8
	Int16 = S16
	Int32 = S32
	Int64 = S64
)
