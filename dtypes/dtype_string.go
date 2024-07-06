// Code generated by "stringer -type=DType gen_dtype_enum.go"; DO NOT EDIT.

package dtypes

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[InvalidDType-0]
	_ = x[Bool-1]
	_ = x[Int8-2]
	_ = x[Int16-3]
	_ = x[Int32-4]
	_ = x[Int64-5]
	_ = x[Uint8-6]
	_ = x[Uint16-7]
	_ = x[Uint32-8]
	_ = x[Uint64-9]
	_ = x[Float16-10]
	_ = x[Float32-11]
	_ = x[Float64-12]
	_ = x[BFloat16-13]
	_ = x[Complex64-14]
	_ = x[Complex128-15]
	_ = x[F8E5M2-16]
	_ = x[F8E4M3FN-17]
	_ = x[F8E4M3B11FNUZ-18]
	_ = x[F8E5M2FNUZ-19]
	_ = x[F8E4M3FNUZ-20]
	_ = x[S4-21]
	_ = x[U4-22]
	_ = x[TOKEN-23]
	_ = x[S2-24]
	_ = x[U2-25]
}

const _DType_name = "InvalidDTypeBoolInt8Int16Int32Int64Uint8Uint16Uint32Uint64Float16Float32Float64BFloat16Complex64Complex128F8E5M2F8E4M3FNF8E4M3B11FNUZF8E5M2FNUZF8E4M3FNUZS4U4TOKENS2U2"

var _DType_index = [...]uint8{0, 12, 16, 20, 25, 30, 35, 40, 46, 52, 58, 65, 72, 79, 87, 96, 106, 112, 120, 133, 143, 153, 155, 157, 162, 164, 166}

func (i DType) String() string {
	if i < 0 || i >= DType(len(_DType_index)-1) {
		return "DType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _DType_name[_DType_index[i]:_DType_index[i+1]]
}
