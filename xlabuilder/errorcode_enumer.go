// Code generated by "enumer -type=errorCode errorcode.go"; DO NOT EDIT.

package xlabuilder

import (
	"fmt"
	"strings"
)

const (
	_errorCodeName_0      = "status_OKstatus_CANCELLEDstatus_UNKNOWNstatus_INVALID_ARGUMENTstatus_DEADLINE_EXCEEDEDstatus_NOT_FOUNDstatus_ALREADY_EXISTSstatus_PERMISSION_DENIEDstatus_RESOURCE_EXHAUSTEDstatus_FAILED_PRECONDITIONstatus_ABORTEDstatus_OUT_OF_RANGE"
	_errorCodeLowerName_0 = "status_okstatus_cancelledstatus_unknownstatus_invalid_argumentstatus_deadline_exceededstatus_not_foundstatus_already_existsstatus_permission_deniedstatus_resource_exhaustedstatus_failed_preconditionstatus_abortedstatus_out_of_range"
	_errorCodeName_1      = "status_UNAUTHENTICATED"
	_errorCodeLowerName_1 = "status_unauthenticated"
)

var (
	_errorCodeIndex_0 = [...]uint8{0, 9, 25, 39, 62, 86, 102, 123, 147, 172, 198, 212, 231}
	_errorCodeIndex_1 = [...]uint8{0, 22}
)

func (i errorCode) String() string {
	switch {
	case 0 <= i && i <= 11:
		return _errorCodeName_0[_errorCodeIndex_0[i]:_errorCodeIndex_0[i+1]]
	case i == 16:
		return _errorCodeName_1
	default:
		return fmt.Sprintf("errorCode(%d)", i)
	}
}

// An "invalid array index" compiler error signifies that the constant values have changed.
// Re-run the stringer command to generate them again.
func _errorCodeNoOp() {
	var x [1]struct{}
	_ = x[status_OK-(0)]
	_ = x[status_CANCELLED-(1)]
	_ = x[status_UNKNOWN-(2)]
	_ = x[status_INVALID_ARGUMENT-(3)]
	_ = x[status_DEADLINE_EXCEEDED-(4)]
	_ = x[status_NOT_FOUND-(5)]
	_ = x[status_ALREADY_EXISTS-(6)]
	_ = x[status_PERMISSION_DENIED-(7)]
	_ = x[status_RESOURCE_EXHAUSTED-(8)]
	_ = x[status_FAILED_PRECONDITION-(9)]
	_ = x[status_ABORTED-(10)]
	_ = x[status_OUT_OF_RANGE-(11)]
	_ = x[status_UNAUTHENTICATED-(16)]
}

var _errorCodeValues = []errorCode{status_OK, status_CANCELLED, status_UNKNOWN, status_INVALID_ARGUMENT, status_DEADLINE_EXCEEDED, status_NOT_FOUND, status_ALREADY_EXISTS, status_PERMISSION_DENIED, status_RESOURCE_EXHAUSTED, status_FAILED_PRECONDITION, status_ABORTED, status_OUT_OF_RANGE, status_UNAUTHENTICATED}

var _errorCodeNameToValueMap = map[string]errorCode{
	_errorCodeName_0[0:9]:          status_OK,
	_errorCodeLowerName_0[0:9]:     status_OK,
	_errorCodeName_0[9:25]:         status_CANCELLED,
	_errorCodeLowerName_0[9:25]:    status_CANCELLED,
	_errorCodeName_0[25:39]:        status_UNKNOWN,
	_errorCodeLowerName_0[25:39]:   status_UNKNOWN,
	_errorCodeName_0[39:62]:        status_INVALID_ARGUMENT,
	_errorCodeLowerName_0[39:62]:   status_INVALID_ARGUMENT,
	_errorCodeName_0[62:86]:        status_DEADLINE_EXCEEDED,
	_errorCodeLowerName_0[62:86]:   status_DEADLINE_EXCEEDED,
	_errorCodeName_0[86:102]:       status_NOT_FOUND,
	_errorCodeLowerName_0[86:102]:  status_NOT_FOUND,
	_errorCodeName_0[102:123]:      status_ALREADY_EXISTS,
	_errorCodeLowerName_0[102:123]: status_ALREADY_EXISTS,
	_errorCodeName_0[123:147]:      status_PERMISSION_DENIED,
	_errorCodeLowerName_0[123:147]: status_PERMISSION_DENIED,
	_errorCodeName_0[147:172]:      status_RESOURCE_EXHAUSTED,
	_errorCodeLowerName_0[147:172]: status_RESOURCE_EXHAUSTED,
	_errorCodeName_0[172:198]:      status_FAILED_PRECONDITION,
	_errorCodeLowerName_0[172:198]: status_FAILED_PRECONDITION,
	_errorCodeName_0[198:212]:      status_ABORTED,
	_errorCodeLowerName_0[198:212]: status_ABORTED,
	_errorCodeName_0[212:231]:      status_OUT_OF_RANGE,
	_errorCodeLowerName_0[212:231]: status_OUT_OF_RANGE,
	_errorCodeName_1[0:22]:         status_UNAUTHENTICATED,
	_errorCodeLowerName_1[0:22]:    status_UNAUTHENTICATED,
}

var _errorCodeNames = []string{
	_errorCodeName_0[0:9],
	_errorCodeName_0[9:25],
	_errorCodeName_0[25:39],
	_errorCodeName_0[39:62],
	_errorCodeName_0[62:86],
	_errorCodeName_0[86:102],
	_errorCodeName_0[102:123],
	_errorCodeName_0[123:147],
	_errorCodeName_0[147:172],
	_errorCodeName_0[172:198],
	_errorCodeName_0[198:212],
	_errorCodeName_0[212:231],
	_errorCodeName_1[0:22],
}

// errorCodeString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func errorCodeString(s string) (errorCode, error) {
	if val, ok := _errorCodeNameToValueMap[s]; ok {
		return val, nil
	}

	if val, ok := _errorCodeNameToValueMap[strings.ToLower(s)]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to errorCode values", s)
}

// errorCodeValues returns all values of the enum
func errorCodeValues() []errorCode {
	return _errorCodeValues
}

// errorCodeStrings returns a slice of all String values of the enum
func errorCodeStrings() []string {
	strs := make([]string, len(_errorCodeNames))
	copy(strs, _errorCodeNames)
	return strs
}

// IsAerrorCode returns "true" if the value is listed in the enum definition. "false" otherwise
func (i errorCode) IsAerrorCode() bool {
	for _, v := range _errorCodeValues {
		if i == v {
			return true
		}
	}
	return false
}
