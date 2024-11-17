#include "common.h"

const PJRT_Api* call_GetPJRTApiFn(GetPJRTApiFn fn) {
    return fn();
}

extern PJRT_NamedValueUnion Extract_PJRT_NamedValue_Union(PJRT_NamedValue *named_value) {
    PJRT_NamedValueUnion result;
    switch (named_value->type) {
    case PJRT_NamedValue_kString:
        result.string_value = named_value->string_value;
        break;
    case PJRT_NamedValue_kInt64:
        result.int64_value = named_value->int64_value;
        break;
    case PJRT_NamedValue_kInt64List:
        result.int64_array_value = named_value->int64_array_value;
        break;
    case PJRT_NamedValue_kFloat:
        result.float_value = named_value->float_value;
        break;
    case PJRT_NamedValue_kBool:
        result.bool_value = named_value->bool_value;
        break;
    }
    return result;
}

// Set the corresponding field in the PJRT_NamedValue structure.
// The one to use is based on named_value->type.
void Set_PJRT_NamedValue_Union(PJRT_NamedValue *named_value, PJRT_NamedValueUnion split_value) {
    switch (named_value->type) {
    case PJRT_NamedValue_kString:
        named_value->string_value = split_value.string_value;
        break;
    case PJRT_NamedValue_kInt64:
        named_value->int64_value = split_value.int64_value;
        break;
    case PJRT_NamedValue_kInt64List:
        named_value->int64_array_value = split_value.int64_array_value;
        break;
    case PJRT_NamedValue_kFloat:
        named_value->float_value = split_value.float_value;
        break;
    case PJRT_NamedValue_kBool:
        named_value->bool_value = split_value.bool_value;
        break;
    }
}