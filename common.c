#include "common.h"

const PJRT_Api* call_GetPJRTApiFn(GetPJRTApiFn fn) {
    return fn();
}