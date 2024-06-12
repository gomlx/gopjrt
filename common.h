#ifndef GOMLX_GOPJRT_COMMON
#define GOMLX_GOPJRT_COMMON
#include "pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef const PJRT_Api* (*GetPJRTApiFn)();


#ifdef __cplusplus
}  // extern "C"
#endif


#endif  // GOMLX_GOPJRT_COMMON