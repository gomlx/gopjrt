/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

#ifndef GOMLX_GOPJRT_COMMON
#define GOMLX_GOPJRT_COMMON
#include "pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef const PJRT_Api* (*GetPJRTApiFn)();
extern const PJRT_Api* call_GetPJRTApiFn(GetPJRTApiFn fn);

// De-union-ed values of PJRT_NamedValue, because Go cannot handle unnamed unions.
typedef struct {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
} PJRT_NamedValueUnion;

// De-union the value of a PRJT_NamedValue, to allow Go access to it.
extern PJRT_NamedValueUnion Extract_PJRT_NamedValue_Union(PJRT_NamedValue *named_value);

// Set the corresponding field in the PJRT_NamedValue structure.
// The one to use is based on named_value->type.
extern void Set_PJRT_NamedValue_Union(PJRT_NamedValue *named_value, PJRT_NamedValueUnion split_value);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif  // GOMLX_GOPJRT_COMMON