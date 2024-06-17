/*
 *	Copyright 2023 Jan Pfeifer
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

// xlabuilder.h holds C API wrapper for xla::Builder and xla::Computation:
//
// - Computation (holds an XlaBuilder)
// - XlaOp
// - XlaBuilder

#ifndef _GOMLX_XLABUILDER_XLABUILDER_H
#define _GOMLX_XLABUILDER_XLABUILDER_H

#include "gomlx/xlabuilder/literal.h"
#include "gomlx/xlabuilder/op.h"
#include "gomlx/xlabuilder/utils.h"

#ifdef __cplusplus
// C++ only: Dependencies.
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"

// Alias to xla::Literal.
typedef xla::XlaOp XlaOp;
typedef xla::XlaBuilder XlaBuilder;

#else
// C only: Forward reference of C++ types.
typedef _Bool bool;
typedef void XlaBuilder;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// NewXlaBuilder returns a new xla::XlaBuilder.
// The caller owns the returned pointer and the name string.
extern XlaBuilder *NewXlaBuilder(char *name);

// XlaBuilderAddOp creates an xla::XlaOp for the given node.
//
// The parameter `node` is used both for input and output.
// It returns the new op and its shape in the fields `node.new_op` and
// `node.new_shape`.
//
// The caller owns `node` before and after the call.
extern XlaStatus *XlaBuilderAddOp(XlaBuilder *builder, SerializedOp *serialized_op);

// DestroyXlaBuilder destroys and frees the builder.
extern void DestroyXlaBuilder(XlaBuilder *builder);

// DestroyXlaOp destroys the XlaOp reference.
extern void DestroyXlaOp(XlaOp *op);

// SerializedHLO converts the computation built using XlaBuilder to a serialized HLO proto, that can be used by PJRT.
//
// It returns an error or a VectorData of bytes, with the serialized HLO proto (format is "hlo" when using in PJRT).
extern StatusOr SerializedHLO(XlaBuilder *builder, XlaOp *output_node);

#ifdef __cplusplus
}
#endif

#endif