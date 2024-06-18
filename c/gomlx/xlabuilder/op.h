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

// node.h holds C API structure to serialized nodes, and related C/C++ types:
//
// - SerializedOp
// - NodeType enum (from gen_node_types.h)
// - XlaLiteral
// - XlaOp

#ifndef _GOMLX_XLABUILDER_SERIALIZED_OP_H
#define _GOMLX_XLABUILDER_SERIALIZED_OP_H

#include <stdlib.h>

#include "gomlx/xlabuilder/gen_op_types.h"
#include "gomlx/xlabuilder/shape.h"
#include "gomlx/xlabuilder/literal.h"

#ifdef __cplusplus
// C++ only includes: these are not seen by the Go compiler.
#include "xla/client/xla_builder.h"
#include "xla/shape.h"

typedef xla::XlaOp XlaOp;

#else
// C and CGO only code.
typedef _Bool bool;
typedef void XlaOp;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef XlaOp *XlaOpPtr;

// SerializedOp represents the Node arguments needed to create an XlaOp. The
// underlying data (pointers) are owned by Go, and shouldn't be freed by C
// functions.
typedef struct {
  int32_t op_type;
  int32_t num_op_inputs;
  XlaOpPtr *op_inputs;

  // When there is a literal involved.
  struct Literal *literal;

  // Extra arguments that depend on the node type:
  int64_t integer;
  int64_t *integer_array;
  int32_t integer_array_size;
  Shape *shape;
  char *string;
  float float_v;

  // Output: information about the new op created, filled in by XlaBuilderAddOp.
  // Space allocated in C, but ownership is transferred back to the caller (in Go).
  XlaOp *new_op;
  Shape *new_shape;
} SerializedOp;

#ifdef __cplusplus
}
#endif

#endif // _GOMLX_XLABUILDER_SERIALIZED_OP_H
