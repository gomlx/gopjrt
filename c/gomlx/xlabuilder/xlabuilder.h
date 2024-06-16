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
#include "gomlx/xlabuilder/node.h"
#include "gomlx/xlabuilder/utils.h"

#ifdef __cplusplus
// C++ dependencies.
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"

// Alias to xla::Literal.
typedef xla::XlaOp XlaOp;

#else
typedef _Bool bool;
typedef void XlaGlobalData;

// Forward reference of C++ types.
struct Computation;
typedef struct Computation Computation;
#endif

#ifdef __cplusplus
extern "C" {

// Computation, internal representation to the C++ binding code. In
// contains the builder and later the compiled computation.
struct Computation {
  ~Computation() {
    if (builder != nullptr) {
      delete builder;
    }
  }

  // Builder, set while the Computation is being built.
  xla::XlaBuilder *builder;
};

#endif

// NewComputation creates a C++ handle to a computation building structure. It
// owns the xla::XlaBuilder, and eventually the xla::XlaComputation and
// xla::ExecutionHandle.
extern Computation *NewComputation(char *name);

// ComputationAddOp creates an xla::XlaOp for the given node description.
// Returns the new op and its shape in the fields `node.new_op` and
// `node.new_shape`. Ownership of the memory is transferred back.
extern XlaStatus *ComputationAddOp(Computation *comp, SerializedNode *node);

// DeleteComputation will destroy associated resources.
extern void DeleteComputation(void *comp);

// DeleteXlaOp delete XlaOp reference.
extern void DeleteXlaOp(XlaOp *op);

// SerializedHLO converts the computation in comp to a serialized HLO proto, that can be used by PJRT.
// It returns an error or a VectorData of bytes, with the serialized HLO proto (format is "hlo" when using in PJRT).
extern StatusOr SerializedHLO(Computation *comp, XlaOp *output);

#ifdef __cplusplus
}
#endif

#endif