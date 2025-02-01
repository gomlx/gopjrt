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

#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>

#include "xlabuilder.h"

#include "gomlx/xlabuilder/literal.h"
#include "gomlx/xlabuilder/utils.h"
#include "gomlx/xlabuilder/op.h"
#include "gomlx/xlabuilder/gen_op_types.h"

#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/literal.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"


using namespace std;

// GopjrtXlaBuilderVersion is the "semantic versioning" numbers (e.g. "v0.6.0") of the C/C++
// XlaBuilder wrapper library for Gopjrt.
//
// This often lags behind Gopjrt version, if/when the C/C++ wrapper doesn't change --
// we don't bump the version of the C/C++ code if it doesn't change.
// But when it changes, it matches the Gopjrt version it's being released with.
const char *GopjrtXlaBuilderVersion = "v0.6.0";

// ShapeFromXlaShape allocates and sets a new Shape struct set with the same
// shape defined by xla::Shape. C++ only.
extern Shape *ShapeFromXlaShape(const xla::Shape &shape);

XlaBuilder *NewXlaBuilder(char *name) {
  return new xla::XlaBuilder(name);
}

void XlaBuilderDestroy(XlaBuilder *builder) {
  if (builder != nullptr) {
    delete builder;
  }
}

char* XlaBuilderName(XlaBuilder *builder) {
    return c_str(builder->name());
}

XlaBuilder * XlaBuilderCreateSubBuilder(XlaBuilder *builder, char* name) {
    auto newBuilder = builder->CreateSubBuilder(name);  // unique_ptr<XlaBuilder>
    return newBuilder.release();  // Ownership is returned.
}

void XlaOpDestroy(XlaOp *op) {
    delete (static_cast<xla::XlaOp *>(op));
}

// TODO: Rewrite with more sane serialized representation, and for simple ops, auto-generated.
XlaStatus *XlaBuilderAddOp(XlaBuilder *builder, SerializedOp *serialized_op) {
  // Create new XlaOp.
  // TODO: A registration mechanism, where one can implement different
  // op_types in different files or libraries even.
  xla::XlaOp op;

  // Extract optional parameters.
  xla::XlaOp **inputs = serialized_op->op_inputs;
  absl::Span<const int64_t> list_of_ints(serialized_op->integer_array,
                                         serialized_op->integer_array_size);
  absl::Span<const int64_t> shape_dimensions;
  xla::Shape shape;
  if (serialized_op->shape != nullptr) {
    shape_dimensions =
        absl::Span<const int64_t>(serialized_op->shape->dimensions, serialized_op->shape->rank);
    shape = MakeXlaShape(serialized_op->shape);
  }

  // Tool to decode contents encoded in the integer array.
  int integer_array_pos = 0;
  auto decode = [&integer_array_pos, serialized_op]() {
    return serialized_op->integer_array[integer_array_pos++];
  };
  auto decodeSpan = [&integer_array_pos, serialized_op](int len) {
    const absl::Span<const int64_t> result(
        serialized_op->integer_array + integer_array_pos, len);
    integer_array_pos += len;
    return result;
  };

  // Switch for each op type.
  switch (serialized_op->op_type) {

  // Special ops:
  case ConstantOp:
    op = xla::ConstantLiteral(builder, *serialized_op->literal->literal);
    break;
  case IotaOp:
    op = xla::Iota(builder, shape, serialized_op->integer);
    break;
  case ParameterOp:
    op = xla::Parameter(builder, serialized_op->integer, shape, serialized_op->string);
    break;
  case ConvertDTypeOp:
    op = xla::ConvertElementType(
        *inputs[0], static_cast<xla::PrimitiveType>(serialized_op->integer));
    break;
  case WhereOp:
    op = xla::Select(*inputs[0], *inputs[1], *inputs[2]);
    break;
  case TupleOp: {
    std::vector<xla::XlaOp> ops;
    ops.reserve(serialized_op->num_op_inputs);
    for (int ii = 0; ii < serialized_op->num_op_inputs; ii++) {
      ops.push_back(xla::XlaOp(*inputs[ii]));
    }
    op = xla::Tuple(builder, ops);
    break;
  }
  case GetTupleElementOp:
    op = xla::GetTupleElement(*inputs[0], serialized_op->integer);
    break;
  case ReshapeOp:
    op = xla::Reshape(*inputs[0], shape_dimensions);
    break;
  case BroadcastOp:
    op = xla::Broadcast(*inputs[0], list_of_ints);
    break;
  case BroadcastInDimOp:
    op = xla::BroadcastInDim(*inputs[0], shape_dimensions, list_of_ints);
    break;
  case CallOp: {
    vector<xla::XlaOp> operands;
    for (int ii = 0; ii < serialized_op->num_op_inputs; ii++) {
      operands.push_back(*inputs[ii]);
    }
    op = xla::Call(builder, *(serialized_op->computation), operands);
    break;
  }
  case ReduceOp: {
    if (serialized_op->integer_array_size > 0) {
      op = xla::Reduce(*inputs[0], *inputs[1], *serialized_op->computation, list_of_ints);
    } else {
      op = xla::ReduceAll(*inputs[0], *inputs[1], *serialized_op->computation);
    }
    break;
  }
  case ArgMinMaxOp: {
    // Inputs:
    //   * inputs[0]: Tensor to `num_inputs_to_reduce` pairs of input/initial
    //   value.
    //   * serialized_op->integer: Axis on which to calculate the argmax/argmin.
    //   * serialized_op->integer_array[0]: `is_min`, whether to do argmin or argmax.
    //   * serialized_op->integer_array[1]: DType of the output.
    int axis(serialized_op->integer);
    bool is_min(serialized_op->integer_array[0]);
    xla::PrimitiveType output_type =
        static_cast<xla::PrimitiveType>(serialized_op->integer_array[1]);
    op = xla::ArgMinMax(*inputs[0], output_type, axis, is_min);
    break;
  }
  case SliceOp: {
    int rank = serialized_op->integer_array_size / 3;
    absl::Span<const int64_t> starts(serialized_op->integer_array, rank);
    absl::Span<const int64_t> limits(serialized_op->integer_array + rank, rank);
    absl::Span<const int64_t> strides(serialized_op->integer_array + 2 * rank, rank);
    op = xla::Slice(*inputs[0], starts, limits, strides);
    break;
  }
  case PadOp: {
    auto &operand = *inputs[0];
    auto &pad_value = *inputs[1];

    xla::PaddingConfig config;
    int rank = serialized_op->integer_array_size / 3;
    for (int ii = 0; ii < rank; ii++) {
      auto axisConfig = config.add_dimensions();
      axisConfig->set_edge_padding_low(decode());
      axisConfig->set_edge_padding_high(decode());
      axisConfig->set_interior_padding(decode());
    }

    op = xla::Pad(operand, pad_value, config);
    break;
  }
  case GatherOp: {
    xla::GatherDimensionNumbers gather_dims;
    int64_t index_vector_dim = serialized_op->integer_array[0];
    gather_dims.set_index_vector_dim(index_vector_dim);
    int64_t len_offset_dims = serialized_op->integer_array[1];
    int64_t len_collapsed_slice_dims = serialized_op->integer_array[2];
    int64_t len_start_index_map = serialized_op->integer_array[3];
    int64_t len_slice_sizes = serialized_op->integer_array[4];
    bool indices_are_sorted = bool(serialized_op->integer_array[5]);
    int pos = 6;
    for (int ii = 0; ii < len_offset_dims; ii++) {
      gather_dims.mutable_offset_dims()->Add(serialized_op->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_collapsed_slice_dims; ii++) {
      gather_dims.mutable_collapsed_slice_dims()->Add(
          serialized_op->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_start_index_map; ii++) {
      gather_dims.mutable_start_index_map()->Add(serialized_op->integer_array[pos++]);
    }
    // Same for collapsed_slice_dims and start_index_map
    absl::Span<const int64_t> slice_sizes(serialized_op->integer_array + pos,
                                          len_slice_sizes);
    op = xla::Gather(*inputs[0], *inputs[1], gather_dims, slice_sizes,
                     indices_are_sorted);
    break;
  }
  case ScatterOp: {
    int pos = 0;
    xla::ScatterDimensionNumbers scatter_dims;
    scatter_dims.set_index_vector_dim(serialized_op->integer_array[pos++]);
    bool unique_indices = bool(serialized_op->integer_array[pos++]);
    bool indices_are_sorted = bool(serialized_op->integer_array[pos++]);
    int64_t len_update_window_dims = serialized_op->integer_array[pos++];
    int64_t len_inserted_window_dims = serialized_op->integer_array[pos++];
    int64_t len_scatter_dims_to_operand_dims = serialized_op->integer_array[pos++];
    for (int ii = 0; ii < len_update_window_dims; ii++) {
      scatter_dims.mutable_update_window_dims()->Add(
          serialized_op->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_inserted_window_dims; ii++) {
      scatter_dims.mutable_inserted_window_dims()->Add(
          serialized_op->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_scatter_dims_to_operand_dims; ii++) {
      scatter_dims.mutable_scatter_dims_to_operand_dims()->Add(
          serialized_op->integer_array[pos++]);
    }
    // Create the update computation: only Add supported for now.
    auto shape_or = builder->GetShape(*inputs[0]);
    if (!shape_or.ok()) {
      return new absl::Status(std::move(shape_or.status()));
    }
    xla::PrimitiveType primitive_type = shape_or.value().element_type();
    auto update_computation =
        CreateScalarAddComputation(primitive_type, builder);
    op = Scatter(*inputs[0], *inputs[1], *inputs[2], update_computation,
                 scatter_dims, indices_are_sorted, unique_indices);
    break;
  }
  case ConcatenateOp: {
    vector<xla::XlaOp> operands;
    for (int ii = 0; ii < serialized_op->num_op_inputs; ii++) {
      operands.push_back(*inputs[ii]);
    }
    op = xla::ConcatInDim(inputs[0]->builder(), operands, serialized_op->integer);
    break;
  }
  case ConvGeneralDilatedOp: {
    int64_t num_spatial_dims = decode();
    int64_t filter_group_count = decode();
    int64_t batch_group_count = decode();

    // Array lengths.
    int64_t len_strides = decode();
    int64_t len_padding = decode();
    int64_t len_input_dilation = decode();
    int64_t len_filter_dilation = decode();

    // Decode ConvolutionDimensionNumbers.
    xla::ConvolutionDimensionNumbers conv_dims;
    conv_dims.set_input_batch_dimension(decode());
    conv_dims.set_input_feature_dimension(decode());
    for (int ii = 0; ii < num_spatial_dims; ii++) {
      conv_dims.mutable_input_spatial_dimensions()->Add(decode());
    }

    conv_dims.set_kernel_input_feature_dimension(decode());
    conv_dims.set_kernel_output_feature_dimension(decode());
    for (int ii = 0; ii < num_spatial_dims; ii++) {
      conv_dims.mutable_kernel_spatial_dimensions()->Add(decode());
    }

    conv_dims.set_output_batch_dimension(decode());
    conv_dims.set_output_feature_dimension(decode());
    for (int ii = 0; ii < num_spatial_dims; ii++) {
      conv_dims.mutable_output_spatial_dimensions()->Add(decode());
    }

    // Unpack various arrays.
    absl::Span<const int64_t> window_strides = decodeSpan(len_strides);
    std::vector<std::pair<int64_t, int64_t>> padding(len_padding);
    for (int ii = 0; ii < len_padding; ii++) {
      padding[ii].first = decode();
      padding[ii].second = decode();
    }
    absl::Span<const int64_t> input_dilation = decodeSpan(len_input_dilation);
    absl::Span<const int64_t> filter_dilation = decodeSpan(len_filter_dilation);

    // Other undocumented parameters not used.
    const xla::PrecisionConfig *precision_config = nullptr;
    std::optional<xla::PrimitiveType> preferred_element_type;

    std::optional<std::vector<bool>> window_reversal = std::nullopt;
    op = ConvGeneralDilated(
        *inputs[0], *inputs[1], window_strides,
        /* absl::Span<const std::pair<int64_t, int64_t>> */ padding,
        input_dilation, filter_dilation, conv_dims, filter_group_count,
        batch_group_count, precision_config, preferred_element_type,
        window_reversal);
    break;
  }
  case ReverseOp: {
    op = Rev(*inputs[0], absl::Span<const int64_t>(serialized_op->integer_array,
                                                   serialized_op->integer_array_size));
    break;
  }
  case TransposeOp: {
    op = Transpose(*inputs[0],
                   absl::Span<const int64_t>(serialized_op->integer_array,
                                             serialized_op->integer_array_size));
    break;
  }
  case ReduceWindowOp: {
    // Decode parameters.
    int64_t rank = serialized_op->integer_array_size / 6;
    absl::Span<const int64_t> window_dimensions = decodeSpan(rank);
    absl::Span<const int64_t> window_strides = decodeSpan(rank);
    absl::Span<const int64_t> base_dilations = decodeSpan(rank);
    absl::Span<const int64_t> window_dilations = decodeSpan(rank);
    std::vector<std::pair<int64_t, int64_t>> paddings(rank);
    for (int ii = 0; ii < rank; ii++) {
      paddings[ii].first = decode();
      paddings[ii].second = decode();
    }
    op = xla::ReduceWindowWithGeneralPadding(
        *inputs[0], *inputs[1], *(serialized_op->computation), window_dimensions, window_strides,
        base_dilations, window_dilations, paddings);
    break;
  }
  case SelectAndScatterOp: {
    // All operands.
    auto &operand = *inputs[0];
    auto &source = *inputs[1];
    auto &init_value = *inputs[2];

    // Create select and scatter comps.
    const xla::XlaComputation &select_comp = *serialized_op->computation;
    const xla::XlaComputation &scatter_comp = *serialized_op->second_computation;

    // Decode parameters.
    int64_t rank = decode();
    int64_t len_paddings = decode();
    absl::Span<const int64_t> window_dimensions = decodeSpan(rank);
    absl::Span<const int64_t> window_strides = decodeSpan(rank);
    std::vector<std::pair<int64_t, int64_t>> paddings(len_paddings);
    for (int ii = 0; ii < len_paddings; ii++) {
      paddings[ii].first = decode();
      paddings[ii].second = decode();
    }

    op = SelectAndScatterWithGeneralPadding(
        operand, select_comp, window_dimensions, window_strides, paddings,
        source, init_value, scatter_comp);
    break;
  }
  case BatchNormInferenceOp: {
    auto &operand = *inputs[0];
    auto &scale = *inputs[1];
    auto &offset = *inputs[2];
    auto &mean = *inputs[3];
    auto &variance = *inputs[4];
    float epsilon = serialized_op->float_v;
    int64_t feature_index = serialized_op->integer;
    op = xla::BatchNormInference(operand, scale, offset, mean, variance,
                                 epsilon, feature_index);
    break;
  }
  case BatchNormTrainingOp: {
    auto &operand = *inputs[0];
    auto &scale = *inputs[1];
    auto &offset = *inputs[2];
    float epsilon = serialized_op->float_v;
    int64_t feature_index = serialized_op->integer;
    op = xla::BatchNormTraining(operand, scale, offset, epsilon, feature_index);
    break;
  }
  case BatchNormGradOp: {
    auto &operand = *inputs[0];
    auto &scale = *inputs[1];
    auto &batch_mean = *inputs[2];
    auto &batch_var = *inputs[3];
    auto &grad_output = *inputs[4];
    float epsilon = serialized_op->float_v;
    int64_t feature_index = serialized_op->integer;
    op = xla::BatchNormGrad(operand, scale, batch_mean, batch_var, grad_output,
                            epsilon, feature_index);
    break;
  }
  case DotGeneralOp: {
    auto &lhs = *inputs[0]; // left-hand-side.
    auto &rhs = *inputs[1];
    xla::DotDimensionNumbers dims;
    std::vector<google::protobuf::RepeatedField<google::protobuf::int64> *>
        lists = {dims.mutable_lhs_contracting_dimensions(),
                 dims.mutable_lhs_batch_dimensions(),
                 dims.mutable_rhs_contracting_dimensions(),
                 dims.mutable_rhs_batch_dimensions()};
    std::vector<int> listsLens;
    for (int ii = 0; ii < lists.size(); ii++) {
      listsLens.push_back(decode());
    }
    for (int ii = 0; ii < lists.size(); ii++) {
      auto &list = lists[ii];
      int len = listsLens[ii];
      for (int elem = 0; elem < len; elem++) {
        list->Add(decode());
      }
    }

    const xla::PrecisionConfig *precision_config = nullptr;
    std::optional<xla::PrimitiveType> preferred_element_type;
    op = xla::DotGeneral(lhs, rhs, dims, precision_config,
                         preferred_element_type);
    break;
  }
  case RngBitGeneratorOp: {
      xla::RandomAlgorithm algo =
          static_cast<xla::RandomAlgorithm>(serialized_op->integer);
      op = xla::RngBitGenerator(algo, *inputs[0], shape);
      break;
  }
  case FftOp: {
    xla::FftType fft_type = static_cast<xla::FftType>(serialized_op->integer);
    op = xla::Fft(*inputs[0], fft_type, list_of_ints);
    break;
  }
  case WhileOp: {
    // Create select and scatter comps.
    const xla::XlaComputation &condition_comp = *serialized_op->computation;
    const xla::XlaComputation &body_comp = *serialized_op->second_computation;
    op = xla::While(condition_comp, body_comp, *inputs[0]);
    break;
  }
  case DynamicSliceOp: {
    auto &operand = *inputs[0];
    vector<XlaOp> start_indices(serialized_op->num_op_inputs-1);
    for (int ii = 0; ii < serialized_op->num_op_inputs-1; ii++) {
        start_indices[ii] = XlaOp(*inputs[ii+1]);
    }
    op = xla::DynamicSlice(operand, start_indices, list_of_ints);
    break;
  }
  case DynamicUpdateSliceOp: {
    auto &operand = *inputs[0];
    auto &update = *inputs[1];
    vector<XlaOp> start_indices(serialized_op->num_op_inputs-2);
    for (int ii = 0; ii < serialized_op->num_op_inputs-2; ii++) {
        start_indices[ii] = XlaOp(*inputs[ii+2]);
    }
    op = xla::DynamicUpdateSlice(operand, update, start_indices);
    break;
  }


  // One-argument ops:
  case AbsOp:
    op = xla::Abs(*inputs[0]);
    break;
  case NegOp:
    op = xla::Neg(*inputs[0]);
    break;
  case ExpOp:
    op = xla::Exp(*inputs[0]);
    break;
  case Expm1Op:
    op = xla::Expm1(*inputs[0]);
    break;
  case FloorOp:
    op = xla::Floor(*inputs[0]);
    break;
  case CeilOp:
    op = xla::Ceil(*inputs[0]);
    break;
  case RoundOp:
    op = xla::Round(*inputs[0]);
    break;
  case LogOp:
    op = xla::Log(*inputs[0]);
    break;
  case Log1pOp:
    op = xla::Log1p(*inputs[0]);
    break;
  case LogicalNotOp:
    op = xla::Not(*inputs[0]);
    break;
  case BitwiseNotOp:
    op = ~(*inputs[0]);
    break;
  case LogisticOp:
    op = xla::Logistic(*inputs[0]);
    break;
  case SignOp:
    op = xla::Sign(*inputs[0]);
    break;
  case ClzOp:
    op = xla::Clz(*inputs[0]);
    break;
  case CosOp:
    op = xla::Cos(*inputs[0]);
    break;
  case SinOp:
    op = xla::Sin(*inputs[0]);
    break;
  case TanhOp:
    op = xla::Tanh(*inputs[0]);
    break;
  case SqrtOp:
    op = xla::Sqrt(*inputs[0]);
    break;
  case RsqrtOp:
    op = xla::Rsqrt(*inputs[0]);
    break;
  case ImagOp:
    op = xla::Imag(*inputs[0]);
    break;
  case RealOp:
    op = xla::Real(*inputs[0]);
    break;
  case ConjOp:
    op = xla::Conj(*inputs[0]);
    break;
  case ErfOp:
    op = xla::Erf(*inputs[0]);
    break;
  case IsFiniteOp:
    op = xla::IsFinite(*inputs[0]);
    break;
  case PopulationCountOp:
    op = xla::PopulationCount(*inputs[0]);
    break;

  // Two-arguments ops
  case AddOp:
    op = xla::Add(*inputs[0], *inputs[1]);
    break;
  case MulOp:
    op = xla::Mul(*inputs[0], *inputs[1]);
    break;
  case SubOp:
    op = xla::Sub(*inputs[0], *inputs[1]);
    break;
  case DivOp:
    op = xla::Div(*inputs[0], *inputs[1]);
    break;
  case RemOp:
    op = xla::Rem(*inputs[0], *inputs[1]);
    break;
  case LogicalAndOp:
    op = xla::And(*inputs[0], *inputs[1]);
    break;
  case LogicalOrOp:
    op = xla::Or(*inputs[0], *inputs[1]);
    break;
  case LogicalXorOp:
    op = xla::Xor(*inputs[0], *inputs[1]);
    break;
  case BitwiseAndOp:
    op = (*inputs[0]) & (*inputs[1]);
    break;
  case BitwiseOrOp:
    op = (*inputs[0]) | (*inputs[1]);
    break;
  case BitwiseXorOp:
    op = (*inputs[0]) ^ (*inputs[1]);
    break;
  case DotOp:
    op = xla::Dot(*inputs[0], *inputs[1]);
    break;
  case MinOp:
    op = xla::Min(*inputs[0], *inputs[1]);
    break;
  case MaxOp:
    op = xla::Max(*inputs[0], *inputs[1]);
    break;
  case PowOp:
    op = xla::Pow(*inputs[0], *inputs[1]);
    break;
  case ComplexOp:
    op = xla::Complex(*inputs[0], *inputs[1]);
    break;
  case ShiftLeftOp:
    op = xla::ShiftLeft(*inputs[0], *inputs[1]);
    break;
  case ShiftRightArithmeticOp:
    op = xla::ShiftRightArithmetic(*inputs[0], *inputs[1]);
    break;
  case ShiftRightLogicalOp:
    op = xla::ShiftRightLogical(*inputs[0], *inputs[1]);
    break;

  // Logical operations.
  case EqualOp:
    op = xla::Eq(*inputs[0], *inputs[1]);
    break;
  case NotEqualOp:
    op = xla::Ne(*inputs[0], *inputs[1]);
    break;
  case GreaterOrEqualOp:
    op = xla::Ge(*inputs[0], *inputs[1]);
    break;
  case GreaterThanOp:
    op = xla::Gt(*inputs[0], *inputs[1]);
    break;
  case LessOrEqualOp:
    op = xla::Le(*inputs[0], *inputs[1]);
    break;
  case LessThanOp:
    op = xla::Lt(*inputs[0], *inputs[1]);
    break;
  case EqualTotalOrderOp:
    op = xla::EqTotalOrder(*inputs[0], *inputs[1]);
    break;
  case NotEqualTotalOrderOp:
    op = xla::NeTotalOrder(*inputs[0], *inputs[1]);
    break;
  case GreaterOrEqualTotalOrderOp:
    op = xla::GeTotalOrder(*inputs[0], *inputs[1]);
    break;
  case GreaterThanTotalOrderOp:
    op = xla::GtTotalOrder(*inputs[0], *inputs[1]);
    break;
  case LessOrEqualTotalOrderOp:
    op = xla::LeTotalOrder(*inputs[0], *inputs[1]);
    break;
  case LessThanTotalOrderOp:
    op = xla::LtTotalOrder(*inputs[0], *inputs[1]);
    break;

  default:
    return new absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("unknown op_type=%d for XlaBuilderAddOp",
                        serialized_op->op_type));
  }
  if (!op.valid()) {
    auto status = builder->first_error();
    if (!status.ok()) {
      return new absl::Status(status);
    }
    return new absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("failed to convert serialized_op to XLA: op_type=%d",
                        serialized_op->op_type));
  }
  serialized_op->new_op = new xla::XlaOp(op);

  // Also retrieve of the shape of the resulting op.
  auto shape_or = builder->GetShapePtr(*serialized_op->new_op);
  if (!shape_or.ok()) {
    return FromStatus(shape_or.status());
  }
  serialized_op->new_shape = ShapeFromXlaShape(*shape_or.value());
  return nullptr;
}

StatusOr XlaBuilderBuildComp(XlaBuilder *builder, XlaOp *output_op) {
  StatusOr r{0, 0};

  // Build XlaComputation.
  auto comp_or = builder->Build(*output_op);
  if (!comp_or.ok()) {
    r.status = FromStatus(comp_or.status());
    return r;
  }
  auto xla_comp = new xla::XlaComputation(std::move(comp_or.value()));
  r.value = xla_comp;
  return r;
}

char* XlaComputationName(XlaComputation *xla_comp) {
    return c_str(xla_comp->name());
}

void XlaComputationDestroy(XlaComputation *xla_comp) {
    delete xla_comp;
}

VectorData* XlaComputationSerializedHLO(XlaComputation *xla_comp) {
  std::string module_str = xla_comp->proto().SerializeAsString();
  return str_to_bytes(module_str);
}

char* XlaComputationTextHLO(XlaComputation *xla_comp) {
  std::string text;
  tsl::protobuf::TextFormat::PrintToString(xla_comp->proto(), &text);
  return c_str(text);
}