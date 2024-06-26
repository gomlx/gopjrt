// Code generated by "stringer -type=OpType gen_op_types.go"; DO NOT EDIT.

package xlabuilder

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[InvalidOp-0]
	_ = x[ParameterOp-1]
	_ = x[IotaOp-2]
	_ = x[ConstantOp-3]
	_ = x[IdentityOp-4]
	_ = x[ConvertDTypeOp-5]
	_ = x[WhereOp-6]
	_ = x[TupleOp-7]
	_ = x[GetTupleElementOp-8]
	_ = x[ReshapeOp-9]
	_ = x[BroadcastOp-10]
	_ = x[BroadcastInDimOp-11]
	_ = x[TransposeOp-12]
	_ = x[CallOp-13]
	_ = x[ReduceOp-14]
	_ = x[ReduceWindowOp-15]
	_ = x[ConcatenateOp-16]
	_ = x[SliceOp-17]
	_ = x[ArgMinMaxOp-18]
	_ = x[PadOp-19]
	_ = x[GatherOp-20]
	_ = x[ScatterOp-21]
	_ = x[SelectAndScatterOp-22]
	_ = x[ConvGeneralDilatedOp-23]
	_ = x[ReverseOp-24]
	_ = x[DotGeneralOp-25]
	_ = x[FftOp-26]
	_ = x[BatchNormTrainingOp-27]
	_ = x[BatchNormInferenceOp-28]
	_ = x[BatchNormGradOp-29]
	_ = x[AbsOp-30]
	_ = x[NegOp-31]
	_ = x[ExpOp-32]
	_ = x[Expm1Op-33]
	_ = x[FloorOp-34]
	_ = x[CeilOp-35]
	_ = x[RoundOp-36]
	_ = x[LogOp-37]
	_ = x[Log1pOp-38]
	_ = x[LogicalNotOp-39]
	_ = x[LogisticOp-40]
	_ = x[SignOp-41]
	_ = x[ClzOp-42]
	_ = x[CosOp-43]
	_ = x[SinOp-44]
	_ = x[TanhOp-45]
	_ = x[SqrtOp-46]
	_ = x[RsqrtOp-47]
	_ = x[ImagOp-48]
	_ = x[RealOp-49]
	_ = x[ConjOp-50]
	_ = x[AddOp-51]
	_ = x[MulOp-52]
	_ = x[SubOp-53]
	_ = x[DivOp-54]
	_ = x[RemOp-55]
	_ = x[AndOp-56]
	_ = x[OrOp-57]
	_ = x[XorOp-58]
	_ = x[DotOp-59]
	_ = x[MinOp-60]
	_ = x[MaxOp-61]
	_ = x[PowOp-62]
	_ = x[ComplexOp-63]
	_ = x[EqualOp-64]
	_ = x[NotEqualOp-65]
	_ = x[GreaterOrEqualOp-66]
	_ = x[GreaterThanOp-67]
	_ = x[LessOrEqualOp-68]
	_ = x[LessThanOp-69]
	_ = x[EqualTotalOrderOp-70]
	_ = x[NotEqualTotalOrderOp-71]
	_ = x[GreaterOrEqualTotalOrderOp-72]
	_ = x[GreaterThanTotalOrderOp-73]
	_ = x[LessOrEqualTotalOrderOp-74]
	_ = x[LessThanTotalOrderOp-75]
	_ = x[RngBitGeneratorOp-76]
	_ = x[RngNormalOp-77]
	_ = x[RngUniformOp-78]
}

const _OpType_name = "InvalidOpParameterOpIotaOpConstantOpIdentityOpConvertDTypeOpWhereOpTupleOpGetTupleElementOpReshapeOpBroadcastOpBroadcastInDimOpTransposeOpCallOpReduceOpReduceWindowOpConcatenateOpSliceOpArgMinMaxOpPadOpGatherOpScatterOpSelectAndScatterOpConvGeneralDilatedOpReverseOpDotGeneralOpFftOpBatchNormTrainingOpBatchNormInferenceOpBatchNormGradOpAbsOpNegOpExpOpExpm1OpFloorOpCeilOpRoundOpLogOpLog1pOpLogicalNotOpLogisticOpSignOpClzOpCosOpSinOpTanhOpSqrtOpRsqrtOpImagOpRealOpConjOpAddOpMulOpSubOpDivOpRemOpAndOpOrOpXorOpDotOpMinOpMaxOpPowOpComplexOpEqualOpNotEqualOpGreaterOrEqualOpGreaterThanOpLessOrEqualOpLessThanOpEqualTotalOrderOpNotEqualTotalOrderOpGreaterOrEqualTotalOrderOpGreaterThanTotalOrderOpLessOrEqualTotalOrderOpLessThanTotalOrderOpRngBitGeneratorOpRngNormalOpRngUniformOp"

var _OpType_index = [...]uint16{0, 9, 20, 26, 36, 46, 60, 67, 74, 91, 100, 111, 127, 138, 144, 152, 166, 179, 186, 197, 202, 210, 219, 237, 257, 266, 278, 283, 302, 322, 337, 342, 347, 352, 359, 366, 372, 379, 384, 391, 403, 413, 419, 424, 429, 434, 440, 446, 453, 459, 465, 471, 476, 481, 486, 491, 496, 501, 505, 510, 515, 520, 525, 530, 539, 546, 556, 572, 585, 598, 608, 625, 645, 671, 694, 717, 737, 754, 765, 777}

func (i OpType) String() string {
	if i < 0 || i >= OpType(len(_OpType_index)-1) {
		return "OpType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _OpType_name[_OpType_index[i]:_OpType_index[i+1]]
}
