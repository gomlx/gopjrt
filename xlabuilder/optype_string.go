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
	_ = x[ConvertTypeOp-5]
	_ = x[WhereOp-6]
	_ = x[TupleOp-7]
	_ = x[GetTupleElementOp-8]
	_ = x[ReshapeOp-9]
	_ = x[BroadcastOp-10]
	_ = x[BroadcastInDimOp-11]
	_ = x[ReduceSumOp-12]
	_ = x[ReduceMaxOp-13]
	_ = x[ReduceMultiplyOp-14]
	_ = x[SliceOp-15]
	_ = x[PadOp-16]
	_ = x[GatherOp-17]
	_ = x[ScatterOp-18]
	_ = x[ConcatenateOp-19]
	_ = x[ConvGeneralDilatedOp-20]
	_ = x[ReverseOp-21]
	_ = x[TransposeOp-22]
	_ = x[ReduceWindowOp-23]
	_ = x[SelectAndScatterOp-24]
	_ = x[BatchNormTrainingOp-25]
	_ = x[BatchNormInferenceOp-26]
	_ = x[BatchNormGradOp-27]
	_ = x[DotGeneralOp-28]
	_ = x[ArgMinMaxOp-29]
	_ = x[FftOp-30]
	_ = x[AbsOp-31]
	_ = x[NegOp-32]
	_ = x[ExpOp-33]
	_ = x[Expm1Op-34]
	_ = x[FloorOp-35]
	_ = x[CeilOp-36]
	_ = x[RoundOp-37]
	_ = x[LogOp-38]
	_ = x[Log1pOp-39]
	_ = x[LogicalNotOp-40]
	_ = x[LogisticOp-41]
	_ = x[SignOp-42]
	_ = x[ClzOp-43]
	_ = x[CosOp-44]
	_ = x[SinOp-45]
	_ = x[TanhOp-46]
	_ = x[SqrtOp-47]
	_ = x[RsqrtOp-48]
	_ = x[ImagOp-49]
	_ = x[RealOp-50]
	_ = x[ConjOp-51]
	_ = x[AddOp-52]
	_ = x[MulOp-53]
	_ = x[SubOp-54]
	_ = x[DivOp-55]
	_ = x[RemOp-56]
	_ = x[AndOp-57]
	_ = x[OrOp-58]
	_ = x[XorOp-59]
	_ = x[DotOp-60]
	_ = x[MinOp-61]
	_ = x[MaxOp-62]
	_ = x[PowOp-63]
	_ = x[ComplexOp-64]
	_ = x[EqualOp-65]
	_ = x[NotEqualOp-66]
	_ = x[GreaterOrEqualOp-67]
	_ = x[GreaterThanOp-68]
	_ = x[LessOrEqualOp-69]
	_ = x[LessThanOp-70]
	_ = x[EqualTotalOrderOp-71]
	_ = x[NotEqualTotalOrderOp-72]
	_ = x[GreaterOrEqualTotalOrderOp-73]
	_ = x[GreaterThanTotalOrderOp-74]
	_ = x[LessOrEqualTotalOrderOp-75]
	_ = x[LessThanTotalOrderOp-76]
	_ = x[RngBitGeneratorOp-77]
	_ = x[RngNormalOp-78]
	_ = x[RngUniformOp-79]
}

const _OpType_name = "InvalidOpParameterOpIotaOpConstantOpIdentityOpConvertTypeOpWhereOpTupleOpGetTupleElementOpReshapeOpBroadcastOpBroadcastInDimOpReduceSumOpReduceMaxOpReduceMultiplyOpSliceOpPadOpGatherOpScatterOpConcatenateOpConvGeneralDilatedOpReverseOpTransposeOpReduceWindowOpSelectAndScatterOpBatchNormTrainingOpBatchNormInferenceOpBatchNormGradOpDotGeneralOpArgMinMaxOpFftOpAbsOpNegOpExpOpExpm1OpFloorOpCeilOpRoundOpLogOpLog1pOpLogicalNotOpLogisticOpSignOpClzOpCosOpSinOpTanhOpSqrtOpRsqrtOpImagOpRealOpConjOpAddOpMulOpSubOpDivOpRemOpAndOpOrOpXorOpDotOpMinOpMaxOpPowOpComplexOpEqualOpNotEqualOpGreaterOrEqualOpGreaterThanOpLessOrEqualOpLessThanOpEqualTotalOrderOpNotEqualTotalOrderOpGreaterOrEqualTotalOrderOpGreaterThanTotalOrderOpLessOrEqualTotalOrderOpLessThanTotalOrderOpRngBitGeneratorOpRngNormalOpRngUniformOp"

var _OpType_index = [...]uint16{0, 9, 20, 26, 36, 46, 59, 66, 73, 90, 99, 110, 126, 137, 148, 164, 171, 176, 184, 193, 206, 226, 235, 246, 260, 278, 297, 317, 332, 344, 355, 360, 365, 370, 375, 382, 389, 395, 402, 407, 414, 426, 436, 442, 447, 452, 457, 463, 469, 476, 482, 488, 494, 499, 504, 509, 514, 519, 524, 528, 533, 538, 543, 548, 553, 562, 569, 579, 595, 608, 621, 631, 648, 668, 694, 717, 740, 760, 777, 788, 800}

func (i OpType) String() string {
	if i < 0 || i >= OpType(len(_OpType_index)-1) {
		return "OpType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _OpType_name[_OpType_index[i]:_OpType_index[i+1]]
}
