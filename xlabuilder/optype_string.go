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
	_ = x[RngBitGeneratorOp-30]
	_ = x[WhileOp-31]
	_ = x[AbsOp-32]
	_ = x[NegOp-33]
	_ = x[ExpOp-34]
	_ = x[Expm1Op-35]
	_ = x[FloorOp-36]
	_ = x[CeilOp-37]
	_ = x[RoundOp-38]
	_ = x[LogOp-39]
	_ = x[Log1pOp-40]
	_ = x[LogicalNotOp-41]
	_ = x[LogisticOp-42]
	_ = x[SignOp-43]
	_ = x[ClzOp-44]
	_ = x[CosOp-45]
	_ = x[SinOp-46]
	_ = x[TanhOp-47]
	_ = x[SqrtOp-48]
	_ = x[RsqrtOp-49]
	_ = x[ImagOp-50]
	_ = x[RealOp-51]
	_ = x[ConjOp-52]
	_ = x[AddOp-53]
	_ = x[MulOp-54]
	_ = x[SubOp-55]
	_ = x[DivOp-56]
	_ = x[RemOp-57]
	_ = x[AndOp-58]
	_ = x[OrOp-59]
	_ = x[XorOp-60]
	_ = x[DotOp-61]
	_ = x[MinOp-62]
	_ = x[MaxOp-63]
	_ = x[PowOp-64]
	_ = x[ComplexOp-65]
	_ = x[EqualOp-66]
	_ = x[NotEqualOp-67]
	_ = x[GreaterOrEqualOp-68]
	_ = x[GreaterThanOp-69]
	_ = x[LessOrEqualOp-70]
	_ = x[LessThanOp-71]
	_ = x[EqualTotalOrderOp-72]
	_ = x[NotEqualTotalOrderOp-73]
	_ = x[GreaterOrEqualTotalOrderOp-74]
	_ = x[GreaterThanTotalOrderOp-75]
	_ = x[LessOrEqualTotalOrderOp-76]
	_ = x[LessThanTotalOrderOp-77]
	_ = x[DynamicSliceOp-78]
	_ = x[DynamicUpdateSliceOp-79]
	_ = x[ErfOp-80]
	_ = x[IsFiniteOp-81]
	_ = x[PopulationCountOp-82]
	_ = x[ShiftLeftOp-83]
	_ = x[ShiftRightArithmeticOp-84]
	_ = x[ShiftRightLogicalOp-85]
}

const _OpType_name = "InvalidOpParameterOpIotaOpConstantOpIdentityOpConvertDTypeOpWhereOpTupleOpGetTupleElementOpReshapeOpBroadcastOpBroadcastInDimOpTransposeOpCallOpReduceOpReduceWindowOpConcatenateOpSliceOpArgMinMaxOpPadOpGatherOpScatterOpSelectAndScatterOpConvGeneralDilatedOpReverseOpDotGeneralOpFftOpBatchNormTrainingOpBatchNormInferenceOpBatchNormGradOpRngBitGeneratorOpWhileOpAbsOpNegOpExpOpExpm1OpFloorOpCeilOpRoundOpLogOpLog1pOpLogicalNotOpLogisticOpSignOpClzOpCosOpSinOpTanhOpSqrtOpRsqrtOpImagOpRealOpConjOpAddOpMulOpSubOpDivOpRemOpAndOpOrOpXorOpDotOpMinOpMaxOpPowOpComplexOpEqualOpNotEqualOpGreaterOrEqualOpGreaterThanOpLessOrEqualOpLessThanOpEqualTotalOrderOpNotEqualTotalOrderOpGreaterOrEqualTotalOrderOpGreaterThanTotalOrderOpLessOrEqualTotalOrderOpLessThanTotalOrderOpDynamicSliceOpDynamicUpdateSliceOpErfOpIsFiniteOpPopulationCountOpShiftLeftOpShiftRightArithmeticOpShiftRightLogicalOp"

var _OpType_index = [...]uint16{0, 9, 20, 26, 36, 46, 60, 67, 74, 91, 100, 111, 127, 138, 144, 152, 166, 179, 186, 197, 202, 210, 219, 237, 257, 266, 278, 283, 302, 322, 337, 354, 361, 366, 371, 376, 383, 390, 396, 403, 408, 415, 427, 437, 443, 448, 453, 458, 464, 470, 477, 483, 489, 495, 500, 505, 510, 515, 520, 525, 529, 534, 539, 544, 549, 554, 563, 570, 580, 596, 609, 622, 632, 649, 669, 695, 718, 741, 761, 775, 795, 800, 810, 827, 838, 860, 879}

func (i OpType) String() string {
	if i < 0 || i >= OpType(len(_OpType_index)-1) {
		return "OpType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _OpType_name[_OpType_index[i]:_OpType_index[i+1]]
}
