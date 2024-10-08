/***** File generated by ./cmd/xlabuilder_codegen, based on op_types.txt. Don't edit it directly. *****/

package xlabuilder

import (
	"github.com/pkg/errors"
)

// Abs returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Abs(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(AbsOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Neg returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Neg(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(NegOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Exp returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Exp(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(ExpOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Expm1 returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Expm1(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(Expm1Op, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Floor returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Floor(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(FloorOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Ceil returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Ceil(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(CeilOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Round returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Round(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(RoundOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Log returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Log(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(LogOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Log1p returns the expression log(x+1).
// The op is created on the same XlaBuilder as used for x.
func Log1p(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(Log1pOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// LogicalNot returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func LogicalNot(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(LogicalNotOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Logistic returns the element-wise expression 1/(1+exp(-x)). Also known as the Sigmoid function.
// The op is created on the same XlaBuilder as used for x.
func Logistic(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(LogisticOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Sign returns element-wise +1, +/-0 or -1 depending on the sign of x. It returns NaN if the input is NaN.
// The op is created on the same XlaBuilder as used for x.
func Sign(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(SignOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Clz returns element-wise the "count leading zeros" bits of input node x -- for integer values.
// The op is created on the same XlaBuilder as used for x.
func Clz(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(ClzOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Cos returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Cos(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(CosOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Sin returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Sin(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(SinOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Tanh returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Tanh(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(TanhOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Sqrt returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x.
func Sqrt(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(SqrtOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Rsqrt returns the element-wise reciprocal of square root operation 1/sqrt(x).
// The op is created on the same XlaBuilder as used for x.
func Rsqrt(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(RsqrtOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Imag returns the imaginary part of a complex number. It returns 0 if the x is a float number.
// The op is created on the same XlaBuilder as used for x.
func Imag(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(ImagOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Real return the real part of a complex number. It returns x if the x is a float number.
// The op is created on the same XlaBuilder as used for x.
func Real(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(RealOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Conj returns the conjugate of a complex number. E.g: Conj(1+3i) = 1-3i
// The op is created on the same XlaBuilder as used for x.
func Conj(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(ConjOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Add returns the element-wise sum of the two values.
// Standard broadcasting rules apply (see documentation).
// The op is created on the same XlaBuilder as used for x0 and x1.
func Add(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Add(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(AddOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Mul returns the element-wise multiplication of the two values.
// Standard broadcasting rules apply (see documentation).
// The op is created on the same XlaBuilder as used for x0 and x1.
func Mul(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Mul(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(MulOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Sub returns the element-wise subtraction of the two values.
// Standard broadcasting rules apply (see documentation).
// The op is created on the same XlaBuilder as used for x0 and x1.
func Sub(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Sub(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(SubOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Div returns the element-wise subtraction of the two values.
// Standard broadcasting rules apply (see documentation).
// The op is created on the same XlaBuilder as used for x0 and x1.
func Div(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Div(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(DivOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Rem returns the remainder operation, also known as modulo (or Mod for short).
// Notice despite the name XLA implements Mod not IEEE754 Remainder operation.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Rem(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Rem(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(RemOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// And returns the element-wise logic "and" operator.
// The op is created on the same XlaBuilder as used for x0 and x1.
func And(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of And(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(AndOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Or returns the element-wise logic "and" operator.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Or(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Or(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(OrOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Xor returns the element-wise logic "and" operator.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Xor(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Xor(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(XorOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Dot returns the "dot product" operation.
// The exact semantics of this operation depend on the ranks of the operands:
//
// | Input | Output | Semantics |
// | vector [n] dot vector [n] | scalar | vector dot product |
// | matrix [m x k] dot vector [k] | vector [m]	matrix-vector multiplication |
// | matrix [m x k] dot matrix [k x n] | matrix [m x n] | matrix-matrix multiplication |
//
// The operation performs sum of products over the second dimension of x0 (or the first if it has rank 1) and
// the first dimension of x1.
// These are the "contracted" dimensions.
// The contracted dimensions of x0 and x1 must be of the same size.
// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications or
// matrix/matrix multiplications.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Dot(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Dot(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(DotOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Min returns the element-wise smallest value among the two.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Min(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Min(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(MinOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Max returns the element-wise highest value among the two.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Max(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Max(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(MaxOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Pow returns the Op that represents the output of the corresponding operation.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Pow(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Pow(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(PowOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Complex returns the complex number taking x0 as the real part and x1 as the imaginary part.
// The real (x0) and imaginary (x1) must have the same dtype, and they must be either `dtypes.Float32` or
// `dtypes.Float64`.
// The output will be either `dtypes.Complex64` or `dtypes.Complex128`, depending on x0 and x1 dtypes.
// The shapes of `real` or `imaginary` must be the same, or one must be a scalar, in which case
// the value is broadcast to every other value.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Complex(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Complex(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(ComplexOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Equal performs element-wise equality check, returns boolean results with the same dimensions as input.
// The op is created on the same XlaBuilder as used for x0 and x1.
func Equal(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of Equal(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(EqualOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// NotEqual performs element-wise inequality check, returns boolean results with the same dimensions as input.
// The op is created on the same XlaBuilder as used for x0 and x1.
func NotEqual(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of NotEqual(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(NotEqualOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// GreaterOrEqual performs element-wise comparison, returns boolean results with the same dimensions as input.
// The op is created on the same XlaBuilder as used for x0 and x1.
func GreaterOrEqual(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of GreaterOrEqual(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(GreaterOrEqualOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// GreaterThan performs element-wise comparison, returns boolean results with the same dimensions as input.
// The op is created on the same XlaBuilder as used for x0 and x1.
func GreaterThan(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of GreaterThan(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(GreaterThanOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// LessOrEqual performs element-wise comparison, returns boolean results with the same dimensions as input.
// The op is created on the same XlaBuilder as used for x0 and x1.
func LessOrEqual(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of LessOrEqual(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(LessOrEqualOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// LessThan performs element-wise comparison, returns boolean results with the same dimensions as input.
// The op is created on the same XlaBuilder as used for x0 and x1.
func LessThan(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of LessThan(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(LessThanOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// EqualTotalOrder returns the element-wise operation.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
// The op is created on the same XlaBuilder as used for x0 and x1.
func EqualTotalOrder(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of EqualTotalOrder(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(EqualTotalOrderOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// NotEqualTotalOrder returns the element-wise operation.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
// The op is created on the same XlaBuilder as used for x0 and x1.
func NotEqualTotalOrder(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of NotEqualTotalOrder(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(NotEqualTotalOrderOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// GreaterOrEqualTotalOrder returns the element-wise operation.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
// The op is created on the same XlaBuilder as used for x0 and x1.
func GreaterOrEqualTotalOrder(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of GreaterOrEqualTotalOrder(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(GreaterOrEqualTotalOrderOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// GreaterThanTotalOrder returns the element-wise operation.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
// The op is created on the same XlaBuilder as used for x0 and x1.
func GreaterThanTotalOrder(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of GreaterThanTotalOrder(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(GreaterThanTotalOrderOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// LessOrEqualTotalOrder returns the element-wise operation.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
// The op is created on the same XlaBuilder as used for x0 and x1.
func LessOrEqualTotalOrder(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of LessOrEqualTotalOrder(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(LessOrEqualTotalOrderOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// LessThanTotalOrder returns the element-wise operation.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
// The op is created on the same XlaBuilder as used for x0 and x1.
func LessThanTotalOrder(x0, x1 *Op) (*Op, error) {
	if x0.builder != x1.builder {
		return nil, errors.New("arguments of LessThanTotalOrder(x0, x1) come from different XlaBuilder objects (or nil)")
	}
	if x0.Shape.DType != x1.Shape.DType {
		return nil, errors.Errorf("dtype of first (%s) and second (%s) operands don't match", x0.Shape.DType, x1.Shape.DType)
	}
	builder := x0.builder
	y := newOp(LessThanTotalOrderOp, x0, x1)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// Erf returns the "error function", defined as erf(x) = 2/Pi * \int_{0}^{x}{e^{-t^2}dt}.
// The op is created on the same XlaBuilder as used for x.
func Erf(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(ErfOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// IsFinite tests whether each element of operand is finite, i.e., is not positive or negative infinity, and is not NaN.
// It returns an array of boolean values with the same shape as the input, where each element is true if and only if
// the corresponding input element is finite.
// The op is created on the same XlaBuilder as used for x.
func IsFinite(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(IsFiniteOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}

// PopulationCount computes the number of bits set in each element of operand.
// The op is created on the same XlaBuilder as used for x.
func PopulationCount(x *Op) (*Op, error) {
	builder := x.builder
	y := newOp(PopulationCountOp, x)
	err := builder.addOp(y)
	if err != nil {
		return nil, err
	}
	return y, nil
}
