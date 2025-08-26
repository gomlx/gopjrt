package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/stablehlo/optypes"
	"github.com/gomlx/gopjrt/stablehlo/shapeinference"
	"github.com/gomlx/gopjrt/stablehlo/shapes"
	"github.com/pkg/errors"
)

// Function represents a `func.func` in ToStableHLO.
type Function struct {
	// Name of the function. It should not include the "@" prefix.
	Name string

	// IsPublic marks the function as public, which is rendered as `func.func public @...`
	IsPublic bool

	// Inputs to the function.
	Inputs []*Value

	// Outputs types of the function.
	Outputs []shapes.Shape

	// Statements in the function body.
	Statements []*Statement

	// values holds all the values (e.g. %0, %1, %arg0) created in the function's scope.
	values []*Value
}

// NewConstant creates a new constant statement and returns the resulting value.
func (f *Function) NewConstant(value any) (*Value, error) {
	// The shape of the constant is inferred from the value.
	dtype := dtypes.FromAny(value)
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant value type %T", value)
	}
	shape := shapes.Make(dtype)
	c := &Statement{
		OpType: optypes.Constant,
		Attributes: map[string]any{
			"value": value,
		},
		Outputs: f.newValue(shape),
	}
	f.Statements = append(f.Statements, c)
	return c.Outputs, nil
}

// AddOp adds a new operation to the function.
func (f *Function) AddOp(opType optypes.OpType, inputs ...*Value) (*Value, error) {
	inputShapes := make([]shapes.Shape, len(inputs))
	for i, input := range inputs {
		inputShapes[i] = input.shape
	}

	outputShape, err := inferShape(opType, inputShapes...)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to infer shape for op %s", opType)
	}

	stmt := &Statement{
		OpType:  opType,
		Inputs:  inputs,
		Outputs: f.newValue(outputShape),
	}
	f.Statements = append(f.Statements, stmt)
	return stmt.Outputs, nil
}

// newValue creates a new unique value within the function's scope.
func (f *Function) newValue(shape shapes.Shape) *Value {
	v := &Value{
		id:    len(f.values),
		shape: shape,
	}
	f.values = append(f.values, v)
	return v
}

// inferShape dispatches to the correct shape inference function based on the opType.
func inferShape(opType optypes.OpType, inputs ...shapes.Shape) (shapes.Shape, error) {
	if shapeinference.StandardUnaryOperations.Has(opType) {
		if len(inputs) != 1 {
			return shapes.Invalid(), errors.Errorf("unary op %s must have 1 input, got %d", opType, len(inputs))
		}
		return shapeinference.UnaryOp(opType, inputs[0])
	}
	if shapeinference.StandardBinaryOperations.Has(opType) {
		if len(inputs) != 2 {
			return shapes.Invalid(), errors.Errorf("binary op %s must have 2 inputs, got %d", opType, len(inputs))
		}
		return shapeinference.BinaryOp(opType, inputs[0], inputs[1])
	}
	if shapeinference.ComparisonOperations.Has(opType) {
		if len(inputs) != 2 {
			return shapes.Invalid(), errors.Errorf("comparison op %s must have 2 inputs, got %d", opType, len(inputs))
		}
		return shapeinference.ComparisonOp(opType, inputs[0], inputs[1])
	}
	return shapes.Invalid(), errors.Errorf("shape inference for op %s not implemented", opType)
}

// Return adds a return statement to the function with the given return values.
func (f *Function) Return(values ...*Value) {
	outputShapes := make([]shapes.Shape, len(values))
	for i, value := range values {
		outputShapes[i] = value.shape
	}
	f.Outputs = outputShapes

	stmt := &Statement{
		OpType: optypes.FuncReturn,
		Inputs: values,
	}
	f.Statements = append(f.Statements, stmt)
}

func (f *Function) Write(w io.Writer) error {
	if _, err := io.WriteString(w, "func.func "); err != nil {
		return err
	}
	if f.IsPublic {
		if _, err := io.WriteString(w, "public "); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "@%s(", f.Name); err != nil {
		return err
	}
	for i, input := range f.Inputs {
		if i > 0 {
			if _, err := io.WriteString(w, ", "); err != nil {
				return err
			}
		}
		if err := input.Write(w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, ": %s", input.shape.ToStableHLO()); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, ") -> ("); err != nil {
		return err
	}
	for i, output := range f.Outputs {
		if i > 0 {
			if _, err := io.WriteString(w, ", "); err != nil {
				return err
			}
		}
		if _, err := io.WriteString(w, output.ToStableHLO()); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, ") {\n"); err != nil {
		return err
	}

	for _, stmt := range f.Statements {
		if err := stmt.Write(w); err != nil {
			return err
		}
		if _, err := io.WriteString(w, "\n"); err != nil {
			return err
		}
	}

	if _, err := io.WriteString(w, "}"); err != nil {
		return err
	}
	return nil
}

// scalarShapeForValue is a local helper to get the shape for a scalar value.
func scalarShapeForValue(value any) (shapes.Shape, error) {
	var dtype dtypes.DType
	switch value.(type) {
	case bool:
		dtype = dtypes.Bool
	case int:
		dtype = dtypes.Int64 // Assume int is 64-bit.
	case int8:
		dtype = dtypes.S8
	case int16:
		dtype = dtypes.S16
	case int32:
		dtype = dtypes.S32
	case int64:
		dtype = dtypes.S64
	case uint8:
		dtype = dtypes.U8
	case uint16:
		dtype = dtypes.U16
	case uint32:
		dtype = dtypes.U32
	case uint64:
		dtype = dtypes.U64
	case float32:
		dtype = dtypes.F32
	case float64:
		dtype = dtypes.F64
	default:
		return shapes.Shape{}, errors.Errorf("unsupported scalar value type %T", value)
	}
	return shapes.Make(dtype), nil
}
