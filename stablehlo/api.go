package stablehlo

import (
	"fmt"
	"strings"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/stablehlo/optypes"
	"github.com/gomlx/gopjrt/stablehlo/shapeinference"
	"github.com/gomlx/gopjrt/stablehlo/shapes"
	"github.com/pkg/errors"
)

// Function represents a `func.func` in StableHLO.
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

// Value represents a value in a StableHLO program, like `%0` or `%arg0`.
// It has a name and a shape.
type Value struct {
	id    int
	shape shapes.Shape
}

// Statement represents a single operation line in StableHLO.
type Statement struct {
	// OpType is the type of the operation.
	OpType optypes.OpType

	// Inputs to the operation.
	Inputs []*Value

	// Attributes of the operation.
	Attributes map[string]any

	// Result of the operation.
	Result *Value
}

// NewConstant creates a new constant statement and returns the resulting value.
func (f *Function) NewConstant(value any) (*Value, error) {
	// The shape of the constant is inferred from the value.
	shape, err := scalarShapeForValue(value)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get shape for constant value")
	}
	c := &Statement{
		OpType: optypes.Constant,
		Attributes: map[string]any{
			"value": value,
		},
		Result: f.newValue(shape),
	}
	f.Statements = append(f.Statements, c)
	return c.Result, nil
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
		OpType: opType,
		Inputs: inputs,
		Result: f.newValue(outputShape),
	}
	f.Statements = append(f.Statements, stmt)
	return stmt.Result, nil
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

// String methods for generation of StableHLO text format.

func (v *Value) String() string {
	return fmt.Sprintf("%%%d", v.id)
}

func (s *Statement) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "  %s = \"stablehlo.%s\"", s.Result, strings.ToLower(s.OpType.String()))
	sb.WriteString("(")
	for i, input := range s.Inputs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(input.String())
	}
	sb.WriteString(")")

	// Attributes
	if len(s.Attributes) > 0 {
		sb.WriteString(" {")
		first := true
		for key, value := range s.Attributes {
			if !first {
				sb.WriteString(", ")
			}
			first = false
			fmt.Fprintf(&sb, "%s = %s", key, attributeToString(value))
		}
		sb.WriteString("}")
	}

	// Signature
	sb.WriteString(" : (")
	for i, input := range s.Inputs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(input.shape.ToStableHLO())
	}
	sb.WriteString(") -> ")
	sb.WriteString(s.Result.shape.ToStableHLO())

	return sb.String()
}

func (f *Function) String() string {
	var sb strings.Builder
	sb.WriteString("func.func ")
	if f.IsPublic {
		sb.WriteString("public ")
	}
	fmt.Fprintf(&sb, "@%s(", f.Name)
	for i, input := range f.Inputs {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(&sb, "%s: %s", input, input.shape.ToStableHLO())
	}
	sb.WriteString(") -> (")
	for i, output := range f.Outputs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(output.ToStableHLO())
	}
	sb.WriteString(") {\n")

	for _, stmt := range f.Statements {
		sb.WriteString(stmt.String())
		sb.WriteString("\n")
	}

	// Return statement
	sb.WriteString(`  "func.return"`)
	if len(f.Outputs) > 0 {
		// Assuming the last statement's result is the return value.
		// This is a simplification and will need to be improved.
		if len(f.Statements) > 0 {
			lastResult := f.Statements[len(f.Statements)-1].Result
			fmt.Fprintf(&sb, "(%s)", lastResult)
		}
	}
	sb.WriteString(" : (")
	for i, output := range f.Outputs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(output.ToStableHLO())
	}
	sb.WriteString(") -> ()\n")

	sb.WriteString("}")
	return sb.String()
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

// attributeToString converts an attribute value to its StableHLO string representation.
// This is a simplified version and will need to be extended.
func attributeToString(attr any) string {
	switch v := attr.(type) {
	case string:
		return fmt.Sprintf(`"%s"`, v)
	case float32, float64:
		shape, err := scalarShapeForValue(v)
		if err != nil {
			panic(err)
		}
		return fmt.Sprintf("dense<%e> : %s", v, shape.ToStableHLO())
	case int, int32, int64:
		shape, err := scalarShapeForValue(v)
		if err != nil {
			panic(err)
		}
		return fmt.Sprintf("dense<%d> : %s", v, shape.ToStableHLO())
	default:
		return fmt.Sprintf("%#v", v)
	}
}
