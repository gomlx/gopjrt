package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/stablehlo/optypes"
	"github.com/gomlx/gopjrt/stablehlo/shapes"
)

// Statement represents a single operation line in ToStableHLO.
type Statement struct {
	// OpType is the type of the operation.
	OpType optypes.OpType

	// Inputs to the operation.
	Inputs []*Value

	// Attributes of the operation.
	Attributes map[string]any

	// Outputs of the operation. It may be nil for operations like func.return.
	Outputs []*Value
}

// Write writes a string representation of the statement to the given writer.
func (s *Statement) Write(writer io.Writer) error {
	var err error
	w := func(format string, args ...any) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}
	we := func(e elementWriter) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer)
	}

	// Output values are written first:
	w("  ") // Indentation of functions.
	if len(s.Outputs) > 0 {
		for i, output := range s.Outputs {
			if i > 0 {
				w(", ")
			}
			we(output)
		}
		w(" = ")
	}

	// Write op name and arguments:
	w("%q(", s.OpType.ToStableHLO())
	for i, input := range s.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input)
	}
	w(")")

	// Write attributes:
	if len(s.Attributes) > 0 {
		w("{")
		first := true
		for key, value := range s.Attributes {
			if !first {
				w(", ")
			}
			first = false
			w("%s = %s", key, literalToStableHLO(value))
		}
		w("}")
	}

	// Write signature:
	w(" : (")
	for i, input := range s.Inputs {
		if i > 0 {
			w(", ")
		}
		w(input.shape.ToStableHLO())
	}
	w(")")
	if len(s.Outputs) > 0 {
		w(" -> (")
		for i, output := range s.Outputs {
			if i > 0 {
				w(", ")
			}
			w(output.shape.ToStableHLO())
		}
		w(")")
	}

	return err
}

// literalToStableHLO converts a literal value, usually used in attributes, to its ToStableHLO string representation.
func literalToStableHLO(attr any) string {
	switch v := attr.(type) {
	case string:
		return fmt.Sprintf("%q", v)
	case float32, float64:
		shape := shapes.Make(dtypes.FromAny(v))
		return fmt.Sprintf("dense<%e> : %s", v, shape.ToStableHLO())
	case int, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
		shape := shapes.Make(dtypes.FromAny(v))
		return fmt.Sprintf("dense<%d> : %s", v, shape.ToStableHLO())
	case bool:
		if v {
			return "true"
		}
		return "false"
	default:
		return fmt.Sprintf("Unknown literal type: %t %#v", v, v)
	}
}
