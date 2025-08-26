package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/gopjrt/stablehlo/shapes"
)

// Value represents a value in a ToStableHLO program, like `%0` or `%arg0`.
// It has a name, shape and an optional descriptive name that can contain letters, digits and underscore.
type Value struct {
	id    int
	shape shapes.Shape
	name  string // Optional name composed of letters, digits and underscore
}

// Write writes the value in ToStableHLO text format to the given writer.
func (v *Value) Write(w io.Writer) error {
	if v.name != "" {
		_, err := fmt.Fprintf(w, "%%%s", v.name)
		return err
	}
	_, err := fmt.Fprintf(w, "%%%d", v.id)
	return err
}
