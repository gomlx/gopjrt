package stablehlo

import (
	"github.com/gomlx/gopjrt/stablehlo/optypes"
	"github.com/gomlx/gopjrt/stablehlo/shapes"
)

// Op holds an operation definition with all its inputs, and the result name and shape.
type Op struct {
	opType optypes.OpType
	inputs []*Op
	shape  shapes.Shape
}
