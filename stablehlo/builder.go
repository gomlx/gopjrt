package stablehlo

import (
	"strings"

	"github.com/gomlx/gopjrt/stablehlo/shapes"
	"github.com/pkg/errors"
)

// Builder is used to construct a StableHLO program.
// See New.
type Builder struct {
	name   string
	parent *Builder

	// functions holds all the functions created in the builder's scope.
	functions []*Function
}

// New creates a new Builder object holding a computation graph in construction.
//
// Add operations (ops) one by one, until you defined the desired graph, and then call the method Build,
// which returns a Computation. A pjrt.Client can use a Computation as input to JIT-compile and execute
// efficiently.
func New(name string) *Builder {
	return &Builder{
		name: name,
	}
}

// NewFunction creates a new function and adds it to the program.
func (b *Builder) NewFunction(name string, isPublic bool, inputs []*Value, outputs []shapes.Shape) *Function {
	fn := &Function{
		Name:    name,
		IsPublic: isPublic,
		Inputs:  inputs,
		Outputs: outputs,
	}
	b.functions = append(b.functions, fn)
	return fn
}

// Build builds the Computation with the requested operations (the outputOp and all its dependencies)
// or returns a non-ok status.
//
// Note that all ops that have been enqueued will be moved to the computation being returned and will no
// longer be valid.
func (b *Builder) Build() (*Computation, error) {
	var sb strings.Builder
	hasMain := false
	for i, fn := range b.functions {
		if fn.Name == "main" {
			hasMain = true
		}
		if i > 0 {
			sb.WriteString("\n\n")
		}
		// Set the outputs of the function to be the result of the last statement.
		// This is a simplification and will be improved later.
		if len(fn.Statements) > 0 {
			lastStmt := fn.Statements[len(fn.Statements)-1]
			fn.Outputs = []shapes.Shape{lastStmt.Result.shape}
		}
		sb.WriteString(fn.String())
	}

	if !hasMain {
		return nil, errors.New("program must have a main function")
	}

	return &Computation{
		Name:      b.name,
		StableHLO: sb.String(),
	}, nil
}
