package stablehlo

// Builder is used to construct a StableHLO program.
// See New.
type Builder struct {
	name   string
	parent *Builder
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

// Build builds the Computation with the requested operations (the outputOp and all its dependencies)
// or returns a non-ok status.
//
// Note that all ops that have been enqueued will be moved to the computation being returned and will no
// longer be valid.
func (b *Builder) Build(outputs ...*Op) (*Computation, error) {
	return &Computation{Name: b.name}, nil
}
