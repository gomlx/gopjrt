package stablehlo

// Computation holds a rendered computation graph, that can be fed to PJRT.
// It is created with Builder.Build.
type Computation struct {
	Name      string
	StableHLO string
}
