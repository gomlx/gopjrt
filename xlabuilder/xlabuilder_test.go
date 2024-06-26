package xlabuilder_test

import (
	"flag"
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

var flagStableHLOOutput = flag.String("hlo", "",
	"Name of the output file that will hold generate hlo content. "+
		"Some test functions (including TestXlaBuilder) will generate StableHLO that can be saved to this file "+
		"and manually checked for their values.")

func TestXlaBuilder(t *testing.T) {
	// f(x) = x^2
	builder := New("x*x")
	x, err := Parameter(builder, "x", 0, MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err)
	fX, err := Mul(x, x)
	require.NoError(t, err)

	// Get computation created.
	comp, err := builder.Build(fX)
	require.NoError(t, err)
	fmt.Printf("HloModule proto:\n%s\n\n", comp.TextHLO())

	stableHLO := comp.SerializedHLO()
	defer stableHLO.Free()
	if *flagStableHLOOutput != "" {
		f, err := os.Create(*flagStableHLOOutput)
		require.NoErrorf(t, err, "Failed to open StableHLO proto output file %q", *flagStableHLOOutput)
		bufBytes := stableHLO.Bytes()
		n, err := f.Write(bufBytes)
		require.NoErrorf(t, err, "Failed to write StableHLO proto output file %q", *flagStableHLOOutput)
		require.Equal(t, len(bufBytes), n)
		require.NoError(t, f.Close(), "Failed to close StableHLO proto output file %q", *flagStableHLOOutput)
	}
}
