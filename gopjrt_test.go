package gopjrt

import (
	"flag"
	"fmt"
	"github.com/stretchr/testify/require"
	"gopjrt/dtypes"
	"gopjrt/pjrt"
	"gopjrt/xlabuilder"
	"testing"
)

var flagPluginName = flag.String("plugin", "cpu", "PRJT plugin name or full path")

func TestEndToEnd(t *testing.T) {
	// f(x) = x^2
	builder := xlabuilder.New("x*x")
	x, err := xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err, "Failed to create Parameter")
	fX, err := xlabuilder.Mul(x, x)
	require.NoError(t, err, "Failed operation Mul")

	// Get computation created.
	comp, err := builder.Build(fX)
	require.NoError(t, err, "Failed to build XlaComputation from ops.")
	fmt.Printf("HloModule proto:\n%s\n\n", comp.TextHLO())

	// Extract HLO buffer.
	hlo := comp.SerializedHLO()
	_ = hlo

	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed GetPlugin")
	fmt.Printf("Loaded %s\n", plugin)

	// Create a client.
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	devices, err := client.AddressableDevices()
	require.NoErrorf(t, err, "Failed to fetch AddressableDevices() from client on %s", plugin)
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	// Destroy the client and leave.
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
