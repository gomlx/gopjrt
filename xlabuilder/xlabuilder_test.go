package xlabuilder_test

import (
	"flag"
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"os"
	"testing"
)

var flagStableHLOOutput = flag.String("hlo", "",
	"Name of the output file that will hold generate hlo content. "+
		"Some test functions (including TestXlaBuilder) will generate StableHLO that can be saved to this file "+
		"and manually checked for their values.")

var flagPluginName = flag.String("plugin", "cpu", "PRJT plugin name or full path to use for XlaBuilder tests that evaluate the program")
var flagUseStableHLO = flag.Bool("stable_hlo", false, "Convert HLO to StableHLO before executing")

type errTester[T any] struct {
	value T
	err   error
}

func init() {
	klog.InitFlags(nil)
}

// capture is a shortcut to test that there is no error and return the value.
func capture[T any](value T, err error) errTester[T] {
	return errTester[T]{value, err}
}

func (e errTester[T]) Test(t *testing.T) T {
	require.NoError(t, e.err)
	return e.value
}

// getPJRTClient loads a PJRT plugin and create a client to run tests on.
// It exits the test if anything goes wrong.
func getPJRTClient(t *testing.T) *pjrt.Client {
	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *flagPluginName)
	plugin.UseStableHLO = *flagUseStableHLO && HasStableHLO()
	if *flagUseStableHLO && !HasStableHLO() {
		klog.Warning("StableHLO disabled because it's not available in build")
	}
	fmt.Printf("Loaded PJRT plugin %s\n", plugin)
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	attributes := plugin.Attributes()
	fmt.Printf("Loaded PJRT plugin %s with %d atributes:\n", plugin, len(attributes))
	for key, value := range attributes {
		fmt.Printf("\t%s: %+v\n", key, value)
	}
	return client
}

// compile compiles the program and returns the executable that can be used for testing.
// It exits the test if anything goes wrong.
func compile(t *testing.T, client *pjrt.Client, comp *XlaComputation) (exec *pjrt.LoadedExecutable) {
	var err error
	exec, err = client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile program")
	return
}

// execWithScalars executes the program on the input value given, and return the output.
// Both input and output expected to be a scalar.
// Any errors fail the test.
func execWithScalars[T dtypes.Supported](t *testing.T, client *pjrt.Client, exec *pjrt.LoadedExecutable, inputs ...T) T {
	inputBuffers := make([]*pjrt.Buffer, len(inputs))
	defer func() {
		for _, buf := range inputBuffers {
			if buf != nil {
				assert.NoError(t, buf.Destroy())
			}
		}
	}()
	var err error
	for ii, input := range inputs {
		inputBuffers[ii], err = pjrt.ScalarToBuffer(client, input)
		require.NoErrorf(t, err, "Failed to create on-device buffer for input %v", input)
	}

	outputBuffers, err := exec.Execute(inputBuffers...).Done()
	require.NoErrorf(t, err, "Failed to execute on inputs %v", inputs)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	output, err := pjrt.BufferToScalar[T](outputBuffers[0])
	fmt.Printf("  > f(%v)=%v\n", inputs, output)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution on inputs %v", exec.Name, inputs)
	return output
}

func execWithSlices[T dtypes.Supported](t *testing.T, client *pjrt.Client, exec *pjrt.LoadedExecutable, input []T) (flat []T, dims []int) {
	inputBuffer, err := pjrt.ArrayToBuffer(client, input, len(input))
	require.NoErrorf(t, err, "Failed to create on-device buffer for input %v", input)
	defer func() { require.NoError(t, inputBuffer.Destroy()) }()

	outputBuffers, err := exec.Execute(inputBuffer).Done()
	require.NoErrorf(t, err, "Failed to execute on input %v", input)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	flat, dims, err = pjrt.BufferToArray[T](outputBuffers[0])
	fmt.Printf("  > f(%v)=(%T%v) %v\n", input, flat[0], dims, flat)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution on input %d", exec.Name, input)
	return
}

func execArrayOutput[T dtypes.Supported](t *testing.T, client *pjrt.Client, exec *pjrt.LoadedExecutable) (flat []T, dims []int) {
	outputBuffers := capture(exec.Execute().Done()).Test(t)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	var err error
	flat, dims, err = pjrt.BufferToArray[T](outputBuffers[0])
	var v T
	fmt.Printf("  > f()=(%T%v) %v\n", v, dims, flat)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution", exec.Name)
	return
}

// execScalarOutput executes the LoadedExecutable with no inputs, and a scalar output of the given type.
func execScalarOutput[T dtypes.Supported](t *testing.T, client *pjrt.Client, exec *pjrt.LoadedExecutable) (value T) {
	outputBuffers := capture(exec.Execute().Done()).Test(t)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	var err error
	value, err = pjrt.BufferToScalar[T](outputBuffers[0])
	require.NoErrorf(t, err, "Failed to transfer results of %q execution", exec.Name)
	fmt.Printf("  > f()=(%T) %v\n", value, value)
	return
}

// TestCVersion checks that the libgomlx_xlabuilder.so matches the expected version.
func TestCVersions(t *testing.T) {
	fmt.Printf("MatchingCVersion=%s, CVersion loaded=%s\n", MatchingCVersion, CVersion())
	require.Equal(t, MatchingCVersion, CVersion())
}

func TestXlaBuilder(t *testing.T) {
	// f(x) = x^2
	builder := New("x*x")
	fmt.Printf("XlaBuilder %q:\n", builder.Name())
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

func TestStableHLO(t *testing.T) {
	if !HasStableHLO() {
		fmt.Println("Skipping TestStableHLO: StableHLO not included in build.")
		return
	}

	// f(x) = x^2
	builder := New("x*x")
	fmt.Printf("XlaBuilder %q:\n", builder.Name())
	x, err := Parameter(builder, "x", 0, MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err)
	fX, err := Mul(x, x)
	require.NoError(t, err)
	comp, err := builder.Build(fX)
	require.NoError(t, err)

	// Convert to StableHLO (text):
	stableHLO, err := comp.TextStableHLO()
	require.NoError(t, err)
	fmt.Printf("StableHLO code:\n=====================================\n%s=====================================\n", stableHLO)
	want := `{
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}`
	require.Contains(t, stableHLO, want)

	// Convert to StableHLO (binary):
	data, err := comp.SerializedStableHLO()
	require.NoError(t, err)
	data.Free()
}

func TestMismatches(t *testing.T) {
	// Dtype mismatches:
	builder := New(t.Name() + ":1")
	lhs := capture(Parameter(builder, "lhs", 0, MakeShape(dtypes.Float32))).Test(t)
	rhs := capture(Parameter(builder, "rhs", 1, MakeShape(dtypes.Int32))).Test(t)
	_, err := Add(lhs, rhs)
	fmt.Printf("Expected error when comparing Float32 > Int32: %v\n", err)
	require.Error(t, err)

	builder2 := New(t.Name() + ":2")
	rhs2 := capture(Parameter(builder2, "rhs2", 1, MakeShape(dtypes.Float32))).Test(t)
	_, err = LessThan(lhs, rhs2)
	fmt.Printf("Expected error when comparing ops from different builders: %v\n", err)
	require.Error(t, err)
}

// TestHasStableHLO just checks that the CGO call doesn't crash or anything.
// Also handy for debugging in some platform, to check if StableHLO converter was linked.
func TestHasStableHLO(t *testing.T) {
	fmt.Printf("HasStableHLO=%v\n", HasStableHLO())
}
