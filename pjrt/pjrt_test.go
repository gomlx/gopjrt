package pjrt

// Common initialization and testing tools for all test files.

import (
	"flag"
	"fmt"

	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var FlagPluginName = flag.String("plugin", "cpu", "plugin name")

func init() {
	klog.InitFlags(nil)
}

type errTester[T any] struct {
	value T
	err   error
}

// capture is a shortcut to test that there is no error and return the value.
func capture[T any](value T, err error) errTester[T] {
	return errTester[T]{value, err}
}

func (e errTester[T]) Test(t *testing.T) T {
	require.NoError(t, e.err)
	return e.value
}

func must(err error) {
	if err != nil {
		panicf("Failed: %+v", errors.WithStack(err))
	}
}

func must1[T any](t T, err error) T {
	must(err)
	return t
}

// getPJRTClient loads a PJRT plugin and create a client to run tests on.
// It exits the test if anything goes wrong.
func getPJRTClient(t *testing.T) *Client {
	// PJRT plugin and create a client.
	plugin, err := GetPlugin(*FlagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *FlagPluginName)
	attributes := plugin.Attributes()
	fmt.Printf("Loaded PJRT plugin %s with %d atributes:\n", plugin, len(attributes))
	for key, value := range attributes {
		fmt.Printf("\t%s: %+v\n", key, value)
	}
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	return client
}

// compile compiles the program and returns the executable that can be used for testing.
// It exits the test if anything goes wrong.
func compile(t *testing.T, client *Client, comp XlaComputation) (exec *LoadedExecutable) {
	var err error
	exec, err = client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile program")
	return
}

// execWithScalars executes the program on the input value given, and return the output.
// Both input and output expected to be a scalar.
// Any errors fail the test.
func execWithScalars[T dtypes.Supported](t *testing.T, client *Client, exec *LoadedExecutable, input T) T {
	inputBuffer, err := ScalarToBuffer(client, input)
	require.NoErrorf(t, err, "Failed to create on-device buffer for input %v", input)
	defer func() { require.NoError(t, inputBuffer.Destroy()) }()

	outputBuffers, err := exec.Execute(inputBuffer).Done()
	require.NoErrorf(t, err, "Failed to execute on input %v", input)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	output, err := BufferToScalar[T](outputBuffers[0])
	fmt.Printf("  > f(%v)=%v\n", input, output)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution on input %d", exec.Name, input)
	return output
}

func execWithSlices[T dtypes.Supported](t *testing.T, client *Client, exec *LoadedExecutable, input []T) (flat []T, dims []int) {
	inputBuffer, err := ArrayToBuffer(client, input, len(input))
	require.NoErrorf(t, err, "Failed to create on-device buffer for input %v", input)
	defer func() { require.NoError(t, inputBuffer.Destroy()) }()

	outputBuffers, err := exec.Execute(inputBuffer).Done()
	require.NoErrorf(t, err, "Failed to execute on input %v", input)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	flat, dims, err = BufferToArray[T](outputBuffers[0])
	fmt.Printf("  > f(%v)=(%T%v) %v\n", input, flat[0], dims, flat)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution on input %d", exec.Name, input)
	return
}

func execArrayOutput[T dtypes.Supported](t *testing.T, client *Client, exec *LoadedExecutable) (flat []T, dims []int) {
	outputBuffers := capture(exec.Execute().Done()).Test(t)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	var err error
	flat, dims, err = BufferToArray[T](outputBuffers[0])
	var v T
	fmt.Printf("  > f()=(%T%v) %v\n", v, dims, flat)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution", exec.Name)
	return
}

// execScalarOutput executes the LoadedExecutable with no inputs, and a scalar output of the given type.
func execScalarOutput[T dtypes.Supported](t *testing.T, client *Client, exec *LoadedExecutable) (value T) {
	outputBuffers := capture(exec.Execute().Done()).Test(t)
	require.Len(t, outputBuffers, 1, "Expected only one output")
	defer func() { require.NoError(t, outputBuffers[0].Destroy()) }()

	// Transfer output on-device buffer to a "host" value (in Go).
	var err error
	value, err = BufferToScalar[T](outputBuffers[0])
	fmt.Printf("  > f()=(%T) %v\n", value, value)
	require.NoErrorf(t, err, "Failed to transfer results of %q execution", exec.Name)
	return
}
