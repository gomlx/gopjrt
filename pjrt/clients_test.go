package pjrt

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	pjrt_proto "gopjrt/proto"
	"os"
	"testing"
)

type testFileInfo struct {
	name        string
	numOutputs  int
	testInputs  []float32
	wantOutputs [][]float32
}

var (
	testHLOPrograms = []testFileInfo{
		{
			name:        "test_hlo.pb",
			numOutputs:  1,
			testInputs:  []float32{1.0, 3.0},
			wantOutputs: [][]float32{{1.0}, {9.0}},
		},
		{
			name:        "test_tuple_hlo.pb",
			numOutputs:  2,
			testInputs:  []float32{1.0, 9.0},
			wantOutputs: [][]float32{{1.0, 1.0}, {81.0, 3.0}},
		},
	}
)

func TestPlugin_NewClient(t *testing.T) {
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices, err := client.AddressableDevices()
	require.NoErrorf(t, err, "Failed to fetch AddressableDevices() from client on %s", plugin)
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}

func TestClientCompileAndExecute(t *testing.T) {
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices, err := client.AddressableDevices()
	require.NoErrorf(t, err, "Failed to fetch AddressableDevices() from client on %s", plugin)
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	for _, programTest := range testHLOPrograms {
		fmt.Printf("Program: %s\n", programTest.name)

		// Load test program.
		hloBin, err := os.ReadFile(programTest.name)
		require.NoError(t, err)
		hloProto := &pjrt_proto.HloModuleProto{}
		require.NoError(t, proto.Unmarshal(hloBin, hloProto), "Unmarshalling HloModuleProto")
		//fmt.Printf("HloModuleProto: {\n%s}\n", prototext.Format(hloProto))

		// Compile program.
		loadedExec, err := client.Compile().WithHLO(hloBin).Done()
		require.NoErrorf(t, err, "Failed to compile %q", programTest)

		// Get executable description and check the number of outputs.
		exec, err := loadedExec.GetExecutable()
		require.NoError(t, err)
		numOutputs, err := exec.NumOutputs()
		require.NoError(t, err)
		require.Equal(t, programTest.numOutputs, numOutputs)
		require.NoErrorf(t, exec.Destroy(), "Failed to destroy Executable on %s", plugin)

		for ii, input := range programTest.testInputs {
			buffer, err := client.BufferFromHost().FromRawData(ScalarToRaw(input)).Done()
			require.NoErrorf(t, err, "Failed to transfer scalar %v", input)
			require.NoErrorf(t, buffer.Destroy(), "Failed to destroy scalar buffer for %v", input)
			want := programTest.wantOutputs[ii]
			_ = want
		}

		// Destroy compiled executables.
		require.NoErrorf(t, loadedExec.Destroy(), "Failed to destroy LoadedExecutable on %s", plugin)
	}

	// Destroy client.
	require.NoErrorf(t, client.Destroy(), "Failed to destroy client on %s", plugin)
}
