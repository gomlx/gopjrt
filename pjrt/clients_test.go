package pjrt

import (
	"fmt"
	pjrt_proto "github.com/gomlx/gopjrt/protos"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
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

func TestCompileAndExecute(t *testing.T) {
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
		require.NoErrorf(t, err, "Failed to compile %q", programTest.name)
		fmt.Printf("\t> name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

		for ii, input := range programTest.testInputs {
			buffer, err := client.BufferFromHost().FromRawData(ScalarToRaw(input)).Done()
			require.NoErrorf(t, err, "Failed to transfer scalar %v", input)
			want := programTest.wantOutputs[ii]
			outputs, err := loadedExec.Execute(buffer).Done()
			require.NoErrorf(t, err, "Failed to execute for %v", input)
			require.Len(t, outputs, len(want))
			if len(outputs) == 1 {
				got, err := BufferToScalar[float32](outputs[0])
				require.NoErrorf(t, err, "Failed to transfer output to host for %v", input)
				fmt.Printf("\t> input=%f, output=%f, want=%f\n", input, got, want[0])
			}
			require.NoErrorf(t, buffer.Destroy(), "Failed to destroy scalar buffer for %v", input)
		}

		// Destroy compiled executables.
		require.NoErrorf(t, loadedExec.Destroy(), "Failed to destroy LoadedExecutable on %s", plugin)
	}

	// Destroy client.
	require.NoErrorf(t, client.Destroy(), "Failed to destroy client on %s", plugin)
}
