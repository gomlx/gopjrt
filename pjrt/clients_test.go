package pjrt

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	pjrt_proto "gopjrt/proto"
	"os"
	"testing"
)

var (
	testHLOProgramFiles = []string{
		"test_hlo.pb",
		"test_tuple_hlo.pb",
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

func TestClientCompile(t *testing.T) {
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices, err := client.AddressableDevices()
	require.NoErrorf(t, err, "Failed to fetch AddressableDevices() from client on %s", plugin)
	require.NotEmptyf(t, devices, "No addressable devices for client on %s", plugin)

	for _, programFile := range testHLOProgramFiles {
		// Load test program.
		hloBin, err := os.ReadFile(programFile)
		require.NoError(t, err)
		hloProto := &pjrt_proto.HloModuleProto{}
		require.NoError(t, proto.Unmarshal(hloBin, hloProto), "Unmarshalling HloModuleProto")
		fmt.Printf("HloModuleProto: {\n%s}\n", prototext.Format(hloProto))

		// Compile program.
		loadedExec, err := client.Compile().WithHLO(hloBin).Done()
		require.NoErrorf(t, err, "Failed to compile %q", programFile)
		exec, err := loadedExec.GetExecutable()
		require.NoError(t, err)

		numOutputs, err := exec.NumOutputs()
		fmt.Printf("\tnum_outputs=%d\n", numOutputs)

		// Destroy compiled executables.
		require.NoErrorf(t, exec.Destroy(), "Failed to destroy Executable on %s", plugin)
		require.NoErrorf(t, loadedExec.Destroy(), "Failed to destroy LoadedExecutable on %s", plugin)
	}

	// Destroy client.
	require.NoErrorf(t, client.Destroy(), "Failed to destroy client on %s", plugin)
}
