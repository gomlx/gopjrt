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

const testHLOProgramFile = "test_hlo.pb"

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

	// Load test program.
	hloBin, err := os.ReadFile(testHLOProgramFile)
	require.NoError(t, err)
	hloProto := &pjrt_proto.HloModuleProto{}
	require.NoError(t, proto.Unmarshal(hloBin, hloProto), "Unmarshalling HloModuleProto")
	fmt.Printf("HloModuleProto: {\n%s}\n", prototext.Format(hloProto))

	// Compile program.
	exec, err := client.Compile().WithHLO(hloBin).Done()
	require.NoErrorf(t, err, "Failed to compile %q", testHLOProgramFile)
	_ = exec

	// Destroy client.
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
