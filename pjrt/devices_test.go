package pjrt

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestClient_Devices(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("\t%s\n", client)

	devices, err := client.AllDevices()
	require.NoError(t, err, "Failed to list devices for %s", client)

	addressableDevices := client.AddressableDevices()
	fmt.Printf("\t%d devices, %d addressable\n", len(devices), len(addressableDevices))

	if client.ProcessIndex() == 0 {
		require.Equal(t, len(devices), len(addressableDevices),
			"In single-process client (process index==0), all devices should be addressable, but only %d out of %d are",
			len(addressableDevices), len(devices))
	}

	var countAddressable int
	for _, d := range devices {
		isAddr, err := d.IsAddressable()
		require.NoError(t, err)
		if isAddr {
			countAddressable++
		}
		desc, err := d.GetDescription()
		require.NoError(t, err)
		fmt.Printf("\t\tDevice Local Hardware Id %d: %s\n", d.LocalHardwareID(), desc.DebugString())
	}
	require.Equal(t, countAddressable, len(addressableDevices))
	require.NoError(t, client.Destroy())
}
