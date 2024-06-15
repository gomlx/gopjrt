package pjrt

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestPlugin_NewClient(t *testing.T) {
	plugin, err := GetPlugin(*flagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
