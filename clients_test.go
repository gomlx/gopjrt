package gopjrt

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestPlugin_NewClient(t *testing.T) {
	plugin, err := loadNamedPlugin("cpu")
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
}
