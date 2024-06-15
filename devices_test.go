package cmd

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestClient_Devices(t *testing.T) {
	plugin, err := loadNamedPlugin("cpu")
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

}
