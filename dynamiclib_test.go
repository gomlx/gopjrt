package gopjrt

import (
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"testing"
)

func init() {
	klog.InitFlags(nil)
}

// TestLoadPlatformCPU requires that PJRT CPU plugin be available.
func TestLoadPlatformCPU(t *testing.T) {
	require.NoError(t, LoadPlatform("CPU"))
	require.NoError(t, LoadPlatform("host")) // Should be found using the alias.
}
