package gopjrt

// Common initialization for all test files.

import (
	"flag"
	"k8s.io/klog/v2"
)

var flagPluginName = flag.String("plugin", "cpu", "plugin name")

func init() {
	klog.InitFlags(nil)
}
