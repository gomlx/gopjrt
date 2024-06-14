package gopjrt

// Common initialization for all test files.

import "k8s.io/klog/v2"

func init() {
	klog.InitFlags(nil)
}
