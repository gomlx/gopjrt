//go:build pjrt_cpu_dynamic

package pjrt

import (
	"k8s.io/klog/v2"
)

// Duplicated from pjrt/cpu/static to avoid cyclic dependencies.
func init() {
	pjrtAPI := cpudynamictest.GetPjrtApi()
	if pjrtAPI == 0 {
		klog.Fatal("Failed to get PJRT API pointer when initializing static PJRT (github.com/gomlx/gopjrt/pjrt/static).")
	}
	err := RegisterPreloadedPlugin("cpu", pjrtAPI)
	if err != nil {
		klog.Fatalf("Failed to register static PJRT plugin for CPU (github.com/gomlx/gopjrt/pjrt/static): %+v", err)
	}
}
