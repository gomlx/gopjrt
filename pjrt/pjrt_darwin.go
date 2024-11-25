//go:build darwin

// Macs don't work with dynamically loading plugins, so it defaults to statically pre-linking
// the CPU plugin (except if explicitly pre-linking it dynamically).

package pjrt

// #cgo LDFLAGS: -lpjrt_c_api_cpu_static
/*
#include "pjrt_c_api.h"

extern const PJRT_Api* GetPjrtApi();
*/
import "C"
import (
	"k8s.io/klog/v2"
	"unsafe"
)

func init() {
	pjrtAPI := uintptr(unsafe.Pointer(C.GetPjrtApi()))
	if pjrtAPI == 0 {
		klog.Fatal("Failed to get PJRT API pointer when initializing statically preloaded PJRT (default for DarwinOS).")
	}
	err := RegisterPreloadedPlugin("cpu", pjrtAPI)
	if err != nil {
		klog.Fatalf("Failed to register statically preloaded PJRT plugin for CPU (default for DarwinOS): %+v", err)
	}
}
