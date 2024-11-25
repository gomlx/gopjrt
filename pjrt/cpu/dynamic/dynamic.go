// Package dynamic will link (preload) a dynamically loaded library `libpjrt_c_api_cpu_dynamic`, that is used
// if the user requests a "cpu" plugin.
//
// To use it simply import with:
//
//	import _ "github.com/gomlx/gopjrt/pjrt/static"
//
// And calls to pjrt.GetPlugin("cpu") will return the dynamically linked one.
//
// It still can load in runtime other plugins if needed.
//
// If you use this, don't use prjt/cpu/static package. Only one can be set.
package dynamic

// #cgo LDFLAGS: -lpjrt_c_api_cpu_dynamic -lstdc++ -lm
/*
typedef void PJRT_Api;

extern const PJRT_Api* GetPjrtApi();
*/
import "C"
import (
	"github.com/gomlx/gopjrt/pjrt"
	"k8s.io/klog/v2"
	"unsafe"
)

func init() {
	pjrtAPI := uintptr(unsafe.Pointer(C.GetPjrtApi()))
	if pjrtAPI == 0 {
		klog.Fatal("Failed to get PJRT API pointer when initializing dynamically preloaded PJRT (github.com/gomlx/gopjrt/pjrt/cpu/dynamic).")
	}
	err := pjrt.RegisterPreloadedPlugin("cpu", pjrtAPI)
	if err != nil {
		klog.Fatalf("Failed to register dynamically preloaded PJRT plugin for CPU (github.com/gomlx/gopjrt/pjrt/cpu/dynamic): %+v", err)
	}
}
