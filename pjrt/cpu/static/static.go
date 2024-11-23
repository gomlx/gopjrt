// Package static statically links a CPU PJRT plugin, and registers with the name "cpu".
//
// To use it simply import with:
//
//	import _ "github.com/gomlx/gopjrt/pjrt/static"
//
// And calls to pjrt.GetPlugin("cpu") will return the statically linked one.
//
// If you use this, don't use prjt/cpu/dynamic package. Only one can be set.
package static

// #cgo LDFLAGS: -lpjrt_c_api_cpu_static -lstdc++ -lm
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
		klog.Fatal("Failed to get PJRT API pointer when initializing statically preloaded PJRT (github.com/gomlx/gopjrt/pjrt/cpu/static).")
	}
	err := pjrt.RegisterPreloadedPlugin("cpu", pjrtAPI)
	if err != nil {
		klog.Fatalf("Failed to register statically preloaded PJRT plugin for CPU (github.com/gomlx/gopjrt/pjrt/cpu/static): %+v", err)
	}
}
