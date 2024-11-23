// Package static statically links a CPU PJRT plugin, and registers with the name "cpu".
//
// To use it simply import with:
//
//	import _ "github.com/gomlx/gopjrt/pjrt/static"
//
// And calls to pjrt.GetPlugin("cpu") will return the statically linked one.
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
		klog.Fatal("Failed to get PJRT API pointer when initializing static PJRT (github.com/gomlx/gopjrt/pjrt/static).")
	}
	err := pjrt.RegisterPreloadedPlugin("cpu", pjrtAPI)
	if err != nil {
		klog.Fatalf("Failed to register static PJRT plugin for CPU (github.com/gomlx/gopjrt/pjrt/static): %+v", err)
	}
}
