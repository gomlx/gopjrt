// Package cpudynamictest is just a hack around Go's limitation to use CGO in tests and to avoid cyclic dependency.
// Don't use it except inside pjrt package tests.
package cpudynamictest

// #cgo LDFLAGS: -lpjrt_c_api_cpu_dynamic -lstdc++ -lm
/*
typedef void PJRT_Api;

extern const PJRT_Api* GetPjrtApi();
*/
import "C"
import "unsafe"

// GetPjrtApi calls the statically linked GetPjrtApi.
// Don't use it, except if inside the pjrt package test.
// This is just a hack around Go's limitation to use CGO in tests.
func GetPjrtApi() uintptr {
	return uintptr(unsafe.Pointer(C.GetPjrtApi()))
}
