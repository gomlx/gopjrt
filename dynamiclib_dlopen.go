//go:build linux

package gopjrt

// This file handles management of loading dynamic libraries.
//
// Modified version of https://github.com/coreos/pkg/blob/main/dlopen/dlopen.go, licenced with Apache 2.0 license
// https://github.com/coreos/pkg/blob/main/LICENSE

// #cgo LDFLAGS: -ldl
/*
#include <stdlib.h>
#include <dlfcn.h>
#include "pjrt_c_api.h"
#include "common.h"
*/
import "C"
import (
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"strings"
	"unsafe"
)

// libHandle represents an open handle to a library (.so)
type libHandle struct {
	Handle unsafe.Pointer
	Name   string
}

// LoadLibrary will return a pointer to a C function that returns the PJRT_API.
// It tries the names in the order given, and will use the first name that finds a matching dynamic library
// that defines GetPJRTApiFunctionName.
//
// Notice that if the library is loaded, the `dlopen`(3) handle is never closed and is lost -- only when the process closes.
// It is assumed that a library is not going to be loaded more than once.
//
// Any handles created and not actually used, are properly closed.
//
// If test is set to true, the library is immediately discarded (and the handle closed), and this can be used
// to check the availability of plugins. It's costly, so one should do this once and cache the results.
func LoadLibrary(test bool, names ...string) (C.GetPJRTApiFn, error) {
	for _, name := range names {
		nameC := C.CString(name)
		handle := C.dlopen(nameC, C.RTLD_LAZY)
		C.free(unsafe.Pointer(nameC))
		if handle == nil {
			continue
		}

		klog.V(1).Infof("loaded library %s\n", name)
		h := &libHandle{
			Handle: handle,
			Name:   name,
		}
		getPJRT, err := h.GetSymbolPointer(GetPJRTApiFunctionName)
		_ = getPJRT
		if err != nil {
			klog.Warningf("Tried to load %q, but failed to find symbol %q, skipping: %v", name, GetPJRTApiFunctionName, err)
			err = h.Close()
			if err != nil {
				klog.Warningf("Failed to close dynamic library %q: %v", name, err)
			}
			continue // Try next path.
		}
		if test {
			err = h.Close()
			if err != nil {
				klog.Warningf("Failed to close dynamic library %q: %v", name, err)
			}
			return nil, nil
		}
		var cPtr C.GetPJRTApiFn
		cPtr = (C.GetPJRTApiFn)(getPJRT)
		return cPtr, nil
	}
	return nil, errors.Errorf("failed to load library with any of the names [%q]", strings.Join(names, ", "))
}

// GetSymbolPointer takes a symbol name and returns a pointer to the symbol.
func (l *libHandle) GetSymbolPointer(symbol string) (unsafe.Pointer, error) {
	sym := C.CString(symbol)
	defer C.free(unsafe.Pointer(sym))

	C.dlerror()
	p := C.dlsym(l.Handle, sym)
	e := C.dlerror()
	if e != nil {
		return nil, errors.Errorf("error resolving symbol %q: %v", symbol, errors.New(C.GoString(e)))
	}

	return p, nil
}

// Close closes a LibHandle.
func (l *libHandle) Close() error {
	C.dlerror()
	C.dlclose(l.Handle)
	e := C.dlerror()
	if e != nil {
		return errors.Errorf("error closing %v: %v", l.Name, errors.New(C.GoString(e)))
	}
	return nil
}
