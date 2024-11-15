//go:build darwin

package pjrt

// This file handles management of loading dynamic libraries for linux
//
// It should implement 3 methods:
//
//	osDefaultLibraryPaths() []string
//	loadPlugin(path string) (C.GetPJRTApiFn, error)
//	checkPlugin(path string) error
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
	"os"
	"path"
	"strings"
	"unsafe"
)

// osDefaultLibraryPaths is called during initialization to set the default search paths.
// It always includes the default "/usr/local/lib/gomlx/pjrt" for linux and darwin, plus
// for Darwin, the contents of teh LD_LIBRARY_PATH and DYLD_LIBRARY_PATH.
func osDefaultLibraryPaths() []string {
	paths := []string{"/usr/local/lib/gomlx/pjrt"}

	// Standard environment variables.
	for _, varName := range []string{"DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"} {
		for _, ldPath := range strings.Split(os.Getenv(varName), string(os.PathListSeparator)) {
			if ldPath == "" || !path.IsAbs(ldPath) {
				// No empty or relative paths.
				continue
			}
			paths = append(paths, ldPath)
		}
	}
	return paths
}

// loadPlugin tries to dlopen the plugin and returns a handle with the pointer to the PJRT api function
func loadPlugin(pluginPath string) (handleWrapper dllHandleWrapper, err error) {
	info, err := os.Stat(pluginPath)
	if err != nil {
		err = errors.Wrapf(err, "failed to stat %q", pluginPath)
		return
	}
	if info.IsDir() {
		err = errors.Errorf("plugin path %q is a directory!?", pluginPath)
		return
	}

	nameC := C.CString(pluginPath)
	klog.V(2).Infof("trying to load library %s\n", pluginPath)
	handle := C.dlopen(nameC, C.RTLD_LAZY|C.RTLD_LOCAL)
	cFree(nameC)
	if handle == nil {
		msg := C.GoString(C.dlerror())
		err = errors.Errorf("failed to dynamically load PJRT plugin from %q: %q -- check with `ldd %s` in case there are missing required libraries.", msg, pluginPath, pluginPath)
		klog.Warningf("%v", err)
		return
	}

	klog.V(1).Infof("loaded library %s\n", pluginPath)
	h := &linuxDLLHandle{
		Handle: handle,
		Name:   pluginPath,
	}
	cPtr, err := h.GetSymbolPointer(GetPJRTApiFunctionName)
	if err != nil {
		err = errors.Errorf("tried to load %q, but failed to find symbol %q, skipping: %v", pluginPath, GetPJRTApiFunctionName, err)
		klog.Warningf("%v", err)
		err2 := h.Close()
		if err2 != nil {
			klog.Warningf("Failed to close dynamic library %q: %v", pluginPath, err2)
		}
		return
	}
	h.PJRTApiFn = (C.GetPJRTApiFn)(cPtr)
	handleWrapper = h
	return
}

// linuxDLLHandle represents an open handle to a library (.dylib)
type linuxDLLHandle struct {
	Handle    unsafe.Pointer
	PJRTApiFn C.GetPJRTApiFn
	Name      string
}

// GetPJRTApiFn returns the pointer to the PJRT API function.
func (l *linuxDLLHandle) GetPJRTApiFn() (C.GetPJRTApiFn, error) {
	return l.PJRTApiFn, nil
}

// GetSymbolPointer takes a symbol name and returns a pointer to the symbol.
func (l *linuxDLLHandle) GetSymbolPointer(symbol string) (unsafe.Pointer, error) {
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
func (l *linuxDLLHandle) Close() error {
	C.dlerror()
	C.dlclose(l.Handle)
	e := C.dlerror()
	if e != nil {
		return errors.Errorf("error closing %v: %v", l.Name, errors.New(C.GoString(e)))
	}
	return nil
}
