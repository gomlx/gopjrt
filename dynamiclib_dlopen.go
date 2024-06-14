//go:build linux

/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package gopjrt

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
	"bufio"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"unsafe"
)

var (
	// dynamicLibrariesPaths is initialized with the values of LD_LIBRARY_PATH and /etc/ld.so.conf file.
	dynamicLibrariesPaths map[string]bool

	reLdConfInclude = regexp.MustCompile(`^\s*include\s*(.*)$`)
	reLdConfComment = regexp.MustCompile(`^\s*#`)
	reLdConfPath    = regexp.MustCompile(`^\s*(.+?)\s*$`)
)

// osDefaultLibraryPaths is called during initialization to set the default search paths.
// It always includes the default "/usr/local/lib/gomlx" for linux.
func osDefaultLibraryPaths() []string {
	paths := []string{"/usr/local/lib/gomlx"}

	// Prefix LD_LIBRARY_PATH to non-absolute entries.
	for _, ldPath := range strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":") {
		if ldPath == "" || !path.IsAbs(ldPath) {
			// No empty or relative paths.
			continue
		}
		paths = append(paths, ldPath)
	}
	return loadLibraryPaths(paths, "/etc/ld.so.conf")
}

func loadLibraryPaths(paths []string, fileWithIncludes string) []string {
	klog.V(2).Infof("Loading paths for libraries from %q", fileWithIncludes)
	file, err := os.Open(fileWithIncludes)
	if err != nil {
		klog.Errorf("Failed to load paths for libraries from %q: %v", fileWithIncludes, err)
		return paths
	}
	defer func() { _ = file.Close() }()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if parts := reLdConfInclude.FindStringSubmatch(line); len(parts) > 0 {
			// Include pattern.
			klog.V(2).Infof("loadLibraryPaths: include %q", parts[1])
			files, err := filepath.Glob(parts[1])
			if err != nil {
				klog.Errorf("Failed to load paths for libraries while expanding include entry %q: %v", parts[1], err)
				continue
			}
			for _, includeFile := range files {
				paths = loadLibraryPaths(paths, includeFile)
			}

		} else if reLdConfComment.MatchString(line) {
			klog.V(2).Infof("loadLibraryPaths: comment %q", line)

		} else if parts := reLdConfPath.FindStringSubmatch(line); len(parts) > 0 {
			klog.V(2).Infof("loadLibraryPaths: path %q", parts[1])
			paths = append(paths, parts[1])

		} else if strings.TrimSpace(line) != "" {
			klog.V(2).Infof("loadLibraryPaths: cannot parse line %q", line)
		}
	}
	if err := scanner.Err(); err != nil {
		klog.Errorf("Error while loading paths for libraries from %q: %v", fileWithIncludes, err)
	}
	return paths
}

// loadPlugin tries to dlopen the plugin and will return a pointer to a C function that returns the PJRT_API.
//
// Notice that if the library is loaded, the `dlopen`(3) handle is never closed and is lost -- only when the process closes.
// It is assumed that a library is not going to be loaded more than once.
func loadPlugin(pluginPath string) (C.GetPJRTApiFn, error) {
	return linuxLoadPlugin(pluginPath, false)
}

// checkPlugin tries to dlopen the plugin and verify that the GetPjrtApi function is exported.
//
// The handle returned by dlopen is properly destroyed.
func checkPlugin(pluginPath string) error {
	_, err := linuxLoadPlugin(pluginPath, true)
	return err
}

func linuxLoadPlugin(pluginPath string, onlyCheck bool) (C.GetPJRTApiFn, error) {
	info, err := os.Stat(pluginPath)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return nil, errors.New("plugin path is a directory!?")
	}

	nameC := C.CString(pluginPath)
	klog.V(2).Infof("trying to load library (onlyCheck=%v) %s\n", onlyCheck, pluginPath)
	handle := C.dlopen(nameC, C.RTLD_LAZY)
	cFree(nameC)
	if handle == nil {
		err = errors.Errorf("failed to dynamically load PJRT plugin from %q: check with `ldd %s`, maybe there are missing required libraries.", pluginPath, pluginPath)
		klog.Warningf("%v", err)
		return nil, err
	}

	klog.V(1).Infof("loaded library %s\n", pluginPath)
	h := &libHandle{
		Handle: handle,
		Name:   pluginPath,
	}
	getPJRT, err := h.GetSymbolPointer(GetPJRTApiFunctionName)
	if err != nil {
		err = errors.Errorf("tried to load %q, but failed to find symbol %q, skipping: %v", pluginPath, GetPJRTApiFunctionName, err)
		klog.Warningf("%v", err)
		err2 := h.Close()
		if err2 != nil {
			klog.Warningf("Failed to close dynamic library %q: %v", pluginPath, err2)
		}
		return nil, err
	}

	if onlyCheck {
		err = h.Close()
		if err != nil {
			klog.Warningf("Failed to close dynamic library %q: %v", pluginPath, err)
		}
		return nil, nil
	}

	// Keep handle alive till the end of the program.
	var cPtr C.GetPJRTApiFn
	cPtr = (C.GetPJRTApiFn)(getPJRT)
	return cPtr, nil
}

// libHandle represents an open handle to a library (.so)
type libHandle struct {
	Handle unsafe.Pointer
	Name   string
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
