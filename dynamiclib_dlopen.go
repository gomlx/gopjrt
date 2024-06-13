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

func initDynamicLibrariesPaths() {
	dynamicLibrariesPaths = make(map[string]bool)

	// Prefix LD_LIBRARY_PATH to non-absolute entries.
	for _, ldPath := range strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":") {
		if ldPath == "" || !path.IsAbs(ldPath) {
			// No empty or relative paths.
			continue
		}
		dynamicLibrariesPaths[ldPath] = true
	}
	loadLibraryPaths("/etc/ld.so.conf")
	if klog.V(1).Enabled() {
		klog.Infof("Library paths: %v", keys(dynamicLibrariesPaths))
	}
}

func loadLibraryPaths(filePath string) {
	klog.V(2).Infof("Loading paths for libraries from %q", filePath)
	file, err := os.Open(filePath)
	if err != nil {
		klog.Errorf("Failed to load paths for libraries from %q: %v", filePath, err)
		return
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
				loadLibraryPaths(includeFile)
			}

		} else if reLdConfComment.MatchString(line) {
			klog.V(2).Infof("loadLibraryPaths: comment %q", line)

		} else if parts := reLdConfPath.FindStringSubmatch(line); len(parts) > 0 {
			klog.V(2).Infof("loadLibraryPaths: path %q", parts[1])
			dynamicLibrariesPaths[parts[1]] = true

		} else if strings.TrimSpace(line) != "" {
			klog.V(2).Infof("loadLibraryPaths: cannot parse line %q", line)
		}
	}
	if err := scanner.Err(); err != nil {
		klog.Errorf("Error while loading paths for libraries from %q: %v", filePath, err)
	}
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
	if dynamicLibrariesPaths == nil {
		initDynamicLibrariesPaths()
	}

	noPaths := []string{""}
	allPaths := keys(dynamicLibrariesPaths)

	for _, name := range names {
		prefixes := allPaths
		if path.IsAbs(name) {
			prefixes = noPaths
		}
		for _, prefix := range prefixes {
			candidateName := path.Join(prefix, name)
			nameC := C.CString(candidateName)
			klog.V(2).Infof("trying to load library %s\n", candidateName)
			handle := C.dlopen(nameC, C.RTLD_LAZY)
			C.free(unsafe.Pointer(nameC))
			if handle == nil {
				if info, err := os.Stat(candidateName); err == nil && !info.IsDir() {
					klog.Warningf("Failed to dynamically load PJRT plugin from %q: check with `ldd %s`, maybe there are missing required libraries.", candidateName, candidateName)
				}
				continue
			}
			klog.V(1).Infof("loaded library %s\n", candidateName)
			h := &libHandle{
				Handle: handle,
				Name:   name,
			}
			getPJRT, err := h.GetSymbolPointer(GetPJRTApiFunctionName)
			if err != nil {
				klog.Warningf("Tried to load %q, but failed to find symbol %q, skipping: %v", candidateName, GetPJRTApiFunctionName, err)
				err = h.Close()
				if err != nil {
					klog.Warningf("Failed to close dynamic library %q: %v", candidateName, err)
				}
				continue // Try next path.
			}
			if test {
				err = h.Close()
				if err != nil {
					klog.Warningf("Failed to close dynamic library %q: %v", candidateName, err)
				}
				return nil, nil
			}
			var cPtr C.GetPJRTApiFn
			cPtr = (C.GetPJRTApiFn)(getPJRT)
			return cPtr, nil
		}
	}
	return nil, errors.Errorf("failed to load library with any of the names [%q]", strings.Join(names, ", "))
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
