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
	"bufio"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"unsafe"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	// dynamicLibrariesPaths is initialized with the values of LD_LIBRARY_PATH and /etc/ld.so.conf file.
	dynamicLibrariesPaths map[string]bool

	reLdConfInclude = regexp.MustCompile(`^\s*include\s*(.*)$`)
	reLdConfComment = regexp.MustCompile(`^\s*#`)
	reLdConfPath    = regexp.MustCompile(`^\s*(.+?)\s*$`)
)

// osDefaultLibraryPaths is called during initialization to set the default search paths.
// It always includes the local default "${HOME}/.local/lib/gomlx/pjrt" and the
// system default "/usr/local/lib/gomlx/pjrt".
func osDefaultLibraryPaths() []string {
	var paths []string

	// Local (XDG) path.
	if homeDir, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(homeDir, ".local", "lib", "gomlx", "pjrt"))
	} else {
		klog.Errorf("Couldn't get user's home directory -- it won't be searched for PJRT plugins: %v", err)
	}

	// Standard system directory, included by default.
	paths = append(paths, "/usr/local/lib/gomlx/pjrt")

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

// linuxDLLHandle represents an open handle to a library (.so)
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

// SuppressAbseilLoggingHack prevents some irrelevant logging from PJRT plugins by duplicating the file descriptor (fd) 2,
// reassigning the new fd to Go's os.Stderr, and then closing fd 2, so PJRT plugins won't be able to write anything.
//
// Usually, this is only needed during the creation of the Client of the CPU plugin. The suggestion it to just
// wrap that part.
//
// Now since many things rely on fd 2 being stderr, it only does that, executes fn given and reverts the change.
//
// The issue of doing this permanently is that Go's default panic handler outputs the stack tracke the the fd 2,
// and this would suppress that as well.
//
// It's an overkill, because this may prevent valid logging in some truly exceptional situations, but it's the only
// solution I can think of for now. See discussion in https://github.com/abseil/abseil-cpp/discussions/1700
//
// Since file descriptors are a global resource, this function is not reentrant, and you should
// make sure no two goroutines are calling this at the same time.
func SuppressAbseilLoggingHack(fn func()) {
	newFd, err := suppressLogging()
	if err != nil {
		klog.Errorf("Failed to temporarily suppress absl::logging: %+v", err)
	} else {
		defer func() {
			// Revert suppression: revert back newFd to 2
			err := syscall.Dup3(newFd, 2, 0)
			if err != nil {
				klog.Errorf("Failed sycall.Dup3 while reverting suppression of logging: %v", err)
			}
		}()
	}

	fn()
}
