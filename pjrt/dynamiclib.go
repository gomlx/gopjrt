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
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"syscall"
)

// This file holds common definitions for the different implementations of dynamiclib (linux, windows, mac?).

const (
	// PJRTPluginPathsEnv is the name of the environment variable that define the search paths for plugins.
	PJRTPluginPathsEnv = "PJRT_PLUGIN_LIBRARY_PATH"

	// GetPJRTApiFunctionName is the name of the function exported by PJRT plugins that returns the API.
	GetPJRTApiFunctionName = "GetPjrtApi"
)

var (
	// pluginSearchPaths is set during initialization by the per-architecture implementations (dynamiclib_<arch>.go files).
	//
	// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
	// If it is not set it will search in `/usr/local/lib/gomlx` and the standard libraries directories of the
	// system (in linux in LD_LIBRARY_CONFIG and /etc/ld.so.conf file).
	pluginSearchPaths []string

	// loadedPlugins caches the plugins already loaded. Protected by muPlugins.
	loadedPlugins = make(map[string]*Plugin)
	muPlugins     sync.Mutex
)

func init() {
	pjrtPaths, found := os.LookupEnv(PJRTPluginPathsEnv)
	if !found {
		pluginSearchPaths = osDefaultLibraryPaths()
	} else {
		pluginSearchPaths = slices.DeleteFunc(strings.Split(pjrtPaths, ":"), func(p string) bool {
			return p == "" // Remove empty paths.
		})
	}
}

// loadNamedPlugin by loading the corresponding plugin.
// It returns an error if it doesn't find it.
//
// It uses a mutex to serialize (make it safe) calls from different goroutines.
func loadNamedPlugin(name string) (*Plugin, error) {
	muPlugins.Lock()
	defer muPlugins.Unlock()

	// Search previously loaded plugin: match by name or by path (if the name given is an absolute path).
	if plugin, found := loadedPlugins[name]; found {
		return plugin, nil
	}
	if path.IsAbs(name) {
		for _, plugin := range loadedPlugins {
			if plugin.Path() == name {
				return plugin, nil
			}
		}
	}

	// Search path to plugin -- except if name is an absolute path.
	pluginPath := name
	if !path.IsAbs(pluginPath) {
		var found bool
		pluginPath, found = searchPlugin(name)
		if !found {
			return nil, errors.Errorf("plugin name %q not found in paths %v: set PJRT_PLUGIN_LIBRARY_PATH to an specific path(s) to search; "+
				"plugins should be named pjrt_c_api_<name>_plugin.so",
				name, pluginSearchPaths)
		}
	}
	klog.V(1).Infof("attempting to laod plugin from %s", pluginPath)

	var err error
	var pjrtAPIFn C.GetPJRTApiFn
	pjrtAPIFn, err = loadPlugin(pluginPath)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to load PJRT plugin for name %q", name)
	}
	api := C.call_GetPJRTApiFn(pjrtAPIFn)
	if api == nil {
		return nil, errors.WithMessagef(err, "loaded PJRT plugin for name %q, but it returned a nil plugin!?", name)
	}
	plugin, err := newPlugin(name, pluginPath, api)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to initialize PJRT plugin for name %q after loading it, this leaves the plugin in an unstable state", name)
	}
	loadedPlugins[name] = plugin
	cudaPluginChecks(name)
	return plugin, nil
}

var (
	// Patterns to extract the name from the plugins.
	rePluginName = []*regexp.Regexp{
		regexp.MustCompile(`^.*/pjrt_c_api_(\w+)_plugin.so$`),
		regexp.MustCompile(`^.*/pjrt[-_]plugin[-_](\w+).so$`),
	}
)

// pathToPluginName returns the name of the plugin if it's a matching plugin path, otherwise returns "".
func pathToPluginName(pPath string) string {
	for _, re := range rePluginName {
		if re.MatchString(pPath) {
			subMatches := re.FindStringSubmatch(pPath)
			return subMatches[1]
		}
	}
	return ""
}

// AvailablePlugins searches for available plugins in the standard directories and returns a map from their name to their paths.
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
// If it is not set it will search in `/usr/local/lib/gomlx` and the standard libraries directories of the
// system (in linux in LD_LIBRARY_PATH and /etc/ld.so.conf file) in that order.
//
// If there are plugins with the same name but different versions in different directories, it respects the order of the directories given by
// PJRT_PLUGIN_LIBRARY_PATH or by the system.
func AvailablePlugins() (pluginsPaths map[string]string) {
	return searchPlugins("")
}

func searchPlugin(searchName string) (path string, found bool) {
	path, found = searchPlugins(searchName)[searchName]
	return
}

func searchPlugins(searchName string) (pluginsPaths map[string]string) {
	pluginsPaths = make(map[string]string)
	for _, pluginPath := range pluginSearchPaths {
		for _, pattern := range []string{"pjrt-plugin-*.so", "pjrt_plugin_*.so", "pjrt_c_api_*_plugin.so"} {
			candidates, err := filepath.Glob(path.Join(pluginPath, pattern))
			if err != nil {
				continue
			}
			for _, candidate := range candidates {
				name := pathToPluginName(candidate)
				if name == "" {
					continue
				}
				if searchName != "" && searchName != name {
					continue
				}
				if _, found := pluginsPaths[name]; found {
					// We already have a plugin with that name.
					continue
				}
				err := checkPlugin(candidate)
				if err != nil {
					continue
				}
				pluginsPaths[name] = candidate
			}
		}
	}
	return
}

// cudaPluginChecks issues warning on cuda plugins if it cannot find the corresponding nvidia library files.
// It should be called after the named plugin is loaded.
//
// This is helpful to try to sort out the mess of path for nvidia libraries. It's something really badly organized
// at multiple levels (just search to see how many questions there are related to where/how install to CUDA libraries).
func cudaPluginChecks(name string) {
	cudaChecks := os.Getenv("GOPJRT_CUDA_CHECKS")
	if cudaChecks != "" && cudaChecks != "1" && strings.ToUpper(cudaChecks) != "TRUE" && strings.ToUpper(cudaChecks) != "YES" {
		// Checks disabled.
		return
	}
	if strings.Index(strings.ToUpper(name), "CUDA") == -1 &&
		strings.Index(strings.ToUpper(name), "NVIDIA") == -1 &&
		strings.Index(strings.ToUpper(name), "GPU") == -1 {
		// Assume not a cuda plugin.
		return
	}

	plugin, ok := loadedPlugins[name]
	if !ok {
		return
	}
	nvidiaPath := path.Join(path.Dir(path.Dir(plugin.Path())), "nvidia")
	fi, err := os.Stat(nvidiaPath)
	if err == nil && fi.IsDir() {
		// We assume the NVIDIA libraries are installed correctly.
		return
	}
	klog.Warningf("Can't find nvidia/ subdirectory next to the cuda plugin (%q) in %q. "+
		"When compiling and executing a program likely the PJRT CUDA plugin will fail to find the many required NVidia's "+
		"sub-libraries: this is confusing, the plugin usually hard code the search path to $ORIGIN/../nvidia/... and $ORIGIN/../../nvidia/... "+
		"in a hardcoded variable called RPATH (it can be checked with `readelf -d %q`, look for RPATH). "+
		"Either install the various nvidia libraries there, or more simply, use the Jax python installation (`pip install -U \"jax[cuda12]\"`), "+
		"and link its nvidia directories (`ln -s \"<python_virtual_environment_path>/lib/python3.12/site-packages/nvidia\" %q`). "+
		"If you have things correctly set up, and you want to disable this warning, just set the environment variable `export GOPJR_CUDA_CHECKS=0`.",
		plugin.Path(), nvidiaPath, plugin.Path(), nvidiaPath)
}

// SuppressAbseilLoggingHack prevents some irrelevant logging from PJRT plugins, by duplicating the file descriptor (fd) 2,
// reassigning the new fd to Go's os.Stderr, and then closing fd 2, so PJRT plugins won't be able to write anything.
//
// It's an overkill, because this may prevent valid logging, in some truly exceptional situation, but it's the only
// solution for now. See discussion in https://github.com/abseil/abseil-cpp/discussions/1700
func SuppressAbseilLoggingHack() {
	// Silence absl::logging hack.
	newStderrFd, err := syscall.Dup(2)
	if err != nil {
		klog.Errorf("Failed to duplicate file descriptor 2 (stderr) in order to silence abseil logging: %v", err)
	}
	err = syscall.Close(2)
	if err != nil {
		klog.Errorf("Failed to close syscall 2: %v", err)
	}
	os.Stderr = os.NewFile(uintptr(newStderrFd), "stderr")
}
