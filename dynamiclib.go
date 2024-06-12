package gopjrt

import (
	"github.com/pkg/errors"
	"os"
	"path"
	"strings"
)

// This file holds common definitions for the different implementations of dynamiclib (linux, windows, mac?).

const (
	// GetPJRTApiFunctionName is the name of the function exported by PJRT plugins that returns the API.
	GetPJRTApiFunctionName = "GetPjrtApi"
)

var (
	// KnownPlugins maps known plugins names (uppercase) to a list of library names that should be loaded.
	//
	// You can add names during initialization, but not after it.
	//
	// TODO: this should be moved to platform specific, since in Windows they will be .dll files, and likely with different names.
	KnownPlugins = map[string][]string{
		"CPU": []string{"gomlx/pjrt_c_api_cpu_plugin.so", "pjrt_c_api_cpu_plugin.so"},
		"GPU": []string{"gomlx/pjrt_c_api_gpu_plugin.so", "pjrt_c_api_gpu_plugin.so"},
	}

	// PluginsAliases map platform names (uppercase) to the corresponding canonical plugin name (also uppercase).
	PluginsAliases = map[string]string{
		"HOST": "CPU",
		"CUDA": "GPU",
	}
)

// LoadPlatform by loading the corresponding plugin.
// It returns an error if it doesn't find it.
func LoadPlatform(platform string) error {
	canonicalPlatform := strings.ToUpper(platform)
	if _, ok := PluginsAliases[canonicalPlatform]; ok {
		canonicalPlatform = PluginsAliases[canonicalPlatform]
	}
	if _, ok := KnownPlugins[canonicalPlatform]; !ok {
		return errors.Errorf("Unknown platform %q (canonical form %q)", platform, canonicalPlatform)
	}
	pluginPaths := KnownPlugins[canonicalPlatform]

	// Prefix LD_LIBRARY_PATH to non-absolute entries.
	for _, ldPath := range strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":") {
		if ldPath == "" || !path.IsAbs(ldPath) {
			// No empty or relative paths.
			continue
		}
		for ii := range len(pluginPaths) {
			p := pluginPaths[ii]
			if path.IsAbs(p) {
				continue
			}
			pluginPaths = append(pluginPaths, path.Join(ldPath, p))
		}
	}
	_, err := LoadLibrary(false, pluginPaths...)
	return err
}
