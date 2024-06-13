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

import (
	"github.com/pkg/errors"
	"strings"
)

// This file holds common definitions for the different implementations of dynamiclib (linux, windows, mac?).

// Generate automatic C-to-Go boilerplate code.
//go:generate go run ./cmd/codegen < pjrt_c_api.h

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
	_, err := LoadLibrary(false, pluginPaths...)
	return err
}
