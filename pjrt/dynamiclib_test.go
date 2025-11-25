//go:build !pjrt_cpu_dynamic && !pjrt_cpu_static && !darwin

// Test doesn't work if plugin is pre-linked into binary.
// In Macs loading plugins after the program start (dlopen) doesn't work.

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

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

// TestLoadNamedPlugin loads the *FlagPluginName plugin, which defaults to "cpu", that should be made available.
func TestLoadNamedPlugin(t *testing.T) {
	plugin, err := loadNamedPlugin(*FlagPluginName)
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)
	fmt.Printf("\tattributes: %v\n", plugin.attributes)

	// Checks cache works.
	plugin2, err := loadNamedPlugin(*FlagPluginName)
	require.NoError(t, err)
	require.Equal(t, plugin, plugin2)
	plugin3, err := loadNamedPlugin(plugin2.Path()) // Try by using the absolute path, should return the same plugin.
	require.NoError(t, err)
	require.Equal(t, plugin2, plugin3)

	// Checks non-existent (yet) plugin.
	plugin, err = loadNamedPlugin("milliways")
	fmt.Printf("Loading milliways plugin, expected error: %v\n", err)
	require.Error(t, err)
}

// TestAvailablePlugins requires that PJRT CPU plugin be available.
func TestAvailablePlugins(t *testing.T) {
	plugins := AvailablePlugins()
	fmt.Printf("Available plugins: %v\n", plugins)
	require.NotEqualf(t, "", plugins[*FlagPluginName], "Can not find %q plugin", *FlagPluginName)
}

// TestSuppressAbseilLoggingHack never fails, since errors are simply logged.
// But we leave it here even if to be manually checked.
func TestSuppressAbseilLoggingHack(t *testing.T) {
	SuppressAbseilLoggingHack(func() { fmt.Println("SuppressAbseilLoggingHack call 1") })
	SuppressAbseilLoggingHack(func() { fmt.Println("SuppressAbseilLoggingHack call 2") })
}
