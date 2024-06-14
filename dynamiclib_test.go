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
	"fmt"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"testing"
)

func init() {
	klog.InitFlags(nil)
}

// TestLoadPlatformCPU requires that PJRT CPU plugin be available.
func TestLoadPlatformCPU(t *testing.T) {
	plugin, err := loadNamedPlugin("cpu")
	require.NoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	// Checks cache..
	plugin2, err := loadNamedPlugin("cpu")
	require.NoError(t, err)

	require.Equal(t, plugin, plugin2)
}

// TestGetPlatforms requires that PJRT CPU plugin be available.
func TestGetPlatforms(t *testing.T) {
	plugins := AvailablePlugins()
	fmt.Printf("Available plugins: %v\n", plugins)
	require.NotEqual(t, "", plugins["cpu"], "Can not find \"cpu\" plugin")
}
