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
	"slices"
	"testing"
)

func init() {
	klog.InitFlags(nil)
}

// TestLoadPlatformCPU requires that PJRT CPU plugin be available.
func TestLoadPlatformCPU(t *testing.T) {
	platform := "CPU"
	ptr, err := LoadPlatformPlugin(platform)
	require.NoError(t, err)
	fmt.Printf("%s plugin v%d.%d\n", platform, ptr.pjrt_api_version.major_version, ptr.pjrt_api_version.minor_version)

	_, err = LoadPlatformPlugin("host")
	require.NoError(t, err) // Should be found using the alias.
}

// TestGetPlatforms requires that PJRT CPU plugin be available.
func TestGetPlatforms(t *testing.T) {
	platforms := GetPlatforms()
	fmt.Printf("Platforms with available plugins: %v\n", platforms)
	require.True(t, slices.Index(platforms, "CPU") != -1, "Can not find CPU plugin")
}
