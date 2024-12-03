package pjrt

import (
	"k8s.io/klog/v2"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// This file includes the required hacks to support Nvidia's Cuda based PJRT plugins.

// isCuda tries to guess that the plugin named is associated with Nvidia Cuda, to apply the corresponding hacks.
func isCuda(name string) bool {
	return strings.Index(strings.ToUpper(name), "CUDA") != -1 ||
		strings.Index(strings.ToUpper(name), "NVIDIA") != -1
}

var hasNvidiaGPUCache *bool

// hasNvidiaGPU tries to guess if there is an actual Nvidia GPU installed (as opposed to only the drivers/PJRT
// file installed, but no actual hardware).
// It does that by checking for the presence of the device files in /dev/nvidia*.
func hasNvidiaGPU() bool {
	if hasNvidiaGPUCache != nil {
		return *hasNvidiaGPUCache
	}
	matches, err := filepath.Glob("/dev/nvidia*")
	if err != nil {
		klog.Errorf("Failed to figure out if there is an Nvidia GPU installed while searching for files matching \"/dev/nvidia*\": %v", err)
		return false
	}
	hasGPU := len(matches) > 0
	if !hasGPU {
		klog.Infof("No NVidia devices found matching \"/dev/nvidia*\", assuming there are no GPU cards installed in the system. " +
			"To force the attempt to use the \"cuda\" PJRT, use its absolute path.")
	}
	hasNvidiaGPUCache = &hasGPU
	return hasGPU
}

// cudaPluginCheckDrivers issues warning on cuda plugins if it cannot find the corresponding nvidia library files.
// It should be called after the named plugin is loaded.
//
// This is helpful to try to sort out the mess of path for nvidia libraries. It's something really badly organized
// at multiple levels (just search to see how many questions there are related to where/how install to CUDA libraries).
//
// To disable this check set GOPJRT_CUDA_CHECKS=no or GOPJRT_CUDA_CHECKS=0.
func cudaPluginCheckDrivers(name string) {
	cudaChecks := os.Getenv("GOPJRT_CUDA_CHECKS")
	if cudaChecks != "" && cudaChecks != "1" && strings.ToUpper(cudaChecks) != "TRUE" && strings.ToUpper(cudaChecks) != "YES" {
		// Checks disabled.
		return
	}
	if !isCuda(name) {
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
		"If you have things correctly set up, and you want to disable this warning, just set the environment variable `export GOPJR_CUDA_CHECKS=0`. "+
		"Alternatively, see gopjrt's cmd/install_cuda.sh for an quick default installation script.",
		plugin.Path(), nvidiaPath, plugin.Path(), nvidiaPath)
}
