//go:build !linux

package pjrt

// isCuda tries to guess that the plugin named is associated with Nvidia Cuda, to apply the corresponding hacks.
func isCuda(name string) bool { return false }

// hasNvidiaGPU tries to guess if there is an actual Nvidia GPU installed (as opposed to only the drivers/PJRT
// file installed, but no actual hardware).
// It does that by checking for the presence of the device files in /dev/nvidia*.
func hasNvidiaGPU() bool { return false }

// cudaPluginCheckDrivers issues a warning on cuda plugins if it cannot find the corresponding nvidia library files.
// It should be called after the named plugin is loaded.
//
// This is helpful to try to sort out the mess of path for nvidia libraries.
// Sadly, NVidia drivers are badly organized at multiple levels -- search to see how many questions there are related
// to where/how to install to CUDA libraries.
//
// To disable this check set GOPJRT_CUDA_CHECKS=no or GOPJRT_CUDA_CHECKS=0.
func cudaPluginCheckDrivers(name string) {
	return
}
