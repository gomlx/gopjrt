package gopjrt

import "C"

// Device is a lightweight reference to a Device managed by a Client.
//
// (Explanation by Gemini)
// The meaning of Device in PJRT/XLA is a bit nuanced, it refers to an individual unit of processing capable of executing XLA computations.
//
// Here's how it breaks down in different scenarios:
//
// - Single-GPU System: In a computer with a single GPU, the device typically corresponds to that entire GPU.
// - Multi-GPU System: In a system with multiple GPUs, each individual GPU is considered a separate device. You would typically create multiple PjrtClient instances, each associated with a different GPU device.
// - TPU Pods/Slices: On Google Cloud TPUs, a device can represent either a whole TPU chip (with multiple cores) or a slice of a TPU chip (a subset of cores).
// - CPU: In the context of the CPU plugin, the device usually refers to the entire CPU or a specific NUMA node (a group of CPU cores with faster access to a particular region of memory).
//
// Device Selection: When creating a PjrtClient, you can either let PjRT choose a default device or explicitly specify which device to use.
// The PjrtClient_Devices function can help you list the available devices.
//
// Device-Specific Operations: Some PjRT operations (like querying device attributes or transferring data to/from the device)
// are device-specific and operate on individual PjrtDevice objects (obtained from the PjrtClient_Devices list).
type Device struct {
	plugin *Plugin
	client *Client
	device *C.PJRT_Device // Pointer owned by PJRT, no need to destroy.
}

// newDevice create a new Device reference.
func newDevice(plugin *Plugin, client *Client, device *C.PJRT_Device) *Device {
	return &Device{plugin: plugin, client: client, device: device}
}
