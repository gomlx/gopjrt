package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"fmt"

	"k8s.io/klog/v2"
)

func pjrtDeviceLocalHardwareId(device *Device) (int, error) {
	args := C.new_PJRT_Device_LocalHardwareId_Args()
	defer cFree(args)
	args.device = device.cDevice
	err := toError(device.plugin, C.call_PJRT_Device_LocalHardwareId(device.plugin.api, args))
	if err != nil {
		return -1, err
	}
	return int(args.local_hardware_id), nil
}

func pjrtDeviceDescriptionProcessIndex(dDesc *DeviceDescription) (int, error) {
	args := C.new_PJRT_DeviceDescription_ProcessIndex_Args()
	defer cFree(args)
	args.device_description = dDesc.deviceDescription
	err := toError(dDesc.plugin, C.call_PJRT_DeviceDescription_ProcessIndex(dDesc.plugin.api, args))
	if err != nil {
		return -1, err
	}
	return int(args.process_index), nil
}

// Device is a lightweight reference to a Device managed by a Client -- it doesn't own the underlying object.
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
// Device-Specific Operations: Some PJRT operations (like querying device attributes or transferring data to/from the device)
// are device-specific and operate on individual PjrtDevice objects (obtained from the PjrtClient_Devices list).
type Device struct {
	plugin          *Plugin
	cDevice         *C.PJRT_Device // (PJRT) `device` has the same lifetime as (PJRT) `client`. It is owned by (PJRT) `client`.
	localHardwareId int
}

// newDevice create a new Device reference.
func newDevice(client *Client, device *C.PJRT_Device) *Device {
	d := &Device{plugin: client.plugin, cDevice: device}
	var err error
	d.localHardwareId, err = pjrtDeviceLocalHardwareId(d)
	if err != nil {
		klog.Errorf("Failed to get device local_hardware_id for client %s: %v", client, err)
	}
	return d
}

// IsAddressable returns whether the device is addressable by this client.
func (d *Device) IsAddressable() (bool, error) {
	args := C.new_PJRT_Device_IsAddressable_Args()
	defer cFree(args)
	args.device = d.cDevice
	err := toError(d.plugin, C.call_PJRT_Device_IsAddressable(d.plugin.api, args))
	if err != nil {
		return false, err
	}
	return bool(args.is_addressable), nil
}

// LocalHardwareID returns an opaque hardware ID, e.g., the CUDA device number. In general, not guaranteed
// to be dense, and -1 if undefined.
func (d *Device) LocalHardwareID() int {
	return d.localHardwareId
}

// GetDescription get a DeviceDescription object associated with this device.
func (d *Device) GetDescription() (*DeviceDescription, error) {
	args := C.new_PJRT_Device_GetDescription_Args()
	defer cFree(args)
	args.device = d.cDevice
	err := toError(d.plugin, C.call_PJRT_Device_GetDescription(d.plugin.api, args))
	if err != nil {
		return nil, err
	}
	return newDeviceDescription(d.plugin, args.device_description), nil
}

// DeviceDescription may be associated with an actual device
// (via PJRT_Device_GetDescription), but they can also be used to describe a
// device that isn't currently available to the plugin. This is useful for
// compiling executables without hardware available, which can then be
// serialized and written somewhere durable, and then loaded and run on actual
// hardware later.
type DeviceDescription struct {
	plugin            *Plugin
	deviceDescription *C.PJRT_DeviceDescription
	processIndex      int
}

// newDeviceDescription create a new Device reference.
func newDeviceDescription(plugin *Plugin, deviceDescription *C.PJRT_DeviceDescription) *DeviceDescription {
	dDesc := &DeviceDescription{plugin: plugin, deviceDescription: deviceDescription}
	var err error
	dDesc.processIndex, err = pjrtDeviceDescriptionProcessIndex(dDesc)
	if err != nil {
		klog.Errorf("Failed to get process index for devicedescription for client %s: %v", dDesc.plugin, err)
	}
	return dDesc
}

// ProcessIndex returns the index of the process that this device belongs to, i.e. is addressable
// from. This is not always identical to PJRT_Client_ProcessIndex in a
// multi-process setting, where each client can see devices from all
// processes, but only a subset of them are addressable and have the same
// process_index as the client.
func (dDesc *DeviceDescription) ProcessIndex() int {
	return dDesc.processIndex
}

// A vendor-dependent string that uniquely identifies the kind of device,
// e.g., "Tesla V100-SXM2-16GB".

// DebugString suitable for logging when errors occur.
// Should be verbose enough to describe the current device unambiguously.
func (dDesc *DeviceDescription) DebugString() string {
	args := C.new_PJRT_DeviceDescription_DebugString_Args()
	defer cFree(args)
	args.device_description = dDesc.deviceDescription
	err := toError(dDesc.plugin, C.call_PJRT_DeviceDescription_DebugString(dDesc.plugin.api, args))
	if err != nil {
		return fmt.Sprintf("DeviceDescription failed to retrieve debug string: %v", err)
	}
	return cCharArray(args.debug_string, args.debug_string_size)
}
