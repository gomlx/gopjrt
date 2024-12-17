package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"runtime"
	"slices"
	"unsafe"
)

func pjrtClientPlatformName(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformName_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_PlatformName(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_name, args.platform_name_size), nil
}

func pjrtClientPlatformVersion(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformVersion_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_PlatformVersion(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_version, args.platform_version_size), nil
}

func pjrtClientProcessIndex(plugin *Plugin, client *Client) (int, error) {
	args := C.new_PJRT_Client_ProcessIndex_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_ProcessIndex(plugin.api, args))
	if err != nil {
		return -1, err
	}
	return int(args.process_index), nil
}

func pjrtClientDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_Devices_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_Devices(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cDevices := cDataToSlice[*C.PJRT_Device](unsafe.Pointer(args.devices), int(args.num_devices))
	devices := make([]*Device, len(cDevices))
	for ii, d := range cDevices {
		devices[ii] = newDevice(client, d)
	}
	return devices, nil
}

func pjrtClientAddressableDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_AddressableDevices_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_AddressableDevices(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cDevices := cDataToSlice[*C.PJRT_Device](unsafe.Pointer(args.addressable_devices), int(args.num_addressable_devices))
	devices := make([]*Device, len(cDevices))
	for ii, d := range cDevices {
		devices[ii] = newDevice(client, d)
	}
	return devices, nil
}

// pjrtClientCompile compiles the program. Remember to make sure that the both the program and and compileOptionsProto
// are pinned until the C function returns.
func pjrtClientCompile(plugin *Plugin, client *Client, program []byte, programFormat string, compileOptionsProto []byte) (*LoadedExecutable, error) {
	// Create program structure.
	var cProgram *C.PJRT_Program
	cProgram = C.new_PJRT_Program()
	defer cFree(cProgram)
	cProgramFormat := C.CString(programFormat)
	defer cFree(cProgramFormat)
	cProgram.format = cProgramFormat
	cProgram.format_size = (C.size_t)(len(programFormat))
	cProgram.code = (*C.char)(unsafe.Pointer(unsafe.SliceData(program)))
	cProgram.code_size = (C.size_t)(len(program))

	// Create args for call.
	args := C.new_PJRT_Client_Compile_Args()
	defer cFree(args)
	args.client = client.client
	args.program = cProgram
	args.compile_options = (*C.char)(unsafe.Pointer(unsafe.SliceData(compileOptionsProto)))
	args.compile_options_size = (C.size_t)(len(compileOptionsProto))
	cErr := C.call_PJRT_Client_Compile(plugin.api, args)
	err := toError(plugin, cErr)
	if err != nil {
		return nil, err
	}
	return newLoadedExecutable(plugin, client, args.executable)
}

// Client manages the resources of one device: its buffers, compilation and execution of HLO code.
type Client struct {
	plugin                    *Plugin
	client                    *C.PJRT_Client
	platform, platformVersion string
	processIndex              int
	addressableDevices        []*Device
	allowBufferViews          bool
}

// newClient is called by Plugin.NewClient to create a new PJRT_Client wrapper.
func newClient(plugin *Plugin, options NamedValuesMap) (*Client, error) {
	// Create C.PJRT_Client object.
	args := C.new_PJRT_Client_Create_Args()
	defer cFree(args)
	var err error
	args.create_options, args.num_options, err = options.mallocArrayPJRT_NamedValue()
	if err != nil {
		return nil, errors.WithMessagef(err, "invalid options when creating a new pjrt.Client")
	}
	// No callback support yet, so we leave the various PJRT_KeyValue... fields empty.
	err = toError(plugin, C.call_PJRT_Client_Create(plugin.api, args))
	if err != nil {
		return nil, err
	}

	// Prepare Client object: not all initialization is fatal to the construction of the client.
	c := &Client{plugin: plugin, client: args.client}
	c.platform, err = pjrtClientPlatformName(plugin, c)
	if err != nil {
		// Non-fatal
		klog.Errorf("Failed to retrieve client platform name (plugin %s): %v", plugin, err)
	}
	c.platformVersion, err = pjrtClientPlatformVersion(plugin, c)
	if err != nil {
		// Non-fatal
		klog.Errorf("Failed to retrieve client platform version (plugin %s): %v", plugin, err)
	}
	c.processIndex, err = pjrtClientProcessIndex(plugin, c)
	if err != nil {
		// Non-fatal
		klog.Errorf("Failed to retrieve client process index (plugin %s): %v", plugin, err)
	}
	c.addressableDevices, err = pjrtClientAddressableDevices(plugin, c)
	if err != nil {
		// Fatal
		err = errors.WithMessagef(err, "failed to retrieve addressable devices for new client (plugin %s) -- can't use client with no addressable device", plugin)
		finalizeClient(c)
		return nil, err
	}

	// Register finalizer.
	runtime.SetFinalizer(c, finalizeClient)
	return c, nil
}

func finalizeClient(c *Client) {
	err := c.Destroy()
	if err != nil {
		klog.Errorf("Client.Destroy failed: %v", err)
	}
}

// Plugin returns the Plugin from which the Client was created.
func (c *Client) Plugin() *Plugin {
	return c.plugin
}

// Destroy the client, release resources, and Client is no longer valid.
// This is automatically called if Client is garbage collected.
func (c *Client) Destroy() error {
	if c.plugin == nil || c.client == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(c)
	args := C.new_PJRT_Client_Destroy_Args()
	defer cFree(args)
	args.client = c.client
	err := toError(c.plugin, C.call_PJRT_Client_Destroy(c.plugin.api, args))
	c.plugin = nil
	c.client = nil
	return err
}

// String implements fmt.Stringer.
func (c *Client) String() string {
	if c.client == nil {
		return "Invalid client"
	}

	pid := c.ProcessIndex()
	var pidStr string
	if pid == 0 {
		pidStr = "single-process"
	} else {
		pidStr = fmt.Sprintf("pid=%d", pid)
	}
	return fmt.Sprintf("Client[plugin=%q, platform=%q, %s]", c.plugin.Name(), c.Platform()+" - "+c.PlatformVersion(), pidStr)
}

// Platform returns the name of the client platform.
func (c *Client) Platform() string {
	return c.platform
}

// PlatformVersion returns the version of the client platform.
func (c *Client) PlatformVersion() string {
	return c.platformVersion
}

// ProcessIndex returns the process index of the client platform.
// Always 0 in single-process settings.
func (c *Client) ProcessIndex() int {
	return c.processIndex
}

// Devices returns a list of all devices visible to the runtime, including addressable
// // and non-addressable devices.
func (c *Client) Devices() ([]*Device, error) {
	return pjrtClientDevices(c.plugin, c)
}

// AddressableDevices returns a list of devices addressable to the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment (Client.ProcessIndex() == 0).
//
// The returned slice and the Devices are owned by the Client, don't change it.
func (c *Client) AddressableDevices() []*Device {
	return c.addressableDevices
}

// NumForDevice returns the "deviceNum" for the given device.
// The value deviceNum is an index to Client.AddressableDevices, and can be used in several other methods.
//
// It returns -1 if device not found in Client.AddressableDevices.
func (c *Client) NumForDevice(device *Device) int {
	for deviceNum, otherDevice := range c.addressableDevices {
		if device.localHardwareId == otherDevice.localHardwareId {
			return deviceNum
		}
	}
	return -1
}

// Compile turn a StableHLO program into a "LoadedExecutable" that is the executable runner.
//
// There are different formats of input, and many different compilation options [1], so this returns
// a CompilationConfig that must be furthered configured. At the very least the program must be given: see
// CompileConfig.WithComputation or CompileConfig.WithHLO. Then the call to CompileConfig.Done triggers
// the compilation into a "LoadedExecutable".
//
// [1] The original compilation options is defined as the proto CompileOptionsProto:
// https://github.com/openxla/xla/blob/main/xla/pjrt/compile_options.proto .
// But the proto itself is not documented, instead see documentation in the C++ xla::CompileOptions class defined in:
// https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_executable.h .
func (c *Client) Compile() *CompileConfig {
	return newCompileConfig(c)
}

// BufferFromHost creates an on-device buffer with the contents copied (optionally reused, if device is CPU) from
// the given host buffer.
//
// It returns a BufferFromHostConfig that must be furthered configured -- at least the host data to transfer must be given.
// Call BufferFromHostConfig.Done to trigger the transfer.
func (c *Client) BufferFromHost() *BufferFromHostConfig {
	return &BufferFromHostConfig{
		client:              c,
		device:              nil,
		hostBufferSemantics: PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
	}
}

// CreateViewOfDeviceBuffer creates a PJRT Buffer that is backed by storage on the same device given by the caller as flatData and shape.
//
// Different PJRT may have different requirements on alignment, but for the CPU PJRT the library provide
// AlignedAlloc and AlignedFree, that can be used to allocate the aligned storage space.
//
// Example of how a typical usage where the same buffer is reused as input in loop:
//
//	dtype := dtypes.Float32
//	dimensions := []int{batchSize, sequenceLength, 384}
//	rawData := pjrt.AlignedAlloc(dtype.SizeForDimensions(dimensions...), pjrt.BufferAlignment)
//	defer pjrt.AlignedFree(rawData)
//	buf := client.CreateViewOfDeviceBuffer(rawData, dtype, dimensions)
//	flat := unsafe.Slice((*float32)(storage), batchSize*sequenceLength*384)
//	for _, batch := range batches {
//		// ... set flat values
//		// ... use buf as input when executing a PJRT program.
//	}
//
// If device is not given (at most one can be given), the first device available for the client is used.
//
// The naming comes from PJRT and is unfortunate, since it's the name from PJRT's perspective (PJRT view of a
// users device buffer).
// Probably, it should have been better named by "ShareDeviceBuffer" or something similar.
//
// This may not be implemented for all hardware (or all PJRT plugins).
//
// This can be useful to avoid the copy of values, by mutating directly in the memory shared with PJRT, to be used
// as input to a computation.
//
// See: dtypes.SizeForDimensions() to calculate the size for an arbitrary shape; AlignedAlloc, AlignedFree and
// BufferAlignment (a constant with the required alignment size) to allocate and free aligned storage.
func (c *Client) CreateViewOfDeviceBuffer(rawData unsafe.Pointer, dtype dtypes.DType, dimensions []int, device ...*Device) (*Buffer, error) {
	var selectedDevice *Device
	if len(device) > 1 {
		return nil, errors.Errorf("only one device can be given to CreateViewOfDeviceBuffer, %d were given", len(device))
	} else if len(device) == 1 {
		selectedDevice = device[0]
	} else {
		devices := c.AddressableDevices()
		if len(devices) == 0 {
			return nil, errors.New("CreateViewOfDeviceBuffer can't find addressable device to transfer to")
		}
		selectedDevice = devices[0]
	}

	// Arena for memory allocations used by CGO.
	arena := getArenaFromPool()
	defer returnArenaToPool(arena)

	// Arguments to PJRT call.
	var args *C.PJRT_Client_CreateViewOfDeviceBuffer_Args
	args = arenaAlloc[C.PJRT_Client_CreateViewOfDeviceBuffer_Args](arena)
	args.struct_size = C.PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE
	args.client = c.client
	args.device_buffer_ptr = rawData
	args.element_type = C.PJRT_Buffer_Type(dtype)
	args.num_dims = C.size_t(len(dimensions))
	if args.num_dims > 0 {
		dims := arenaAllocSlice[C.int64_t](arena, int(args.num_dims))
		for ii, dim := range dimensions {
			dims[ii] = C.int64_t(dim)
		}
		args.dims = unsafe.SliceData(dims)
	}
	args.device = selectedDevice.cDevice
	err := toError(c.plugin, C.call_PJRT_Client_CreateViewOfDeviceBuffer(c.plugin.api, args))
	if err != nil {
		return nil, err
	}
	buffer := newBuffer(c, args.buffer)
	buffer.dims = slices.Clone(dimensions)
	buffer.dimsSet = true
	buffer.dtype = dtype
	buffer.dtypeSet = true
	return buffer, nil
}
