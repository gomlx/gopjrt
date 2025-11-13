package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func pjrtClientPlatformName(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformName_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_PlatformName(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_name, args.platform_name_size), nil
}

func pjrtClientPlatformVersion(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformVersion_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_PlatformVersion(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_version, args.platform_version_size), nil
}

func pjrtClientProcessIndex(plugin *Plugin, client *Client) (int, error) {
	args := C.new_PJRT_Client_ProcessIndex_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_ProcessIndex(plugin.api, args))
	if err != nil {
		return -1, err
	}
	return int(args.process_index), nil
}

func pjrtClientDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_Devices_Args()
	defer cFree(args)
	args.client = client.client.c
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
	args.client = client.client.c
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

func pjrtClientDefaultDeviceAssignment(plugin *Plugin, client *Client, numReplicas, numPartitions int) ([]int, error) {
	args := C.new_PJRT_Client_DefaultDeviceAssignment_Args()
	defer cFree(args)
	args.client = client.client.c
	args.num_replicas = C.int(numReplicas)
	args.num_partitions = C.int(numPartitions)
	assignmentSize := numReplicas * numPartitions
	args.default_assignment_size = (C.size_t)(assignmentSize)
	args.default_assignment = cMallocArray[C.int](assignmentSize)
	defer cFree(args.default_assignment)
	err := toError(plugin, C.call_PJRT_Client_DefaultDeviceAssignment(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cAssignment := cDataToSlice[C.int](unsafe.Pointer(args.default_assignment), assignmentSize)
	assignment := make([]int, assignmentSize)
	for i, v := range cAssignment {
		assignment[i] = int(v)
	}
	return assignment, nil
}

// pjrtClientCompile compiles the program. Make sure that both the program and the compileOptionsProto
// are pinned until the C function returns.
func pjrtClientCompile(plugin *Plugin, client *Client, program []byte, programFormat string,
	compileOptionsProto []byte) (*LoadedExecutable, error) {

	// Create the program struct.
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
	args.client = client.client.c
	args.program = cProgram

	if len(compileOptionsProto) != 0 {
		args.compile_options = (*C.char)(C.CBytes(compileOptionsProto))
		//args.compile_options = (*C.char)(unsafe.Pointer(unsafe.SliceData(compileOptionsProto)))
		args.compile_options_size = (C.size_t)(len(compileOptionsProto))
		defer cFree(args.compile_options)
	}
	cErr := C.call_PJRT_Client_Compile(plugin.api, args)
	runtime.KeepAlive(program) // Makes sure it is alive during the C call.
	err := toError(plugin, cErr)
	if err != nil {
		return nil, err
	}
	return newLoadedExecutable(plugin, client, args.executable)
}

// Client manages the resources of one device: its buffers, compilation and execution of HLO code.
type Client struct {
	plugin                    *Plugin
	client                    *clientC
	platform, platformVersion string
	processIndex              int
	addressableDevices        []*Device
	allowBufferViews          bool
}

type clientC struct {
	// c holds the pointer to the C/C++ structure.
	c *C.PJRT_Client
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

	// Prepare the Client object: not all initializations are fatal to the construction of the client.
	c := &Client{
		plugin: plugin,
		client: &clientC{c: args.client},
	}
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
		c.client.Destroy(plugin)
		return nil, err
	}

	// Register cleanup.
	runtime.AddCleanup(c, func(client *clientC) {
		err := client.Destroy(plugin)
		if err != nil {
			klog.Errorf("Failed to destroy client (plugin %s): %v", plugin, err)
		}
	}, c.client)
	return c, nil
}

// Plugin returns the Plugin from which the Client was created.
func (c *Client) Plugin() *Plugin {
	return c.plugin
}

func (client *clientC) Destroy(plugin *Plugin) error {
	if plugin == nil || client == nil || client.c == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(client)
	args := C.new_PJRT_Client_Destroy_Args()
	defer cFree(args)
	args.client = client.c
	err := toError(plugin, C.call_PJRT_Client_Destroy(plugin.api, args))
	client.c = nil
	return err
}

// IsValid returns if client has been properly created and not yet destroyed.
func (c *Client) IsValid() bool {
	return c != nil && c.client != nil && c.client.c != nil
}

// Destroy the client, release resources, and Client is no longer valid.
// This is automatically called if Client is garbage collected.
func (c *Client) Destroy() error {
	if c.plugin == nil || c.client == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(c)
	return c.client.Destroy(c.plugin)
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
	return fmt.Sprintf("Client[plugin=%q, platform=%q, %s, %d device(s)]",
		c.plugin.Name(), c.Platform()+" - "+c.PlatformVersion(), pidStr, len(c.addressableDevices))
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

// AllDevices returns a list of all devices visible to the runtime, including addressable
// and non-addressable devices.
//
// Usually, you want to use the AddressableDevices method instead.
func (c *Client) AllDevices() ([]*Device, error) {
	return pjrtClientDevices(c.plugin, c)
}

// AddressableDevices returns a list of devices addressable to the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment (Client.ProcessIndex() == 0).
//
// The Client owns the returned slice and the devices. Don't change them.
func (c *Client) AddressableDevices() []*Device {
	return c.addressableDevices
}

// NumDevices returns the number of addressable devices.
func (c *Client) NumDevices() int {
	return len(c.addressableDevices)
}

// DefaultDeviceAssignment for the given number of replicas and partitions.
//
// Replicas refer to data parallelism: the number of identical copies of the program that will be run on
// different data.
//
// Partitions refer to model parallelism: the number of independent copies of the program that will be run on
// different parts of the model. For SPMD programs, this is always 1.
//
// The returned slice is of length numReplicas * numPartitions.
func (c *Client) DefaultDeviceAssignment(numReplicas, numPartitions int) ([]int, error) {
	return pjrtClientDefaultDeviceAssignment(c.plugin, c, numReplicas, numPartitions)
}

// NumForDevice returns the "deviceNum" for the given device.
// The value deviceNum is an index to Client.AddressableDevices, and can be used in several other methods.
//
// It returns -1 if the device is not found in Client.AddressableDevices.
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
