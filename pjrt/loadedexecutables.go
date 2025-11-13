package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"

PJRT_Error* ExecuteAndWait(const PJRT_Api *api, PJRT_LoadedExecutable_Execute_Args* args) {
	PJRT_Error *err = api->PJRT_LoadedExecutable_Execute(args);
	if (err) {
		return err;
	}

	if (args->device_complete_events) {
		// Wait for devices to complete executions.
		for (int ii = 0; ii < args->num_devices; ii++) {
			PJRT_Event_Await_Args event_args = {0};
			event_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
			event_args.event = args->device_complete_events[ii];
			err = api->PJRT_Event_Await(&event_args);
			PJRT_Event_Destroy_Args efree_args;
			efree_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
			efree_args.event = args->device_complete_events[ii];
			api->PJRT_Event_Destroy(&efree_args);
			if (err) {
				return err;
			}
		}
	}
	return NULL;
}


*/
import "C"
import (
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// LoadedExecutable is a reference to a compiled program ready to be executed.
//
// All public attributes are read-only.
type LoadedExecutable struct {
	wrapper *loadedExecutableC
	plugin  *Plugin
	client  *Client

	// executable is extracted as soon as LoadedExecutable is created, and with it all its fields.
	executable *Executable

	// Name of the executable.
	Name string

	// NumOutputs of the executable.
	NumOutputs int

	// numReplicas/numPartitions = 1,1 for single device execution, the default.
	// For SPMD (Single-Data, Multiple-Data), numPartitions=1.
	numReplicas, numPartitions int
	deviceAssignment           []int
	isPortable                 bool

	// OnDeviceMemoryUsageStats, OnHostMemoryUsageStats can be used to estimate the required memory usage for the executable on device (and on host).
	OnDeviceMemoryUsageStats, OnHostMemoryUsageStats ExecutableMemoryUsageStats
}

// loadedExecutableC wraps the C pointer, so we can use runtime.addCleanUp.
type loadedExecutableC struct {
	c      *C.PJRT_LoadedExecutable
	plugin *Plugin
}

func (wrapper *loadedExecutableC) Destroy() error {
	if wrapper == nil || wrapper.plugin == nil || wrapper.c == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(wrapper)
	args := C.new_PJRT_LoadedExecutable_Destroy_Args()
	defer cFree(args)
	args.executable = wrapper.c
	err := toError(wrapper.plugin, C.call_PJRT_LoadedExecutable_Destroy(wrapper.plugin.api, args))
	wrapper.plugin = nil
	wrapper.c = nil
	numLoadedExecutables.Add(-1)
	return err
}

var numLoadedExecutables atomic.Int64

// LoadedExecutablesAlive returns a count of the numbers of LoadedExecutables currently in memory and tracked by gopjrt.
func LoadedExecutablesAlive() int64 {
	return numLoadedExecutables.Load()
}

// newLoadedExecutable creates LoadedExecutable and registers it for freeing.
func newLoadedExecutable(plugin *Plugin, client *Client, cLoadedExecutable *C.PJRT_LoadedExecutable) (*LoadedExecutable, error) {
	e := &LoadedExecutable{
		plugin:        plugin,
		client:        client,
		wrapper:       &loadedExecutableC{c: cLoadedExecutable, plugin: plugin},
		numPartitions: 1,
		numReplicas:   1,
	}
	numLoadedExecutables.Add(1)
	runtime.AddCleanup(e, func(e *loadedExecutableC) {
		err := e.Destroy()
		if err != nil {
			klog.Errorf("Failed to destroy pjrt.LoadedExecutable: %+v", err)
		}
	}, e.wrapper)

	// Gather information about executable:
	var err error
	e.executable, err = e.getExecutable()
	if err != nil {
		e.destroyOrLog()
		return nil, errors.WithMessagef(err, "failed to GetExecutable from compiled LoadedExecutable")
	}
	e.Name, err = e.executable.Name()
	if err != nil {
		e.destroyOrLog()
		return nil, errors.WithMessagef(err, "failed to get Executable.Name from compiled LoadedExecutable")
	}
	e.NumOutputs, err = e.executable.NumOutputs()
	if err != nil {
		e.destroyOrLog()
		return nil, errors.WithMessagef(err, "failed to Executable.NumOutputs from compiled LoadedExecutable")
	}

	e.OnDeviceMemoryUsageStats, e.OnHostMemoryUsageStats, err = e.executable.GetMemoryStats()
	if err != nil {
		e.destroyOrLog()
		return nil, errors.WithMessagef(err, "failed to Executable.GetMemoryStats from compiled LoadedExecutable")
	}
	return e, nil
}

// Destroy the LoadedExecutable, release resources, and LoadedExecutable is no longer valid.
// This is automatically called if LoadedExecutable is garbage collected.
func (e *LoadedExecutable) Destroy() error {
	if e == nil || e.plugin == nil || e.wrapper == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(e)
	if e.executable != nil {
		e.executable.destroyOrLog()
	}
	err := e.wrapper.Destroy()
	e.plugin = nil
	e.wrapper = nil
	return err
}

// destroyOrLog destroys the LoadedExecutable and log any errors.
func (e *LoadedExecutable) destroyOrLog() {
	err := e.Destroy()
	if err != nil {
		klog.Errorf("LoadedExecutable.Destroy failed: %v", err)
	}
}

// getExecutable returns the Executable associated with the LoadedExecutable.
func (e *LoadedExecutable) getExecutable() (*Executable, error) {
	if e == nil || e.plugin == nil || e.wrapper == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_LoadedExecutable_GetExecutable_Args()
	defer cFree(args)
	args.loaded_executable = e.wrapper.c
	err := toError(e.plugin, C.call_PJRT_LoadedExecutable_GetExecutable(e.plugin.api, args))
	if err != nil {
		return nil, err
	}
	return newExecutable(e.plugin, args.executable), nil
}

// GetDeviceAssignment returns the device assignment of the executable.
//
// This is used when using multiple-devices. The assignment is a list of device indices, ordered by replica first
// and then by partition number.
//
// For example, if the executable 2 replicas and 2 partitions, and the assignment is [0, 1, 2, 3], that means:
//
// - Partition 0 uses devices [0, 2] for its replicas.
// - Partition 1 uses devices [1, 3] for its replicas.
//
// If the number of partitions is 0, this is working in SPMD (single-program, multiple-data), and there are no
// partitions, all devices are replicas.
//
// If computation was compiled in a portable fashion, the assignment is nil. See IsPortable.
func (e *LoadedExecutable) GetDeviceAssignment() (numReplicas, numPartitions int, assignment []int, err error) {
	if e == nil || e.plugin == nil || e.wrapper == nil {
		return 0, 0, nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	return e.numReplicas, e.numPartitions, e.deviceAssignment, nil
}

// IsPortable returns whether the computation was compiled to be device-portable -- it can run on any device.
func (e *LoadedExecutable) IsPortable() bool {
	return e.isPortable
}

// Execute the compiled computation. It returns an ExecutionConfig for further configuration.
// Call ExecutionConfig.Done and the computation is executed.
//
// It provides good defaults, so in the common case nothing else is needed:
//
// - Using the first addressable device.
// - All input buffers marked as not-donated (see discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation) input buffers.
//
// See ExecutionConfig for more details and options.
//
// Example:
//
//	outputBuffers, err := loadedExec.Execute(inputBuffer).Done()
//
// Multiple-devices execution (SPMD or MPMD): if using multiple devices, the inputs slice is split equally, one part
// per device assignment (see GetDeviceAssignment). Similarly, Done will return a slice of buffers, one per device.
// Care must be taken to use the outputs in the correct order.
//
// Example: if executing f(x,y) on two replicas, you should call Execute(x_0, y_0, x_1, y_1), where f(x_0, y_0)
// will be executed on the first replica and f(x_1, y_1) on the second replica.
func (e *LoadedExecutable) Execute(inputs ...*Buffer) *ExecutionConfig {
	c := &ExecutionConfig{
		executable: e,
		inputs:     inputs,
	}
	c.DonateNone()
	if e.isPortable {
		// Default to the first addressable device.
		_ = c.OnDeviceByNum(0)
	}
	return c
}

// ExecutionConfig holds the configuration for executing a LoadedExecutable.
// It is created with LoadedExecutable.Execute.
//
// After configuring it, call Done to actually trigger the execution.
//
// TODO: add support for multi-device execution, with some inputs shared across devices, and some per-device specific.
type ExecutionConfig struct {
	executable         *LoadedExecutable
	onDevice           *Device
	inputs             []*Buffer
	nonDonatableInputs []int

	// portableDevice is the device to execute the computation on, if it is portable.
	portableDevice int

	// err saves an error during the configuration.
	err error
}

// OnDevice selects which devices to execute.
// Usually only 1, but more than one can be configured.
//
// The default is to use the first addressable device.
// See also OnDeviceByNum.
func (c *ExecutionConfig) OnDevice(device *Device) *ExecutionConfig {
	if c.err != nil {
		return c
	}
	if device == nil {
		c.err = errors.New("LoadedExecutable.Execute().OnDevice() given a nil device")
		return c
	}
	addressable, err := device.IsAddressable()
	if err != nil {
		c.err = errors.WithMessagef(err, "LoadedExecutable.Execute().OnDevice() failed to check whether device is addressable")
		return c
	}
	if !addressable {
		c.err = errors.New("LoadedExecutable.Execute().OnDevice() given a non addressable device")
		return c
	}
	c.onDevice = device
	return c
}

// OnDeviceByNum selects which devices to execute.
// The devicesNum point to the device in the list returned by Client.AddressableDevices.
// Usually only 1, but more than one can be configured.
//
// The default is to use the first addressable device.
//
// This is only used for portable executables (see LoadedExecutable.IsPortable) that execute on exactly one device.
// For multi-device computations, or if the device was specified during the compilation, setting the device
// will lead to an error.
//
// See also OnDevice.
func (c *ExecutionConfig) OnDeviceByNum(deviceNum int) *ExecutionConfig {
	if c.err != nil {
		return c
	}
	addressableDevices := c.executable.client.addressableDevices
	if deviceNum < 0 || deviceNum >= len(addressableDevices) {
		c.err = errors.Errorf("LoadedExecutable.Execute().OnDevice() invalid deviceNum=%d, only %d addressable devices available", deviceNum, len(addressableDevices))
		return c
	}
	return c.OnDevice(addressableDevices[deviceNum])
}

// DonateAll marks all inputs to be "donated".
//
// Donated inputs become invalid after the execution, they are automatically destroyed.
// Often donated arguments are also the output of a computation and are updated in place.
// See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) DonateAll() *ExecutionConfig {
	c.nonDonatableInputs = nil
	return c
}

// DonateNone makes all inputs to be marked as non-donatable. This is the default.
//
// Donated inputs become invalid after the execution. Often donated arguments are also the output of a computation
// and are updated in place. See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) DonateNone() *ExecutionConfig {
	c.nonDonatableInputs = make([]int, len(c.inputs))
	for ii := range c.inputs {
		c.nonDonatableInputs[ii] = ii
	}
	return c
}

// Donate marks the inputs (referred to its indices) to be donated.
//
// This can be called more than once for different inputsIndices.
//
// Donated inputs become invalid after the execution, they are automatically destroyed.
// Often donated arguments are also the output of a computation and are updated in place.
// See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) Donate(inputsIndices ...int) *ExecutionConfig {
	c.nonDonatableInputs = slices.DeleteFunc(c.nonDonatableInputs, func(i int) bool {
		return slices.Index(inputsIndices, i) != -1
	})
	return c
}

// SetDonate set the "donate" status of all inputs in one call. The default is no input is donated.
//
// Donated inputs become invalid after the execution, they are automatically destroyed.
// Often donated arguments are also the output of a computation and are updated in place.
// See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) SetDonate(donate []bool) *ExecutionConfig {
	if c.err != nil {
		return c
	}
	if len(donate) != len(c.inputs) {
		c.err = errors.Errorf("LoadedExecutable.Execute().SetDonate() requires one value for each input, but there are %d inputs, and %d donate values given", len(c.inputs), len(donate))
		return c
	}
	c.nonDonatableInputs = make([]int, 0, len(c.inputs))
	for idx, donateIdx := range donate {
		if !donateIdx {
			c.nonDonatableInputs = append(c.nonDonatableInputs, idx)
		}
	}
	return c
}

// Done triggers the execution of the compiled computation.
func (c *ExecutionConfig) Done() ([]*Buffer, error) {
	if c.err != nil {
		return nil, c.err
	}
	e := c.executable
	plugin := e.plugin

	if plugin == nil || e.wrapper == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)

	// Dimensions of inputs/outputs.
	numDevices := e.numReplicas * e.numPartitions
	numInputs := len(c.inputs)
	if numInputs%numDevices != 0 {
		return nil, errors.Errorf("LoadedExecutable.Execute() requires that the number of inputs be "+
			"divisible by the number of devices, but got %d inputs and %d devices", numInputs, numDevices)
	}
	numInputsPerDevice := numInputs / numDevices
	numOutputsPerDevice := e.NumOutputs
	numOutputs := numOutputsPerDevice * numDevices

	// Allocations that CGO will use.
	// Except if the number of inputs/outputs is very large, used the default arena size.
	minSize := (numInputs+numOutputs)*3*8 /*pointer size*/ + 1024
	arena := plugin.getArena(minSize)
	defer plugin.returnArena(arena)

	// Create arguments structures for call to Execute.
	var args *C.PJRT_LoadedExecutable_Execute_Args
	args = arenaAlloc[C.PJRT_LoadedExecutable_Execute_Args](arena)
	args.struct_size = C.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
	args.executable = e.wrapper.c
	args.num_devices = C.size_t(numDevices)
	if e.isPortable {
		if numDevices > 1 {
			return nil, errors.Errorf("invalid number of devices for portable executable, portable "+
				"executables only work for one device, got %d devices", numDevices)
		}
		if c.onDevice == nil {
			return nil, errors.Errorf("LoadedExecutable.Execute() requires that OnDevice to be set to" +
				" non-nil device before Done")
		}
		args.execute_device = c.onDevice.cDevice
	} else {
		if c.onDevice != nil {
			return nil, errors.Errorf("LoadedExecutable.Execute(): non-portable computation cannot set " +
				"OnDevice or OnDeviceNum: the device(s) was(were) determined during the compilation")
		}
		args.execute_device = nil
	}

	var options *C.PJRT_ExecuteOptions
	options = arenaAlloc[C.PJRT_ExecuteOptions](arena) // Extra args that for some reason(?) go on a separate struct.
	options.struct_size = C.PJRT_ExecuteOptions_STRUCT_SIZE
	args.options = options

	// Configure (non-)donatable inputs.
	if len(c.nonDonatableInputs) > 0 {
		options.num_non_donatable_input_indices = C.size_t(len(c.nonDonatableInputs))
		nonDonatableIndices := arenaAllocSlice[C.int64_t](arena, len(c.nonDonatableInputs))
		for ii := range nonDonatableIndices {
			nonDonatableIndices[ii] = C.int64_t(c.nonDonatableInputs[ii])
		}
		options.non_donatable_input_indices = &nonDonatableIndices[0]
	}

	// Inputs organized per device.
	args.num_args = C.size_t(numInputsPerDevice)
	if args.num_args > 0 {
		args.argument_lists = allocatePerDeviceBufferListWithArena(arena, numDevices, numInputsPerDevice, c.inputs)
		if args.argument_lists == nil {
			return nil, errors.Errorf("LoadedExecutable.Execute() failed to allocate argument_lists")
		}
	}

	// For some reason the line below doesn't work. I think something is wrong with PJRT ... but I'm not sure.
	if numOutputs > 0 {
		args.output_lists = allocatePerDeviceBufferListWithArena(arena, numDevices, numOutputsPerDevice, nil)
	}

	// Create events to wait for the end of execution: leaving this as NULL is allowed, but what happens then
	// (does it wait or not, and then what?) is not documented in PJRT.
	perDeviceEvents := arenaAllocSlice[*C.PJRT_Event](arena, numDevices)
	args.device_complete_events = (**C.PJRT_Event)(unsafe.SliceData(perDeviceEvents))

	err := toError(e.plugin, C.ExecuteAndWait(e.plugin.api, args))
	if err != nil {
		return nil, err
	}

	// We only support one device for now, so we return the results from the first device.
	outputs := make([]*Buffer, numOutputs)
	outputBuffers := unsafe.Slice(*args.output_lists, numOutputs)
	for ii := range outputs {
		outputs[ii] = newBuffer(e.client, outputBuffers[ii])
	}

	// Destroy donated inputs, since they are no longer valid.
	for idx, input := range c.inputs {
		if c.nonDonatableInputs == nil || slices.Index(c.nonDonatableInputs, idx) == -1 {
			err := input.Destroy()
			if err != nil {
				err = errors.WithMessagef(err, "LoadedExecutable.Execute().Done() failed to destroy donated input %d: %v", idx, err)
				return nil, err
			}
		}
	}
	return outputs, nil
}

// Allocate [numDevices][numBuffers]*Buffer C 2D-array to be used by PJRT C API.
//
// If buffers != nil it is used to initialize the newly allocated 2D-array. If buffers != nil it must be of size
// numDevices * numBuffersPerDevice.
func allocatePerDeviceBufferListWithArena(
	arena *arenaContainer, numDevices int, numBuffersPerDevice int, buffers []*Buffer) ***C.PJRT_Buffer {
	// Top level:
	bufferIdx := 0
	perDevice := arenaAllocSlice[**C.PJRT_Buffer](arena, numDevices)
	for deviceIdx := range perDevice {
		deviceBuffers := arenaAllocSlice[*C.PJRT_Buffer](arena, numBuffersPerDevice)
		perDevice[deviceIdx] = &deviceBuffers[0]
		if buffers != nil {
			for deviceBufferIdx := range deviceBuffers {
				buf := buffers[bufferIdx]
				bufferIdx++
				if buf == nil || buf.wrapper == nil {
					deviceBuffers[deviceBufferIdx] = nil
					continue
				}
				deviceBuffers[deviceBufferIdx] = buf.wrapper.c
			}
		}
	}
	return &perDevice[0]
}
