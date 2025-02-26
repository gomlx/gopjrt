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
			efree_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
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
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"
)

// LoadedExecutable is a reference to a compiled program ready to be executed.
//
// All public attributes are read-only.
type LoadedExecutable struct {
	cLoadedExecutable *C.PJRT_LoadedExecutable
	plugin            *Plugin
	client            *Client

	// executable is extracted as soon as LoadedExecutable is created, and with it all its fields.
	executable *Executable

	// Name of the executable.
	Name string

	// NumOutputs of the executable.
	NumOutputs int

	// OnDeviceMemoryUsageStats, OnHostMemoryUsageStats can be used to estimate the required memory usage for the executable on device (and on host).
	OnDeviceMemoryUsageStats, OnHostMemoryUsageStats ExecutableMemoryUsageStats
}

var numLoadedExecutables atomic.Int64

// LoadedExecutablesAlive returns a count of the numbers of LoadedExecutables currently in memory and tracked by gopjrt.
func LoadedExecutablesAlive() int64 {
	return numLoadedExecutables.Load()
}

// newLoadedExecutable creates LoadedExecutable and registers it for freeing.
func newLoadedExecutable(plugin *Plugin, client *Client, cLoadedExecutable *C.PJRT_LoadedExecutable) (*LoadedExecutable, error) {
	e := &LoadedExecutable{
		plugin:            plugin,
		client:            client,
		cLoadedExecutable: cLoadedExecutable,
	}
	numLoadedExecutables.Add(1)
	runtime.SetFinalizer(e, func(e *LoadedExecutable) { e.destroyOrLog() })

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
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(e)
	if e.executable != nil {
		e.executable.destroyOrLog()
	}
	args := C.new_PJRT_LoadedExecutable_Destroy_Args()
	defer cFree(args)
	args.executable = e.cLoadedExecutable
	err := toError(e.plugin, C.call_PJRT_LoadedExecutable_Destroy(e.plugin.api, args))
	e.plugin = nil
	e.cLoadedExecutable = nil

	numLoadedExecutables.Add(-1)
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
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_LoadedExecutable_GetExecutable_Args()
	defer cFree(args)
	args.loaded_executable = e.cLoadedExecutable
	err := toError(e.plugin, C.call_PJRT_LoadedExecutable_GetExecutable(e.plugin.api, args))
	if err != nil {
		return nil, err
	}
	return newExecutable(e.plugin, args.executable), nil
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
func (e *LoadedExecutable) Execute(inputs ...*Buffer) *ExecutionConfig {
	c := &ExecutionConfig{
		executable: e,
		inputs:     inputs,
	}
	c.DonateNone()
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
	devices            []*Device
	inputs             []*Buffer
	nonDonatableInputs []int

	// err saves an error during the configuration.
	err error
}

// OnDevices selects which devices to execute.
// Usually only 1, but more than one can be configured.
//
// The default is to use the first addressable device.
// See also OnDevicesByNum.
func (c *ExecutionConfig) OnDevices(devices ...*Device) *ExecutionConfig {
	if c.err != nil {
		return c
	}
	if len(devices) == 0 {
		// Trivial case.
		c.devices = nil
		return c
	}
	c.devices = make([]*Device, len(devices))
	for ii, device := range devices {
		if device == nil {
			c.err = errors.New("LoadedExecutable.Execute().OnDevices() given a nil device")
			return c
		}
		addressable, err := device.IsAddressable()
		if err != nil {
			c.err = errors.WithMessagef(err, "LoadedExecutable.Execute().OnDevices() failed to check whether device is addressable")
			return c
		}
		if !addressable {
			c.err = errors.New("LoadedExecutable.Execute().OnDevices() given a non addressable device")
			return c
		}
		c.devices[ii] = device
	}
	return c
}

// OnDevicesByNum selects which devices to execute.
// The devicesNum point to the device in the list returned by Client.AddressableDevices.
// Usually only 1, but more than one can be configured.
//
// The default is to use the first addressable device.
// See also OnDevices.
func (c *ExecutionConfig) OnDevicesByNum(devicesNum ...int) *ExecutionConfig {
	if c.err != nil {
		return c
	}
	if len(devicesNum) == 0 {
		c.devices = nil
		return c
	}
	addressableDevices := c.executable.client.addressableDevices
	devices := make([]*Device, len(devicesNum))
	for ii, deviceNum := range devicesNum {
		if deviceNum < 0 || deviceNum >= len(addressableDevices) {
			c.err = errors.Errorf("LoadedExecutable.Execute().OnDevices() invalid deviceNum=%d, only %d addressable devices available", deviceNum, len(addressableDevices))
			return c
		}
		devices[ii] = addressableDevices[deviceNum]
	}
	return c.OnDevices(devices...)
}

// DonateAll marks all inputs to be "donated".
//
// Donated inputs become invalid after the execution. Often donated arguments are also the output of a computation
// and are updated in place. See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
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
// Donated inputs become invalid after the execution. Often donated arguments are also the output of a computation
// and are updated in place. See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) Donate(inputsIndices ...int) *ExecutionConfig {
	c.nonDonatableInputs = slices.DeleteFunc(c.nonDonatableInputs, func(i int) bool {
		return slices.Index(inputsIndices, i) != -1
	})
	return c
}

// SetDonate set the donate status of all inputs in one call. The default is no input is donated.
//
// Donated inputs become invalid after the execution. Often donated arguments are also the output of a computation
// and are updated in place. See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
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

func (c *ExecutionConfig) Done() ([]*Buffer, error) {
	if c.err != nil {
		return nil, c.err
	}
	e := c.executable
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)

	// If no devices were given, use the first addressable one.
	if len(c.devices) == 0 {
		devices := e.client.AddressableDevices()
		if len(devices) == 0 {
			return nil, errors.New("LoadedExecutable.Execute can't find addressable device to execute")
		}
		c.devices = []*Device{devices[0]}
	}

	// Dimensions of inputs/outputs.
	numInputs := len(c.inputs)
	numOutputs := e.NumOutputs

	// Allocations that will be used by CGO.
	// Except if the number of inputs/outputs is very large, used the default arena size.
	var arena *arenaContainer
	minSize := (numInputs+numOutputs)*3*8 /*pointer size*/ + 1024
	if minSize > arenaDefaultSize {
		arena = newArena(arenaDefaultSize + minSize)
		defer arena.Free()
	} else {
		arena = getArenaFromPool()
		defer returnArenaToPool(arena)
	}

	// Create arguments structures for call to Execute.
	var args *C.PJRT_LoadedExecutable_Execute_Args
	args = arenaAlloc[C.PJRT_LoadedExecutable_Execute_Args](arena)
	args.struct_size = C.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
	args.executable = e.cLoadedExecutable

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

	numDevices := 1
	args.num_devices = C.size_t(numDevices)
	args.execute_device = c.devices[0].cDevice

	args.num_args = C.size_t(numInputs)
	if args.num_args > 0 {
		args.argument_lists = allocatePerDeviceBufferListWithArena(arena, numDevices, numInputs, c.inputs)
	}

	// For some reason the line below doesn't work. I think something is wrong with PJRT ... but I'm not sure.
	if numOutputs > 0 {
		args.output_lists = allocatePerDeviceBufferListWithArena(arena, numDevices, numOutputs, nil)
	}

	// Create events to wait for the end of execution: leaving this as NULL is allowed, but what happens then
	// (does it wait or not, and then what?) is not documented in PJRT.
	perDeviceEvents := arenaAllocSlice[*C.PJRT_Event](arena, numDevices)
	args.device_complete_events = (**C.PJRT_Event)(unsafe.SliceData(perDeviceEvents))
	//args.device_complete_events = cMallocArray[*C.PJRT_Event](numDevices)
	//defer cFree(args.device_complete_events)

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
	return outputs, nil
}

// Allocate [numDevices][numBuffers]*Buffer C 2D-array to be used by PJRT C API, with the given Buffer pointers.
func allocatePerDeviceBufferListWithArena(arena *arenaContainer, numDevices int, numBuffers int, buffers []*Buffer) ***C.PJRT_Buffer {
	// Top level:
	perDevice := arenaAllocSlice[**C.PJRT_Buffer](arena, numDevices)
	for deviceIdx := range perDevice {
		deviceBuffers := arenaAllocSlice[*C.PJRT_Buffer](arena, numBuffers)
		perDevice[deviceIdx] = &deviceBuffers[0]
		if buffers != nil {
			for bufferIdx := range deviceBuffers {
				if buffers[bufferIdx] == nil {
					deviceBuffers[bufferIdx] = nil
					continue
				}
				if buffers[bufferIdx].cBuffer == nil {
					// Buffer given, but it's cBuffer is nil -> probably it has already been destroyed.
					panicf("buffers[%d].cBuffer is nil, has it already been destroyed!?", bufferIdx)
				}
				deviceBuffers[bufferIdx] = buffers[bufferIdx].cBuffer
			}
		}
	}
	return &perDevice[0]
}
