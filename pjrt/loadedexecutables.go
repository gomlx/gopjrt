package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"github.com/gomlx/exceptions"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"runtime"
	"slices"
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
}

// newLoadedExecutable creates LoadedExecutable and registers it for freeing.
func newLoadedExecutable(plugin *Plugin, client *Client, cLoadedExecutable *C.PJRT_LoadedExecutable) (*LoadedExecutable, error) {
	e := &LoadedExecutable{
		plugin:            plugin,
		client:            client,
		cLoadedExecutable: cLoadedExecutable,
	}
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
	c.NotDonatable()
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

// DonateAll marks all inputs to be "donated".
//
// Donated inputs become invalid after the execution. Often donated arguments are also the output of a computation
// and are updated in place. See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) DonateAll() *ExecutionConfig {
	c.nonDonatableInputs = nil
	return c
}

// NotDonatable makes all inputs to be marked as non-donatable. This is the default.
//
// Donated inputs become invalid after the execution. Often donated arguments are also the output of a computation
// and are updated in place. See discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation
func (c *ExecutionConfig) NotDonatable() *ExecutionConfig {
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

func (c *ExecutionConfig) Done() ([]*Buffer, error) {
	e := c.executable
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)

	// Find device:
	if len(c.devices) == 0 {
		var err error
		c.devices, err = e.client.AddressableDevices()
		if err != nil {
			return nil, errors.WithMessage(err, "LoadedExecutable.Execute failed while finding addressable device to execute")
		}
		if len(c.devices) == 0 {
			return nil, errors.New("LoadedExecutable.Execute can't find addressable device to execute")
		}
	}

	// Create arguments structures for call to Execute.
	args := C.new_PJRT_LoadedExecutable_Execute_Args()
	defer cFree(args)
	args.executable = e.cLoadedExecutable
	options := C.new_PJRT_ExecuteOptions() // Like more args that for some reason(?) go on a separate struct.
	defer cFree(options)
	args.options = options

	// Configure (non-)donatable inputs.
	if len(c.nonDonatableInputs) > 0 {
		options.num_non_donatable_input_indices = C.size_t(len(c.nonDonatableInputs))
		options.non_donatable_input_indices = cMallocArrayAndSet[C.int64_t](len(c.nonDonatableInputs), func(ii int) C.int64_t {
			return C.int64_t(c.nonDonatableInputs[ii])
		})
		defer cFree(options.non_donatable_input_indices)
	}

	numDevices := 1
	args.num_devices = C.size_t(numDevices)
	args.execute_device = c.devices[0].cDevice
	args.num_args = C.size_t(len(c.inputs))
	args.argument_lists = allocatePerDeviceBufferList(numDevices, c.inputs)
	defer freePerDeviceBufferList(args.argument_lists, numDevices)
	args.output_lists = allocatePerDeviceBufferList(numDevices, make([]*Buffer, e.NumOutputs))
	defer freePerDeviceBufferList(args.output_lists, numDevices)
	//args.device_complete_events = cMallocArray[*C.PJRT_Event](numDevices)
	//defer cFree(args.device_complete_events)

	err := toError(e.plugin, C.call_PJRT_LoadedExecutable_Execute(e.plugin.api, args))
	if err != nil {
		return nil, err
	}

	perDevice := gatherPerDeviceBufferList(e.plugin, args.output_lists, numDevices, e.NumOutputs)
	return perDevice[0], nil
}

// Allocate [numDevices][numBuffers]*Buffer C 2D-array to be used by PJRT C API, with the given Buffer pointers.
func allocatePerDeviceBufferList(numDevices int, buffers []*Buffer) ***C.PJRT_Buffer {
	// Top level:
	perDevice := make([]**C.PJRT_Buffer, numDevices)
	for deviceIdx := range perDevice {
		perDevice[deviceIdx] = cMallocArrayAndSet[*C.PJRT_Buffer](len(buffers), func(idxBuffer int) *C.PJRT_Buffer {
			if buffers[idxBuffer] == nil {
				// No buffer given for structure.
				return nil
			}
			if buffers[idxBuffer].cBuffer == nil {
				// Buffer given, but it's cBuffer is nil -> probably it has already been destroyed.
				exceptions.Panicf("buffers[%d].cBuffer is nil, has it already been destroyed!?", idxBuffer)
			}
			return buffers[idxBuffer].cBuffer
		})
	}
	return cMallocArrayFromSlice(perDevice)
}

// freePerDeviceBufferList frees the intermediary array pointers, but it doesn't touch the buffers themselves.
func freePerDeviceBufferList(data ***C.PJRT_Buffer, numDevices int) {
	perDevice := cDataToSlice[**C.PJRT_Buffer](unsafe.Pointer(data), numDevices)
	for _, list := range perDevice {
		cFree(list)
	}
	cFree(data)
}

// gatherPerDeviceBufferList returns a [numDevices][numBuffers]*Buffer given the C 2D array.
func gatherPerDeviceBufferList(plugin *Plugin, data ***C.PJRT_Buffer, numDevices, numBuffers int) [][]*Buffer {
	perDevice := make([][]*Buffer, numDevices)
	cPerDevice := cDataToSlice[**C.PJRT_Buffer](unsafe.Pointer(data), numDevices)
	for ii, cBufferListPtr := range cPerDevice {
		perDevice[ii] = make([]*Buffer, numBuffers)
		cBuffers := cDataToSlice[*C.PJRT_Buffer](unsafe.Pointer(cBufferListPtr), numBuffers)
		for jj, cBuffer := range cBuffers {
			perDevice[ii][jj] = newBuffer(plugin, cBuffer)
		}
	}
	return perDevice
}
