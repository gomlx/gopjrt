package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"runtime"
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

// Execute the compile program using the most standard options:
//
// - Using the first addressable device.
// - All input buffers marked as not-donated (see discussion in https://jax.readthedocs.io/en/latest/faq.html#buffer-donation) input buffers.
// -
func (e *LoadedExecutable) Execute(inputs ...*Buffer) ([]*Buffer, error) {
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)

	// Find device:
	devices, err := e.client.AddressableDevices()
	if err != nil {
		return nil, errors.WithMessage(err, "LoadedExecutable.Execute failed while finding addressable device to execute")
	}
	if len(devices) == 0 {
		return nil, errors.New("LoadedExecutable.Execute can't find addressable device to execute")
	}

	args := C.new_PJRT_LoadedExecutable_Execute_Args()
	defer cFree(args)
	options := C.new_PJRT_ExecuteOptions() // Like more args that for some reason(?) go on a separate struct.
	defer cFree(options)
	args.executable = e.cLoadedExecutable
	args.options = options
	numDevices := 1
	args.num_devices = C.size_t(numDevices)
	args.execute_device = devices[0].cDevice
	args.num_args = C.size_t(len(inputs))
	args.argument_lists = allocatePerDeviceBufferList(numDevices, inputs)
	defer freePerDeviceBufferList(args.argument_lists, numDevices)
	args.output_lists = allocatePerDeviceBufferList(numDevices, make([]*Buffer, e.NumOutputs))
	defer freePerDeviceBufferList(args.output_lists, numDevices)
	err = toError(e.plugin, C.call_PJRT_LoadedExecutable_Execute(e.plugin.api, args))
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
		perDevice[deviceIdx] = cMallocArrayAndSet[*C.PJRT_Buffer](len(buffers), func(i int) *C.PJRT_Buffer {
			if buffers[i] == nil {
				return nil
			}
			return buffers[i].cBuffer
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
