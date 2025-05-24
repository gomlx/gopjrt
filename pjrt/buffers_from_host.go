package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"

PJRT_Error* BufferFromHostAndWait(const PJRT_Api *api, PJRT_Client_BufferFromHostBuffer_Args *args) {
	PJRT_Error* err = api->PJRT_Client_BufferFromHostBuffer(args);
	if (err) {
		return err;
	}
	PJRT_Event_Await_Args event_args = {0};
	event_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
	event_args.event = args->done_with_host_buffer;
	err = api->PJRT_Event_Await(&event_args);

	PJRT_Event_Destroy_Args efree_args;
	efree_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
	efree_args.event = args->done_with_host_buffer;
	api->PJRT_Event_Destroy(&efree_args);

	return err;
}

PJRT_Error *dummy_error;
PJRT_Error *Dummy(void *api) {
	if (api == NULL) {
		return NULL;
	}
	return dummy_error;
}

*/
import "C"
import (
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"reflect"
	"runtime"
	"slices"
	"unsafe"
)

// BufferFromHostConfig is used to configure the transfer from a buffer from host memory to on-device memory, it is
// created with Client.BufferFromHost.
//
// The data to transfer from host can be set up with one of the following methods:
//
// - FromRawData: it takes as inputs the bytes and shape (dtype and dimensions).
// - FromFlatDataWithDimensions: it takes as inputs a flat slice and shape (dtype and dimensions).
//
// The device defaults to 0, but it can be configured with BufferFromHostConfig.ToDevice or BufferFromHostConfig.ToDeviceNum.
//
// At the end call BufferFromHostConfig.Done to actually initiate the transfer.
//
// TODO: Implement async transfers, arbitrary memory layout, etc.
type BufferFromHostConfig struct {
	client     *Client
	data       []byte
	dtype      dtypes.DType
	dimensions []int
	device     *Device

	hostBufferSemantics PJRT_HostBufferSemantics

	// err stores the first error that happened during configuration.
	// If it is not nil, it is immediately returned by the Done call.
	err error
}

// FromRawData configures the data from host to copy: a pointer to bytes that must be kept alive (and constant)
// during the call. The parameters dtype and dimensions provide the shape of the array.
func (b *BufferFromHostConfig) FromRawData(data []byte, dtype dtypes.DType, dimensions []int) *BufferFromHostConfig {
	if b.err != nil {
		return b
	}
	b.data = data
	b.dtype = dtype
	b.dimensions = dimensions
	return b
}

// ToDevice configures which device to copy the host data to.
//
// If left un-configured, it will pick the first device returned by Client.AddressableDevices.
//
// You can also provide a device by their index in Client.AddressableDevices.
func (b *BufferFromHostConfig) ToDevice(device *Device) *BufferFromHostConfig {
	if b.err != nil {
		return b
	}
	if device == nil {
		b.err = errors.New("BufferFromHost().ToDevice() given a nil device")
		return b
	}
	addressable, err := device.IsAddressable()
	if err != nil {
		b.err = errors.WithMessagef(err, "BufferFromHost().ToDevice() failed to check whether device is addressable")
		return b
	}
	if !addressable {
		b.err = errors.New("BufferFromHost().ToDevice() given a non addressable device")
		return b
	}
	b.device = device
	return b
}

// ToDeviceNum configures which device to copy the host data to, given a deviceNum pointing to the device in the
// list returned by Client.AddressableDevices.
//
// If left un-configured, it will pick the first device returned by Client.AddressableDevices.
//
// You can also provide a device by their index in Client.AddressableDevices.
func (b *BufferFromHostConfig) ToDeviceNum(deviceNum int) *BufferFromHostConfig {
	if b.err != nil {
		return b
	}
	if deviceNum < 0 || deviceNum >= len(b.client.addressableDevices) {
		b.err = errors.Errorf("BufferFromHost().ToDeviceNum() invalid deviceNum=%d, only %d addressable devices available", deviceNum, len(b.client.addressableDevices))
		return b
	}
	return b.ToDevice(b.client.addressableDevices[deviceNum])
}

// FromFlatDataWithDimensions configures the data to come from a flat slice of the desired data type, and the underlying
// dimensions.
// The flat slice size must match the product of the dimension.
// If no dimensions are given, it is assumed to be a scalar, and flat should have length 1.
func (b *BufferFromHostConfig) FromFlatDataWithDimensions(flat any, dimensions []int) *BufferFromHostConfig {
	if b.err != nil {
		return b
	}
	// Checks dimensions.
	expectedSize := 1
	for _, dim := range dimensions {
		if dim <= 0 {
			b.err = errors.Errorf("FromFlatDataWithDimensions cannot be given zero or negative dimensions, got %v", dimensions)
			return b
		}
		expectedSize *= dim
	}

	// Check the flat slice has the right shape.
	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		b.err = errors.Errorf("FromFlatDataWithDimensions was given a %s for flat, but it requires a slice", flatV.Kind())
		return b
	}
	if flatV.Len() != expectedSize {
		b.err = errors.Errorf("FromFlatDataWithDimensions(flat, dimensions=%v) needs %d values to match dimensions, but got len(flat)=%d", dimensions, expectedSize, flatV.Len())
		return b
	}

	// Check validity of the slice elements type.
	element0 := flatV.Index(0)
	element0Type := element0.Type()
	dtype := dtypes.FromGoType(element0Type)
	if dtype == dtypes.InvalidDType {
		b.err = errors.Errorf("FromFlatDataWithDimensions(flat, dimensions%v) got flat=[]%s, expected a slice of a Go tyep that can be converted to a valid DType", dimensions, element0Type)
		return b
	}

	// Create slice of bytes and use b.FromRawData.
	sizeBytes := uintptr(flatV.Len()) * element0Type.Size()
	data := unsafe.Slice((*byte)(element0.Addr().UnsafePointer()), sizeBytes)
	return b.FromRawData(data, dtype, dimensions)
}

// Done will use the configuration to start the transfer from host to device.
// It's synchronous: it awaits the transfer to finish and then returns.
func (b *BufferFromHostConfig) Done() (*Buffer, error) {
	if b.err != nil {
		// Return first error saved during configuration.
		return nil, b.err
	}
	if len(b.data) == 0 {
		return nil, errors.New("BufferFromHost requires one to configure the host data to transfer, none was configured.")
	}
	defer runtime.KeepAlive(b)

	// Makes sure program data is not moved around by the GC during the C/C++ call.
	var pinner runtime.Pinner
	defer pinner.Unpin()
	dataPtr := unsafe.SliceData(b.data)
	pinner.Pin(dataPtr)

	// Set default device.
	if b.device == nil {
		devices := b.client.AddressableDevices()
		if len(devices) == 0 {
			return nil, errors.New("BufferFromHost can't find addressable device to transfer to")
		}
		b.device = devices[0]
	}

	// Arena for memory allocations used by CGO.
	arena := b.client.plugin.getArenaFromPool()
	defer b.client.plugin.returnArenaToPool(arena)

	// Arguments to PJRT call.
	var args *C.PJRT_Client_BufferFromHostBuffer_Args
	args = arenaAlloc[C.PJRT_Client_BufferFromHostBuffer_Args](arena)
	args.struct_size = C.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
	args.client = b.client.client.c
	args.data = unsafe.Pointer(dataPtr)
	args._type = C.PJRT_Buffer_Type(b.dtype)
	args.num_dims = C.size_t(len(b.dimensions))
	if len(b.dimensions) > 0 {
		dims := arenaAllocSlice[C.int64_t](arena, len(b.dimensions))
		for ii, dim := range b.dimensions {
			dims[ii] = C.int64_t(dim)
		}
		args.dims = unsafe.SliceData(dims)
	}
	args.host_buffer_semantics = C.PJRT_HostBufferSemantics(b.hostBufferSemantics)
	args.device = b.device.cDevice
	err := toError(b.client.plugin, C.BufferFromHostAndWait(b.client.plugin.api, args))
	if err != nil {
		return nil, err
	}

	buffer := newBuffer(b.client, args.buffer)
	buffer.dims = slices.Clone(b.dimensions)
	buffer.dimsSet = true
	buffer.dtype = b.dtype
	buffer.dtypeSet = true
	return buffer, nil
}

// dummyCGO calls a minimal C function and doesn't do anything.
// Here for the purpose of benchmarking CGO calls.
func dummyCGO(pointer unsafe.Pointer) {
	_ = C.Dummy(pointer)
}
