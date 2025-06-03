package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"

extern void OnDeleteSharedBuffer(void* device_buffer_ptr, void* user_arg);

// OnDeleteSharedBuffer is a no-op.
void OnDeleteSharedBuffer(void* device_buffer_ptr, void* user_arg) {
	return;
}

extern void (*OnDeleteSharedBufferPtr)(void* device_buffer_ptr, void* user_arg);
void (*OnDeleteSharedBufferPtr)(void* device_buffer_ptr, void* user_arg) = &OnDeleteSharedBuffer;

*/
import "C"
import (
	"reflect"
	"slices"
	"unsafe"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// CreateViewOfDeviceBuffer creates a PJRT Buffer that is backed by storage on the same device given by the caller as flatData and shape.
// Consider using the simpler API NewSharedBuffer.
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
//		// ... use buf as input when executing a PJRT program
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
	arena := c.plugin.getArenaFromPool()
	defer c.plugin.returnArenaToPool(arena)

	// Arguments to PJRT call.
	var args *C.PJRT_Client_CreateViewOfDeviceBuffer_Args
	args = arenaAlloc[C.PJRT_Client_CreateViewOfDeviceBuffer_Args](arena)
	args.struct_size = C.PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE
	args.client = c.client.c
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
	args.on_delete_callback = (*[0]byte)(unsafe.Pointer(C.OnDeleteSharedBufferPtr))
	args.on_delete_callback_arg = nil
	err := toError(c.plugin, C.call_PJRT_Client_CreateViewOfDeviceBuffer(c.plugin.api, args))
	if err != nil {
		return nil, err
	}
	buffer := newBuffer(c, args.buffer)
	buffer.isShared = true
	buffer.dims = slices.Clone(dimensions)
	buffer.dimsSet = true
	buffer.dtype = dtype
	buffer.dtypeSet = true
	return buffer, nil
}

// NewSharedBuffer returns a buffer that can be used for execution and share the underlying
// memory space with the host/local, which can be read and mutated directly.
//
// Shared buffers cannot be donated to executions.
//
// The buffer should not be mutated while it is used by an execution.
//
// When the buffer is finalized, the shared memory is also de-allocated.
//
// It returns a handle to the buffer and a slice of the corresponding data type pointing
// to the shared data.
func (c *Client) NewSharedBuffer(dtype dtypes.DType, dimensions []int, device ...*Device) (buffer *Buffer, flat any, err error) {
	memorySize := uintptr(dtype.SizeForDimensions(dimensions...))
	rawStorage := AlignedAlloc(memorySize, BufferAlignment)
	buffer, err = c.CreateViewOfDeviceBuffer(rawStorage, dtype, dimensions, device...)
	if err != nil {
		AlignedFree(rawStorage)
		err = errors.WithMessagef(err, "NewSharedBuffer failed creating new buffer")
		buffer = nil
		return
	}
	buffer.wrapper.sharedRawStorage = rawStorage
	buffer.isShared = true

	goDType := dtype.GoType()
	flat = reflect.SliceAt(dtype.GoType(), rawStorage, int(memorySize/goDType.Size())).Interface()
	return
}

// IsShared returns whether this buffer was created with Client.NewSharedBuffer.
// These buffers cannot be donated in execution.
func (b *Buffer) IsShared() bool {
	return b.isShared
}

// UnsafePointer returns platform-dependent address for the given buffer that is often but
// not guaranteed to be the physical/device address.
// Consider using the more convenient DirectAccess.
//
// Probably, this should only be used by CPU plugins.
//
// To be on the safe side, only use this if Client.HasSharedBuffers is true.
// It uses the undocumented PJRT_Buffer_UnsafePointer.
func (b *Buffer) UnsafePointer() (unsafe.Pointer, error) {
	plugin, err := b.getPlugin()
	if err != nil {
		return nil, err
	}

	// Arena for memory allocations used by CGO.
	arena := plugin.getArenaFromPool()
	defer plugin.returnArenaToPool(arena)

	// Arguments to PJRT call.
	var args *C.PJRT_Buffer_UnsafePointer_Args
	args = arenaAlloc[C.PJRT_Buffer_UnsafePointer_Args](arena)
	args.struct_size = C.PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(plugin, C.call_PJRT_Buffer_UnsafePointer(plugin.api, args))
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(uintptr(args.buffer_pointer)), nil
}

// Data returns the flat slice pointing to the underlying storage data for the buffer.
//
// This is an undocumented feature of PJRT and likely only works for CPU platforms.
// The flat slice returned is only valid while the buffer is alive.
func (b *Buffer) Data() (flat any, err error) {
	var rawStorage unsafe.Pointer
	rawStorage, err = b.UnsafePointer()
	if err != nil {
		return nil, err
	}
	dims := b.dims
	dtype := b.dtype
	if !b.dimsSet {
		dims, err = b.Dimensions()
		if err != nil {
			return nil, err
		}
	}
	if !b.dtypeSet {
		dtype, err = b.DType()
		if err != nil {
			return nil, err
		}
	}

	numElements := 1
	for _, dim := range dims {
		numElements *= dim
	}
	return reflect.SliceAt(dtype.GoType(), rawStorage, numElements).Interface(), nil
}
