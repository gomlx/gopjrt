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
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"reflect"
	"slices"
	"unsafe"
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
	buffer.sharedRawStorage = rawStorage
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
