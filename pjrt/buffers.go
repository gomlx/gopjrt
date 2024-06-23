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
	"gopjrt/dtypes"
	"k8s.io/klog/v2"
	"runtime"
	"unsafe"
)

// Buffer is a reference to an array storage (buffer) on device.
type Buffer struct {
	cBuffer *C.PJRT_Buffer
	plugin  *Plugin
}

// newBuffer creates Buffer and registers it for freeing.
func newBuffer(plugin *Plugin, cBuffer *C.PJRT_Buffer) *Buffer {
	b := &Buffer{
		plugin:  plugin,
		cBuffer: cBuffer,
	}
	runtime.SetFinalizer(b, func(b *Buffer) {
		err := b.Destroy()
		if err != nil {
			klog.Errorf("Buffer.Destroy failed: %v", err)
		}
	})
	return b
}

// Destroy the Buffer, release resources, and Buffer is no longer valid.
// This is automatically called if Buffer is garbage collected.
func (b *Buffer) Destroy() error {
	if b == nil || b.plugin == nil || b.cBuffer == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(b)
	args := C.new_PJRT_Buffer_Destroy_Args()
	defer cFree(args)
	args.buffer = b.cBuffer
	err := toError(b.plugin, C.call_PJRT_Buffer_Destroy(b.plugin.api, args))
	b.plugin = nil
	b.cBuffer = nil
	return err
}

// BufferFromHostConfig is used to configure the transfer from a buffer from host memory to on-device memory, it is
// created with Client.CreateBufferFromHost.
//
// The host data source must be configured with either HostRawData or HostFlatData. All other configurations
// are optional.
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
}

// FromRawData configures the data from host to copy: a pointer to bytes that must be kept alive (and constant)
// during the call. The parameters dtype and dimensions provide the shape of the array.
func (b *BufferFromHostConfig) FromRawData(data []byte, dtype dtypes.DType, dimensions []int) *BufferFromHostConfig {
	b.data = data
	b.dtype = dtype
	b.dimensions = dimensions
	return b
}

// ToDevice configures which device to copy the host data to.
// If left un-configured, it will pick the first device returned by Client.AddressableDevices.
func (b *BufferFromHostConfig) ToDevice(device *Device) *BufferFromHostConfig {
	b.device = device
	return b
}

// Done will use the configuration to start the transfer from host to device.
// It's synchronous: it awaits the transfer to finish and then returns.
func (b *BufferFromHostConfig) Done() (*Buffer, error) {
	if len(b.data) == 0 {
		return nil, errors.New("BufferFromHost requires one to configure the host data to transfer, none was configured.")
	}

	// Makes sure program data is not moved around by the GC during the C/C++ call.
	var pinner runtime.Pinner
	dataPtr := unsafe.SliceData(b.data)
	pinner.Pin(b)
	pinner.Pin(dataPtr)
	defer func() {
		pinner.Unpin()
	}()

	// Set default device.
	if b.device == nil {
		devices, err := b.client.AddressableDevices()
		if err != nil {
			return nil, errors.WithMessage(err, "BufferFromHost can't find addressable device to transfer to")
		}
		if len(devices) == 0 {
			return nil, errors.New("BufferFromHost can't find addressable device to transfer to")
		}
		b.device = devices[0]
	}
	pinner.Pin(b.device)

	// Start the call.
	args := C.new_PJRT_Client_BufferFromHostBuffer_Args()
	defer cFree(args)
	args.client = b.client.client
	args.data = unsafe.Pointer(dataPtr)
	args._type = C.PJRT_Buffer_Type(b.dtype)
	args.num_dims = C.size_t(len(b.dimensions))
	if len(b.dimensions) > 0 {
		args.dims = cMallocArrayAndSet[C.int64_t](len(b.dimensions), func(i int) C.int64_t {
			return C.int64_t(b.dimensions[i])
		})
	}
	if args.dims != nil {
		cFree(args.dims)
	}
	args.host_buffer_semantics = C.PJRT_HostBufferSemantics(b.hostBufferSemantics)
	args.device = b.device.cDevice
	err := toError(b.client.plugin, C.call_PJRT_Client_BufferFromHostBuffer(b.client.plugin.api, args))
	if err != nil {
		return nil, err
	}

	// We get a PJRT_Buffer even before it's fully transferred.
	buffer := newBuffer(b.client.plugin, args.buffer)

	// Await for transfer to finish.
	doneEvent := newEvent(b.client.plugin, args.done_with_host_buffer)
	defer func() { _ = doneEvent.Destroy() }()
	err = doneEvent.Await()
	if err != nil {
		err2 := buffer.Destroy()
		if err2 != nil {
			klog.Errorf("Failed to destroy buffer that didn't finish to transfer from host: %+v", err2)
		}
		return nil, errors.WithMessage(err, "Failed to finish Client.BufferFromHost transfer")
	}
	return buffer, nil
}

// FlatDataToRaw takes a flat slice of values and the target dimensions of the underlying array and convert
// to the raw data, dtype and dimensions needed by BufferFromHostConfig.FromRawData.
//
// If len(flat) != Product(dimensions) or if any dimension is 0, it panics.
//
// Scalars can be defined with len(dimensions) == 0 and len(flat) == 1.
func FlatDataToRaw[T dtypes.Supported](flat []T, dimensions ...int) ([]byte, dtypes.DType, []int) {
	// Checks dimensions.
	expectedSize := 1
	for _, dim := range dimensions {
		if dim <= 0 {
			exceptions.Panicf("FlatDataToRaw cannot be given zero or negative dimensions, got %v", dimensions)
		}
		expectedSize *= dim
	}
	if len(flat) != expectedSize {
		exceptions.Panicf("FlatDataToRaw given a flat slice of size %d that doesn't match dimensions %v (total size %d)",
			len(flat), dimensions, expectedSize)
	}
	dtype := dtypes.DTypeGeneric[T]()
	if len(flat) == 0 {
		return nil, dtype, dimensions
	}
	flatBytesSize := len(flat) * int(unsafe.Sizeof(flat[0]))
	rawPointer := unsafe.Pointer(unsafe.SliceData(flat))
	rawSlice := unsafe.Slice((*byte)(rawPointer), flatBytesSize)
	return rawSlice, dtype, dimensions
}

// ScalarToRaw generates the raw values needed by BufferFromHostConfig.FromRawData to feed a simple scalar value.
func ScalarToRaw[T dtypes.Supported](value T) ([]byte, dtypes.DType, []int) {
	dtype := dtypes.DTypeGeneric[T]()
	rawSlice := unsafe.Slice((*byte)(unsafe.Pointer(&value)), int(unsafe.Sizeof(value)))
	return rawSlice, dtype, nil // empty dimensions for scalar
}
