package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"reflect"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"
)

// Buffer is a reference to an on-device array storage (buffer).
type Buffer struct {
	wrapper *bufferWrapper
	client  *Client

	// For "shared buffers", with a direct pointer to the underlying data.
	// This is nil for non-shared-buffers.
	isShared bool

	dimsSet bool // Whether dims is set.
	dims    []int

	dtypeSet bool // Whether dtype is set.
	dtype    dtypes.DType
	// DEBUG: creationStackTrace error
}

// bufferWrapper wraps the C/C++ data that requires clean up.
type bufferWrapper struct {
	c                *C.PJRT_Buffer
	sharedRawStorage unsafe.Pointer
	plugin           *Plugin
}

func (wrapper *bufferWrapper) IsValid() bool {
	return wrapper != nil && wrapper.c != nil
}

func (wrapper *bufferWrapper) Destroy() error {
	if wrapper == nil || wrapper.plugin == nil || wrapper.c == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(wrapper)

	arena := wrapper.plugin.getArenaFromPool()
	defer wrapper.plugin.returnArenaToPool(arena)
	args := arenaAlloc[C.PJRT_Buffer_Destroy_Args](arena)
	args.struct_size = C.PJRT_Buffer_Destroy_Args_STRUCT_SIZE
	args.buffer = wrapper.c
	err := toError(wrapper.plugin, C.call_PJRT_Buffer_Destroy(wrapper.plugin.api, args))
	wrapper.plugin = nil
	wrapper.c = nil
	buffersAlive.Add(-1)

	if wrapper.sharedRawStorage != nil {
		// Shared storage can only be freed after the buffer is destroyed.
		AlignedFree(wrapper.sharedRawStorage)
		wrapper.sharedRawStorage = nil
	}
	return err
}

var buffersAlive atomic.Int64

// BuffersAlive returns the number of PJRT Buffers in memory and currently tracked by gopjrt.
func BuffersAlive() int64 {
	return buffersAlive.Load()
}

// newBuffer creates Buffer and registers it for freeing.
func newBuffer(client *Client, cBuffer *C.PJRT_Buffer) *Buffer {
	b := &Buffer{
		client:  client,
		wrapper: &bufferWrapper{plugin: client.plugin, c: cBuffer},
		// DEBUG: creationStackTrace: errors.New("bufferCreation"),
	}
	buffersAlive.Add(1)

	runtime.AddCleanup(b, func(wrapper *bufferWrapper) {
		err := wrapper.Destroy()
		if err != nil {
			klog.Errorf("pjrt.Buffer.Destroy failed: %v", err)
		}
	}, b.wrapper)
	return b
}

// Destroy the Buffer, release resources, and Buffer is no longer valid.
// This is automatically called if Buffer is garbage collected.
func (b *Buffer) Destroy() error {
	if !b.wrapper.IsValid() {
		return nil
	}
	err := b.wrapper.Destroy()
	b.client = nil
	return err
}

// Dimensions of the Buffer.
// Returned slice is owned by the buffer, to avoid creating a copy. Don't change it.
func (b *Buffer) Dimensions() (dims []int, err error) {
	if b == nil || b.client == nil || b.client.plugin == nil || !b.wrapper.IsValid() {
		err = errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	if b.dimsSet {
		return b.dims, nil
	}
	defer runtime.KeepAlive(b)

	arena := b.client.plugin.getArenaFromPool()
	defer b.client.plugin.returnArenaToPool(arena)
	args := arenaAlloc[C.PJRT_Buffer_Dimensions_Args](arena)
	args.struct_size = C.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(b.client.plugin, C.call_PJRT_Buffer_Dimensions(b.client.plugin.api, args))
	if err != nil {
		return
	}
	if args.num_dims == 0 {
		return // dims = nil
	}
	b.dims = slices.Clone(cDataToSlice[int](unsafe.Pointer(args.dims), int(args.num_dims)))
	b.dimsSet = true
	return b.dims, nil
}

// DType of the Buffer (PJRT_Buffer_ElementType).
func (b *Buffer) DType() (dtype dtypes.DType, err error) {
	dtype = dtypes.InvalidDType
	if b == nil || b.client == nil || b.client.plugin == nil || !b.wrapper.IsValid() {
		err = errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	defer runtime.KeepAlive(b)
	if b.dtypeSet {
		return b.dtype, nil
	}

	arena := b.client.plugin.getArenaFromPool()
	defer b.client.plugin.returnArenaToPool(arena)
	args := arenaAlloc[C.PJRT_Buffer_ElementType_Args](arena)
	args.struct_size = C.PJRT_Buffer_ElementType_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(b.client.plugin, C.call_PJRT_Buffer_ElementType(b.client.plugin.api, args))
	if err != nil {
		return
	}
	dtype = dtypes.DType(args._type)
	b.dtype = dtype
	b.dtypeSet = true
	return
}

// Device returns the device the buffer is stored.
func (b *Buffer) Device() (device *Device, err error) {
	if b == nil || b.client == nil || b.client.plugin == nil || !b.wrapper.IsValid() {
		err = errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	defer runtime.KeepAlive(b)

	arena := b.client.plugin.getArenaFromPool()
	defer b.client.plugin.returnArenaToPool(arena)
	args := arenaAlloc[C.PJRT_Buffer_Device_Args](arena)
	args.struct_size = C.PJRT_Buffer_Device_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(b.client.plugin, C.call_PJRT_Buffer_Device(b.client.plugin.api, args))
	if err != nil {
		return
	}
	device = newDevice(b.client, args.device)
	return
}

// Client returns the client that created this Buffer.
func (b *Buffer) Client() *Client {
	return b.client
}

// ScalarToRaw generates the raw values needed by BufferFromHostConfig.FromRawData to feed a simple scalar value.
func ScalarToRaw[T dtypes.Supported](value T) ([]byte, dtypes.DType, []int) {
	dtype := dtypes.FromGenericsType[T]()
	rawSlice := unsafe.Slice((*byte)(unsafe.Pointer(&value)), int(unsafe.Sizeof(value)))
	return rawSlice, dtype, nil // empty dimensions for scalar
}

// Size returns the size in bytes if required for the buffer to be transferred with ToHost.
func (b *Buffer) Size() (int, error) {
	defer runtime.KeepAlive(b)
	if b == nil || b.client.plugin == nil || !b.wrapper.IsValid() {
		// Already destroyed ?
		return 0, errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(b)

	arena := b.client.plugin.getArenaFromPool()
	defer b.client.plugin.returnArenaToPool(arena)

	// It uses a PJRT_Buffer_ToHostBuffer_Args but it doesn't transfer, only inquire about size.
	args := arenaAlloc[C.PJRT_Buffer_ToHostBuffer_Args](arena)
	args.struct_size = C.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE
	args.src = b.wrapper.c
	args.dst = nil // Don't transfer, only inquire about size.
	err := toError(b.client.plugin, C.call_PJRT_Buffer_ToHostBuffer(b.client.plugin.api, args))
	if err != nil {
		return 0, errors.WithMessage(err, "Failed to call PJRT_Buffer_ToHostBuffer for inquiring size of the buffer")
	}
	return int(args.dst_size), nil
}

// BufferToScalar is a generic function that transfer a Buffer back to host as a scalar of the given type.
func BufferToScalar[T dtypes.Supported](b *Buffer) (value T, err error) {
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	err = b.ToHost(dst)
	return
}

// ScalarToBuffer transfers the scalar value to a Buffer on the default device.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ScalarToBuffer[T dtypes.Supported](client *Client, value T) (b *Buffer, err error) {
	dtype := dtypes.FromGenericsType[T]()
	src := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return client.BufferFromHost().FromRawData(src, dtype, nil).Done()
}

// ScalarToBufferOnDeviceNum transfers the scalar value to a Buffer on the given device.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ScalarToBufferOnDeviceNum[T dtypes.Supported](client *Client, deviceNum int, value T) (b *Buffer, err error) {
	dtype := dtypes.FromGenericsType[T]()
	src := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return client.BufferFromHost().FromRawData(src, dtype, nil).ToDeviceNum(deviceNum).Done()
}

// ArrayToBuffer transfer a slice to a Buffer on the default device.
// The underlying array is provided with its flat values as a slice, and the underlying dimensions.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ArrayToBuffer[T dtypes.Supported](client *Client, flatValues []T, dimensions ...int) (b *Buffer, err error) {
	return client.BufferFromHost().FromFlatDataWithDimensions(flatValues, dimensions).Done()
}

// BufferToArray transfers the buffer to an array defined by a slice with its flat values, and returns also its underlying dimensions.
func BufferToArray[T dtypes.Supported](buffer *Buffer) (flatValues []T, dimensions []int, err error) {
	var dtype dtypes.DType
	dtype, err = buffer.DType()
	if err != nil {
		return
	}
	requestedDType := dtypes.FromGenericsType[T]()
	if dtype != requestedDType {
		var dummy T
		err = errors.Errorf("called BufferToArray[%T](...), but underlying buffer has dtype %s", dummy, dtype)
		return
	}
	dimensions, err = buffer.Dimensions()
	if err != nil {
		return
	}
	totalSize := 1
	for _, dim := range dimensions {
		totalSize *= dim
	}
	if totalSize <= 0 {
		// Odd empty buffer (likely one of the dimensions was 0), we return nil for the flatValues, the reported dimensions
		// and no error.
		return
	}
	flatValues = make([]T, totalSize)
	flatValuesPtr := unsafe.SliceData(flatValues)
	dst := unsafe.Slice((*byte)(
		unsafe.Pointer(flatValuesPtr)),
		totalSize*int(unsafe.Sizeof(flatValues[0])))
	err = buffer.ToHost(dst)
	return
}

// ToFlatDataAndDimensions transfers the buffer to a flat slice and returns also its underlying dimensions.
//
// Similar to the generic BufferToArray[T], but this returns an anonymous typed (`any`) flat slice instead of using generics.
func (b *Buffer) ToFlatDataAndDimensions() (flat any, dimensions []int, err error) {
	var dtype dtypes.DType
	dtype, err = b.DType()
	if err != nil {
		return
	}

	dimensions, err = b.Dimensions()
	if err != nil {
		return
	}
	totalSize := 1
	for _, dim := range dimensions {
		totalSize *= dim
	}
	if totalSize <= 0 {
		// Odd empty b (likely one of the dimensions was 0), we return nil for the flat, the reported dimensions
		// and no error.
		return
	}
	goType := dtype.GoType()
	flatV := reflect.MakeSlice(reflect.SliceOf(goType), totalSize, totalSize)
	element0 := flatV.Index(0)
	flatValuesPtr := element0.Addr().UnsafePointer()
	sizeBytes := uintptr(flatV.Len()) * element0.Type().Size()

	dst := unsafe.Slice((*byte)(flatValuesPtr), sizeBytes)
	err = b.ToHost(dst)
	flat = flatV.Interface()
	return
}
