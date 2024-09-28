package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"

PJRT_Buffer_MemoryLayout_Tiled *GetTiledLayoutUnion(PJRT_Buffer_MemoryLayout *layout) {
	return &(layout->tiled);
}
*/
import "C"
import (
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"reflect"
	"runtime"
	"slices"
	"unsafe"
)

// Buffer is a reference to an array storage (buffer) on device.
type Buffer struct {
	cBuffer *C.PJRT_Buffer
	client  *Client
	// DEBUG: creationStackTrace error
}

// newBuffer creates Buffer and registers it for freeing.
func newBuffer(client *Client, cBuffer *C.PJRT_Buffer) *Buffer {
	b := &Buffer{
		client:  client,
		cBuffer: cBuffer,
		// DEBUG: creationStackTrace: errors.New("bufferCreation"),
	}
	runtime.SetFinalizer(b, func(b *Buffer) {
		/* DEBUG:
		if b != nil && cBuffer != nil && b.client != nil && b.client.plugin != nil {
			dims, _ := b.Dimensions()
			dtype, _ := b.DType()
			fmt.Printf("\nGC buffer: (%s)%v\n", dtype, dims)
			fmt.Printf("\tStack trace:\n%+v\n", b.creationStackTrace)
		}
		*/
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
	if b == nil || b.client == nil || b.client.plugin == nil || b.cBuffer == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(b)
	args := C.new_PJRT_Buffer_Destroy_Args()
	defer cFree(args)
	args.buffer = b.cBuffer
	err := toError(b.client.plugin, C.call_PJRT_Buffer_Destroy(b.client.plugin.api, args))
	b.client = nil
	b.cBuffer = nil
	return err
}

// Dimensions of the Buffer.
func (b *Buffer) Dimensions() (dims []int, err error) {
	if b == nil || b.client == nil || b.client.plugin == nil || b.cBuffer == nil {
		err = errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	defer runtime.KeepAlive(b)

	args := C.new_PJRT_Buffer_Dimensions_Args()
	defer cFree(args)
	args.buffer = b.cBuffer
	err = toError(b.client.plugin, C.call_PJRT_Buffer_Dimensions(b.client.plugin.api, args))
	if err != nil {
		return
	}
	if args.num_dims == 0 {
		return // dims = nil
	}
	dims = slices.Clone(cDataToSlice[int](unsafe.Pointer(args.dims), int(args.num_dims)))
	return
}

// DType of the Buffer (PJRT_Buffer_ElementType).
func (b *Buffer) DType() (dtype dtypes.DType, err error) {
	dtype = dtypes.InvalidDType
	if b == nil || b.client == nil || b.client.plugin == nil || b.cBuffer == nil {
		err = errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	defer runtime.KeepAlive(b)

	args := C.new_PJRT_Buffer_ElementType_Args()
	defer cFree(args)
	args.buffer = b.cBuffer
	err = toError(b.client.plugin, C.call_PJRT_Buffer_ElementType(b.client.plugin.api, args))
	if err != nil {
		return
	}
	dtype = dtypes.DType(args._type)
	return
}

// Device returns the device the buffer is stored.
func (b *Buffer) Device() (device *Device, err error) {
	if b == nil || b.client == nil || b.client.plugin == nil || b.cBuffer == nil {
		err = errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	defer runtime.KeepAlive(b)

	args := C.new_PJRT_Buffer_Device_Args()
	defer cFree(args)
	args.buffer = b.cBuffer
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
		devices := b.client.AddressableDevices()
		if len(devices) == 0 {
			return nil, errors.New("BufferFromHost can't find addressable device to transfer to")
		}
		b.device = devices[0]
	}
	pinner.Pin(b.device)

	// Start the call.
	args := C.new_PJRT_Client_BufferFromHostBuffer_Args()
	defer cFree(args)
	pinner.Pin(b.client)
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
		defer cFree(args.dims)
	}
	args.host_buffer_semantics = C.PJRT_HostBufferSemantics(b.hostBufferSemantics)
	args.device = b.device.cDevice
	pinner.Pin(b.client.plugin)
	err := toError(b.client.plugin, C.call_PJRT_Client_BufferFromHostBuffer(b.client.plugin.api, args))
	if err != nil {
		return nil, err
	}

	// We get a PJRT_Buffer even before it's fully transferred.
	buffer := newBuffer(b.client, args.buffer)

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

// ScalarToRaw generates the raw values needed by BufferFromHostConfig.FromRawData to feed a simple scalar value.
func ScalarToRaw[T dtypes.Supported](value T) ([]byte, dtypes.DType, []int) {
	dtype := dtypes.FromGenericsType[T]()
	rawSlice := unsafe.Slice((*byte)(unsafe.Pointer(&value)), int(unsafe.Sizeof(value)))
	return rawSlice, dtype, nil // empty dimensions for scalar
}

// Size returns the size in bytes if required for the buffer to be transferred with ToHost.
func (b *Buffer) Size() (int, error) {
	if b == nil || b.client.plugin == nil || b.cBuffer == nil {
		// Already destroyed ?
		return 0, errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(b)
	args := C.new_PJRT_Buffer_ToHostBuffer_Args()
	defer cFree(args)
	args.src = b.cBuffer
	args.dst = nil // Don't transfer, only inquire about size.
	err := toError(b.client.plugin, C.call_PJRT_Buffer_ToHostBuffer(b.client.plugin.api, args))
	if err != nil {
		return 0, errors.WithMessage(err, "Failed to call PJRT_Buffer_ToHostBuffer for inquiring size of the buffer")
	}
	return int(args.dst_size), nil
}

// ToHost transfers the contents of buffer stored on device to the host.
// The space in dst has to hold enough space (see Buffer.Size) to hold the required data, or an error is returned.
//
// This always request a major-to-minor layout, the assumption of the layout in host memory -- TPUs are known to
// reorganize the layout.
func (b *Buffer) ToHost(dst []byte) error {
	if b == nil || b.client.plugin == nil || b.cBuffer == nil {
		// Already destroyed ?
		return errors.New("Buffer is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}

	// Make sure garbage collection doesn't free or move data before they are used by C/C++.
	var pinner runtime.Pinner
	pinner.Pin(b)
	pinner.Pin(unsafe.SliceData(dst))
	defer pinner.Unpin()

	// We'll need the buffer rank to set up the layout.
	dims, err := b.Dimensions()
	if err != nil {
		return err
	}
	rank := len(dims)

	// Prepare arguments for the buffer-to-host call.
	args := C.new_PJRT_Buffer_ToHostBuffer_Args()
	defer cFree(args)
	args.src = b.cBuffer
	args.dst = unsafe.Pointer(unsafe.SliceData(dst))
	args.dst_size = C.size_t(len(dst))

	// Layout argument.
	layoutArgs := C.new_PJRT_Buffer_MemoryLayout()
	defer cFree(layoutArgs)
	args.host_layout = layoutArgs

	// Tiled layout must be present, even if there are no tiles (tileArgs.num_tiles==0).
	layoutArgs._type = C.PJRT_Buffer_MemoryLayout_Type_Tiled
	tileArgs := C.GetTiledLayoutUnion(layoutArgs)

	// Configure major-to-minor layout into tileArgs, if not scalar.
	tileArgs.minor_to_major_size = C.size_t(rank)
	if rank > 0 {
		tileArgs.minor_to_major = cMallocArray[C.int64_t](rank)
		minorToMajorMapping := unsafe.Slice(tileArgs.minor_to_major, rank)
		defer cFree(tileArgs.minor_to_major)
		for axisIdx := range len(dims) {
			minorToMajorMapping[axisIdx] = C.int64_t(rank - axisIdx - 1)
		}
	}

	err = toError(b.client.plugin, C.call_PJRT_Buffer_ToHostBuffer(b.client.plugin.api, args))
	if err != nil {
		return errors.WithMessage(err, "Failed to call PJRT_Buffer_ToHostBuffer to transfer the buffer to host")
	}

	// Await for transfer to finish.
	doneEvent := newEvent(b.client.plugin, args.event)
	defer func() { _ = doneEvent.Destroy() }()
	err = doneEvent.Await()
	if err != nil {
		return errors.WithMessage(err, "Failed to wait Buffer.ToHost transfer to finish")
	}
	return nil
}

// BufferToScalar is a generic function that transfer a Buffer back to host as a scalar of the given type.
func BufferToScalar[T dtypes.Supported](b *Buffer) (value T, err error) {
	var pinner runtime.Pinner
	pinner.Pin(b)
	pinner.Pin(&value)
	defer pinner.Unpin()

	dst := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	err = b.ToHost(dst)
	return
}

// ScalarToBuffer transfers the scalar value to a Buffer on the default device.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ScalarToBuffer[T dtypes.Supported](client *Client, value T) (b *Buffer, err error) {
	var pinner runtime.Pinner
	pinner.Pin(client)
	pinner.Pin(&value)
	defer pinner.Unpin()

	dtype := dtypes.FromGenericsType[T]()
	src := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return client.BufferFromHost().FromRawData(src, dtype, nil).Done()
}

// ScalarToBufferOnDeviceNum transfers the scalar value to a Buffer on the given device.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ScalarToBufferOnDeviceNum[T dtypes.Supported](client *Client, deviceNum int, value T) (b *Buffer, err error) {
	var pinner runtime.Pinner
	pinner.Pin(client)
	pinner.Pin(&value)
	defer pinner.Unpin()

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

	var pinner runtime.Pinner
	pinner.Pin(buffer)
	pinner.Pin(flatValuesPtr)
	defer pinner.Unpin()

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

	var pinner runtime.Pinner
	pinner.Pin(b)
	pinner.Pin(flatValuesPtr)
	defer pinner.Unpin()
	dst := unsafe.Slice((*byte)(flatValuesPtr), sizeBytes)
	err = b.ToHost(dst)
	flat = flatV.Interface()
	return
}
