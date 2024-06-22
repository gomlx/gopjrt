package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"github.com/pkg/errors"
	"gopjrt/dtypes"
	"k8s.io/klog/v2"
	"runtime"
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
// Once all options are configured (all are optional), call BufferFromHostConfig.Done to actually initiate the transfer.
type BufferFromHostConfig struct {
	client     *Client
	data       []byte
	dtype      dtypes.DType
	dimensions []int
	device     *Device

	hostBufferSemantics PJRT_HostBufferSemantics
}

// OnDevice configures which device to copy the host data to.
// If left un-configured, it will pick the first device returned by Client.AddressableDevices.
func (b *BufferFromHostConfig) OnDevice(device *Device) *BufferFromHostConfig {
	b.device = device
	return b
}

// Done will use the configuration to start the transfer from host to device.
// It waits it to finish and then returns.
// TODO: Implement AsyncTransfer.
func (b *BufferFromHostConfig) Done() (*Buffer, error) {
	return nil, errors.New("Not implemented")
}
