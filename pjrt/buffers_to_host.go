package pjrt

import (
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
)

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"

PJRT_Error* BufferToHost(const PJRT_Api *api, PJRT_Buffer *buffer, void *dst, int64_t dst_size, int rank) {
	PJRT_Buffer_ToHostBuffer_Args args = {0};

	args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
	args.src = buffer;
	args.dst = dst;
	args.dst_size = dst_size;
	PJRT_Buffer_MemoryLayout layout_args = {0};
	layout_args.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
	args.host_layout = &layout_args;
	layout_args.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
	layout_args.tiled.minor_to_major_size = rank;
	int64_t minor_to_major[rank > 0 ? rank : 1];
	if (rank > 0) {
		for (int axisIdx = 0; axisIdx < rank; axisIdx++) {
			minor_to_major[axisIdx] = rank - axisIdx - 1;
		}
		layout_args.tiled.minor_to_major = &minor_to_major[0];
	}
	PJRT_Error* err = api->PJRT_Buffer_ToHostBuffer(&args);
	if (err) {
		return err;
	}

	PJRT_Event_Await_Args event_args = {0};
	event_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
	event_args.event = args.event;
	err = api->PJRT_Event_Await(&event_args);
	PJRT_Event_Destroy_Args efree_args;
	efree_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
	efree_args.event = args.event;
	api->PJRT_Event_Destroy(&efree_args);

	return err;
}

*/
import "C"

// ToHost transfers the contents of buffer stored on device to the host.
// The space in dst has to hold enough space (see Buffer.Size) to hold the required data, or an error is returned.
//
// This always request a major-to-minor layout, the assumption of the layout in host memory -- TPUs are known to
// reorganize the layout.
func (b *Buffer) ToHost(dst []byte) error {
	plugin, err := b.getPlugin()
	if err != nil {
		return err
	}
	defer runtime.KeepAlive(b)

	// We'll need the buffer rank to set up the layout.
	dims, err := b.Dimensions()
	if err != nil {
		return err
	}
	rank := len(dims)

	dstBytes := unsafe.Pointer(unsafe.SliceData(dst))
	var pinner runtime.Pinner
	pinner.Pin(dstBytes)
	defer pinner.Unpin()

	pErr := C.BufferToHost(plugin.api, b.wrapper.c, dstBytes, C.int64_t(len(dst)), C.int(rank))
	err = toError(plugin, pErr)
	if err != nil {
		return errors.WithMessage(err, "Failed to call PJRT_Buffer_ToHostBuffer to transfer the buffer to host")
	}
	return nil
}
