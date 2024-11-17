package pjrt

/*
#include <stdlib.h>
#include <dlfcn.h>
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"github.com/pkg/errors"
	"unsafe"
)

// pjrtErrorDestroy calls C.PJRT_Error_Destroy.
func pjrtErrorDestroy(plugin *Plugin, pErr *C.PJRT_Error) {
	args := C.new_PJRT_Error_Destroy_Args()
	defer cFree(args)
	args.error = pErr
	C.call_PJRT_Error_Destroy(plugin.api, args)
}

// pjrtErrorMessage calls C.PJRT_Error_Message and returns the message from an error structure.
func pjrtErrorMessage(plugin *Plugin, pErr *C.PJRT_Error) string {
	args := C.new_PJRT_Error_Message_Args()
	defer cFree(args)
	args.error = pErr
	C.call_PJRT_Error_Message(plugin.api, args)
	msg := string(unsafe.Slice((*byte)(unsafe.Pointer(args.message)), int(args.message_size))) // Notice bytes are copied.
	// I'm guessing the C-message memory is still owned by the PJRT_Error structure, so I'm not freeing it.
	// The C++ PJRT Client library also doesn't free the message.
	return msg
}

// pjrtErrorGetCode calls C.PJRT_Error_GetCode.
func pjrtErrorGetCode(plugin *Plugin, pErr *C.PJRT_Error) PJRT_Error_Code {
	args := C.new_PJRT_Error_GetCode_Args()
	defer cFree(args)
	args.error = pErr
	C.call_PJRT_Error_GetCode(plugin.api, args)
	code := PJRT_Error_Code(args.code)
	return code
}

// toError converts a *C.PJRT_Error to a Go error, with a stack trace (see 	github.com/pkg/errors package).
// If the incoming error is nil or not an error, it returns nil as well.
// At the end this frees the returned error.
func toError(plugin *Plugin, pErr *C.PJRT_Error) error {
	if pErr == nil {
		return nil
	}
	msg := pjrtErrorMessage(plugin, pErr)
	code := pjrtErrorGetCode(plugin, pErr)
	pjrtErrorDestroy(plugin, pErr)
	return errors.Errorf("PJRT error (code=%d): %s", code, msg)
}
