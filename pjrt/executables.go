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
)

// Executable is a reference that describes a compiled program -- it cannot be executed, only introspected.
// It is created by a LoadedExecutable. LoadedExecutable is also the one that can be executed.
type Executable struct {
	cExecutable *C.PJRT_Executable
	plugin      *Plugin
}

// newExecutable creates Executable and registers it for freeing.
func newExecutable(plugin *Plugin, cExecutable *C.PJRT_Executable) *Executable {
	e := &Executable{
		plugin:      plugin,
		cExecutable: cExecutable,
	}
	runtime.SetFinalizer(e, func(e *Executable) {
		err := e.Destroy()
		if err != nil {
			klog.Errorf("Executable.Destroy failed: %v", err)
		}
	})
	return e
}

// Destroy the Executable, release resources, and Executable is no longer valid.
// This is automatically called if Executable is garbage collected.
func (e *Executable) Destroy() error {
	if e == nil || e.plugin == nil || e.cExecutable == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_Executable_Destroy_Args()
	defer cFree(args)
	args.executable = e.cExecutable
	err := toError(e.plugin, C.call_PJRT_Executable_Destroy(e.plugin.api, args))
	e.plugin = nil
	e.cExecutable = nil
	return err
}

// NumOutputs returns the number of outputs for the given executable.
func (e *Executable) NumOutputs() (int, error) {
	if e == nil || e.plugin == nil || e.cExecutable == nil {
		return 0, errors.New("Executable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_Executable_NumOutputs_Args()
	defer cFree(args)
	args.executable = e.cExecutable
	err := toError(e.plugin, C.call_PJRT_Executable_NumOutputs(e.plugin.api, args))
	if err != nil {
		return 0, err
	}
	return int(args.num_outputs), nil
}
