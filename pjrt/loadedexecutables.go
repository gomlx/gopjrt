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

// LoadedExecutable is a reference to a compiled program ready to be executed.
type LoadedExecutable struct {
	cLoadedExecutable *C.PJRT_LoadedExecutable
	plugin            *Plugin
}

// newLoadedExecutable creates LoadedExecutable and registers it for freeing.
func newLoadedExecutable(plugin *Plugin, cLoadedExecutable *C.PJRT_LoadedExecutable) *LoadedExecutable {
	e := &LoadedExecutable{
		plugin:            plugin,
		cLoadedExecutable: cLoadedExecutable,
	}
	runtime.SetFinalizer(e, func(e *LoadedExecutable) {
		err := e.Destroy()
		if err != nil {
			klog.Errorf("LoadedExecutable.Destroy failed: %v", err)
		}
	})
	return e
}

// Destroy the LoadedExecutable, release resources, and LoadedExecutable is no longer valid.
// This is automatically called if LoadedExecutable is garbage collected.
func (e *LoadedExecutable) Destroy() error {
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_LoadedExecutable_Destroy_Args()
	defer cFree(args)
	args.executable = e.cLoadedExecutable
	err := toError(e.plugin, C.call_PJRT_LoadedExecutable_Destroy(e.plugin.api, args))
	e.plugin = nil
	e.cLoadedExecutable = nil
	return err
}

func (e *LoadedExecutable) GetExecutable() (*Executable, error) {
	if e == nil || e.plugin == nil || e.cLoadedExecutable == nil {
		return nil, errors.New("LoadedExecutable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_LoadedExecutable_GetExecutable_Args()
	defer cFree(args)
	args.loaded_executable = e.cLoadedExecutable
	err := toError(e.plugin, C.call_PJRT_LoadedExecutable_GetExecutable(e.plugin.api, args))
	if err != nil {
		return nil, err
	}
	return newExecutable(e.plugin, args.executable), nil
}
