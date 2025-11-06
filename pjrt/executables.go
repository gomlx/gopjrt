package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"runtime"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Executable is a reference that describes a compiled program -- it cannot be executed, only introspected.
//
// This is usually not directly used: the LoadedExecutable when create automatically extracts the Executable
// and related information.
type Executable struct {
	wrapper *executableWrapper
}

type executableWrapper struct {
	c      *C.PJRT_Executable
	plugin *Plugin
}

// ExecutableMemoryUsageStats reports the static memory usage for a compiled program, in bytes.
// The on-device memory needed to run an executable is at least:
// GeneratedCode + Inputs + Outputs - Aliases + Temporary. See ExecutableMemoryUsageStats.Requirements.
//
// Aliases is how much memory of the input is reused as output (?).
//
// The documentation is sparse in XLA, here are the links:
//
//   - xla::CompiledMemoryStats: https://github.com/openxla/xla/blob/2fff53249ed49930de14b235f50ed2235e69df8b/xla/pjrt/pjrt_executable.h#L284
//   - PJRT C API:https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h#L1668
type ExecutableMemoryUsageStats struct {
	GeneratedCode, Inputs, Outputs, Aliases, Temporary int64
}

// Requirements returns an estimate of memory requirements for the executable.
func (m ExecutableMemoryUsageStats) Requirements() int64 {
	return m.GeneratedCode + m.Inputs + m.Outputs - m.Aliases + m.Temporary
}

// newExecutable creates Executable and registers it for freeing.
func newExecutable(plugin *Plugin, cExecutable *C.PJRT_Executable) *Executable {
	e := &Executable{wrapper: &executableWrapper{
		plugin: plugin,
		c:      cExecutable,
	}}
	runtime.AddCleanup(e, func(wrapper *executableWrapper) {
		err := wrapper.Destroy()
		if err != nil {
			klog.Errorf("pjrt.Executable.Destroy failed: %v", err)
		}
	}, e.wrapper)
	return e
}

func (wrapper *executableWrapper) IsValid() bool {
	return wrapper != nil && wrapper.c != nil && wrapper.plugin != nil
}

func (wrapper *executableWrapper) Destroy() error {
	if !wrapper.IsValid() {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(wrapper)
	args := C.new_PJRT_Executable_Destroy_Args()
	defer cFree(args)
	args.executable = wrapper.c
	err := toError(wrapper.plugin, C.call_PJRT_Executable_Destroy(wrapper.plugin.api, args))
	wrapper.plugin = nil
	wrapper.c = nil
	return err
}

// Destroy the Executable releasing immediately its resources.
// After this the Executable is no longer valid, and shouldn't be used.
// This is automatically called if Executable is garbage collected.
func (e *Executable) Destroy() error {
	if e == nil {
		return nil
	}
	return e.wrapper.Destroy()
}

// destroyOrLog destroys the Executable and log any errors.
func (e *Executable) destroyOrLog() {
	err := e.Destroy()
	if err != nil {
		klog.Errorf("Executable.Destroy failed: %v", err)
	}
}

// NumOutputs returns the number of outputs for the given executable.
func (e *Executable) NumOutputs() (int, error) {
	if e == nil || !e.wrapper.IsValid() {
		return 0, errors.New("Executable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_Executable_NumOutputs_Args()
	defer cFree(args)
	args.executable = e.wrapper.c
	err := toError(e.wrapper.plugin, C.call_PJRT_Executable_NumOutputs(e.wrapper.plugin.api, args))
	if err != nil {
		return 0, err
	}
	return int(args.num_outputs), nil
}

// Name returns the name of the executable.
func (e *Executable) Name() (string, error) {
	if e == nil || !e.wrapper.IsValid() {
		return "", errors.New("Executable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_Executable_Name_Args()
	defer cFree(args)
	args.executable = e.wrapper.c
	err := toError(e.wrapper.plugin, C.call_PJRT_Executable_Name(e.wrapper.plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.executable_name, args.executable_name_size), nil
}

// GetMemoryStats returns the sizes (in bytes) for the compiled code, inputs, outputs, aliases and temporary memory
// used both in host and on device.
//
// This can be used to estimate memory requirements for the program.
func (e *Executable) GetMemoryStats() (onDevice, onHost ExecutableMemoryUsageStats, err error) {
	if e == nil || !e.wrapper.IsValid() {
		err = errors.New("Executable is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
		return
	}
	defer runtime.KeepAlive(e)

	arena := e.wrapper.plugin.getDefaultArena()
	defer e.wrapper.plugin.returnArena(arena)

	var args *C.PJRT_Executable_GetCompiledMemoryStats_Args

	args = arenaAlloc[C.PJRT_Executable_GetCompiledMemoryStats_Args](arena)
	args.struct_size = C.PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE
	args.executable = e.wrapper.c
	err = toError(e.wrapper.plugin, C.call_PJRT_Executable_GetCompiledMemoryStats(e.wrapper.plugin.api, args))
	if err != nil {
		return
	}

	onDevice = ExecutableMemoryUsageStats{
		GeneratedCode: int64(args.generated_code_size_in_bytes),
		Inputs:        int64(args.argument_size_in_bytes),
		Outputs:       int64(args.output_size_in_bytes),
		Aliases:       int64(args.alias_size_in_bytes),
		Temporary:     int64(args.temp_size_in_bytes),
	}
	onHost = ExecutableMemoryUsageStats{
		GeneratedCode: int64(args.host_generated_code_size_in_bytes),
		Inputs:        int64(args.host_argument_size_in_bytes),
		Outputs:       int64(args.host_output_size_in_bytes),
		Aliases:       int64(args.host_alias_size_in_bytes),
		Temporary:     int64(args.host_temp_size_in_bytes),
	}
	return
}
