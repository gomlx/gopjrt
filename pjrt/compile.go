package pjrt

import (
	"github.com/gomlx/exceptions"
	"github.com/pkg/errors"
	"gopjrt/cbuffer"
	"runtime"
	"unsafe"
)

// CompileConfig is created with Client.Compile, and is a "builder pattern" to configure a compilation call.
//
// At a minimum one has to set the program to compile (use CompileConfig.WithHLO or CompileConfig.WithComputation).
// Optionally, many other options can be set.
//
// Once finished call CompileConfig.Done to trigger the compilation and get back a LoadedExecutable or an error.
type CompileConfig struct {
	plugin *Plugin
	client *Client

	// program can be a pointer to C/C++ data, but it must be kept alive until after CompileConfig.Done is called.
	program []byte

	// programType supported formats are:
	// "hlo": code string takes serialized HloModuleProto.
	// "hlo_with_config": code string takes serialized HloModuleProtoWithConfig.
	// "mlir": code string takes MLIR module bytecode (or string).
	// Ownership of `format` varies across API functions.
	// See PJRT_Program struct in pjrt_c_api.h
	programType string

	// cbufferToFree is going to be freed after Done is called, if set.
	cbufferToFree *cbuffer.CBuffer
}

// Done triggers the compilation of the program. If the compilation succeeds a LoadedExecutable is returned, otherwise
// an error is returned.
func (cc *CompileConfig) Done() (*LoadedExecutable, error) {
	if cc.client == nil || cc.plugin == nil {
		return nil, errors.New("misconfigured CompileConfig, or an attempt of using it more than once, which is not supported -- call Client.Compile() again")
	}

	// Make sure things are cleaned up before leaving:
	defer func() {
		if cc.cbufferToFree != nil {
			cc.cbufferToFree.Free()
		}

		// CompileConfig can only be used once.
		cc.client = nil
		cc.plugin = nil
	}()

	// Other sanity checks.
	if cc.programType == "" || len(cc.program) == 0 {
		return nil, errors.New("no program given to Client.Compile(), use Client.Compile().WithComputation() or ClientCompile().WithSLO() " +
			"to specify a program, before calling Done()")
	}

	// Makes sure program data is not moved around by the GC during the C/C++ call.
	var pinner runtime.Pinner
	programPtr := unsafe.SliceData(cc.program)
	pinner.Pin(programPtr)
	defer pinner.Unpin()

	// Get options.
	// pjrtClientCompile(...)
	return nil, nil
}

// WithHLO configures the program to the serialized HLO (HloModule proto).
// The serialized proto blob can allocated in Go or in C/C++, and must be kept alive (and unchanged) until the
// call to Done is returned.
//
// Either WithHLO or WithComputation must be set, before Done can be called to trigger the computation, but not both.
// It panics if more than one WithHLO or WithComputation is called.
//
// It returns itself (CompileConfig) to allow cascading configuration calls.
func (cc *CompileConfig) WithHLO(serialized []byte) *CompileConfig {
	if len(cc.program) > 0 || cc.programType != "" {
		exceptions.Panicf("pjrt.Client.Compile() was given the program more than once using WithHLO or WithComputation")
	}

	cc.program = serialized
	cc.programType = "hlo"
	return cc
}

// XlaComputation is an interface that matches xlabuilder.XlaComputation method needed by PJRT.
//
// Created here to avoid creating a hard dependency to the xlabuilder package.
type XlaComputation interface {
	SerializedHLO() *cbuffer.CBuffer
}

// WithComputation configures the program to the xlabuilder.XlaComputation -- see xlabuilder package.
// Behind the scenes it is an HLO program (HloModule proto), but this handles the details.
//
// Either WithHLO or WithComputation must be set, before Done can be called to trigger the computation, but not both.
// It panics if more than one WithHLO or WithComputation is called.
//
// It returns itself (CompileConfig) to allow cascading configuration calls.
func (cc *CompileConfig) WithComputation(computation XlaComputation) *CompileConfig {
	if len(cc.program) > 0 || cc.programType != "" {
		exceptions.Panicf("pjrt.Client.Compile() was given the program more than once using WithHLO or WithComputation")
	}

	// Get HLO program from computation.
	cc.cbufferToFree = computation.SerializedHLO()
	return cc.WithHLO(cc.cbufferToFree.AsBytes())
}
