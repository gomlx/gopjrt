package pjrt

import "C"
import (
	"fmt"
	"github.com/gomlx/gopjrt/cbuffer"
	"github.com/gomlx/gopjrt/protos/compile_options"
	"github.com/gomlx/gopjrt/protos/xla"
	"github.com/pkg/errors"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"
	"os"
	"runtime"
	"unsafe"
)

// EnvXlaDebugOptions is an environment variable that can be defined to set XLA DebugOptions proto when compiling
// a program.
const EnvXlaDebugOptions = "XLA_DEBUG_OPTIONS"

// CompileConfig is created with Client.Compile, and is a "builder pattern" to configure a compilation call.
//
// At a minimum one has to set the program to compile (use CompileConfig.WithHLO or CompileConfig.WithComputation).
// Optionally, many other options can be set.
//
// Once finished call CompileConfig.Done to trigger the compilation and get back a LoadedExecutable or an error.
//
// TODO: expose all (or more) configuration options with "WithX" methods.
type CompileConfig struct {
	plugin *Plugin
	client *Client

	// program can be a pointer to C/C++ data, but it must be kept alive until after CompileConfig.Done is called.
	program []byte

	// programFormat supported formats are:
	// "hlo": code string takes serialized HloModuleProto.
	// "hlo_with_config": code string takes serialized HloModuleProtoWithConfig.
	// "mlir": code string takes MLIR module bytecode (or string).
	// Ownership of `format` varies across API functions.
	// See PJRT_Program struct in pjrt_c_api.h
	programFormat string

	// options is the CompileOptionsProto being configured.
	options *compile_options.CompileOptionsProto

	// cbufferToFree is going to be freed after Done is called, if set.
	cbufferToFree *cbuffer.CBuffer

	// err is the first error that occurred during setup.
	err error
}

func newCompileConfig(client *Client) (cc *CompileConfig) {
	cc = &CompileConfig{
		plugin: client.plugin,
		client: client,
		options: &compile_options.CompileOptionsProto{
			ArgumentLayouts:            nil,
			ParameterIsTupledArguments: false,
			ExecutableBuildOptions:     nil,
			CompilePortableExecutable:  true,
			ProfileVersion:             0,
			SerializedMultiSliceConfig: nil,
			EnvOptionOverrides:         nil,
			TargetConfig:               nil,
		},
	}
	// Default values specified in the comments of the proto (but not as proper proto defaults).
	cc.options.ExecutableBuildOptions = &compile_options.ExecutableBuildOptionsProto{
		DeviceOrdinal: -1, // -1 means not set.
		NumReplicas:   1,
		NumPartitions: 1,
	}

	debugOptionsStr := os.Getenv(EnvXlaDebugOptions)
	if debugOptionsStr != "" {
		debugOptions := &xla.DebugOptions{}
		err := prototext.Unmarshal([]byte(debugOptionsStr), debugOptions)
		if err != nil {
			cc.err = errors.Wrapf(err, "Failed to parse xla.DebugOptions protobuf from $%s=%q", EnvXlaDebugOptions, debugOptionsStr)
			return cc
		}
		// Print configuration.
		textBytes, err := prototext.MarshalOptions{Multiline: true}.Marshal(debugOptions)
		if err != nil {
			klog.Infof("Failed to convert xla.DebugOptions proto to text: %v", err)
		} else {
			klog.Infof("This adds 10ms!! of time to the execution of the compiled graph!")
			klog.Infof("%s=%s -> %s", EnvXlaDebugOptions, debugOptionsStr, textBytes)
		}

		// Set parsed configuration.
		cc.options.ExecutableBuildOptions.DebugOptions = debugOptions
	}
	return cc
}

// Done triggers the compilation of the program. If the compilation succeeds a LoadedExecutable is returned, otherwise
// an error is returned.
func (cc *CompileConfig) Done() (*LoadedExecutable, error) {
	if cc.client == nil || cc.plugin == nil {
		return nil, errors.New("misconfigured CompileConfig, or an attempt of using it more than once, which is not supported -- call Client.Compile() again")
	}
	if cc.err != nil {
		return nil, cc.err
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
	if cc.programFormat == "" || len(cc.program) == 0 {
		return nil, errors.New("no program given to Client.Compile(), use Client.Compile().WithComputation() or ClientCompile().WithSLO() " +
			"to specify a program, before calling Done()")
	}

	// Makes sure program data is not moved around by the GC during the C/C++ call.
	var pinner runtime.Pinner
	programPtr := unsafe.SliceData(cc.program)
	pinner.Pin(programPtr)
	defer pinner.Unpin()
	pinner.Pin(cc)

	// Get options and pin it.
	binOptions, err := proto.Marshal(cc.options)
	if klog.V(1).Enabled() {
		klog.Infof("CompileOptions: {\n%s}\n", prototext.Format(cc.options))
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to marshal the CompileOptionsProto to be passed to the PJRT plugin")
	}
	pinner.Pin(unsafe.SliceData(binOptions))

	klog.V(2).Infof("calling pjrtClientCompile()")
	exec, err := pjrtClientCompile(cc.plugin, cc.client, cc.program, cc.programFormat, binOptions)
	if err != nil {
		klog.V(1).Infof("pjrtClientCompile() failed: %v", err)
		return nil, errors.WithMessagef(err, "failed to compile the program")
	}
	klog.V(2).Infof("pjrtClientCompile() succeeded")
	return exec, nil
}

// WithHLO configures the program to the serialized HLO (HloModule proto).
// The serialized proto blob can allocated in Go or in C/C++, and must be kept alive (and unchanged) until the
// call to Done is returned.
//
// Either WithHLO, WithStableHLO or WithComputation must be set, before Done can be called
// to trigger the computation, but not both.
// It panics if more than one version of the program is called.
//
// It returns itself (CompileConfig) to allow cascading configuration calls.
func (cc *CompileConfig) WithHLO(serialized []byte) *CompileConfig {
	if cc.err != nil {
		return cc
	}
	if len(cc.program) > 0 || cc.programFormat != "" {
		cc.err = errors.Errorf("pjrt.Client.Compile() was given the program more than once using WithHLO or WithComputation")
		return cc
	}

	cc.program = serialized
	cc.programFormat = "hlo"
	return cc
}

// WithStableHLO configures the program with a StableHLO program, encoded as a serialized `mlir.ModuleOp` object.
// The serialized proto blob can be allocated in Go or in C/C++, and must be kept alive (and unchanged) until the
// call to Done is returned.
//
// This also works with the program as a human-readable string (with StableHLO code).
//
// Either WithHLO, WithStableHLO or WithComputation must be set, before Done can be called
// to trigger the computation, but not both.
// It panics if more than one version of the program is called.
func (cc *CompileConfig) WithStableHLO(serialized []byte) *CompileConfig {
	if cc.err != nil {
		return cc
	}
	if len(cc.program) > 0 || cc.programFormat != "" {
		cc.err = errors.Errorf("pjrt.Client.Compile() was given the program more than once using WithHLO or WithComputation")
		return cc
	}
	fmt.Printf("Program:\n%s\n", string(serialized))

	cc.program = serialized
	cc.programFormat = "mlir"
	return cc
}

// XlaComputation is an interface that matches xlabuilder.XlaComputation method needed by PJRT.
//
// Created here to avoid creating a hard dependency to the xlabuilder package.
//
// The returned buffer ownership is transferred to the caller -- the caller must free the buffer later.
type XlaComputation interface {
	// SerializedHLO exports the computation as a serialized `HloModuleProto`.
	SerializedHLO() *cbuffer.CBuffer

	// HasStableHLO returns whether StableHLO is supported.
	HasStableHLO() bool

	// SerializedStableHLO exports the computation as a StableHLO as an `mlir:ModuleOp`.
	SerializedStableHLO() (*cbuffer.CBuffer, error)

	// TextStableHLO returns the computation as a "human readable" StableHLO.
	TextStableHLO() (string, error)
}

// WithComputation configures the program to the xlabuilder.XlaComputation -- see xlabuilder package.
// Behind the scenes it is an HLO program (HloModule proto), but this handles the details.
// If plugin.UseStableHLO is set to true, it takes the StableHLO (in MLIR format) program instead.
//
// Either WithHLO, WithStableHLO or WithComputation must be set, before Done can be called
// to trigger the computation, but not both.
// It panics if more than one version of the program is called.
//
// It returns itself (CompileConfig) to allow cascading configuration calls.
func (cc *CompileConfig) WithComputation(computation XlaComputation) *CompileConfig {
	if cc.err != nil {
		return cc
	}
	if len(cc.program) > 0 || cc.programFormat != "" {
		cc.err = errors.Errorf("pjrt.Client.Compile() was given the program more than once using WithHLO or WithComputation")
		return cc
	}

	fmt.Printf("prjt.Compile().WithComputation(): HasStableHLO=%v, UseStableHLO=%v, UseTextStableHLO=%v\n",
		computation.HasStableHLO(), cc.plugin.UseStableHLO, cc.plugin.UseTextStableHLO)

	// Get HLO program from computation.
	if cc.plugin.UseTextStableHLO && computation.HasStableHLO() {
		// Convert computation graph to StableHLO
		text, err := computation.TextStableHLO()
		if err != nil {
			cc.err = err
			return cc
		}
		cc.cbufferToFree = cbuffer.NewFromString(text, true)
		cc.WithStableHLO(cc.cbufferToFree.Bytes())
	} else if cc.plugin.UseStableHLO && computation.HasStableHLO() {
		// Convert computation graph to StableHLO
		var err error
		cc.cbufferToFree, err = computation.SerializedStableHLO()
		if err != nil {
			cc.err = err
			return cc
		}
		cc.WithStableHLO(cc.cbufferToFree.Bytes())
	} else {
		// Use XlaBuilder HLO:
		cc.cbufferToFree = computation.SerializedHLO()
		cc.WithHLO(cc.cbufferToFree.Bytes())
	}
	return cc
}
