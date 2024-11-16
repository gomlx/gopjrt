package pjrt

import (
	"flag"
	"fmt"
	"github.com/gomlx/gopjrt/protos/hlo"
	"github.com/janpfeifer/must"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"os"
	"runtime"
	"testing"
)

var flagLoadHLO = flag.String("loadhlo", "", "Load HLO to test from file (as opposed to the one created with XlaBuilder.")

// hloText describes the following program:
//
//	HloModule x_x_1.5, entry_computation_layout={(f32[])->f32[]}
//	ENTRY x_x_1.5 {
//		x.1 = f32[] parameter(0)
//		multiply.2 = f32[] multiply(x.1, x.1)
//		constant.3 = f32[] constant(1)
//		ROOT add.4 = f32[] add(multiply.2, constant.3)
//	}
var hloText = `name:"x*x+1.5" entry_computation_name:"x*x+1.5" entry_computation_id:5 computations:{name:"x*x+1.5" ` +
	`instructions:{name:"x.1" opcode:"parameter" shape:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} metadata:{} id:1 frontend_attributes:{}} instructions:{name:"multiply.2" opcode:"multiply" shape:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} metadata:{} id:2 operand_ids:1 operand_ids:1 frontend_attributes:{}} instructions:{name:"constant.3" opcode:"constant" shape:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} metadata:{} literal:{shape:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} f32s:1} id:3 frontend_attributes:{}} instructions:{name:"add.4" opcode:"add" shape:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} metadata:{} id:4 operand_ids:2 operand_ids:3 frontend_attributes:{}} program_shape:{parameters:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} result:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} parameter_names:"x"} id:5 root_id:4} host_program_shape:{parameters:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} result:{element_type:F32 layout:{tail_padding_alignment_in_elements:1}} parameter_names:"x"} id:5`

// TestMinimal is a minimal end-to-end test of loading, compiling and executing a small program using PJRT.
//
// Set (export) XLA_FLAGS=--xla_dump_to=/tmp/xla_dump to get details of its compilation.
func TestMinimal(t *testing.T) {
	// Load HLO program.
	var hloSerialized []byte
	var hloModule hlo.HloModuleProto
	if *flagLoadHLO != "" {
		fmt.Printf("Loading HLO program from %s...\n", *flagLoadHLO)
		hloSerialized = must.M1(os.ReadFile(*flagLoadHLO))
		must.M(proto.Unmarshal(hloSerialized, &hloModule))
	} else {
		// Serialize HLO program from hloText:
		var hloModule hlo.HloModuleProto
		must.M(prototext.Unmarshal([]byte(hloText), &hloModule))
		hloSerialized = must.M1(proto.Marshal(&hloModule))
	}
	fmt.Printf("HLO Program:\n%s\n\n", hloModule.String())

	// `dlopen` PJRT plugin.
	plugin := must.M1(GetPlugin("cpu"))
	defer runtime.KeepAlive(plugin)
	fmt.Printf("PJRT: %s\n", plugin.String())

	// Create client.
	client := must.M1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)
	devices := client.AddressableDevices()
	for ii, dev := range devices {
		desc := must.M1(dev.GetDescription())
		fmt.Printf("\tDevice #%d: %s\n", ii, desc.DebugString())
	}

	// Compile.
	defer runtime.KeepAlive(hloSerialized)
	loadedExec := must.M1(client.Compile().WithHLO(hloSerialized).Done())
	defer runtime.KeepAlive(loadedExec)
	fmt.Printf("\t- program compiled successfully.\n")

	// Test values:
	inputs := []float32{0.1, 1, 3, 4, 5}
	wants := []float32{1.01, 2, 10, 17, 26}
	fmt.Printf("f(x) = x^2 + 1:\n")
	for ii, input := range inputs {
		// Transfer input to an on-device buffer.
		inputBuffer, err := ScalarToBufferOnDeviceNum(client, 0, input)
		require.NoErrorf(t, err, "Failed to create on-device buffer for input %v, deviceNum=%d", input, 0)

		// Execute: it returns the output on-device buffer(s).
		outputBuffers, err := loadedExec.Execute(inputBuffer).OnDevicesByNum(0).Done()
		require.NoErrorf(t, err, "Failed to execute on input %d, deviceNum=%d", input, 0)

		// Transfer output on-device buffer to a "host" value (in Go).
		output, err := BufferToScalar[float32](outputBuffers[0])
		require.NoErrorf(t, err, "Failed to transfer results of execution on input %d", input)

		// Print and check value is what we wanted.
		fmt.Printf("\tf(x=%g) = %g\n", input, output)
		require.InDelta(t, output, wants[ii], 0.001)

		// Release inputBuffer -- and don't wait for the GC.
		require.NoError(t, inputBuffer.Destroy())
	}
}
