// exec_hlo is a trivial testing program to execute HLO programs that take as input only one value.
package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"os"
	"strconv"
)

var (
	flagPluginName          = flag.String("plugin", "cpu", "PRJT plugin name or full path")
	flagHloModuleFile       = flag.String("hlo", "", "File with the serialized HloModuleProto")
	flagSuppressAbslLogging = flag.Bool("suppress_absl_logging", true, "Suppress Abseil logging from the PJRT plugin -- generally it's just noise")
)

// TODO: Accept multiple inputs and multiple outputs, various dtypes, and arrays.

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `exec_hlo will load a binary HLO (HloModuleProto) and execute it.

$ exec_hlo -hlo=<serialized_hlo_file_name> <x>

One can generate HLO using the github.com/gomlx/gopjrt/xlabuilder package, or alternatively using Jax,
see example in https://github.com/gomlx/gopjrt.

For now this tool only works with programs that take one float32 scalar value in, and outputs another float32
scalar value.

Usage:
`)
		flag.PrintDefaults()
	}
	klog.InitFlags(flag.CommandLine)
	flag.Parse()

	if *flagHloModuleFile == "" {
		fmt.Fprintln(os.Stderr, "The HLO program as serialized proto must be given with the --hlo flag!")
		fmt.Fprintln(os.Stderr)
		flag.Usage()
		return
	}
	numStr := flag.Arg(0)
	if numStr == "" {
		fmt.Fprintln(os.Stderr, "No input value given.")
		fmt.Fprintln(os.Stderr)
		flag.Usage()
		return
	}

	// Parse inputs.
	hloBlob := must.M1(os.ReadFile(*flagHloModuleFile))
	input := float32(must.M1(strconv.ParseFloat(numStr, 32)))

	// PJRT plugin and create a client.
	plugin := must.M1(pjrt.GetPlugin(*flagPluginName))
	var client *pjrt.Client
	if *flagSuppressAbslLogging {
		pjrt.SuppressAbseilLoggingHack(func() {
			client = must.M1(plugin.NewClient(nil))
		})
	} else {
		client = must.M1(plugin.NewClient(nil))
	}
	loadedExec := must.M1(client.Compile().WithHLO(hloBlob).Done())

	// Test values:
	inputBuffer := must.M1(pjrt.ScalarToBuffer(client, input))
	outputBuffers := must.M1(loadedExec.Execute(inputBuffer).Done())
	output := must.M1(pjrt.BufferToScalar[float32](outputBuffers[0]))
	fmt.Printf("\tf(x=%g) = %g\n", input, output)
}
