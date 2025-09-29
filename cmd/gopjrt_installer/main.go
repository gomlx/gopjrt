package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

const AmazonLinux = "amazonlinux"

var (
	pluginValues = []string{"linux", AmazonLinux, "cuda12", "cuda13"}
	flagPlugin   = flag.String("plugin", "",
		fmt.Sprintf("PJRT plugin to install, one of: %s. "+
			"The CUDA plugins will download the PJRT and Nvidia drivers included in Jax distribution for "+
			"pypi.org (but it doesn't use Python to install them)", strings.Join(pluginValues, ", ")))

	pathSuggestions = []string{"/usr/local/lib/", "~/.local/lib"}
	flagPath        = flag.String("path", "~/.local/lib",
		fmt.Sprintf("Path where the PJRT plugin will be installed. "+
			"It will create a sub-directory gomlx/prjt for the PJRT plugin, and in case of CUDA plugins, gomlx/nvidia for "+
			"Nvidia's matching drivers. Suggestions: %s. It will require the adequate privileges if installing in system directories.",
			strings.Join(pathSuggestions, ", ")))

	flagVersion = flag.String("version", "latest",
		"The version of the PJRT plugin to install. It defaults to the latest version.")
)

func main() {
	flag.Parse()

	if *flagPlugin == "" || *flagPath == "" || *flagVersion == "" {
		questions := []Question{
			{Name: "Plugin to install", Flag: flag.CommandLine.Lookup("plugin"), Values: pluginValues, CustomValues: false},
			{Name: "Plugin version", Flag: flag.CommandLine.Lookup("version"), Values: []string{"latest"}, CustomValues: true,
				ValidateFn: ValidateVersion},
			{Name: "Path where to install", Flag: flag.CommandLine.Lookup("path"), Values: pathSuggestions, CustomValues: true},
		}
		err := Interact(os.Args[0], questions)
		if err != nil {
			klog.Fatal(err)
		}
	}

	pluginName := *flagPlugin
	version := *flagVersion
	installPath := ReplaceTildeInDir(*flagPath)
	fmt.Printf("Installing PJRT plugin %q@%q to %q \n", pluginName, version, installPath)
}

func ValidateVersion() error {
	if *flagPlugin == "linux" || *flagPlugin == AmazonLinux {
		return LinuxValidateVersion()
	}
	return errors.Errorf("version validation not implemented for plugin %q", *flagPlugin)
}
