package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
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

	pathSuggestions = []string{"/usr/local/", "~/.local"}
	flagPath        = flag.String("path", "~/.local",
		fmt.Sprintf("Path where the PJRT plugin will be installed. "+
			"It will create a sub-directory gomlx/prjt for the PJRT plugin, and in case of CUDA plugins, gomlx/nvidia for "+
			"Nvidia's matching drivers. Suggestions: %s. It will require the adequate privileges if installing in system directories.",
			strings.Join(pathSuggestions, ", ")))

	flagVersion = flag.String("version", "latest",
		"The version of the PJRT plugin to install. It defaults to the latest version.")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagPlugin == "" || *flagPath == "" || *flagVersion == "" {
		questions := []Question{
			{Name: "Plugin to install", Flag: flag.CommandLine.Lookup("plugin"), Values: pluginValues, CustomValues: false},
			{Name: "Plugin version", Flag: flag.CommandLine.Lookup("version"), Values: []string{"latest"}, CustomValues: true,
				ValidateFn: ValidateVersion},
			{Name: "Path where to install", Flag: flag.CommandLine.Lookup("path"), Values: pathSuggestions, CustomValues: true,
				ValidateFn: ValidatePathPermission},
		}
		err := Interact(os.Args[0], questions)
		if err != nil {
			klog.Fatal(err)
		}
	}

	pluginName := *flagPlugin
	version := *flagVersion
	installPath := ReplaceTildeInDir(*flagPath)
	fmt.Printf("Installing PJRT plugin %s@%s to %s:\n", pluginName, version, installPath)

	var err error
	switch pluginName {
	case "linux", AmazonLinux:
		err = LinuxInstall()
	default:
		err = errors.Errorf("plugin %q installation not implemented", pluginName)
	}
	if err != nil {
		klog.Fatal(err)
	}
}

func ValidateVersion() error {
	if *flagPlugin == "linux" || *flagPlugin == AmazonLinux {
		return LinuxValidateVersion()
	}
	return errors.Errorf("version validation not implemented for plugin %q", *flagPlugin)
}

func ValidatePathPermission() error {
	installPath := ReplaceTildeInDir(*flagPath)

	// Check permissions in lib/ and include/ subdirectories
	dirsToCheck := []string{
		filepath.Join(installPath, "lib/"),
		filepath.Join(installPath, "include/gomlx"),
	}

	for _, dir := range dirsToCheck {
		// Check if the directory exists: if it doesn't exist, try to create it in the parent directories
		_, err := os.Stat(dir)
		if err != nil {
			// If the directory doesn't exist, try parent directories
			parent := dir
			for {
				parent = filepath.Dir(parent)
				if parent == "/" || parent == "." {
					return errors.New("could not find an existing parent directory")
				}
				if _, err := os.Stat(parent); err == nil {
					dir = parent
					break
				}
			}
		}

		// Try to create a temporary file to verify write permissions
		testFile := filepath.Join(dir, ".gopjrt_write_test")
		f, err := os.Create(testFile)
		if err != nil {
			return errors.Wrapf(err, "no write permission in directory %q, do you need \"sudo\" ?", dir)
		}
		f.Close()

		// Clean up test file
		if err = os.Remove(testFile); err != nil {
			return errors.Wrapf(err, "failed to remove test file %q", testFile)
		}
	}

	return nil
}
