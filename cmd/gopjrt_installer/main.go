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

var (
	pluginValues           []string
	pluginDescriptions     []string
	pluginPriorities       []int // Order to display the plugins: smaller values are displayed first.
	pluginInstallers       = make(map[string]func(plugin, version, installPath string) error)
	pluginValidators       = make(map[string]func(plugin, version string) error)
	installPathSuggestions []string

	flagPlugin, flagPath *string
	flagVersion          = flag.String("version", "latest",
		"In most PJRT this is the Gopjrt release version in https://github.com/gomlx/gopjrt (e.g.: v0.8.4) from where to download the plugin. "+
			"For the CUDA PJRT this is based on the Jax version in https://pypi.org/project/jax/ (e.g.: 0.7.2), which is where it "+
			"downloads the plugin and Nvidia libraries from.")
)

func main() {
	// Initialize and set default values for flags
	klog.InitFlags(nil)

	// Sort plugins by priority
	for i := 0; i < len(pluginPriorities); i++ {
		for j := i + 1; j < len(pluginPriorities); j++ {
			if pluginPriorities[i] > pluginPriorities[j] {
				pluginPriorities[i], pluginPriorities[j] = pluginPriorities[j], pluginPriorities[i]
				pluginValues[i], pluginValues[j] = pluginValues[j], pluginValues[i]
				pluginDescriptions[i], pluginDescriptions[j] = pluginDescriptions[j], pluginDescriptions[i]
			}
		}
	}

	// Define flags with plugins configured for GOOS/GOARCH used to build this binary:
	flagPlugin = flag.String("plugin", "", "Plugin to install. Possible values: "+strings.Join(pluginValues, ", "))
	flagPath = flag.String("path", "~/.local",
		fmt.Sprintf("Installation base path, under which the required libraries and include files are installed. "+
			"It installs files under lib/ and include/ subdirectories. "+
			"For the PJRT plugins it creates a sub-directory lib/gomlx/prjt, and in case of CUDA plugins, gomlx/nvidia for "+
			"Nvidia's matching drivers. Suggestions: %s. "+
			"It will require the adequate privileges (sudo) if installing in a system directories.",
			strings.Join(installPathSuggestions, ", ")))
	*flagPath = installPathSuggestions[0]

	// Parse flags.
	flag.Parse()

	if *flagPlugin == "" || *flagPath == "" || *flagVersion == "" {
		questions := []Question{
			{Title: "Plugin to install", Flag: flag.CommandLine.Lookup("plugin"),
				Values: pluginValues, ValuesDescriptions: pluginDescriptions, CustomValues: false},
			{Title: "Plugin version", Flag: flag.CommandLine.Lookup("version"), Values: []string{"latest"}, CustomValues: true,
				ValidateFn: ValidateVersion},
			{Title: "Path where to install", Flag: flag.CommandLine.Lookup("path"), Values: installPathSuggestions, CustomValues: true,
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

	pluginInstaller, ok := pluginInstallers[pluginName]
	if !ok {
		klog.Fatalf("Installer for plugin %q not found", pluginName)
	}
	if err := pluginInstaller(pluginName, version, installPath); err != nil {
		klog.Fatal(err)
	}
}

// ValidateVersion is called to validate the version of the plugin chosen by the user during the interactive mode.
func ValidateVersion() error {
	validator, ok := pluginValidators[*flagPlugin]
	if !ok {
		return errors.Errorf("version validation not implemented for plugin %q", *flagPlugin)
	}
	return validator(*flagPlugin, *flagVersion)
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
		ReportError(f.Close())

		// Clean up test file
		if err = os.Remove(testFile); err != nil {
			return errors.Wrapf(err, "failed to remove test file %q", testFile)
		}
	}

	return nil
}
