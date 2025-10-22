package main

import (
	"flag"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	pluginValues           []string
	pluginDescriptions     []string
	pluginPriorities       []int // Order to display the plugins: smaller values are displayed first.
	pluginInstallers       = make(map[string]func() error)
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
	installPathSuggestions = DefaultInstallPaths()
	*flagPath = installPathSuggestions[0]

	flagPlugin = flag.String("plugin", "", "Plugin to install. Possible values: linux, cuda12, cuda13")
	flagPath = flag.String("path", "~/.local",
		fmt.Sprintf("Installation base path, under which the required libraries and include files are installed. "+
			"It installs files under lib/ and include/ subdirectories. "+
			"For the PJRT plugins it creates a sub-directory lib/gomlx/prjt, and in case of CUDA plugins, gomlx/nvidia for "+
			"Nvidia's matching drivers. Suggestions: %s. "+
			"It will require the adequate privileges (sudo) if installing in a system directories.",
			strings.Join(installPathSuggestions, ", ")))

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

	var err error
	switch pluginName {
	case "linux", AmazonLinux:
		err = LinuxInstall()
	case "cuda12", "cuda13":
		err = CudaInstall()
	default:
		err = errors.Errorf("plugin %q installation not implemented", pluginName)
	}
	if err != nil {
		klog.Fatal(err)
	}
}

// DefaultInstallPaths is called before parsing of the flags to set the available installation paths, as well as
// setting the default one.
func DefaultInstallPaths() []string {
	currentUser, err := user.Current()
	if err != nil {
		klog.Errorf("Failed to get current user: %v", err)
		return []string{"~/.local", "/usr/local"}
	}

	switch runtime.GOOS {
	case "darwin":
		return []string{
			filepath.Join(currentUser.HomeDir, "Library/Application Support"),
			"/usr/local",
		}
	default: // Assuming Linux
		return []string{"~/.local", "/usr/local"}
	}
}

// ValidateVersion is called to validate the version of the plugin chosen by the user during the interactive mode.
func ValidateVersion() error {
	switch *flagPlugin {
	case "linux", AmazonLinux:
		return LinuxValidateVersion()
	case "cuda12", "cuda13":
		return CudaValidateVersion()
	default:
		return errors.Errorf("version validation not implemented for plugin %q", *flagPlugin)
	}
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
