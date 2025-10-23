//go:build darwin && arm64

package main

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
)

func init() {
	for _, plugin := range []string{"darwin"} {
		pluginInstallers[plugin] = DarwinInstall
		pluginValidators[plugin] = DarwinValidateVersion
	}
	pluginValues = append(pluginValues, "darwin")
	pluginDescriptions = append(pluginDescriptions, "CPU PJRT (darwin/arm64)")
	pluginPriorities = append(pluginPriorities, 0)
	installPathSuggestions = append(installPathSuggestions, "/usr/local/", "~/Library/Application Support/GoMLX")
}

// LinuxValidateVersion checks whether the linux version selected by "-version" exists.
func DarwinValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	_, err := DarwinGetDownloadURL(plugin, version)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch PJRT plugin from Gopjrt version %q, see "+
			"https://github.com/gomlx/gopjrt/releases for a list of release versions to choose from", version)
	}
	return err
}

// DarwinGetDownloadURL returns the download URL for the given version and plugin.
func DarwinGetDownloadURL(plugin, version string) (url string, err error) {
	var assets []string
	assets, err = GitHubDownloadReleaseAssets(version)
	if err != nil {
		return
	}

	var wantAsset string
	switch plugin {
	case "darwin":
		wantAsset = "gopjrt_cpu_darwin_arm64.tar.gz"
	default:
		err = errors.Errorf("version validation not implemented for plugin %q in version %s", plugin, version)
		return
	}
	for _, assetURL := range assets {
		if strings.HasSuffix(assetURL, wantAsset) {
			return assetURL, nil
		}
	}
	return "", errors.Errorf("Plugin %q version %q doesn't seem to have the required asset (%q) -- assets found: %v", plugin, version, wantAsset, assets)
}

// DarwinInstall the assets on the target directory.
func DarwinInstall(plugin, version, installPath string) error {
	var err error
	if version == "latest" || version == "" {
		version, err = GitHubGetLatestVersion()
		if err != nil {
			return err
		}
	}
	assetURL, err := DarwinGetDownloadURL(plugin, version)
	if err != nil {
		return err
	}
	assetName := filepath.Base(assetURL)

	// Create the target directory.
	installPath = ReplaceTildeInDir(installPath)
	intsallPath = filepath.Join(installPath, "PJRT")
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrap(err, "failed to create install directory")
	}

	// Download the asset to a temporary file.
	sha256hash := "" // TODO: no hash for github releases. Is there a way to get them (or get a hardcoded table for all versions?)
	downloadedFile, inCache, err := DownloadURLToTemp(assetURL, fmt.Sprintf("gopjrt_%s_%s.tar.gz", plugin, version), sha256hash)
	if err != nil {
		return err
	}
	if !inCache {
		defer func() { ReportError(os.Remove(downloadedFile)) }()
	}

	// Create a temporary file to store the extracted files list.
	listFile, err := os.CreateTemp("", "gopjrt_list_files.*")
	if err != nil {
		return errors.Wrap(err, "failed to create list file")
	}
	if !klog.V(1).Enabled() {
		defer func() { ReportError(os.Remove(listFile.Name())) }()
	}

	// Extract files
	fmt.Printf("- Extracting files in %s to %s\n", downloadedFile, installPath)
	cmd := exec.Command("tar", "xvzf", downloadedFile, "-C", installPath)
	cmd.Stdout = listFile
	klog.V(1).Infof("Running command: %v > %s", cmd.Args, listFile.Name())
	if err := cmd.Run(); err != nil {
		return errors.Wrap(err, "failed to extract files")
	}
	_ = listFile.Close()

	// Read and print the list of extracted files
	fmt.Println("- Extracted files:")
	fileContents, err := os.ReadFile(listFile.Name())
	if err != nil {
		return errors.Wrap(err, "failed to read list of extracted files")
	}
	fmt.Print(string(fileContents))

	fmt.Printf("\n✅ Installed Gopjrt %s \"cpu\" PJRT to %s (%s/%s)\n\n", version, installPath, runtime.GOOS, runtime.GOARCH)
	return nil
}
