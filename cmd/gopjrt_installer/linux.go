//go:build (linux && amd64) || all

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

const AmazonLinux = "amazonlinux"

func init() {
	for _, plugin := range []string{"linux", AmazonLinux} {
		pluginInstallers[plugin] = LinuxInstall
		pluginValidators[plugin] = LinuxValidateVersion
	}
	pluginValues = append(pluginValues, "linux", AmazonLinux)
	pluginDescriptions = append(pluginDescriptions,
		"CPU PJRT for Linux/amd64 (glibc >= 2.41",
		"CPU PJRT for AmazonLinux/amd64 and Ubuntu 22 (GCP host systems for TPUs) (glibc >= 2.35)")
	pluginPriorities = append(pluginPriorities, 0, 1)
	installPathSuggestions = append(installPathSuggestions, "/usr/local/", "~/.local")
}

// LinuxValidateVersion checks whether the linux version selected by "-version" exists.
func LinuxValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	_, err := LinuxGetDownloadURL(plugin, version)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch PJRT plugin from Gopjrt version %q, see "+
			"https://github.com/gomlx/gopjrt/releases for a list of release versions to choose from", version)
	}
	return err
}

// LinuxGetDownloadURL returns the download URL for the given version and plugin.
func LinuxGetDownloadURL(plugin, version string) (url string, err error) {
	var assets []string
	assets, err = GitHubDownloadReleaseAssets(version)
	if err != nil {
		return
	}

	var wantAsset string
	switch plugin {
	case "linux":
		wantAsset = "gomlx_xlabuilder_linux_amd64.tar.gz"
	case AmazonLinux:
		wantAsset = "gomlx_xlabuilder_linux_amd64_amazonlinux.tar.gz"
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

// LinuxInstall the assets on the target directory.
func LinuxInstall(plugin, version, installPath string) error {
	var err error
	if version == "latest" || version == "" {
		version, err = GitHubGetLatestVersion()
		if err != nil {
			return err
		}
	}
	assetURL, err := LinuxGetDownloadURL(plugin, version)
	if err != nil {
		return err
	}
	assetName := filepath.Base(assetURL)

	// Create the target directory.
	installPath = ReplaceTildeInDir(installPath)
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrap(err, "failed to create install directory")
	}

	// Download the asset to a temporary file.
	sha256hash := "" // TODO: no hash for github releases. Is there a way to get them (or get a hardcoded table for all versions?)
	downloadedFile, inCache, err := DownloadURLToTemp(assetURL, fmt.Sprintf("%s_%s", version, assetName), sha256hash)
	if err != nil {
		return err
	}
	if !inCache {
		defer func() { ReportError(os.Remove(downloadedFile)) }()
	}

	// Extract files
	fmt.Printf("- Extracting files in %s to %s\n", downloadedFile, installPath)
	extractedFiles, err := Untar(downloadedFile, installPath)
	if err != nil {
		return err
	}
	if len(extractedFiles) == 0 {
		return errors.Errorf("failed to extract files from %s", downloadedFile)
	}
	fmt.Printf("- Extracted %d file(s):\n", len(extractedFiles))
	for _, file := range extractedFiles {
		fmt.Printf("  - %s\n", file)
	}

	// Remove older version using dynamically linked library
	oldLib := filepath.Join(installPath, "lib/libgomlx_xlabuilder.so")
	if err := os.Remove(oldLib); err != nil && !os.IsNotExist(err) {
		ReportError(err)
	}

	fmt.Printf("\n✅ Installed Gopjrt %s libraries and \"cpu\" PJRT to %s (%s platform)\n\n", version, installPath, plugin)

	return nil
}
