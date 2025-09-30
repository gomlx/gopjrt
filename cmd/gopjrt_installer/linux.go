package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// LinuxValidateVersion checks whether the linux version selected by "-version" exists.
func LinuxValidateVersion() error {
	// "latest" is always valid.
	if *flagVersion == "latest" {
		return nil
	}

	_, err := LinuxGetDownloadURL(*flagPlugin, *flagVersion)
	return err
}

// LinuxGetDownloadURL returns the download URL for the given version and plugin.
func LinuxGetDownloadURL(plugin, version string) (string, error) {
	assets, err := LinuxDownloadReleaseAssets(version)
	if err != nil {
		return "", err
	}

	var wantAsset string
	switch plugin {
	case "linux":
		wantAsset = "gomlx_xlabuilder_linux_amd64.tar.gz"
	case AmazonLinux:
		wantAsset = "gomlx_xlabuilder_linux_amd64_amazonlinux.tar.gz"
	default:
		return "", errors.Errorf("version validation not implemented for plugin %q in version %s", plugin, version)
	}
	for _, asset := range assets {
		if strings.HasSuffix(asset, wantAsset) {
			return asset, nil
		}
	}
	return "", errors.Errorf("Plugin %q version %q doesn't seem to have the required asset (%q) -- assets found: %v", plugin, version, wantAsset, assets)
}

// LinuxDownloadReleaseAssets downloads the list of assets available for the given Gopjrt release version.
func LinuxDownloadReleaseAssets(version string) ([]string, error) {
	// Construct release URL based on version
	releaseURL := "https://api.github.com/repos/gomlx/gopjrt/releases/latest"
	if version != "latest" {
		releaseURL = fmt.Sprintf("https://api.github.com/repos/gomlx/gopjrt/releases/tags/%s", *flagVersion)
	}

	// Make HTTP request
	resp, err := http.Get(releaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch release data: %v", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	// Parse JSON response
	var release struct {
		Assets []struct {
			BrowserDownloadURL string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(body, &release); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response: %v", err)
	}

	// Extract .tar.gz download URLs
	var urls []string
	for _, asset := range release.Assets {
		if strings.HasSuffix(asset.BrowserDownloadURL, ".tar.gz") {
			urls = append(urls, asset.BrowserDownloadURL)
		}
	}

	return urls, nil
}

// LinuxInstall the assets on the target directory.
func LinuxInstall() error {
	assetURL, err := LinuxGetDownloadURL(*flagPlugin, *flagVersion)
	if err != nil {
		return err
	}

	// Create the target directory.
	installPath := ReplaceTildeInDir(*flagPath)
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrap(err, "failed to create install directory")
	}

	// Download the asset to a temporary file.
	downloadedFile, err := DownloadURLToTemp(assetURL, "gopjrt_download.*")
	if err != nil {
		return err
	}
	if !klog.V(1).Enabled() {
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

	// Remove older version using dynamically linked library
	oldLib := filepath.Join(installPath, "lib/libgomlx_xlabuilder.so")
	os.Remove(oldLib)

	// Read and print the list of extracted files
	fmt.Println("- Extracted files:")
	fileContents, err := os.ReadFile(listFile.Name())
	if err != nil {
		return errors.Wrap(err, "failed to read list of extracted files")
	}
	fmt.Print(string(fileContents))

	return nil
}
