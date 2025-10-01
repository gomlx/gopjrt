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

func LinuxGetLatestVersion() (string, error) {
	const latestURL = "https://api.github.com/repos/gomlx/gopjrt/releases/latest"
	// Make HTTP request
	resp, err := http.Get(latestURL)
	if err != nil {
		return "", errors.Wrapf(err, "failed to fetch release data from %q", latestURL)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", errors.Wrapf(err, "failed to read data from %q", latestURL)
	}

	// Parse JSON response
	var info struct {
		TagName string `json:"tag_name"`
	}
	if err := json.Unmarshal(body, &info); err != nil {
		return "", fmt.Errorf("failed to parse JSON response: %v", err)
	}
	version := info.TagName
	if version == "" {
		return "", errors.Errorf("failed to get version from %q", latestURL)
	}
	return version, nil
}

// LinuxGetDownloadURL returns the download URL for the given version and plugin.
func LinuxGetDownloadURL(plugin, version string) (url string, err error) {
	var assets []string
	assets, err = LinuxDownloadReleaseAssets(version)
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

// LinuxDownloadReleaseAssets downloads the list of assets available for the given Gopjrt release version.
func LinuxDownloadReleaseAssets(version string) ([]string, error) {
	// Construct release URL based on the version -- "latest" is not supported at this point.
	releaseURL := fmt.Sprintf("https://api.github.com/repos/gomlx/gopjrt/releases/tags/%s", version)

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
	version, err := LinuxGetLatestVersion()
	if err != nil {
		return err
	}
	assetURL, err := LinuxGetDownloadURL(*flagPlugin, version)
	if err != nil {
		return err
	}

	// Create the target directory.
	installPath := ReplaceTildeInDir(*flagPath)
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrap(err, "failed to create install directory")
	}

	// Download the asset to a temporary file.
	sha256hash := "" // TODO: no hash for github releases. Is there a way to get them (or get a hardcoded table for all versions?)
	downloadedFile, inCache, err := DownloadURLToTemp(assetURL, fmt.Sprintf("gopjrt_xlabuilder_%s.tar.gz", version), sha256hash)
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

	fmt.Printf("\n✅ Installed Gopjrt %s libraries and \"cpu\" PJRT to %s (%s platform)\n\n", version, installPath, *flagPlugin)

	return nil
}
