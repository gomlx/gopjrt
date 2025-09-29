package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"
)

// LinuxValidateVersion checks whether the linux version selected in -version exists.
func LinuxValidateVersion() error {
	// "latest" is always valid.
	if *flagVersion == "latest" {
		return nil
	}

	assets, err := LinuxDownloadReleaseAssets(*flagVersion)
	if err != nil {
		return err
	}

	wantAsset := fmt.Sprintf("gomlx_xlabuilder_%s.tar.gz", *flagPlugin)
	for _, asset := range assets {
		if strings.Contains(asset, wantAsset) {
			return nil
		}
	}
	return errors.Errorf("Plugin %q version %q doesn't seem to have the required asset (%q) -- assets found: %v", *flagPlugin, *flagVersion, wantAsset, assets)
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
