package main

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var pipPackageLinuxAMD64 = regexp.MustCompile(`-manylinux.*x86_64`)

// CudaInstall installs the cuda PJRT from the Jax PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Author email is from the Jax team
// - Downloaded files sha256 match the ones on pypi.org
func CudaInstall() error {
	// Create the target directory.
	installPath := ReplaceTildeInDir(*flagPath)
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrap(err, "failed to create install directory")
	}

	// Get CUDA PJRT wheel from pypi.org
	info, packageName, err := CudaGetPipInfo(*flagPlugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %s", *flagPlugin)
	}
	if info.Info.AuthorEmail != "jax-dev@google.com" {
		return errors.Errorf("package %s is not from Jax team, something is very suspicious!?", packageName)
	}

	// Translate "latest" to the actual version if needed.
	version := *flagVersion
	if version == "latest" {
		version = info.Info.Version
	}

	releaseInfos, ok := info.Releases[version]
	if !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return errors.Errorf("version %q not found for %q (from pip package %q) -- lastest is %q and existing versions are: %s",
			*flagVersion, *flagPlugin, packageName, info.Info.Version, strings.Join(versions, ", "))
	}

	releaseInfo, err := pipSelectRelease(releaseInfos, pipPackageLinuxAMD64)
	if err != nil {
		return errors.Wrapf(err, "failed to find release for %s, version %s", *flagPlugin, *flagVersion)
	}
	if releaseInfo.PackageType != "bdist_wheel" {
		return errors.Errorf("release %s is not a \"binary wheel\" type", releaseInfo.Filename)
	}

	downloadedJaxPJRTWHL, err := DownloadURLToTemp(releaseInfo.URL, "gopjrt_jax_pjrt_cuda_*.whl")
	if err != nil {
		return errors.Wrap(err, "failed to download cuda PJRT wheel")
	}
	if !klog.V(1).Enabled() {
		defer func() { ReportError(os.Remove(downloadedJaxPJRTWHL)) }()
	}
	pjrtOutputPath := filepath.Join(installPath, "/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so")
	err = ExtractFileFromZip(downloadedJaxPJRTWHL, "xla_cuda_plugin.so", pjrtOutputPath)
	if err != nil {
		return errors.Wrapf(err, "failed to extract CUDA PJRT file from %q wheel", packageName)
	}
	fmt.Printf("- Installed CUDA PJRT to %s\n", pjrtOutputPath)
	return nil
}

// CudaValidateVersion checks whether the cuda version selected by "-version" exists.
func CudaValidateVersion() error {
	// "latest" is always valid.
	if *flagVersion == "latest" {
		return nil
	}

	info, packageName, err := CudaGetPipInfo(*flagPlugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %s", *flagPlugin)
	}
	if info.Info.AuthorEmail != "jax-dev@google.com" {
		return errors.Errorf("package %s is not from Jax team, something is very suspicious!?", packageName)
	}

	if _, ok := info.Releases[*flagVersion]; !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return errors.Errorf("version %s not found for %s (from pip package %q) -- existing versions: %s",
			*flagVersion, *flagPlugin, packageName, strings.Join(versions, ", "))
	}

	// Version found.
	return nil
}

// CudaGetPipInfo returns the JSON info for the PIP package that corresponds to the plugin.
func CudaGetPipInfo(plugin string) (*PipPackageInfo, string, error) {
	var packageName string
	switch *flagPlugin {
	case "cuda12":
		packageName = "jax-cuda12-pjrt"
	case "cuda13":
		packageName = "jax-cuda13-pjrt"
	default:
		return nil, "", errors.Errorf("unknown plugin %q selected", plugin)
	}
	info, err := getPipInfo(packageName)
	if err != nil {
		return nil, "", errors.Wrapf(err, "failed to get package info for %s", packageName)
	}
	return info, packageName, nil
}

func getPipInfo(packageName string) (*PipPackageInfo, error) {
	url := "https://pypi.org/pypi/" + packageName + "/json"

	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch package info from %s", url)
	}
	defer func() { ReportError(resp.Body.Close()) }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read response body")
	}

	var result PipPackageInfo
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, errors.Wrap(err, "failed to parse JSON response")
	}

	return &result, nil
}

// PipPackageInfo is the JSON response from pypi.org for a given package.
type PipPackageInfo struct {
	// Top-level object returned by the API
	Info struct {
		Name           string `json:"name"`
		Version        string `json:"version"` // This is the LATEST version
		Summary        string `json:"summary"`
		HomePage       string `json:"home_page"`
		Author         string `json:"author"`
		AuthorEmail    string `json:"author_email"`
		License        string `json:"license"`
		ProjectURL     string `json:"project_url"`
		RequiresPython string `json:"requires_python"`

		// This is a list of dependencies/requirements
		RequiresDist []string `json:"requires_dist"`
	} `json:"info"`

	// Releases is a map of all versions, where the key is the version string (e.g., "1.2.3")
	Releases map[string][]PipReleaseInfo `json:"releases"`
}

// PipReleaseInfo is the JSON response from pypi.org for a given package version, for some platform.
type PipReleaseInfo struct {
	// You can add more fields here if you need file-specific details
	PackageType string            `json:"packagetype"`
	Filename    string            `json:"filename"`
	URL         string            `json:"url"`
	Digests     map[string]string `json:"digests"`
}

func pipSelectRelease(releaseInfos []PipReleaseInfo, platform *regexp.Regexp) (*PipReleaseInfo, error) {
	var result *PipReleaseInfo
	for i, release := range releaseInfos {
		if platform.MatchString(release.Filename) {
			if result != nil {
				return nil, errors.Errorf("multiple releases found for platform %q: %q and %q", platform, result.Filename, release.Filename)
			}
			result = &releaseInfos[i]
		}
	}
	if result == nil {
		return nil, errors.Errorf("no release found for platform %q", platform)
	}
	return result, nil
}
