//go:build (linux && amd64) || all

package main

import (
	"fmt"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/pkg/errors"
)

func init() {
	for _, plugin := range []string{"tpu"} {
		pluginInstallers[plugin] = TPUInstall
		pluginValidators[plugin] = TPUValidateVersion
	}
	pluginValues = append(pluginValues, "tpu")
	pluginDescriptions = append(pluginDescriptions,
		"TPU PJRT for Linux/amd64 host machines (glibc >= 2.31)")
	pluginPriorities = append(pluginPriorities, 20)
}

// TPUInstall installs the cuda PJRT from the "libtpu" PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Downloaded files sha256 match the ones on pypi.org
func TPUInstall(plugin, version, installPath string) error {
	// Create the target directory.
	installPath = ReplaceTildeInDir(installPath)
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create install directory in %s", installPath)
	}

	pjrtDir := filepath.Join(installPath, "/lib/gomlx/pjrt")
	pjrtOutputPath := path.Join(pjrtDir, "pjrt_c_api_tpu_plugin.so")
	if err := os.MkdirAll(pjrtDir, 0755); err != nil {
		return errors.Wrapf(err, "failed to create PJRT install directory in %s", pjrtDir)
	}

	// Get CUDA PJRT wheel from pypi.org
	info, packageName, err := TPUGetPJRTPipInfo(plugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %s", plugin)
	}

	// Translate "latest" to the actual version if needed.
	if version == "latest" {
		version = info.Info.Version
	}

	releaseInfos, ok := info.Releases[version]
	if !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return errors.Errorf("version %q not found for %q (from pip package %q) -- lastest is %q and existing versions are: %s",
			version, plugin, packageName, info.Info.Version, strings.Join(versions, ", "))
	}

	releaseInfo, err := PipSelectRelease(releaseInfos, pipPackageLinuxAMD64Glibc231, true)
	if err != nil {
		return errors.Wrapf(err, "failed to find release for %s, version %s", plugin, version)
	}
	if releaseInfo.PackageType != "bdist_wheel" {
		return errors.Errorf("release %s is not a \"binary wheel\" type", releaseInfo.Filename)
	}

	sha256hash := releaseInfo.Digests["sha256"]
	downloadedJaxPJRTWHL, fileCached, err := DownloadURLToTemp(releaseInfo.URL, fmt.Sprintf("gopjrt_%s_%s.whl", packageName, version), sha256hash)
	if err != nil {
		return errors.Wrap(err, "failed to download cuda PJRT wheel")
	}
	if !fileCached {
		defer func() { ReportError(os.Remove(downloadedJaxPJRTWHL)) }()
	}
	err = ExtractFileFromZip(downloadedJaxPJRTWHL, "libtpu.so", pjrtOutputPath)
	if err != nil {
		return errors.Wrapf(err, "failed to extract TPU PJRT file from %q wheel", packageName)
	}
	fmt.Printf("- Installed %s %s to %s\n", plugin, version, pjrtOutputPath)

	fmt.Printf("\n✅ Installed \"tpu\" PJRT based on PyPI version %s\n\n", version)
	return nil
}

// TPUValidateVersion checks whether the cuda version selected by "-version" exists.
func TPUValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	info, packageName, err := TPUGetPJRTPipInfo(plugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %q", plugin)
	}

	if _, ok := info.Releases[version]; !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return errors.Errorf("version %s not found for %s (from pip package %q) -- existing versions: %s",
			version, plugin, packageName, strings.Join(versions, ", "))
	}

	// Version found.
	return nil
}

// TPUGetPJRTPipInfo returns the JSON info for the PIP package that corresponds to the plugin.
func TPUGetPJRTPipInfo(plugin string) (*PipPackageInfo, string, error) {
	var packageName string
	switch plugin {
	case "tpu":
		packageName = "libtpu"
	default:
		return nil, "", errors.Errorf("unknown plugin %q selected", plugin)
	}
	info, err := GetPipInfo(packageName)
	if err != nil {
		return nil, "", errors.Wrapf(err, "failed to get package info for %s", packageName)
	}
	return info, packageName, nil
}
