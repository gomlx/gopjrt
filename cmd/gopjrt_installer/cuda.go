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
	for _, plugin := range []string{"cuda13", "cuda12"} {
		pluginInstallers[plugin] = CudaInstall
		pluginValidators[plugin] = CudaValidateVersion
	}
	pluginValues = append(pluginValues, "cuda13", "cuda12")
	pluginDescriptions = append(pluginDescriptions,
		"CUDA PJRT for Linux/amd64, using CUDA 13",
		"CUDA PJRT for Linux/amd64, using CUDA 12, deprecated")
	pluginPriorities = append(pluginPriorities, 10, 11)
}

// CudaInstall installs the cuda PJRT from the Jax PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Author email is from the Jax team
// - Downloaded files sha256 match the ones on pypi.org
func CudaInstall(plugin, version, installPath string) error {
	// Create the target directory.
	installPath = ReplaceTildeInDir(installPath)
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create install directory in %s", installPath)
	}

	var err error
	version, err = CudaInstallPJRT(plugin, version, installPath)
	if err != nil {
		return err
	}

	if err := CudaInstallNvidiaLibraries(plugin, version, installPath); err != nil {
		return err
	}

	cudaVersion := "13"
	if plugin == "cuda12" {
		cudaVersion = "12"
	}
	fmt.Printf("\n✅ Installed \"cuda\" PJRT and Nvidia libraries based on Jax version %s and CUDA version %s\n\n", version, cudaVersion)
	return nil
}

// CudaInstallPJRT installs the cuda PJRT from the Jax PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Author email is from the Jax team
// - Downloaded files sha256 match the ones on pypi.org
//
// Returns the version that was installed -- it can be different if the requested version was "latest", in which case it
// is translated to the actual version.
func CudaInstallPJRT(plugin, version, installPath string) (string, error) {
	// Make the directory that will hold the PJRT files.
	pjrtDir := filepath.Join(installPath, "/lib/gomlx/pjrt")
	pjrtOutputPath := path.Join(pjrtDir, "pjrt_c_api_cuda_plugin.so")
	if err := os.MkdirAll(pjrtDir, 0755); err != nil {
		return "", errors.Wrapf(err, "failed to create PJRT install directory in %s", pjrtDir)
	}

	// Get CUDA PJRT wheel from pypi.org
	info, packageName, err := CudaGetPJRTPipInfo(plugin)
	if err != nil {
		return "", errors.WithMessagef(err, "can't fetch pypi.org information for %s", plugin)
	}
	if info.Info.AuthorEmail != "jax-dev@google.com" {
		return "", errors.Errorf("package %s is not from Jax team, but it's signed by %q: something is suspicious!?",
			packageName, info.Info.AuthorEmail)
	}

	// Translate "latest" to the actual version if needed.
	if version == "latest" {
		version = info.Info.Version
	}

	releaseInfos, ok := info.Releases[version]
	if !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return "", errors.Errorf("version %q not found for %q (from pip package %q) -- lastest is %q and existing versions are: %s",
			version, plugin, packageName, info.Info.Version, strings.Join(versions, ", "))
	}

	releaseInfo, err := PipSelectRelease(releaseInfos, pipPackageLinuxAMD64, false)
	if err != nil {
		return "", errors.Wrapf(err, "failed to find release for %s, version %s", plugin, version)
	}
	if releaseInfo.PackageType != "bdist_wheel" {
		return "", errors.Errorf("release %s is not a \"binary wheel\" type", releaseInfo.Filename)
	}

	sha256hash := releaseInfo.Digests["sha256"]
	downloadedJaxPJRTWHL, fileCached, err := DownloadURLToTemp(releaseInfo.URL, fmt.Sprintf("gopjrt_%s_%s.whl", packageName, version), sha256hash)
	if err != nil {
		return "", errors.Wrap(err, "failed to download cuda PJRT wheel")
	}
	if !fileCached {
		defer func() { ReportError(os.Remove(downloadedJaxPJRTWHL)) }()
	}
	err = ExtractFileFromZip(downloadedJaxPJRTWHL, "xla_cuda_plugin.so", pjrtOutputPath)
	if err != nil {
		return "", errors.Wrapf(err, "failed to extract CUDA PJRT file from %q wheel", packageName)
	}
	fmt.Printf("- Installed %s %s to %s\n", plugin, version, pjrtOutputPath)
	return version, nil
}

// CudaValidateVersion checks whether the cuda version selected by "-version" exists.
func CudaValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	info, packageName, err := CudaGetPJRTPipInfo(plugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %s", plugin)
	}
	if info.Info.AuthorEmail != "jax-dev@google.com" {
		return errors.Errorf("package %s is not from Jax team, but it's signed by %q: something is suspicious!?",
			packageName, info.Info.AuthorEmail)
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

// CudaGetPJRTPipInfo returns the JSON info for the PIP package that corresponds to the plugin.
func CudaGetPJRTPipInfo(plugin string) (*PipPackageInfo, string, error) {
	var packageName string
	switch plugin {
	case "cuda12":
		packageName = "jax-cuda12-pjrt"
	case "cuda13":
		packageName = "jax-cuda13-pjrt"
	default:
		return nil, "", errors.Errorf("unknown plugin %q selected", plugin)
	}
	info, err := GetPipInfo(packageName)
	if err != nil {
		return nil, "", errors.Wrapf(err, "failed to get package info for %s", packageName)
	}
	return info, packageName, nil
}

func CudaInstallNvidiaLibraries(plugin, version, installPath string) error {
	// Remove any previous version of the nvidia libraries and recreate it.
	nvidiaLibsDir := filepath.Join(installPath, "/lib/gomlx/nvidia")
	if err := os.RemoveAll(nvidiaLibsDir); err != nil {
		return errors.Wrapf(err, "failed to remove existing nvidia libraries directory %s", nvidiaLibsDir)
	}
	if err := os.MkdirAll(nvidiaLibsDir, 0755); err != nil {
		return errors.Wrapf(err, "failed to create nvidia libraries directory in %s", nvidiaLibsDir)
	}

	// Find required nvidia packages:
	packageName := "jax-" + plugin + "-plugin"
	jaxCudaPluginInfo, err := GetPipInfo(packageName)
	if err != nil {
		return errors.Wrapf(err, "failed to fetch the package info for %s", packageName)
	}
	fmt.Println("Dependencies:")
	deps, err := jaxCudaPluginInfo.ParseDependencies()
	if err != nil {
		return errors.Wrapf(err, "failed to parse the dependencies for %s", packageName)
	}
	nvidiaDependencies := slices.DeleteFunc(deps, func(dep PipDependency) bool {
		// This is a simplification that works for now: in the future we many need to check "sys_platform" conditions.
		if !strings.HasPrefix(dep.Package, "nvidia") {
			return true
		}
		return false
	})

	// Install the nvidia libraries found in the dependencies.
	for _, dep := range nvidiaDependencies {
		err = cudaInstallNvidiaLibrary(nvidiaLibsDir, dep)
		if err != nil {
			return err
		}
	}

	// Create a link to the binary ptxas, required by the nvidia libraries.
	nvidiaBinPath := filepath.Join(nvidiaLibsDir, "bin")
	if err := os.MkdirAll(nvidiaBinPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create nvidia bin directory in %s", nvidiaBinPath)
	}

	// Create symbolic link to ptxas.
	var ptxasPath string
	switch plugin {
	case "cuda12":
		ptxasPath = filepath.Join(nvidiaLibsDir, "cuda_nvcc/bin/ptxas")
	case "cuda13":
		ptxasPath = filepath.Join(nvidiaLibsDir, "cu13/bin/ptxas")
	default:
		return errors.Errorf("version validation not implemented for plugin %q in version %s", plugin, version)
	}
	ptxasLinkPath := filepath.Join(nvidiaBinPath, "ptxas")
	if err := os.Symlink(ptxasPath, ptxasLinkPath); err != nil {
		return errors.Wrapf(err, "failed to create symbolic link to ptxas in %s", ptxasLinkPath)
	}
	return nil
}

func cudaInstallNvidiaLibrary(nvidiaLibsDir string, dep PipDependency) error {
	info, err := GetPipInfo(dep.Package)
	if err != nil {
		return errors.Wrapf(err, "failed to fetch the package info for %s", dep.Package)
	}

	// Find the highest version that meets constraints.
	var selectedVersion string
	var selectedReleaseInfo *PipReleaseInfo
	for version, releases := range info.Releases {
		if !dep.IsValid(version) {
			continue
		}
		releaseInfo, err := PipSelectRelease(releases, pipPackageLinuxAMD64, false)
		if err != nil {
			continue
		}
		if selectedVersion == "" || PipCompareVersion(version, selectedVersion) > 0 {
			selectedVersion = version
			selectedReleaseInfo = releaseInfo
		}
	}
	if selectedVersion == "" {
		return errors.Errorf("no matching version found for package %s with constraints %+v", dep.Package, dep)
	}

	// Download the ".whl" file (zip file format) for the selected version of the nvidia library..
	sha256hash := selectedReleaseInfo.Digests["sha256"]
	downloadedWHL, whlIsCached, err := DownloadURLToTemp(selectedReleaseInfo.URL, fmt.Sprintf("gopjrt_%s_%s.whl", dep.Package, selectedVersion), sha256hash)
	if err != nil {
		return errors.Wrapf(err, "failed to download %s wheel", dep.Package)
	}
	if !whlIsCached {
		defer func() { ReportError(os.Remove(downloadedWHL)) }()
	}

	// Extract all files under "nvidia/" for this package.
	if err := ExtractDirFromZip(downloadedWHL, "nvidia", nvidiaLibsDir); err != nil {
		return errors.Wrapf(err, "failed to extract nvidia libraries from %s", downloadedWHL)
	}
	fmt.Printf("- Installed %s@%s\n", dep.Package, selectedVersion)
	return nil
}
