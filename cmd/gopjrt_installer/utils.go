package main

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/user"
	"path"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/huh/spinner"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var flagCache = flag.Bool("cache", true, "Use cache to store downloaded files. It defaults to ")

// ReportError prints an error if it is not nil, but otherwise does nothing.
func ReportError(err error) {
	if err != nil {
		klog.Warningf("Error: %v", err)
	}
}

// ReplaceTildeInDir by the user's home directory. Returns dir if it doesn't start with "~".
//
// It may panic with an error if `dir` has an unknown user (e.g: `~unknown/...`)
func ReplaceTildeInDir(dir string) string {
	if len(dir) == 0 {
		return dir
	}
	if dir[0] != '~' {
		return dir
	}
	var userName string
	if dir != "~" && !strings.HasPrefix(dir, "~/") {
		sepIdx := strings.IndexRune(dir, '/')
		if sepIdx == -1 {
			userName = dir[1:]
		} else {
			userName = dir[1:sepIdx]
		}
	}
	var usr *user.User
	var err error
	if userName == "" {
		usr, err = user.Current()
	} else {
		usr, err = user.Lookup(userName)
	}
	if err != nil {
		panic(errors.Wrapf(err, "failed to lookup home directory for user in path %q", dir))
	}
	homeDir := usr.HomeDir
	return path.Join(homeDir, dir[1+len(userName):])
}

// GetCachePath finds and prepares the cache directory for gopjrt.
//
// It uses os.UserCacheDir() for portability:
//
// - Linux: $XDG_CACHE_HOME or $HOME/.cache
// - Darwin: $HOME/Library/Caches
// - Windows: %LocalAppData% (e.g., C:\Users\user\AppData\Local)
func GetCachePath(fileName string) (filePath string, cached bool, err error) {
	baseCacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", false, errors.Wrap(err, "failed to find user cache directory")
	}
	cacheDir := filepath.Join(baseCacheDir, "gopjrt")
	if err = os.MkdirAll(cacheDir, 0755); err != nil {
		return "", false, errors.Wrapf(err, "failed to create cache directory %s", cacheDir)
	}
	filePath = filepath.Join(cacheDir, fileName)
	if stat, err := os.Stat(filePath); err == nil {
		cached = stat.Mode().IsRegular()
	}
	return
}

// DownloadURLToTemp downloads a file from a given URL to a temporary file.
//
// It displays a spinner while downloading and outputs some information about the download.
//
// If -cache is set (the default) it will save the file in a cache directory and try to reuse it if already downloaded.
//
// If wantSHA256 is not empty, it will verify the hash of the downloaded file.
//
// It returns the path where the file was downloaded, and if the downloaded file is in a cache (so it shouldn't be removed after use).
func DownloadURLToTemp(url, fileName, wantSHA256 string) (filePath string, cached bool, err error) {
	// Download the asset to a temporary file
	var downloadedFile *os.File

	if *flagCache {
		filePath, cached, err = GetCachePath(fileName)
		if err != nil {
			return "", false, err
		}
		if !cached {
			downloadedFile, err = os.Create(filePath)
			if err != nil {
				return "", false, errors.Wrapf(err, "failed to create cache file %s", filePath)
			}
		}

	} else {
		// Create a temporary file.
		filePattern := fileName + ".*"
		downloadedFile, err = os.CreateTemp("", filePattern)
		if err != nil {
			return "", false, errors.Wrap(err, "failed to create temporary file")
		}
		filePath = downloadedFile.Name()
	}

	var downloadedBytesStr string
	if !cached {
		// Actually download the file.
		var bytesDownloaded int64
		spinnerErr := spinner.New().
			Title(fmt.Sprintf("Downloading %s….", url)).
			Action(func() {
				var resp *http.Response
				resp, err = http.Get(url)
				if err != nil {
					err = errors.Wrapf(err, "failed to download asset %s", url)
					return
				}
				defer func() { ReportError(resp.Body.Close()) }()

				bytesDownloaded, err = io.Copy(downloadedFile, resp.Body)
				if err != nil {
					err = errors.Wrapf(err, "failed to write asset %s to temporary file %s", url, downloadedFile.Name())
					return
				}
				ReportError(downloadedFile.Close())
			}).
			Run()
		if spinnerErr != nil {
			err = errors.Wrapf(spinnerErr, "failed run spinner for download from %s", url)
			return
		}
		if err != nil {
			return "", false, err
		}

		// Print the downloaded size
		switch {
		case bytesDownloaded < 1024:
			downloadedBytesStr = fmt.Sprintf("%d B", bytesDownloaded)
		case bytesDownloaded < 1024*1024:
			downloadedBytesStr = fmt.Sprintf("%.1f KB", float64(bytesDownloaded)/1024)
		case bytesDownloaded < 1024*1024*1024:
			downloadedBytesStr = fmt.Sprintf("%.1f MB", float64(bytesDownloaded)/(1024*1024))
		default:
			downloadedBytesStr = fmt.Sprintf("%.1f GB", float64(bytesDownloaded)/(1024*1024*1024))
		}
	}

	// Verify SHA256 hash if provided -- also for cached files.
	verifiedStatus := ""
	if wantSHA256 != "" {
		// Open the file for reading.
		f, err := os.Open(filePath)
		if err != nil {
			return "", false, errors.Wrap(err, "failed to open file for hash verification")
		}
		defer func() { ReportError(f.Close()) }()

		// Calculate SHA256 hash using 1MB buffer
		hasher := sha256.New()
		buffer := make([]byte, 1024*1024) // 1MB buffer
		for {
			n, err := f.Read(buffer)
			if n > 0 {
				hasher.Write(buffer[:n])
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", false, errors.Wrap(err, "failed to read file for hash verification")
			}
		}

		actualHash := hex.EncodeToString(hasher.Sum(nil))
		if actualHash != wantSHA256 {
			return "", false, errors.Errorf("SHA256 hash mismatch for %s: expected %q, got %q", filePath, wantSHA256, actualHash)
		}
		verifiedStatus = " (hash checked)"
	}

	if cached {
		fmt.Printf("- Reusing %s from cache%s\n", filePath, verifiedStatus)
	} else {
		fmt.Printf("- Downloaded %s to %s%s\n", downloadedBytesStr, filePath, verifiedStatus)
		if *flagCache {
			// Now the file is cached.
			cached = true
		}
	}
	return
}

// ExtractFileFromZip searches for a file named targetFileName within the zipFilePath
// and extracts the first one found to the outputPath.
func ExtractFileFromZip(zipFilePath, targetFileName, outputPath string) error {
	r, err := zip.OpenReader(zipFilePath)
	if err != nil {
		return err
	}
	defer func() { ReportError(r.Close()) }()

	// Normalize the target file name for comparison
	normalizedTarget := filepath.Clean(targetFileName)

	// Iterate through the files in the archive
	for _, f := range r.File {
		// Get the base name (the file name without any directory path)
		_, name := filepath.Split(f.Name)

		// Check if the base name matches the target file name
		if name == normalizedTarget {
			return extractZipFile(f, outputPath)
		}
	}
	return os.ErrNotExist // File was not found in the archive.
}

// ExtractDirFromZip extracts from zipFilePath all files and directories under dirInZipFile and saves them with the
// same directory structure under outputPath.
//
// Notice dirInZipFile is not repeated in outputPath.
func ExtractDirFromZip(zipFilePath, dirInZipFile, outputPath string) error {
	r, err := zip.OpenReader(zipFilePath)
	if err != nil {
		return err
	}
	defer func() { ReportError(r.Close()) }()

	// Normalize paths for comparison
	normalizedPrefix := filepath.Clean(dirInZipFile) + "/"

	// Iterate through the files in the archive
	for _, f := range r.File {
		// Normalize the file path
		normalizedPath := filepath.Clean(f.Name)

		// Check if this file is under the requested directory
		if !strings.HasPrefix(normalizedPath, normalizedPrefix) {
			continue
		}

		// Calculate relative path from the prefix
		relPath := strings.TrimPrefix(normalizedPath, normalizedPrefix)
		if relPath == "" {
			continue // Skip the directory itself
		}

		// Create the full output path
		fullPath := filepath.Join(outputPath, relPath)

		if f.FileInfo().IsDir() {
			// Create directory
			if err := os.MkdirAll(fullPath, 0755); err != nil {
				return err
			}
			continue
		}

		// Create parent directories if they don't exist
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			return err
		}

		if err := extractZipFile(f, fullPath); err != nil {
			return err
		}
	}

	return nil
}

// extractZipFile is a helper to perform the actual extraction
//
// It adds execution permissions to the file if it is in the bin directory.
func extractZipFile(f *zip.File, outputPath string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer func() { ReportError(rc.Close()) }()

	// Create the output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer func() { ReportError(outFile.Close()) }()

	// Copy the contents
	_, err = io.Copy(outFile, rc)
	if err != nil {
		return err
	}

	// Make file executable if in a bin directory
	if strings.Contains(outputPath, "/bin/") {
		if err := os.Chmod(outputPath, 0755); err != nil {
			return err
		}
	}
	return nil
}

func GitHubGetLatestVersion() (string, error) {
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

// GitHubDownloadReleaseAssets downloads the list of assets available for the given Gopjrt release version.
func GitHubDownloadReleaseAssets(version string) ([]string, error) {
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
