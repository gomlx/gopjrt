package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
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
	retries := 0
	const maxRetries = 2
retry:
	for {
		// Make HTTP request with optional authorization header
		req, err := http.NewRequest("GET", latestURL, nil)
		if err != nil {
			return "", errors.Wrapf(err, "failed to create request for %q", latestURL)
		}
		req.Header.Add("Accept", "application/vnd.github+json")
		if token, found := os.LookupEnv("GH_TOKEN"); found {
			if token == "" {
				klog.Infof("GH_TOKEN is empty, skipping authentication")
			} else {
				req.Header.Add("Authorization", "Bearer "+token)
				klog.Infof("Using GitHub token for authentication")
			}
		} else {
			klog.Infof("GH_TOKEN is not set, skipping authentication")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return "", errors.Wrapf(err, "failed to fetch release data from %q", latestURL)
		}

		// Read response body
		body, err := io.ReadAll(resp.Body)
		ReportError(resp.Body.Close())
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data from %q", latestURL)
		}
		if resp.StatusCode != http.StatusOK {
			return "", errors.Errorf("failed to get version from %q, got status code %d -- message %q", latestURL, resp.StatusCode, body)
		}

		// Parse JSON response
		var info struct {
			TagName string `json:"tag_name"`
		}
		if err := json.Unmarshal(body, &info); err != nil {
			return "", errors.Wrapf(err, "failed to parse JSON response")
		}
		version := info.TagName
		if version == "" {
			if retries == maxRetries {
				return "", errors.Errorf("failed to get version from %q, it is missing the field `tag_name`", latestURL)
			}
			retries++
			klog.Warningf("failed to get version from %q, it is missing the field `tag_name`, retrying...", latestURL)
			fmt.Printf("Body: %s\n", string(body))
			continue retry
		}
		return version, nil
	}
}

// GitHubDownloadReleaseAssets downloads the list of assets available for the given Gopjrt release version.
func GitHubDownloadReleaseAssets(version string) ([]string, error) {
	// Construct release URL based on the version -- "latest" is not supported at this point.
	releaseURL := fmt.Sprintf("https://api.github.com/repos/gomlx/gopjrt/releases/tags/%s", version)

	// Make HTTP request with optional authorization header
	req, err := http.NewRequest("GET", releaseURL, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create request for %q", releaseURL)
	}
	req.Header.Add("Accept", "application/vnd.github+json")
	if token := os.Getenv("GH_TOKEN"); token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
		klog.Infof("Using GitHub token for authentication")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch release data from %q", releaseURL)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	ReportError(resp.Body.Close())
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read response body")
	}

	// Parse JSON response
	var release struct {
		Assets []struct {
			BrowserDownloadURL string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(body, &release); err != nil {
		return nil, errors.Wrapf(err, "failed to parse JSON response")
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

// Untar takes a path to a tar/gzip file and an output directory.
// It returns a list of extracted files and any error encountered.
func Untar(tarballPath, outputDirPath string) ([]string, error) {
	// Make sure the output directory is absolute.
	if !filepath.IsAbs(outputDirPath) {
		var err error
		outputDirPath, err = filepath.Abs(outputDirPath)
		if err != nil {
			return nil, errors.Wrapf(err, "Untar failed to get absolute path for output directory %q", outputDirPath)
		}
	}

	// 1. Open the tarball file
	file, err := os.Open(tarballPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open tarball in %s for reading", tarballPath)
	}
	defer func() { ReportError(file.Close()) }()

	// 2. Setup the Gzip reader (assuming it's a .tar.gz)
	// If it's just a .tar file, you would skip this step and use 'file' directly below.
	var fileReader io.Reader = file
	if filepath.Ext(tarballPath) == ".gz" || filepath.Ext(tarballPath) == ".tgz" {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to create gzip reader")
		}
		defer func() { ReportError(gzReader.Close()) }()
		fileReader = gzReader
	}

	// 3. Setup the Tar reader
	tarReader := tar.NewReader(fileReader)

	// Track extracted files
	var extractedFiles []string

	// 4. Iterate through the archive entries
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			return nil, errors.Wrapf(err, "tar reading error")
		}

		// Clean the targetPath and make sure it falls within outputDirPath.
		targetPath := filepath.Join(outputDirPath, header.Name)
		targetPath = filepath.Clean(targetPath)
		if !strings.HasPrefix(targetPath, outputDirPath) {
			return nil, errors.Errorf("tar entry path is unsafe: %s", header.Name)
		}

		// Create parent directories if they don't exist
		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return nil, errors.Wrapf(err, "failed to create directory %s", filepath.Dir(targetPath))
		}

		switch header.Typeflag {
		case tar.TypeDir:
			// Handle directories
			if err := os.MkdirAll(targetPath, os.FileMode(header.Mode)); err != nil {
				return nil, errors.Wrapf(err, "failed to create directory %s", targetPath)
			}
			extractedFiles = append(extractedFiles, targetPath)

		case tar.TypeReg:
			// Handle regular files and links
			outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				// If file exists and is read-only, remove it and try again.
				if os.IsPermission(err) {
					if err = os.Remove(targetPath); err != nil {
						return nil, errors.Wrapf(err, "failed to remove read-only file %s", targetPath)
					}
					outFile, err = os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
					if err != nil {
						return nil, errors.Wrapf(err, "failed to create file %s after removing read-only version", targetPath)
					}
				} else {
					return nil, errors.Wrapf(err, "failed to create file %s", targetPath)
				}
			}
			// Copy file contents
			if _, err := io.Copy(outFile, tarReader); err != nil {
				ReportError(outFile.Close())
				return nil, errors.Wrapf(err, "failed to copy file contents to %s", targetPath)
			}
			ReportError(outFile.Close())
			extractedFiles = append(extractedFiles, targetPath)

		case tar.TypeSymlink:

			// Sanitize the symlink's target path to ensure it stays within the output directory
			linkTarget := filepath.Clean(header.Linkname)
			if filepath.IsAbs(linkTarget) {
				return nil, errors.Errorf("absolute symlink target path unsafe and not allowed: %s", linkTarget)
			}
			// Calculate the absolute path of the link target relative to the symlink's location
			absLinkTarget := filepath.Join(filepath.Dir(targetPath), linkTarget)
			cleanAbsTarget, err := filepath.EvalSymlinks(absLinkTarget)
			if err != nil && !os.IsNotExist(err) {
				return nil, errors.Wrapf(err, "failed to evaluate symlink target for %s", targetPath)
			}
			if cleanAbsTarget != "" && !strings.HasPrefix(cleanAbsTarget, outputDirPath) {
				return nil, errors.Errorf("symlink target path escapes output directory: %s", linkTarget)
			}
			// Remove the target file if it exists.
			if err := os.Remove(targetPath); err != nil && !os.IsNotExist(err) {
				return nil, errors.Wrapf(err, "failed to remove existing file at symlink target %s", targetPath)
			}
			// Create the symlink
			if err := os.Symlink(linkTarget, targetPath); err != nil {
				return nil, errors.Wrapf(err, "failed to create symlink %s -> %s", targetPath, linkTarget)
			}
			extractedFiles = append(extractedFiles, targetPath)

		default:
			klog.Errorf("Skipping unsupported type: %c for file %s\n", header.Typeflag, header.Name)
		}
	}
	return extractedFiles, nil
}
