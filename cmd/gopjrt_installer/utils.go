package main

import (
	"archive/zip"
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

// DownloadURLToTemp downloads a file from a given URL to a temporary file.
// It displays a spinner while downloading, and outputs some information about the download.
func DownloadURLToTemp(url, tempFileNamePattern string) (filePath string, err error) {
	// Download the asset to a temporary file
	var tmpFile *os.File
	tmpFile, err = os.CreateTemp("", tempFileNamePattern)
	if err != nil {
		return "", errors.Wrap(err, "failed to create temporary file")
	}
	filePath = tmpFile.Name()
	defer func() {
		// If an error was returned, remove the temporary file.
		if err != nil {
			ReportError(os.Remove(filePath))
		}
	}()

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
			defer resp.Body.Close()

			bytesDownloaded, err = io.Copy(tmpFile, resp.Body)
			if err != nil {
				err = errors.Wrapf(err, "failed to write asset %s to temporary file %s", url, tmpFile.Name())
				return
			}
			ReportError(tmpFile.Close())
		}).
		Run()
	if spinnerErr != nil {
		err = errors.Wrapf(spinnerErr, "failed run spinner for download from %s", url)
		return
	}
	if err != nil {
		return "", err
	}

	// Print the downloaded size
	bytesStr := ""
	switch {
	case bytesDownloaded < 1024:
		bytesStr = fmt.Sprintf("%d B", bytesDownloaded)
	case bytesDownloaded < 1024*1024:
		bytesStr = fmt.Sprintf("%.1f KB", float64(bytesDownloaded)/1024)
	case bytesDownloaded < 1024*1024*1024:
		bytesStr = fmt.Sprintf("%.1f MB", float64(bytesDownloaded)/(1024*1024))
	default:
		bytesStr = fmt.Sprintf("%.1f GB", float64(bytesDownloaded)/(1024*1024*1024))
	}
	fmt.Printf("- Downloaded %s to %s\n", bytesStr, filePath)
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

	return os.ErrNotExist // File not found in the archive
}

// extractZipFile is a helper to perform the actual extraction
func extractZipFile(f *zip.File, outputPath string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()

	// Create the output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	// Copy the contents
	_, err = io.Copy(outFile, rc)
	return err
}
