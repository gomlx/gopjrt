package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/pkg/errors"
)

func GetPipInfo(packageName string) (*PipPackageInfo, error) {
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

type PipDependency struct {
	Package                                         string
	Minimum, LowerBound, Maximum, UpperBound, Exact string
	Condition                                       string
}

// depRegex: Splits the raw string into Name, Specifiers, and Condition.
var depRegex = regexp.MustCompile(`^([\w.-]+)\s*([^;]*)?(?:;\s*(.+))?$`)

// specRegex: Captures individual version specifiers and their operators.
// Matches groups like: [Operator (==, <, >=, etc.)][Version Value]
var specRegex = regexp.MustCompile(`([<=>!~=]+)([^,]+)`)

func (info *PipPackageInfo) ParseDependencies() ([]PipDependency, error) {
	if info.Info.RequiresDist == nil {
		return nil, nil
	}

	var dependencies []PipDependency

	for _, rawDep := range info.Info.RequiresDist {
		rawDep = strings.TrimSpace(rawDep)
		matches := depRegex.FindStringSubmatch(rawDep)

		if len(matches) == 0 {
			return nil, fmt.Errorf("failed basic split of dependency string: %s", rawDep)
		}

		dep := PipDependency{
			Package:   matches[1],
			Condition: strings.TrimSpace(matches[3]),
		}

		// Extract and clean the full version specification string
		fullSpec := strings.TrimSpace(matches[2])
		if fullSpec != "" {
			// Remove the package name part if the initial regex captured it
			cleanSpec := strings.TrimPrefix(fullSpec, dep.Package)
			cleanSpec = strings.TrimSpace(cleanSpec)

			// Handle multiple comma-separated specifiers
			// Example: "<10.0,>=9.8"
			specParts := strings.Split(cleanSpec, ",")

			for _, part := range specParts {
				part = strings.TrimSpace(part)
				if part == "" {
					continue
				}

				specMatches := specRegex.FindStringSubmatch(part)
				if len(specMatches) < 3 {
					// This is still fragile; logging or ignoring complex/unmatched parts
					continue
				}

				operator := strings.TrimSpace(specMatches[1])
				versionValue := strings.TrimSpace(specMatches[2])

				// Assign the version value to the appropriate field based on the operator
				switch operator {
				case "==":
					dep.Exact = versionValue
				case ">=":
					dep.Minimum = versionValue
				case ">":
					dep.LowerBound = versionValue
				case "<=":
					dep.Maximum = versionValue
				case "<":
					dep.UpperBound = versionValue
				case "~=":
					// Compatible release operator (~=) often implies a Minimum bound
					dep.Minimum = versionValue
					// We ignore the '!=' (exclusive exclusion) operator for this simplified struct
				}
			}
		}

		dependencies = append(dependencies, dep)
	}

	return dependencies, nil
}

// PipReleaseInfo is the JSON response from pypi.org for a given package version, for some platform.
type PipReleaseInfo struct {
	// You can add more fields here if you need file-specific details
	PackageType string            `json:"packagetype"`
	Filename    string            `json:"filename"`
	URL         string            `json:"url"`
	Digests     map[string]string `json:"digests"`
}

func PipSelectRelease(releaseInfos []PipReleaseInfo, platform *regexp.Regexp) (*PipReleaseInfo, error) {
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
