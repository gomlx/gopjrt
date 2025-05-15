package main

import (
	"flag"
	"fmt"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"os"
	"path/filepath"
	"regexp"
)

var (
	flagXLARoot = flag.String("xla", "", "XLA source code root, where to search for proto files. "+
		"It is required, but if not defined it will try to read from the XLA_SRC environment variable.")

	reGoPackage = regexp.MustCompile(
		`(?m)(` +
			`^\s*option\s+go_package\s*=(\s*".*?")+([ \t]+)?;\s*?\n` + // Enum definition
			`)`) // Enum type name
)

func main() {
	// Override the default Usage function
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `strip_protos strips the "option go_package ..." entries from all "*.proto" files
under the root directory given by --xla or by the XLA_SRC environment variable.

Usage:
`)
		flag.PrintDefaults()
	}

	flag.Parse()
	xlaRoot := *flagXLARoot
	if xlaRoot == "" {
		xlaRoot = os.Getenv("XLA_SRC")
	}
	if xlaRoot == "" {
		klog.Fatal("Please provide XLA codebase root path through --xla, or by setting XLA_SRC env variable.")
	}

	// Find proto files.
	err := EnumerateFiles(xlaRoot, ".proto", func(filePath string) error {
		klog.V(1).Infof("Checking %q", filePath)
		contents, err := os.ReadFile(filePath)
		if err != nil {
			return err
		}
		matches := reGoPackage.FindStringSubmatch(string(contents))
		if len(matches) == 0 {
			return nil
		}
		fmt.Printf("Found in %q:\n\t%q\n\n", filePath, matches[0])
		contents = reGoPackage.ReplaceAll(contents, nil)

		fi := must.M1(os.Stat(filePath))
		must.M(os.Rename(filePath, filePath+"~"))
		must.M(os.WriteFile(filePath, contents, fi.Mode()))
		return nil
	})
	if err != nil {
		klog.Fatalf("Error: %+v", err)
	}
}

func EnumerateFiles(root, extension string, callback func(filePath string) error) error {
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() && filepath.Ext(path) == extension {
			return callback(path)
		}

		return nil
	})
	return err
}
