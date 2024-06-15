package main

import (
	"bytes"
	"flag"
	"fmt"
	"github.com/janpfeifer/must"
	"io"
	"os"
	"path"
	"regexp"
	"strings"
)

var (
	flagOriginalGoFile = flag.String("original", "",
		"Original file name (or full path) to copy to the current directory. "+
			"If not an absolute path, the file is searched for in parent directories.")
	flagTargetGoFile = flag.String("target", "gen_{{original}}",
		"Target file name (not the path). It is written in the current directory. "+
			"The string `{{original}}` is replaced with the value set in --original. ")
	flagPackageName = flag.String("package", "", "Package name to use on copy. If empty uses current directory name.")
	flagPrefix      = flag.String("prefix", `/* DO NOT EDIT: this is a copy from {{original}} file */\n`,
		"Prefix text to include in copy. "+
			"The string `{{original}}` is replaced with the value set in --original. "+
			"The strings \\t and \\n are also replaced.")
)

func main() {
	flag.Parse()

	if *flagOriginalGoFile == "" {
		fmt.Printf("--original is required\n")
		flag.PrintDefaults()
		return
	}

	originalName := path.Base(*flagOriginalGoFile)
	targetName := strings.Replace(*flagTargetGoFile, "{{original}}", originalName, -1)
	prefix := strings.Replace(*flagPrefix, "{{original}}", originalName, -1)
	prefix = strings.Replace(prefix, `\t`, "\t", -1)
	prefix = strings.Replace(prefix, `\n`, "\n", -1)
	packageName := *flagPackageName
	if packageName == "" {
		packageName = path.Base(must.M1(os.Getwd()))
	}

	// Find original file
	originalPath := path.Base(*flagOriginalGoFile)
	if !path.IsAbs(originalPath) {
		originalPath = path.Join(must.M1(os.Getwd()), originalPath)
		for {
			_, err := os.Stat(originalPath)
			if err == nil {
				break // Found it.
			}

			if path.Dir(originalPath) == "/" {
				panic("Can't find original file " + originalName + " in any of the parent directories from the current directory.")
			}
			baseDir := path.Dir(path.Dir(originalPath))
			originalPath = path.Join(baseDir, originalName)
			continue
		}
	}

	// Read original file.
	f := must.M1(os.OpenFile(originalPath, os.O_RDONLY, os.ModePerm))
	var b bytes.Buffer
	_ = must.M1(io.Copy(&b, f))
	must.M(f.Close())
	contents := strings.Join([]string{prefix, b.String()}, "\n")

	// Replace package name.
	rePackage := regexp.MustCompile(`(?m)^package\s+\w+$`)
	contents = rePackage.ReplaceAllString(contents, fmt.Sprintf("package %s", packageName))

	// Write to target file.
	must.M(os.WriteFile(targetName, []byte(contents), 0644))

	fmt.Printf("Generated %q from %q, with package name %q\n", targetName, originalPath, packageName)
}
