package main

import (
	"fmt"
	"regexp"
	"strings"
)

const (
	enumsFromCGoFileName = "gen_c_enums.go"
)

var (
	reEnums = regexp.MustCompile(
		`(?m)(typedef enum \{\n([^}]+)}` + // Enum definition
			`\s+(\w+)\s*;)`) // Enum type name
	reEnumComment    = regexp.MustCompile(`^\s*(//.*)$`)
	reEnumDefinition = regexp.MustCompile(`^\s*(\w+(\s*=\s*(\w+)?),)$`)
)

type enumTypeInfo struct {
	Name    string
	Entries []enumEntry
}

type enumEntry struct {
	Name         string
	Comments     []string
	Value        int
	ValueDefined bool
}

func generateEnums(contents string) {
	fmt.Printf("Enums:\n")
	var allEnums []enumTypeInfo
	for _, matches := range reEnums.FindAllStringSubmatch(contents, -1) {
		eType := enumTypeInfo{
			Name: matches[3],
		}
		eEntry := enumEntry{
			Comments: []string{""},
		}
		for _, line := range strings.Split(matches[3], "\n") {
			if line == "" {
				continue
			}

		}
		allEnums = append(allEnums, eType)
		fmt.Printf("Enum %s: \n%s\n", matches[3], matches[2])
	}
}
