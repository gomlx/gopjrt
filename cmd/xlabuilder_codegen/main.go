// codegen parses the node_types.txt and generates boilerplate code both C and Go.
package main

import (
	"bufio"
	"github.com/janpfeifer/must"
	"os"
	"strings"
)

const OpTypesFileName = "op_types.txt"

func main() {
	// Read node_types.xt
	opTypeNames := make([]string, 0, 200)
	f := must.M1(os.OpenFile(OpTypesFileName, os.O_RDONLY, os.ModePerm))
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			// Skip empty lines
			continue
		}
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") {
			// Skip comments.
			continue
		}
		opTypeNames = append(opTypeNames, line)
	}
	must.M(scanner.Err())

	// Create various Go generate files.
	generateOpsEnums(opTypeNames)
}
