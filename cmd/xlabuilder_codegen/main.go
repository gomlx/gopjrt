// codegen parses the node_types.txt and generates boilerplate code both C and Go.
package main

import (
	"bufio"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/must"
	"os"
	"strings"
)

const OpTypesFileName = "op_types.txt"

type OpInfo struct {
	Name, Type string
}

func main() {
	// Read op_types.xt
	opsInfo := make([]OpInfo, 0, 200)
	f := must.M1(os.OpenFile(OpTypesFileName, os.O_RDONLY, os.ModePerm))
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		if line == "" {
			// Skip empty lines
			continue
		}
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") {
			// Skip comments.
			continue
		}
		parts := strings.Split(line, ":")
		if len(parts) != 2 {
			exceptions.Panicf("Invalid op definition in %q:%d : %q", OpTypesFileName, lineNum, line)
		}
		opsInfo = append(opsInfo, OpInfo{Name: parts[0], Type: parts[1]})
	}
	must.M(scanner.Err())

	// Create various Go generate files.
	generateOpsEnums(opsInfo)
	GenerateSimpleGoOps(opsInfo)
}
