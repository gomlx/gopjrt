// codegen parses the node_types.txt and generates boilerplate code both C and Go.
package main

import (
	"bufio"
	"fmt"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	"os"
	"strings"
)

const OpTypesFileName = "op_types.txt"

type OpInfo struct {
	Name, Type string
	Comments   []string
}

// panicf panics with formatted description.
//
// It is only used for "bugs in the code" -- when parameters don't follow the specifications.
// In principle, it should never happen -- the same way nil-pointer panics should never happen.
func panicf(format string, args ...any) {
	panic(errors.Errorf(format, args...))
}

func main() {
	// Read op_types.xt
	opsInfo := make([]OpInfo, 0, 200)
	f := must.M1(os.OpenFile(OpTypesFileName, os.O_RDONLY, os.ModePerm))
	scanner := bufio.NewScanner(f)
	lineNum := 0
	var comments []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		if line == "" || strings.HasPrefix(line, "#") {
			// Skip empty lines and reset comments
			comments = comments[:0]
			continue
		}
		if strings.HasPrefix(line, "//") {
			// Save but skip comments.
			line = strings.TrimSpace(line[2:])
			comments = append(comments, line)
			continue
		}
		parts := strings.Split(line, ":")
		if len(parts) != 2 {
			panicf("Invalid op definition in %q:%d : %q", OpTypesFileName, lineNum, line)
		}
		if len(comments) == 0 {
			comments = append(comments, fmt.Sprintf("%s returns the Op that represents the output of the corresponding operation.", parts[0]))
		}
		opsInfo = append(opsInfo, OpInfo{Name: parts[0], Type: parts[1], Comments: comments})
		comments = nil
	}
	must.M(scanner.Err())

	// Create various Go generate files.
	generateOpsEnums(opsInfo)
	GenerateSimpleGoOps(opsInfo)
}
