// codegen parses the node_types.txt and generates boilerplate code both C and Go.
package main

import (
	"bufio"
	"fmt"
	"github.com/janpfeifer/must"
	"os"
	"strings"
)

func main() {
	// Read node_types.xt
	nodeTypeNames := make([]string, 0, 200)
	f := must.M1(os.OpenFile("node_types.txt", os.O_RDONLY, os.ModePerm))
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
		nodeTypeNames = append(nodeTypeNames, line)
	}
	must.M(scanner.Err())
	fmt.Printf("Node types: %v\n", nodeTypeNames)

	// Create various Go generate files.
	generateNodeEnums(nodeTypeNames)
}
