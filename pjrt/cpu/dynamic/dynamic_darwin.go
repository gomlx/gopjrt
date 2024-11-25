//go:build darwin

package dynamic

import "fmt"

func init() {
	fmt.Println("*** WARNING: Darwin(Mac) fails several tests while JIT-compiling if dynamically linking the PJRT CPU plugin. " +
		"Except if actually debugging this issue, consider the default statically linked CPU plugin ***")
}
